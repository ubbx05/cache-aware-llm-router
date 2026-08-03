"""
semantic_worker_router.py
----------------------------
"Mini model" prototipi: bir istegin icerigini (retrieval calistirmadan ONCE)
anlayip, hangi worker'in cache'inin bu istekle semantik olarak en yakin
oldugunu tahmin eden aday-secim katmani.

NEREDE DEVREYE GIRIYOR (mevcut mimariye gore):

    Sorgu geldi
        |
        v
    [BU MODUL] Semantik aday tahmini (embedding benzerligi)
        |         -> candidate_worker_names (orn. en yakin 2 worker)
        v
    per_worker_tree_router.choose(candidate_worker_names=..., ...)
        |         -> CacheWeaver-stili reorder + skor karsilastirmasi
        v          (SADECE bu adaylar icin, n*O(k^2) yerine 2*O(k^2))
    secilen worker + sira

NEDEN GEREKLI (literatur taramasindan cikan bosluk):
    DualMap'in hash-tabanli aday secimi (Eq.5) icerigi hic anlamiyor --
    rastgele dagitiyor. RAGRoute (Efficient Federated Search, 2026) ve
    Adaptive-RAG gibi calismalar "retrieval'dan once ucuz bir siniflandirici
    calistir" fikrinin ISE YARADIGINI kanitlamis (RAGRoute <30ms'de), ama
    HICBIRI bunu "hangi worker'in cache'i alakali" sorusuna uygulamamis --
    hepsi "hangi veri kaynagi/strateji" seciyor, dagitik cache-affinity
    routing'e hic baglanmamis. Excel'deki 18 makalede de bu kombinasyon yok.

ONEMLI TASARIM NOTU -- dogru okuyun:
    Gercek sistemde embed() fonksiyonu, sizin pipeline'inizdaki gercek
    embedding modeliyle (multilingual-e5-base, gun-raporunuzda gecen)
    degistirilmeli. Burada, GPU/model indirme gerektirmeden test edilebilsin
    diye bagimsiz bir "hashing bag-of-words" vektorlestirici kullanildi --
    bu SADECE mekanizmayi dogrulamak icin, gercek performans/dogruluk
    sayisi URETMEZ. Bu ayrimi __main__ blogunda da ayrica isaretledim.

    `real_embed()` bu degisimin gercek hali -- multilingual-e5-base'i
    (bench/build_corpus.py ve bench/replay.py'de kullanilan ayni model)
    lazy-load eder, ayni "query: " onekini uygular (e5 ailesi bu oneki
    zorunlu kiliyor, atlanirsa benzerlik olculebilir sekilde bozuluyor --
    replay.py:211'deki ayni not gecerli). SemanticPerWorkerTreeStrategy
    (strategies.py) varsayilan olarak bunu kullanir; cheap_hashing_embed
    sadece bagimsiz/hizli test icin varsayilan constructor degeri olarak
    kaliyor.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Vector = List[float]


# ======================================================================
# Placeholder embedding -- SADECE test icin, gercek modelin yerine DEGIL
# ======================================================================
def cheap_hashing_embed(text: str, dim: int = 128, ngram: int = 3) -> Vector:
    """Bagimsizliksiz (dependency-free), deterministik bir 'hashing trick'
    KARAKTER n-gram vektorlestirici. Kelime-seviyesinde DEGIL, karakter
    n-gram seviyesinde calisiyor -- bunun nedeni Turkce gibi eklemeli
    dillerde ('bolunme' vs 'bolunmesi', 'mitoz' vs 'mayoz') kelime-tam-
    eslesmesinin cok kirilgan olmasi (ilk versiyonda bunu test ederken
    yakaladik -- kelime-bazli hash, farkli ek almis ayni kok icin sifir
    benzerlik veriyordu). Karakter n-gram'lar kismi/kok ortusmesini de
    yakaliyor.

    GERCEK KULLANIMDA BUNU DEGISTIRIN: bkz. real_embed() asagida.
    """
    vec = [0.0] * dim
    text = text.lower().strip()
    if len(text) < ngram:
        grams = [text] if text else []
    else:
        grams = [text[i:i + ngram] for i in range(len(text) - ngram + 1)]
    if not grams:
        return vec
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: Vector, b: Vector) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (norm_a * norm_b)


# ======================================================================
# Gercek embedding -- multilingual-e5-base (lazy-loaded, bench/ ile ayni model)
# ======================================================================
_e5_model_cache: Dict[str, object] = {}


def real_embed(text: str, model_name: str = "intfloat/multilingual-e5-base") -> Vector:
    """Gercek semantik embedding. sentence-transformers'i sadece ilk
    cagrida import eder (prefix_tracker.HFTokenizer ile ayni "lazy import"
    deseni) -- boylece bu modulu sadece cheap_hashing_embed ile kullanan
    kod (or. self-test'in ilk yarisi) sentence-transformers'i hic yuklemek
    zorunda kalmaz.

    Model isim-bazli cache'leniyor: ayni surecte birden fazla
    SemanticWorkerRouter ayni model_name ile kurulursa (veya self-test iki
    kez cagirirsa) ikinci kez GPU/CPU'ya yuklenmez.
    """
    model = _e5_model_cache.get(model_name)
    if model is None:
        from sentence_transformers import SentenceTransformer  # lazy, opsiyonel bagimlilik

        model = SentenceTransformer(model_name)
        _e5_model_cache[model_name] = model

    # e5 ailesi "query: " onekini zorunlu kiliyor -- bkz. bench/replay.py:211
    vec = model.encode([f"query: {text}"], normalize_embeddings=True,
                       convert_to_numpy=True)[0]
    return vec.tolist()


# ======================================================================
# Worker basina "merkez nokta" (centroid) takibi -- EWMA ile guncellenir
# ======================================================================
@dataclass
class WorkerCentroidTracker:
    """Her worker'in YAKIN ZAMANDA islediÄi sorgularin 'ortalama' anlamsal
    konumunu (centroid) tutar. adaptive_drift_model.py'deki EWMA deseniyle
    AYNI mantik -- burada da 'lr' (learning rate) ne kadar hizli unutulacagini
    belirliyor, boylece worker'in konu dagilimi zamanla kayarsa (drift)
    centroid de onu takip ediyor.

    Cold-start: bir worker'in henuz hic sorgusu yoksa centroid None kalir --
    bu worker semantik olarak "notr" sayilir (ne avantajli ne dezavantajli),
    yeni/bos worker'lari ac birakmamak icin ONEMLI bir guvenlik onlemi.
    """

    lr: float = 0.15
    _centroids: Dict[str, Optional[Vector]] = field(default_factory=dict)
    _num_updates: Dict[str, int] = field(default_factory=dict)

    def update(self, worker_name: str, query_vec: Vector) -> None:
        current = self._centroids.get(worker_name)
        if current is None:
            self._centroids[worker_name] = list(query_vec)
        else:
            self._centroids[worker_name] = [
                (1 - self.lr) * c + self.lr * q for c, q in zip(current, query_vec)
            ]
        self._num_updates[worker_name] = self._num_updates.get(worker_name, 0) + 1

    def centroid(self, worker_name: str) -> Optional[Vector]:
        return self._centroids.get(worker_name)

    def is_cold(self, worker_name: str) -> bool:
        return self._centroids.get(worker_name) is None

    def num_updates(self, worker_name: str) -> int:
        return self._num_updates.get(worker_name, 0)


# ======================================================================
# Aday secim mantigi
# ======================================================================
@dataclass
class CandidatePrediction:
    ranked_workers: List[str]           # en alakalidan en alakasiza
    similarities: Dict[str, float]      # her worker icin benzerlik skoru
    cold_workers: List[str]             # centroid'i henuz olusmamis worker'lar
    confidence: float                   # top-1 ile top-2 arasindaki fark (margin)


class SemanticWorkerRouter:
    """Ana sinif: sorgu embedding'ini worker centroid'leriyle karsilastirip
    aday siralamasi uretir. Bu, DualMap'in hash-tabanli aday secimine (Eq.5)
    bir ALTERNATIF ya da EK sinyal olarak kullanilabilir."""

    def __init__(self, worker_names: Sequence[str], embed_fn=cheap_hashing_embed,
                 lr: float = 0.15):
        self._embed_fn = embed_fn
        self._tracker = WorkerCentroidTracker(lr=lr)
        for name in worker_names:
            self._tracker._centroids.setdefault(name, None)

    def predict_candidates(self, query_text: str, top_k: int = 2) -> CandidatePrediction:
        query_vec = self._embed_fn(query_text)
        sims: Dict[str, float] = {}
        cold: List[str] = []

        # Once bilinen (cold olmayan) worker'larin gercek benzerliklerini hesapla
        known_sims: Dict[str, float] = {}
        for name in self._tracker._centroids.keys():
            centroid = self._tracker.centroid(name)
            if centroid is None:
                cold.append(name)
            else:
                known_sims[name] = cosine_similarity(query_vec, centroid)

        # Cold worker'lara "iyimser ortalama" ata (bandit'lerdeki "belirsizlik
        # karsisinda iyimserlik" ilkesi): bilinen worker'larin ORTALAMA
        # benzerligini ver -- ne cezalandirilsin (sabit 0.0, cogu zaman
        # gercek zayif eslesmelerden bile dusuk olabiliyordu) ne de
        # otomatik favori olsun (sabit yuksek deger). Hicbir known worker
        # yoksa (baslangictaki gibi) hepsi esit (0.0) kalir, siralama
        # anlamsiz olur ama en azindan tum worker'lar esit sansli olur.
        optimistic_default = (
            sum(known_sims.values()) / len(known_sims) if known_sims else 0.0
        )
        for name in cold:
            sims[name] = optimistic_default
        sims.update(known_sims)

        ranked = sorted(sims.keys(), key=lambda n: sims[n], reverse=True)
        candidates = ranked[:top_k]
        # ARTIK zorla degistirme YOK -- cold worker'lar dogal olarak
        # siralamaya optimistic_default skoruyla giriyor, gercek bir
        # eslesmeyi asla otomatik olarak ezmiyor.

        confidence = 0.0
        if len(ranked) >= 2:
            confidence = sims[ranked[0]] - sims[ranked[1]]

        return CandidatePrediction(
            ranked_workers=candidates,
            similarities=sims,
            cold_workers=cold,
            confidence=round(confidence, 4),
        )

    def on_request_finished(self, worker_name: str, query_text: str) -> None:
        """Dispatch-time convention (diger modullerle tutarli): istek
        tamamlaninca, o worker'in centroid'ini bu sorguyla guncelle."""
        query_vec = self._embed_fn(query_text)
        self._tracker.update(worker_name, query_vec)


# ======================================================================
# Kendi-kendini-test: iki "konu kumesi" simule edip, sistemin doÄru
# worker'i tahmin ettigini dogrular + cold-start davranisini kontrol eder.
# ======================================================================
def _run_self_test(embed_fn, label: str) -> None:
    router = SemanticWorkerRouter(worker_names=["w1", "w2", "w3"], embed_fn=embed_fn)

    topic_a_queries = [
        "Osmanli Imparatorlugu nasil kuruldu",
        "Fatih Sultan Mehmet Istanbul'u nasil fethetti",
        "Osmanli padisahlari kimlerdir",
    ]
    topic_b_queries = [
        "hucre bolunmesi nasil gerceklesir",
        "mitoz ve mayoz farki nedir",
        "DNA replikasyonu nasil olur",
    ]

    print(f"\n########## {label} ##########")

    print("=== Cold-start testi ===")
    pred_cold = router.predict_candidates("Osmanli tarihi hakkinda bir soru", top_k=2)
    print(f"Hicbir worker'da veri yokken tahmin: {pred_cold.ranked_workers}")
    print(f"Cold worker'lar: {pred_cold.cold_workers}")
    assert len(pred_cold.cold_workers) == 3, "Baslangicta tum worker'lar cold olmali"

    print("\n=== Egitim: w1'e konu-A, w2'ye konu-B sorgulari isleniyor ===")
    for q in topic_a_queries:
        router.on_request_finished("w1", q)
    for q in topic_b_queries:
        router.on_request_finished("w2", q)
    # w3 kasitli olarak hic veri almiyor -- cold-start guvenligini test etmek icin

    print("\n=== Test 1: yeni bir konu-A sorgusu ===")
    pred_a = router.predict_candidates("Osmanli sultanlarinin listesi nedir", top_k=2)
    print(f"Siralama: {pred_a.ranked_workers}")
    print(f"Benzerlikler: {pred_a.similarities}")
    print(f"Guven (top1-top2 farki): {pred_a.confidence}")
    assert pred_a.ranked_workers[0] == "w1", "Konu-A sorgusu w1'i ilk sirada onermeli"

    print("\n=== Test 2: yeni bir konu-B sorgusu ===")
    pred_b = router.predict_candidates("mayoz bolunme evreleri nelerdir", top_k=2)
    print(f"Siralama: {pred_b.ranked_workers}")
    print(f"Benzerlikler: {pred_b.similarities}")
    assert pred_b.ranked_workers[0] == "w2", "Konu-B sorgusu w2'yi ilk sirada onermeli"

    print("\n=== Test 3: w3 (cold) hala aday havuzunda mi (ac birakmama kontrolu) ===")
    print(f"Test 1'deki adaylar w3'u iceriyor mu: {'w3' in pred_a.ranked_workers}")
    # top_k=2 oldugu icin w1 kesin ilk sirada, w3 cold oldugu icin ikinci
    # adaya girme sansi olmali (w2'nin konu-A ile ilgisiz olmasi sayesinde)

    print(f"\n{label}: self-test PASSED.")


if __name__ == "__main__":
    _run_self_test(cheap_hashing_embed, "cheap_hashing_embed (bagimsiz, karakter n-gram)")

    print("\nNOT: Yukaridaki testte 'cheap_hashing_embed' kullanildi -- gercek")
    print("anlamsal benzerlik degil, kelime/kok ortusmesi yakaliyor.")
    print("Simdi AYNI senaryo real_embed() (multilingual-e5-base) ile tekrarlaniyor")
    print("-- Turkce konu ayriminin gercek embedding ile de dogru calistigini")
    print("dogrulamak icin (sentence-transformers + model indirmesi gerektirir).")

    try:
        _run_self_test(real_embed, "real_embed (multilingual-e5-base)")
    except ImportError as exc:
        print(f"\nreal_embed testi ATLANDI -- sentence-transformers kurulu degil ({exc}).")
        print("pip install sentence-transformers ile kurup tekrar calistirin.")
