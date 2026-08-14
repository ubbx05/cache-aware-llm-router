"""
per_worker_tree_router.py
--------------------------
"Worker-basina agac + iki-aday reorder" tasarimi.

Motivasyon: cacheweaver_dualmap_router.py'nin dual-hash aday secimi n=2'de
matematiksel olarak HER ZAMAN {0,1} uretiyor (kanitlandi -- duzeltme kurali
r1==r2 ise r2=(r1+1) mod n, n=2'de bu her zaman tum cluster'i verir). Yani
consistent-hashing makinesi bu rejimde geregsiz. Bu modul onun yerine
DOGRUDAN bir tasarim sunuyor: n kac olursa olsun, her worker icin "bu
worker'in KENDI cache durumuna gore en iyi siralama ne olurdu" hesaplanir,
hangi worker+siralama en yuksek skoru veriyorsa o secilir. Hash yok, aday
kismi yok -- direkt karsilastirma.

Fark, cacheweaver_dualmap_router.py'nin duzeltilmis halinden (replica-basina
agac) su noktada: orada TEK bir paylasilan agacla (Algoritma 1) tek bir
siralama uretilip SONRA hangi worker'in o sirayi daha iyi tuttuguna
bakiliyordu. Burada HER worker KENDI agacina gore KENDI en iyi siralamasini
uretiyor -- yani "dogru sira" worker'dan bagimsiz tek bir sey degil, worker
secimiyle birlikte ortaya cikiyor.

=======================================================================
MIMARI NOT -- kodu calistirmadan once mutlaka okuyun (bkz. 2026-07-28/29
sohbeti, "worker basina agac" fikri):
=======================================================================
Bu modul SADECE karar mantigini icerir. main.py'ye baktim (main.py:141-149,
_choose()): prompt metni (_prompt_text) ve ondan turetilen hash'ler,
strategy.select() cagrilmadan ONCE sabitleniyor -- yani su an bu modulun
urettigi `ordered_chunk_ids` gercek main.py akisinda vLLM'e giden prompt'u
ETKILEMIYOR. Ayrica ctx.chunk_hashes main.py tarafindan hic doldurulmuyor
(hep None geliyor, RequestContext'teki default). Gercek entegrasyon icin:

  1. main.py, RAG asamasinda retrieval'dan donen chunk_id listesini
     (SIRALANMAMIS, retrieval sirasinda) ctx.chunk_hashes'e koymali.
  2. strategies.py'deki Decision'a `ordered_chunk_ids: list[str] | None`
     eklendi (asagida) -- main.py bu alani OKUYUP, secilen worker icin
     donen sirayla prompt'u (yeniden) insa edip OYLE vLLM'e gondermeli.
     Su an main.py bu alani hic okumuyor.
  3. replay.py, --order canonical/relevance ile ONCEDEN sirali, bitmis bir
     prompt gonderiyor. Bu tasarimin test edilebilmesi icin replay.py'nin
     HAM (siralanmamis) chunk kimliklerini gonderip sirayi router'a
     birakmasi gerekiyor -- yani --order flag'i bu strateji icin anlamsiz
     hale geliyor, ayri bir yol gerekiyor.

Yani: mantik burada, gercek trafige bagli degil. Asagidaki __main__ blogu
GPU/main.py olmadan mekanizmayi dogruluyor -- gercek uctan-uca test ayri bir
is (yukaridaki 3 degisiklik).

=======================================================================
TOKEN-BAZLI cache_gain -- ikinci bir bilinen sinirlama (duzeltildi, ama
tam olarak degil -- asagidaki paragrafi okuyun):
=======================================================================
Eski hata: gain = hit_len / len(retrieved_chunk_ids), yani hit_len CHUNK
SAYISINI sayiyordu, token uzunluklarini degil. build_corpus.py'nin
urettigi corpus'ta chunk'lar SABIT uzunlukta DEGIL (dogal TQuAD paragraf
uzunlugu korunuyor, sadece block-size'a padleniyor -- olculdu: 2619
chunk, 9-754 token arasi, ortalama 182, std 108). Yani "3 kucuk chunk"
"2 buyuk chunk"tan chunk-SAYISI olarak fazla ama token olarak cok daha
az prefill tasarrufu saglayabiliyor -- eski kod bunu yanlis tarafa
karar veriyordu.

Duzeltme: best_reorder_for/choose artik OPSIYONEL bir chunk_texts:
dict[str, str] (chunk_id -> gercek chunk metni) parametresi aliyor.
Verilirse, prefix_tracker.get_tokenizer() (YENIDEN KULLANILIYOR, yeni
tokenizer YUKLENMIYOR) ile her chunk_id'nin GERCEK token sayisi
hesaplanir (instance-basina memoize edilir) ve gain artik
hit_tokens/total_tokens olur.

DURUSTCE ISARETLENMESI GEREKEN SINIRLAMA: router'in KENDISI chunk
metnini hic bilmiyor -- retrieval ve corpus SADECE client-side'da
(bench/replay.py) var. main.py'nin /router/decide_order endpoint'i su an
SADECE chunk_ids gonderiyor (main.py:354, chunk_texts yok), yani
strategies.py'deki PerWorkerTreeStrategy.decide_order() bu fonksiyonlari
chunk_texts=None ile cagiriyor. O yolla gelen her cagri icin butun
chunk'lar 1 token agirlikli sayilir -- yani CANLI main.py akisinda
davranis eskisiyle (chunk-sayisi) AYNI kaliyor, sessizce degil, acikca
boyle. Token-agirlikli hesap su an SADECE bu modulun kendi
self-test'inde (asagida, Senaryo 4) chunk_texts verilerek dogrulaniyor.
main.py/replay.py'nin /router/decide_order sozlesmesini chunk_texts
tasiyacak sekilde genisletmek AYRI bir is (semantic_worker_router.py'nin
query_text sinirlamasiyla ayni desen -- bkz. strategies.py
SemanticPerWorkerTreeStrategy docstring'i).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from cacheweaver_util import CacheWeaverKnowledgeTree
from prefix_tracker import get_tokenizer


@dataclass
class PerWorkerDecision:
    worker_name: str
    ordered_chunk_ids: List[str]
    cache_gain: float                        # [0,1], secilen worker'in KENDI agacina gore
    scores: Dict[str, float]                 # butun adaylarin skoru (debug/log)
    all_orderings: Dict[str, List[str]] = field(default_factory=dict)
    hit_tokens: int = 0                      # secilen worker icin: cache'ten karsilanan token sayisi
    total_tokens: int = 0                    # secilen worker icin: retrieve edilen toplam token sayisi


class PerWorkerTreeRouter:
    """Her worker kendi CacheWeaverKnowledgeTree'sini tutar. Yeni istek
    geldiginde HER healthy worker icin, o worker'in kendi agacina gore en iyi
    siralama + normalize cache_gain hesaplanir; skor = alpha*cache_gain -
    beta*load_norm -- strategies.py'deki CacheAware ile AYNI formul, ayni
    config.ALPHA/BETA/LOAD_REF ile cagrilmasi tutarlilik icin onerilir (bu
    modul kendi sabitini tanimlamiyor, parametre olarak aliyor)."""

    def __init__(self, worker_names: Sequence[str], cache_ttl_seconds: float = 30.0):
        self._trees: Dict[str, CacheWeaverKnowledgeTree] = {
            name: CacheWeaverKnowledgeTree(cache_ttl_seconds=cache_ttl_seconds)
            for name in worker_names
        }
        # chunk_id -> token count, memoize edilir (bkz. modul docstring'i,
        # "TOKEN-BAZLI cache_gain"). Instance-basina: farkli PerWorkerTreeRouter
        # ornekleri (or. testlerde) birbirinin cache'ini kirletmemeli.
        self._token_cache: Dict[str, int] = {}
        self._tokenizer = None  # lazy -- sadece chunk_texts VERILIRSE yuklenir

    def _token_counts(
        self, chunk_ids: Sequence[str], chunk_texts: Optional[Dict[str, str]]
    ) -> Dict[str, int]:
        """chunk_id -> token sayisi. chunk_texts (id->metin) verilmisse
        prefix_tracker.get_tokenizer() ile GERCEK token sayisini hesaplar
        (YENIDEN KULLANILIYOR -- main.py de ayni get_tokenizer() ornegini
        kullaniyor, burada AYRI bir tokenizer YUKLENMIYOR). Metni olmayan
        (veya chunk_texts hic verilmemis) id'ler icin 1 token varsayar --
        bu, eski "chunk sayisi" davranisiyla TAM olarak ayni sonucu verir,
        yani chunk_texts=None iken bu fonksiyon eski koda geriye donuk
        %100 esdegerdir."""
        counts: Dict[str, int] = {}
        for cid in chunk_ids:
            text = (chunk_texts or {}).get(cid)
            if text is None:
                # Fallback (1 token) is NEVER written to the cache -- otherwise
                # a call made without chunk_texts would permanently poison a
                # later call FOR THE SAME chunk_id that does supply real text
                # (bit us in Senaryo 4's self-test: the chunk-count baseline
                # call ran first, cached every id at 1, and the token-based
                # call that followed silently reused those 1s). Reading from
                # the cache here is still safe/desirable: if a real count was
                # already measured for this id, use it instead of guessing 1.
                counts[cid] = self._token_cache.get(cid, 1)
                continue
            if cid in self._token_cache:
                counts[cid] = self._token_cache[cid]
                continue
            if self._tokenizer is None:
                self._tokenizer = get_tokenizer()
            n = len(self._tokenizer.encode(text))
            self._token_cache[cid] = n
            counts[cid] = n
        return counts

    def _cache_hit_tokens(
        self,
        tree: CacheWeaverKnowledgeTree,
        ordered_chunk_ids: Sequence[str],
        token_counts: Dict[str, int],
    ) -> int:
        """greedy_reorder zaten en uzun onbelleklenmis yolu kullanarak
        siraliyor ama depth'i geri dondurmuyor -- ayni agactan, uretilen
        siradan tekrar yururuz (cacheweaver_dualmap_router.py'deki
        _approx_cache_hit_len ile ayni desen, kasitli olarak tekrar
        kullanildi ki iki modul arasinda tutarsizlik olmasin). Fark:
        artik matched chunk SAYISI degil, matched chunk'larin TOKEN
        toplami donuyor (bkz. modul docstring'i)."""
        node = tree._root
        hit_tokens = 0
        for chunk_id in ordered_chunk_ids:
            child = node.get_child(chunk_id)
            if child is None or not tree._is_cached(child):
                break
            node = child
            hit_tokens += token_counts.get(chunk_id, 1)
        return hit_tokens

    def best_reorder_for(
        self,
        worker_name: str,
        retrieved_chunk_ids: Sequence[str],
        chunk_texts: Optional[Dict[str, str]] = None,
        protect_top_k: int = 0,
    ) -> Tuple[List[str], float, int, int]:
        """Tek bir worker icin: o worker'in kendi agacina gore en iyi sira +
        TOKEN-bazli normalize cache_gain (chunk_texts verilmisse gercek
        token sayilariyla, verilmezse eski chunk-sayisi davranisiyla ayni).
        Doner: (ordered_chunk_ids, gain, hit_tokens, total_tokens).

        protect_top_k: cacheweaver_util.greedy_reorder'a oldugu gibi
        gecirilir -- retrieval sirasindaki ilk K chunk'i cache durumuna
        bakmadan olmasi gereken yerde birakir (kalite-koruma, bkz.
        cacheweaver_util.py). K=0 (varsayilan): eski davranis, degismez."""
        tree = self._trees[worker_name]
        ordered = tree.greedy_reorder(list(retrieved_chunk_ids), protect_top_k=protect_top_k)
        token_counts = self._token_counts(retrieved_chunk_ids, chunk_texts)
        hit_tokens = self._cache_hit_tokens(tree, ordered, token_counts)
        total_tokens = sum(token_counts.get(c, 1) for c in retrieved_chunk_ids)
        gain = hit_tokens / total_tokens if total_tokens else 0.0
        return ordered, gain, hit_tokens, total_tokens

    def choose(
        self,
        candidate_worker_names: Sequence[str],
        retrieved_chunk_ids: Sequence[str],
        load_norm: Dict[str, float],
        alpha: float = 1.0,
        beta: float = 1.0,
        chunk_texts: Optional[Dict[str, str]] = None,
        protect_top_k: int = 0,
    ) -> PerWorkerDecision:
        """Her aday worker icin ayri best_reorder_for cagirir, skorlari
        karsilastirir, en iyisini doner.

        Maliyet notu: n aday icin n*O(k^2) -- n=2'de ucuz (2*k^2), n
        buyudukce (>~5-10) pahali olabilir; bu tasarim kucuk-n rejimi icin
        dusunuldu (senin projenin tam odagi), buyuk cluster'a genellenmeden
        once maliyet olcumu yapilmali.
        """
        if not candidate_worker_names:
            raise ValueError("candidate_worker_names bos olamaz")

        orderings: Dict[str, List[str]] = {}
        gains: Dict[str, float] = {}
        hit_tok: Dict[str, int] = {}
        total_tok: Dict[str, int] = {}
        for name in candidate_worker_names:
            ordered, gain, ht, tt = self.best_reorder_for(
                name, retrieved_chunk_ids, chunk_texts, protect_top_k
            )
            orderings[name] = ordered
            gains[name] = gain
            hit_tok[name] = ht
            total_tok[name] = tt

        scores = {
            name: alpha * gains[name] - beta * load_norm.get(name, 0.0)
            for name in candidate_worker_names
        }

        # DUZELTME (kendi max()/w1-yanliligi bugumuzu tekrar etmemek icin):
        # plain max(scores, key=scores.get) ilk maksimumu doner, worker
        # listesi hep ayni sirada oldugu icin beraberlikte hep ilk worker
        # kazanir. Rastgele kirilim, strategies.py'deki _argmax ile ayni.
        best = max(scores.values())
        best_name = random.choice([n for n, v in scores.items() if v >= best - 1e-12])

        return PerWorkerDecision(
            worker_name=best_name,
            ordered_chunk_ids=orderings[best_name],
            cache_gain=gains[best_name],
            scores=scores,
            all_orderings=orderings,
            hit_tokens=hit_tok[best_name],
            total_tokens=total_tok[best_name],
        )

    def on_request_finished(self, worker_name: str, ordered_chunk_ids: Sequence[str]) -> None:
        """Dispatch-time convention (bkz. cacheweaver_dualmap_router.py'deki
        ayni isimli metod) -- ayni yerde, ayni zamanlamada cagrilmali, yoksa
        cacheweaver_dualmap ile bu strateji arasindaki karsilastirma da
        timing farkindan kirlenir."""
        self._trees[worker_name].insert(list(ordered_chunk_ids))


# ----------------------------------------------------------------------
# Kendi-kendini-test (GPU/main.py gerektirmez)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    router = PerWorkerTreeRouter(["w1", "w2"])

    # Senaryo 1: w1 daha once {A,B,C,D} servis etmis, w2'nin agaci bos.
    # Yuk esitse (load_norm=0), w1'in cache_gain'i kazanmali.
    router.on_request_finished("w1", ["A", "B", "C", "D"])

    d1 = router.choose(
        candidate_worker_names=["w1", "w2"],
        retrieved_chunk_ids=["B", "A", "C", "F", "D"],
        load_norm={"w1": 0.0, "w2": 0.0},
        alpha=1.0, beta=1.0,
    )
    print("Senaryo 1 (esit yuk, w1'de cache var):")
    print(f"  secilen worker : {d1.worker_name}")
    print(f"  cache_gain     : {d1.cache_gain:.2f}")
    print(f"  siralama       : {d1.ordered_chunk_ids}")
    print(f"  skorlar        : {d1.scores}")
    assert d1.worker_name == "w1", "cache avantaji varken w1 kazanmali"
    assert d1.cache_gain > 0.0

    # Senaryo 2: ayni istek ama w1 asiri yuklu (load_norm=0.9). Cache
    # avantajini yuk dengelemeli/gecmeli -- ALPHA=BETA=1.0'da skor:
    #   w1: 1*0.8 - 1*0.9 = -0.1   w2: 1*0.0 - 1*0.0 = 0.0  -> w2 kazanmali
    d2 = router.choose(
        candidate_worker_names=["w1", "w2"],
        retrieved_chunk_ids=["B", "A", "C", "F", "D"],
        load_norm={"w1": 0.9, "w2": 0.0},
        alpha=1.0, beta=1.0,
    )
    print("\nSenaryo 2 (w1 asiri yuklu, cache avantajina ragmen):")
    print(f"  secilen worker : {d2.worker_name}")
    print(f"  skorlar        : {d2.scores}")
    assert d2.worker_name == "w2", "yuk farki cache avantajini gecmeli"

    # Senaryo 3: w2'ye dispatch-time kayit -- artik onun da agacinda bir
    # sey var, iki worker'in birbirinden bagimsiz durumlarini dogrular.
    router.on_request_finished(d2.worker_name, d2.ordered_chunk_ids)
    d3 = router.choose(
        candidate_worker_names=["w1", "w2"],
        retrieved_chunk_ids=["A", "B", "C", "F", "D"],
        load_norm={"w1": 0.0, "w2": 0.0},
        alpha=1.0, beta=1.0,
    )
    print("\nSenaryo 3 (w2'nin de artik kendi cache'i var):")
    print(f"  secilen worker : {d3.worker_name}")
    print(f"  cache_gain     : {d3.cache_gain:.2f}")

    # Senaryo 4: token-bazli gain, chunk-sayisi-bazli gainden FARKLI bir
    # worker secmeli. w1 az-ama-BUYUK chunk'lari cache'liyor (2 chunk, cok
    # token); w2 cok-ama-KUCUK chunk'lari cache'liyor (3 chunk, az token).
    # Eski (chunk-sayisi) kod 3>2 oldugu icin w2'yi secerdi -- ama gercek
    # prefill tasarrufu w1'de cok daha buyuk. Ayni senaryoyu chunk_texts
    # VERMEDEN (eski davranis) tekrar kosup iki sonucun FARKLI oldugunu
    # kanitliyoruz.
    router2 = PerWorkerTreeRouter(["w1", "w2"])
    router2.on_request_finished("w1", ["BIG1", "BIG2"])
    router2.on_request_finished("w2", ["small1", "small2", "small3"])

    retrieved_4 = ["BIG1", "BIG2", "small1", "small2", "small3"]
    chunk_texts_4 = {
        # BIG1+BIG2 toplam token sayisi, small1+small2+small3 toplaminin
        # kat kat uzerinde olacak sekilde kasitli uzun -- oran, tokenizer
        # approx/hf farketmeksizin ayni yonde kalsin diye buyuk tutuldu.
        "BIG1": ("Osmanli Imparatorlugu'nun kurulusu, genisleme donemi ve saray "
                "teskilati hakkinda uzun bir tarihsel aciklama metni. ") * 30,
        "BIG2": "Fatih Sultan Mehmet'in Istanbul'u fethi hakkinda orta uzunlukta bir metin. " * 6,
        "small1": "kisa not",
        "small2": "kisa not",
        "small3": "kisa not",
    }

    d4_old = router2.choose(
        candidate_worker_names=["w1", "w2"],
        retrieved_chunk_ids=retrieved_4,
        load_norm={"w1": 0.0, "w2": 0.0},
        alpha=1.0, beta=1.0,
        # chunk_texts=None -- eski (chunk-sayisi) davranisi, geriye donuk uyumluluk
    )
    d4_new = router2.choose(
        candidate_worker_names=["w1", "w2"],
        retrieved_chunk_ids=retrieved_4,
        load_norm={"w1": 0.0, "w2": 0.0},
        alpha=1.0, beta=1.0,
        chunk_texts=chunk_texts_4,
    )

    print("\nSenaryo 4 (az-ama-buyuk chunk vs cok-ama-kucuk chunk):")
    print(f"  chunk_texts=None  (eski, chunk-sayisi) -> secilen: {d4_old.worker_name}  "
          f"gain(w1)={d4_old.scores['w1']:.2f} gain(w2)={d4_old.scores['w2']:.2f}")
    print(f"  chunk_texts=verildi (yeni, token-bazli) -> secilen: {d4_new.worker_name}  "
          f"gain(w1)={d4_new.scores['w1']:.2f} gain(w2)={d4_new.scores['w2']:.2f}  "
          f"hit_tokens={d4_new.hit_tokens}/{d4_new.total_tokens}")
    assert d4_old.worker_name == "w2", (
        "eski (chunk-sayisi) davranis w2'yi secmeliydi (3 chunk > 2 chunk) -- "
        "beklenti degisti, senaryoyu kontrol et"
    )
    assert d4_new.worker_name == "w1", (
        "token-bazli gain w1'i secmeliydi (1100+ token >> 300- token) -- "
        "token agirliklandirma calismiyor olabilir"
    )
    assert d4_old.worker_name != d4_new.worker_name, (
        "token-bazli ve chunk-sayisi-bazli hesap AYNI worker'i secti -- "
        "bu senaryo bir fark kanitlamiyor, chunk uzunluk oranini buyut"
    )
    assert d4_new.hit_tokens > len(["BIG1", "BIG2"]), (
        "hit_tokens hala chunk SAYISI gibi davraniyor (2), gercek token "
        "toplami degil"
    )

    # Senaryo 5: protect_top_k -- retrieval'da EN ALAKALI (ilk) chunk cache'de
    # DEGILKEN, saf greedy reorder onu geriye itebiliyor (cacheweaver_util.py
    # docstring'indeki risk). protect_top_k=1 bunu engellemeli; K=0 eski
    # (korumasiz) davranisi degistirmemeli.
    router5 = PerWorkerTreeRouter(["w1", "w2"])
    router5.on_request_finished("w1", ["X", "Y", "Z"])

    retrieved_5 = ["Q", "X", "Y", "Z"]  # Q en alakali (retrieval sirasinda ilk) ama cache'de degil

    d5_k0 = router5.choose(
        candidate_worker_names=["w1"], retrieved_chunk_ids=retrieved_5,
        load_norm={"w1": 0.0}, alpha=1.0, beta=1.0, protect_top_k=0,
    )
    d5_k1 = router5.choose(
        candidate_worker_names=["w1"], retrieved_chunk_ids=retrieved_5,
        load_norm={"w1": 0.0}, alpha=1.0, beta=1.0, protect_top_k=1,
    )

    print("\nSenaryo 5 (protect_top_k -- en alakali ama cache'de olmayan chunk'i koru):")
    print(f"  K=0 (korumasiz) siralama: {d5_k0.ordered_chunk_ids}")
    print(f"  K=1 (Q korunuyor)  siralama: {d5_k1.ordered_chunk_ids}")
    assert d5_k0.ordered_chunk_ids[0] != "Q", (
        "K=0'da Q'nun one gecmemesi beklenirdi (saf greedy reorder onu geriye "
        "itiyor olmali) -- senaryo degisti, kontrol et"
    )
    assert d5_k1.ordered_chunk_ids[0] == "Q", (
        "K=1 iken Q ILK SIRADA kalmali -- protect_top_k dogru calismiyor olabilir"
    )
    assert set(d5_k1.ordered_chunk_ids) == set(retrieved_5), (
        "protect_top_k KUMEYI degistirmemeli, sadece sirayi"
    )
    print("  OK -- K=0'da Q geriye itiliyor, K=1'de korunuyor")

    print("\nSelf-test PASSED.")
