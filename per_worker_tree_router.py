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
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from cacheweaver_util import CacheWeaverKnowledgeTree


@dataclass
class PerWorkerDecision:
    worker_name: str
    ordered_chunk_ids: List[str]
    cache_gain: float                        # [0,1], secilen worker'in KENDI agacina gore
    scores: Dict[str, float]                 # butun adaylarin skoru (debug/log)
    all_orderings: Dict[str, List[str]] = field(default_factory=dict)


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

    def _cache_hit_len(
        self, tree: CacheWeaverKnowledgeTree, ordered_chunk_ids: Sequence[str]
    ) -> int:
        """greedy_reorder zaten en uzun onbelleklenmis yolu kullanarak
        siraliyor ama depth'i geri dondurmuyor -- ayni agactan, uretilen
        siradan tekrar yururuz (cacheweaver_dualmap_router.py'deki
        _approx_cache_hit_len ile ayni desen, kasitli olarak tekrar
        kullanildi ki iki modul arasinda tutarsizlik olmasin)."""
        node = tree._root
        depth = 0
        for chunk_id in ordered_chunk_ids:
            child = node.get_child(chunk_id)
            if child is None or not tree._is_cached(child):
                break
            node = child
            depth += 1
        return depth

    def best_reorder_for(
        self, worker_name: str, retrieved_chunk_ids: Sequence[str]
    ) -> Tuple[List[str], float]:
        """Tek bir worker icin: o worker'in kendi agacina gore en iyi sira +
        normalize cache_gain (prefix_tracker.chunk_gain ile ayni tanim:
        matched / len(retrieved), [0,1] araliginda)."""
        tree = self._trees[worker_name]
        ordered = tree.greedy_reorder(list(retrieved_chunk_ids))
        hit_len = self._cache_hit_len(tree, ordered)
        gain = hit_len / len(retrieved_chunk_ids) if retrieved_chunk_ids else 0.0
        return ordered, gain

    def choose(
        self,
        candidate_worker_names: Sequence[str],
        retrieved_chunk_ids: Sequence[str],
        load_norm: Dict[str, float],
        alpha: float = 1.0,
        beta: float = 1.0,
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
        for name in candidate_worker_names:
            ordered, gain = self.best_reorder_for(name, retrieved_chunk_ids)
            orderings[name] = ordered
            gains[name] = gain

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

    print("\nSelf-test PASSED.")
