"""
cacheweaver_dualmap_router.py
-------------------------------
CacheWeaver'ın greedy chunk-reorder katmanını (cacheweaver_util.py),
DualMap'in (Yuan et al., ICLR 2026 — github.com/ASISys/DualMap, MIT lisans)
dual-hash + SLO-aware routing + hotspot-rebalancing mantığıyla birleştiren
entegrasyon katmanı.

TASARIM NOTU (önemli): Bu dosya, DualMap'in gerçek `SharedState` /
`replica_budgets` altyapısına DOĞRUDAN bağımlı DEĞİLDİR — kasıtlı olarak
framework-agnostic yazıldı, çünkü:
  (1) Sizin kendi router'ınız (gün-raporunuzdaki FastAPI proxy) farklı bir
      state-tutma mekanizmasına sahip olabilir,
  (2) DualMap'in orijinal kodundaki bazı davranışları BİLEREK
      değiştiriyoruz (aşağıya bakın), o yüzden birebir kopyalamak yerine
      temiz bir yeniden-implementasyon tercih edildi.

DUALMAP'İN ORİJİNAL KODUNA GÖRE BİLİNÇLİ FARKLAR
--------------------------------------------------
1) Hash key: session-kalıcı `hash_session_id` YERİNE, her istekte yeniden
   hesaplanan CacheWeaver-reorder çıktısı kullanılıyor (bkz. cacheweaver_util.py
   docstring'i). RAG'de oturum kalıcılığı varsayımı geçersiz.

2) Hotspot rebalancing SADECE {I1, I2} çifti içinde yapılıyor. DualMap'in
   PAYLAŞILAN KODUNDA `enable_migrate_to_neighbor_replica=True` iken üçüncü
   bir komşu instance'a da taşıma denendiğini gördük — bu, makalenin kendi
   iddiasıyla ("we do not search over all instances; we preserve the
   prefix-bound candidate pair") ÇELİŞİYORDU. Biz makalenin iddiasına sadık
   kalıp SADECE 2 aday içinde taşıma yapıyoruz. Bu bilinçli bir sadeleştirme
   ve raporda ayrıca not edilmeli.

3) `rb_cost1` ailesindeki (paper'da bahsi geçmeyen) recompute-ceza terimi
   burada YOK — sadece paper'ın ana metninde raporlanan basit
   `cost = target_ttft - source_ttft` (Eq.6) kullanılıyor.

KULLANIM
--------
Bu modül, kendi FastAPI router'ınızın içinde her istek geldiğinde
`router.route_request(...)` çağrısıyla kullanılacak şekilde tasarlandı.
Gerçek TTFT tahmini için `ReplicaState.estimate_ttft()` metodunu kendi
vLLM metriklerinizle (örn. /metrics endpoint'inden okuduğunuz
`num_requests_running`, `gpu_cache_usage_perc` vb.) doldurmanız gerekir —
aşağıda placeholder bir hesap var, TODO ile işaretlendi.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from cacheweaver_util import CacheWeaverKnowledgeTree, build_hash_key


# ======================================================================
# 1. Basit hash halkası (uhashring'in yerine, dependency-free)
# ======================================================================
class _SimpleHashRing:
    """DualMap'in `uhashring.HashRing` kullanımına eşdeğer, bağımlılıksız
    bir consistent-hashing implementasyonu. İki farklı `seed` ile iki
    BAĞIMSIZ halka kurup DualMap'in f1/f2'sini taklit ediyoruz.

    DÜZELTME (orijinalde builtin hash() kullanılıyordu): Python'da string
    hash()'i process başına PYTHONHASHSEED ile rastgele tuzlanır -- yani
    router'ı her yeniden başlattığında aynı hash_key farklı bir node'a
    düşebilirdi, bu da deterministik olması gereken benchmark koşularını
    (senin trace/replay harness'ın gibi) tekrarlanamaz yapardı. blake2b ile
    değiştirildi: prefix_tracker.py'deki _hash() ile aynı yaklaşım, process
    yeniden başlasa da aynı sonucu verir.
    """

    def __init__(self, num_nodes: int, seed: int):
        self._num_nodes = num_nodes
        self._seed = seed

    def get_node(self, key: str) -> int:
        h = hashlib.blake2b(f"{self._seed}:{key}".encode(), digest_size=8)
        return int.from_bytes(h.digest(), "big") % self._num_nodes


@dataclass
class ReplicaState:
    """Tek bir vLLM instance'ının (replica) izlenen durumu.

    Gerçek sisteme bağlarken bu alanları vLLM'in /metrics endpoint'inden
    veya kendi router'ınızın topladığı metriklerden doldurun.
    """

    replica_id: int
    num_pending_prefill_tokens: int = 0      # kuyrukta bekleyen prefill token sayısı
    # DÜZELTME: DualMap'in kendi donanımında ölçtüğü 0.00016s/token yerine,
    # kendi sweep_batch.py çıktından ölçülen değer kullanılıyor (cudagraph
    # sweep, c=1, TPOT p50 = 16.7ms = 0.0167s/token, 2026-07-26). Eski
    # sabit ~100x daha iyimserdi -- her TTFT tahmini o kadar yanlış çıkardı.
    # Donanım/motor modu değişirse (örn. --enforce-eager kapatılırsa,
    # farklı bir GPU'ya geçilirse) sweep_batch.py --rescore ile yeniden
    # ölç ve bu sabiti güncelle.
    prefill_tpot_seconds: float = 0.0167
    ttft_slo_seconds: float = 5.0

    def estimate_recompute_tokens(self, ordered_chunk_ids: Sequence[str],
                                   cached_prefix_len_chunks: int) -> int:
        """TODO: Gerçek vLLM cache durumuna göre doldurun. Basit yaklaşık:
        chunk başına ortalama token sayısı * (toplam chunk - cache'de olan)."""
        avg_tokens_per_chunk = 200  # CacheWeaver'ın sentetik pasaj uzunluğuna yakın bir varsayım
        num_uncached_chunks = max(0, len(ordered_chunk_ids) - cached_prefix_len_chunks)
        return num_uncached_chunks * avg_tokens_per_chunk

    def estimate_ttft(self, recompute_tokens: int) -> float:
        """Eq.7 (DualMap): TTFT(r,i) = Tq(r,i) + Tc(r,i).
        Burada basitleştirilmiş: Tq ~ kuyruktaki token sayısı, Tc ~ bu
        isteğin yeniden hesaplanacak token sayısı."""
        tq = self.num_pending_prefill_tokens * self.prefill_tpot_seconds
        tc = recompute_tokens * self.prefill_tpot_seconds
        return tq + tc

    def is_overloaded(self, ttft_slo_threshold_tokens: int) -> bool:
        return self.num_pending_prefill_tokens > ttft_slo_threshold_tokens


@dataclass
class RoutingDecision:
    request_id: str
    primary_replica: int
    second_replica: int
    ordered_chunk_ids: List[str]
    hash_key: str
    used_cache_affinity: bool  # True: cache-affinity ile seçildi, False: SLO nedeniyle min-TTFT'e geçildi
    migrated: bool = False


# ======================================================================
# 2. Asıl router: CacheWeaver reorder + DualMap dual-hash + SLO-routing
# ======================================================================
class CacheWeaverDualMapRouter:
    def __init__(
        self,
        num_replicas: int,
        cache_ttl_seconds: float = 30.0,
        ttft_slo_threshold_tokens: int = 20_000,
        rebalance_threshold_tokens: int = 30_000,
    ):
        self._num_replicas = num_replicas
        # Paylaşılan ağaç: SADECE Algoritma 1'in sıralama kararı için (CacheWeaver
        # §3.2 -- "hangi sırayla yazarsam cache'e daha iyi biner" sorusu, hangi
        # fiziksel replica'nın servis edeceğinden bağımsız bir soru).
        self._knowledge_tree = CacheWeaverKnowledgeTree(cache_ttl_seconds=cache_ttl_seconds)
        # DÜZELTME (asıl bug buradaydı): DualMap'in "bu replica'da bu prefix
        # cache'te mi" sorusu ile CacheWeaver'ın "hangi sırayla yazayım"
        # sorusu FARKLI sorular ve farklı state gerektiriyor. Orijinal kodda
        # ikisi de tek bir self._knowledge_tree'ye bakıyordu; bu yüzden
        # _approx_cache_hit_len(r1) ve _approx_cache_hit_len(r2) HER ZAMAN
        # aynı sonucu veriyordu, recompute_r1 == recompute_r2 oluyordu, ve
        # "<=" karşılaştırması beraberlikte hep r1'i seçiyordu -- yani cache
        # affinity hiçbir zaman gerçek bir karar vermiyordu, router sessizce
        # "r1'i tercih et, overloaded olana kadar" heuristiğine dönüşüyordu.
        # Şimdi her replica kendi ağacını tutuyor.
        self._replica_cache_trees: Dict[int, CacheWeaverKnowledgeTree] = {
            i: CacheWeaverKnowledgeTree(cache_ttl_seconds=cache_ttl_seconds)
            for i in range(num_replicas)
        }
        self._ring1 = _SimpleHashRing(num_replicas, seed=1)
        self._ring2 = _SimpleHashRing(num_replicas, seed=2)
        self._replicas: Dict[int, ReplicaState] = {
            i: ReplicaState(replica_id=i) for i in range(num_replicas)
        }
        # TODO(kalibrasyon): bu iki eşik DualMap'in kendi paper'ından
        # borçlanılmış, senin donanımına göre ölçülmemiş -- LOAD_REF=16
        # kalibrasyonuyla aynı mantıkla (sweep_batch.py power knee) senin
        # gerçek RAG prompt uzunluğuna göre yeniden ayarlanmalı. Şu an
        # sadece placeholder, paper'a girmeden önce ölçülmeli.
        self._ttft_slo_threshold_tokens = ttft_slo_threshold_tokens
        self._rebalance_threshold_tokens = rebalance_threshold_tokens

    # ------------------------------------------------------------------
    # Adım 1-4: reorder + dual-hash aday seçimi (Eq.5)
    # ------------------------------------------------------------------
    def _select_candidates(self, ordered_chunk_ids: Sequence[str]) -> Tuple[int, int, str]:
        hash_key = build_hash_key(ordered_chunk_ids)
        r1 = self._ring1.get_node(hash_key)
        r2 = self._ring2.get_node(hash_key)
        if r1 == r2:
            # DualMap Eq.5: instance_id2 = (instance_id1 + 1) mod num_instances
            r2 = (r1 + 1) % self._num_replicas
        return r1, r2, hash_key

    # ------------------------------------------------------------------
    # Adım 5: SLO-aware seçim (DualMap §3.2)
    # ------------------------------------------------------------------
    def _slo_aware_select(
        self, r1: int, r2: int, ordered_chunk_ids: Sequence[str]
    ) -> Tuple[int, int, bool]:
        rep1, rep2 = self._replicas[r1], self._replicas[r2]

        # Basit cache-hit tahmini: her replica'nın KENDİ ağacındaki en uzun
        # eşleşen prefix'i kaba bir "kaç chunk cache'de" sayısına çeviriyoruz.
        # TODO: gerçek vLLM prefix-cache sorgusuyla değiştirin.
        cache_hit_len_r1 = self._approx_cache_hit_len(r1, ordered_chunk_ids)
        cache_hit_len_r2 = self._approx_cache_hit_len(r2, ordered_chunk_ids)

        recompute_r1 = rep1.estimate_recompute_tokens(ordered_chunk_ids, cache_hit_len_r1)
        recompute_r2 = rep2.estimate_recompute_tokens(ordered_chunk_ids, cache_hit_len_r2)

        # Cache-affinity: daha yüksek cache-hit'e sahip (= daha az recompute
        # gereken) adayı önce tercih et.
        if recompute_r1 <= recompute_r2:
            cache_pref_primary, cache_pref_second = r1, r2
        else:
            cache_pref_primary, cache_pref_second = r2, r1

        if rep1.is_overloaded(self._ttft_slo_threshold_tokens) if cache_pref_primary == r1 \
                else rep2.is_overloaded(self._ttft_slo_threshold_tokens):
            # DualMap §3.2: cache-affine aday overloaded ise min-TTFT'e geç
            ttft_r1 = rep1.estimate_ttft(recompute_r1)
            ttft_r2 = rep2.estimate_ttft(recompute_r2)
            if ttft_r1 <= ttft_r2:
                return r1, r2, False
            else:
                return r2, r1, False

        return cache_pref_primary, cache_pref_second, True

    def _approx_cache_hit_len(self, replica_id: int, ordered_chunk_ids: Sequence[str]) -> int:
        """DÜZELTME: artık `replica_id`'nin KENDİ ağacına bakıyor, paylaşılan
        `self._knowledge_tree`'ye değil. Bu ayrım olmadan r1 ve r2 için hep
        aynı sayı çıkıyordu ve cache-affinity karşılaştırması hiçbir zaman
        gerçek bir bilgi taşımıyordu (bkz. __init__'teki not)."""
        tree = self._replica_cache_trees[replica_id]
        node = tree._root  # basit iskelet; prod'da public API ekleyin
        depth = 0
        for chunk_id in ordered_chunk_ids:
            child = node.get_child(chunk_id)
            if child is None or not tree._is_cached(child):
                break
            node = child
            depth += 1
        return depth

    # ------------------------------------------------------------------
    # Adım 6: Hotspot rebalancing (Eq.6) — SADECE {primary, second} içinde
    # ------------------------------------------------------------------
    def _maybe_rebalance(self, primary: int, second: int, ordered_chunk_ids) -> Tuple[int, bool]:
        rep_p, rep_s = self._replicas[primary], self._replicas[second]
        if not (rep_p.is_overloaded(self._rebalance_threshold_tokens)
                and rep_s.is_overloaded(self._rebalance_threshold_tokens)):
            return primary, False  # ikisi de overloaded değilse taşıma yok

        recompute_p = rep_p.estimate_recompute_tokens(ordered_chunk_ids, 0)
        recompute_s = rep_s.estimate_recompute_tokens(ordered_chunk_ids, 0)
        ttft_p = rep_p.estimate_ttft(recompute_p)
        ttft_s = rep_s.estimate_ttft(recompute_s)

        # Eq.6: B_r^(i->j) = TTFT(r,i) - TTFT(r,j). B>0 ise j'ye taşımak kazançlı.
        benefit_to_second = ttft_p - ttft_s
        if benefit_to_second > 0:
            return second, True
        return primary, False

    # ------------------------------------------------------------------
    # Dışa açık ana metod
    # ------------------------------------------------------------------
    def route_request(self, request_id: str, retrieved_chunk_ids: Sequence[str]) -> RoutingDecision:
        # 1) CacheWeaver: greedy reorder (Algoritma 1)
        ordered = self._knowledge_tree.greedy_reorder(retrieved_chunk_ids)

        # 2) DualMap: dual-hash aday seçimi (Eq.5)
        r1, r2, hash_key = self._select_candidates(ordered)

        # 3) SLO-aware seçim
        primary, second, used_cache_affinity = self._slo_aware_select(r1, r2, ordered)

        # 4) Hotspot rebalancing (sadece {primary, second} içinde)
        final_primary, migrated = self._maybe_rebalance(primary, second, ordered)

        return RoutingDecision(
            request_id=request_id,
            primary_replica=final_primary,
            second_replica=second if final_primary == primary else primary,
            ordered_chunk_ids=ordered,
            hash_key=hash_key,
            used_cache_affinity=used_cache_affinity,
            migrated=migrated,
        )

    def on_request_finished(self, decision: RoutingDecision) -> None:
        """CacheWeaver §3.2: istek bitince sıralı diziyi ağaca ekle.

        İki ayrı ekleme yapılıyor: paylaşılan ağaç (gelecekteki sıralama
        kararları için, replica'dan bağımsız) ve SADECE isteği gerçekten
        servis eden replica'nın kendi ağacı (gelecekteki cache-affinity
        karşılaştırmaları için). İkinciyi atlarsan bug geri döner.
        """
        self._knowledge_tree.insert(decision.ordered_chunk_ids)
        self._replica_cache_trees[decision.primary_replica].insert(decision.ordered_chunk_ids)

    def update_replica_load(self, replica_id: int, num_pending_prefill_tokens: int) -> None:
        """Kendi router'ınızdaki metrik toplama döngüsünden çağırın."""
        self._replicas[replica_id].num_pending_prefill_tokens = num_pending_prefill_tokens


# ----------------------------------------------------------------------
# Küçük bir uçtan-uca demo (GPU/vLLM gerektirmez, mantığı doğrular)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    router = CacheWeaverDualMapRouter(num_replicas=2)

    # İstek 1: A,B,C,D chunk'larını getiriyor
    d1 = router.route_request("req-1", ["A", "B", "C", "D"])
    print("req-1 ->", d1)
    router.on_request_finished(d1)

    # İstek 2: aynı chunk'ların çoğunu FARKLI sırada + 1 yeni chunk getiriyor
    d2 = router.route_request("req-2", ["C", "A", "B", "F", "D"])
    print("req-2 ->", d2)
    print("  (reorder öncesi sıra: C,A,B,F,D -> reorder sonrası:", d2.ordered_chunk_ids, ")")

    # Yükü yapay olarak yükseltip SLO geçişini tetikleyelim
    router.update_replica_load(d1.primary_replica, num_pending_prefill_tokens=25_000)
    d3 = router.route_request("req-3", ["A", "B", "C", "D"])
    print("req-3 (yüklü instance sonrası) ->", d3, "cache_affinity_used=", d3.used_cache_affinity)

    print("\nDemo PASSED (hata fırlatmadan tamamlandı).")
