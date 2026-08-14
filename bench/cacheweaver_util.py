"""
cacheweaver_util.py
--------------------
CacheWeaver (Tan, Gu, Li — arXiv:2606.19667) makalesindeki Algoritma 1'in
bağımsız (framework-agnostic) Python implementasyonu.

Makalede kod paylaşılmadığı için burası SIFIRDAN, makalenin pseudocode'una
(Algorithm 1, Bölüm 3.3) sadık kalınarak yazılmıştır:

    O* = argmax_{O in Perm(D)} ell(O, T)

    ell(O, T): O sıralamasının, T (knowledge tree) içindeki önbelleklenmiş
    bir kök-düğüm yolunu takip eden en uzun baştan-başlayan alt dizisinin
    uzunluğu (= "reusable prefix depth").

Karmaşıklık: top-k retrieval için O(k^2) çocuk-arama (paper'daki iddiayla
aynı); ağaç O(H*k) alan kaplar (H = tutulan istek-yolu sayısı).

Bu dosya DualMap entegrasyonundan TAMAMEN bağımsız çalışır — sadece
chunk_id listesi alıp yeniden sıralanmış chunk_id listesi döndürür.
Test/geliştirme için GPU veya vLLM gerektirmez.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


ChunkId = str


@dataclass
class _TreeNode:
    """Knowledge tree'deki tek bir düğüm. Bir chunk_id'yi ve o düğüme en son
    ne zaman gelindiğini (recency) tutar. CacheWeaver'ın 'cache state is
    approximated by recency of service' varsayımı burada _last_used ile
    modelleniyor (Bölüm 3.4)."""

    children: Dict[ChunkId, "_TreeNode"] = field(default_factory=dict)
    last_used: float = 0.0

    def get_child(self, chunk_id: ChunkId) -> Optional["_TreeNode"]:
        return self.children.get(chunk_id)

    def get_or_create_child(self, chunk_id: ChunkId) -> "_TreeNode":
        if chunk_id not in self.children:
            self.children[chunk_id] = _TreeNode()
        return self.children[chunk_id]


class CacheWeaverKnowledgeTree:
    """CacheWeaver'ın 'knowledge tree'sinin (Bölüm 3.2) implementasyonu.

    Parametreler
    ----------
    cache_ttl_seconds:
        Bir düğümün "hâlâ önbellekte olduğu" varsayılan süre. Gerçek vLLM
        cache durumunu bilmediğimiz için (CacheWeaver'ın kendi sınırlaması,
        Bölüm 3.4) recency ile yaklaşıklıyoruz. Bu, DualMap tarafındaki
        prefix_cache_hit_len_list hesaplamasından BAĞIMSIZ bir tahmin —
        entegrasyonda ikisi çelişebilir, bkz. aşağıdaki not.
    max_tracked_paths:
        Ağacın büyümesini sınırlamak için en eski yolları budama eşiği
        (basit bir LRU yaklaşımı; CacheWeaver'da "recency window" olarak
        adlandırılıyor).
    """

    def __init__(self, cache_ttl_seconds: float = 30.0, max_tracked_paths: int = 5000):
        self._root = _TreeNode()
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_tracked_paths = max_tracked_paths
        self._num_inserted_paths = 0

    # ------------------------------------------------------------------
    # Insert: bir istek tamamlandığında sıralı chunk dizisini kaydet
    # ------------------------------------------------------------------
    def insert(self, ordered_chunk_ids: Sequence[ChunkId]) -> None:
        """CacheWeaver Bölüm 3.2: 'after a request finishes, its ordered
        document sequence is inserted'. Biz bunu istek TAMAMLANDIĞINDA
        çağırıyoruz (bkz. cacheweaver_dualmap_scheduler_utils.finish_request),
        çünkü ancak o zaman ilgili KV bloklarının gerçekten hesaplanıp
        önbelleğe alınmış olduğunu varsayabiliriz."""
        node = self._root
        now = time.time()
        for chunk_id in ordered_chunk_ids:
            node = node.get_or_create_child(chunk_id)
            node.last_used = now
        self._num_inserted_paths += 1
        # Basit budama: çok büyürse en eski dalları temizlemek isteyebilirsiniz.
        # (Bu iskelette bilinçli olarak basit tutuldu; prod için bir LRU
        # kuyruk eklenmeli.)

    # ------------------------------------------------------------------
    # Cached mi?
    # ------------------------------------------------------------------
    def _is_cached(self, node: _TreeNode) -> bool:
        return (time.time() - node.last_used) < self._cache_ttl_seconds

    # ------------------------------------------------------------------
    # Greedy reorder: Algoritma 1
    # ------------------------------------------------------------------
    def greedy_reorder(
        self, retrieved_chunk_ids: Sequence[ChunkId], protect_top_k: int = 0
    ) -> List[ChunkId]:
        """CacheWeaver Algoritma 1'in birebir karşılığı, artı opsiyonel bir
        kalite-koruma parametresi.

        protect_top_k: retrieval sırasındaki İLK K chunk'ı (en alakalı
        kabul edilenler) OLDUĞU YERDE bırakır -- cache durumuna bakmadan.
        Sadece K'dan sonraki chunk'lar cache-dostu şekilde yeniden dizilir.
        K=0 (varsayılan): eski davranış, TAMAMEN değişmez.

        Neden gerekli: saf greedy reorder, çok alakalı ama cache'de olmayan
        bir chunk'ı sona itebilir, az alakalı ama cache'de olan bir chunk'ı
        öne çıkarabilir -- CacheWeaver'ın kendi makalesi bile bunun cevap
        kalitesini etkileme RİSKİNİ tam kapatmadığını kabul ediyor (Tablo 7,
        "dar bir iddia... sıranın HİÇBİR ZAMAN önemli olmadığını kanıtlamıyor").
        protect_top_k, bu riski basit bir üst sınırla sınırlıyor.

        Girdi: retrieval sırasındaki chunk_id listesi (top-k).
        Çıktı: cache-dostu yeniden sıralanmış chunk_id listesi.

        ÖNEMLİ: Bu fonksiyon SADECE sırayı değiştirir, top-k KÜMESİNİ
        değiştirmez (CacheWeaver Eq.1'deki Π(D) kısıtı — modelin gördüğü
        içerik aynı kalır, sadece prefill maliyeti değişir).
        """
        protect_top_k = max(0, min(protect_top_k, len(retrieved_chunk_ids)))
        protected = list(retrieved_chunk_ids[:protect_top_k])
        rest = retrieved_chunk_ids[protect_top_k:]

        if not rest:
            return protected

        # Korunan önek, ağaçta HANGİ DÜĞÜME denk geldiğini bulmak için
        # (protected zaten sabit sırada, ama takip eden reorder'ın doğru
        # ağaç konumundan devam etmesi için ağaçta bu öneği "yürürüz" --
        # cache'de olsun olmasın, sadece pozisyonu buluyoruz, değiştirmiyoruz).
        node = self._root
        for chunk_id in protected:
            child = node.get_child(chunk_id)
            if child is None:
                # korunan önek ağaçta hiç yok -- kalan kısmı kökten başlat
                node = self._root
                break
            node = child

        remaining = list(rest)  # retrieval sırası korunuyor
        ordered: List[ChunkId] = []

        while remaining:
            found_idx: Optional[int] = None
            found_child: Optional[_TreeNode] = None

            # En fazla |remaining| çocuk kontrolü -> O(k) bu adımda,
            # toplamda O(k^2) (paper'daki karmaşıklık iddiasıyla aynı).
            for idx, chunk_id in enumerate(remaining):
                child = node.get_child(chunk_id)
                if child is not None and self._is_cached(child):
                    found_idx = idx
                    found_child = child
                    break  # ilk eşleşen yeterli (greedy, dallanma-belirsizliği
                    # nadir varsayımı — CacheWeaver Bölüm 3.3)

            if found_idx is None:
                # Hiçbir devam önbellekte değil -> kalanları retrieval
                # sırasında ekle ve dur (güvenli fallback).
                ordered.extend(remaining)
                break

            ordered.append(remaining.pop(found_idx))
            node = found_child

        return protected + ordered

    def stats(self) -> dict:
        return {
            "num_inserted_paths": self._num_inserted_paths,
            "cache_ttl_seconds": self._cache_ttl_seconds,
        }


def build_hash_key(ordered_chunk_ids: Sequence[ChunkId]) -> str:
    """Sıralanmış chunk_id listesinden DualMap'in hash_function1/2'sine
    girecek TEK bir string üretir.

    KRİTİK FARK — DualMap'in orijinal process_dataset.py'sindeki
    `hash_session_id`'den farklı olarak, bu fonksiyon HER İSTEKTE yeniden
    çağrılır ve HİÇBİR session-cache'e yazılmaz. RAG'de her sorgu bağımsız
    bir retrieval yaptığı için (aynı oturumun farklı turları bile tamamen
    farklı chunk kümeleri getirebilir), DualMap'in "session_id görülmüşse
    eski hash_session_id'yi kullan" davranışı burada BİLEREK uygulanmıyor.
    """
    return "|".join(ordered_chunk_ids)


# ----------------------------------------------------------------------
# Küçük bir kendi-kendini-test bloğu (GPU/vLLM gerektirmez)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    tree = CacheWeaverKnowledgeTree(cache_ttl_seconds=30.0)

    # Örnek: makaledeki Şekil 2 senaryosuna benzer bir durum kuralım.
    tree.insert(["A", "B", "C", "D"])  # önceki istek bu sırayla servis edildi

    # Yeni istek, {A,B,C,F,D} kümesini retrieval sırasıyla getiriyor
    retrieved = ["B", "A", "C", "F", "D"]
    reordered = tree.greedy_reorder(retrieved)

    print("Retrieval sırası :", retrieved)
    print("Yeniden sıralanmış:", reordered)
    # Beklenen (makaledeki Şekil 2 mantığıyla aynı): A,B,C,D önce (en uzun
    # önbelleklenmiş yol), F sona (kalan tek chunk, retrieval sırasında).
    assert reordered[:4] == ["A", "B", "C", "D"]
    assert reordered[4] == "F"

    hash_key_old = build_hash_key(retrieved)
    hash_key_new = build_hash_key(reordered)
    print("Eski (retrieval-sıralı) hash_key:", hash_key_old)
    print("Yeni (reorder edilmiş) hash_key :", hash_key_new)

    # --- protect_top_k testleri ---------------------------------------
    print("\n=== protect_top_k testleri ===")

    # K=0: davranış AYNEN eskisi gibi olmalı (regresyon-yok garantisi)
    reordered_k0 = tree.greedy_reorder(retrieved, protect_top_k=0)
    assert reordered_k0 == reordered, "K=0 eski davranışı BOZMAMALI"
    print(f"K=0 (eski davranış): {reordered_k0}  -- degismedi, OK")

    # K=2: ilk 2 chunk (B, A) retrieval sırasında SABIT kalmalı, kalan
    # {C,F,D} cache'e göre yeniden dizilmeli.
    reordered_k2 = tree.greedy_reorder(retrieved, protect_top_k=2)
    print(f"K=2 (ilk 2 korunuyor): {reordered_k2}")
    assert reordered_k2[:2] == ["B", "A"], "İlk 2 chunk retrieval sırasında sabit kalmalı"
    assert set(reordered_k2[2:]) == {"C", "F", "D"}, "Kalan chunk kümesi değişmemeli"
    assert reordered_k2 != reordered, "K=2, K=0'dan FARKLI bir sıra üretmeli (bu örnekte)"

    # K=len(retrieved): hiçbir şey yeniden sıralanmamalı, tamamen retrieval sırası
    reordered_kall = tree.greedy_reorder(retrieved, protect_top_k=len(retrieved))
    assert reordered_kall == retrieved, "K=tümü iken retrieval sırası hiç değişmemeli"
    print(f"K=tümü (hiç reorder yok): {reordered_kall}  -- retrieval sırasıyla birebir aynı, OK")

    print("\nSelf-test PASSED.")
