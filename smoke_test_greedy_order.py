"""Offline checks for replay.py's `--order greedy` arm (GPU/vLLM/router-suz).

Uc sey dogrulaniyor, ucu de canli kosuda sessizce yanlis olabilecek seyler:

1. retrieve(order="greedy") RELEVANCE sirasi donduruyor mu -- greedy_reorder'in
   "hicbir sey cache'de degilse girdi sirasini koru" fallback'i ve
   protect_top_k'nin "ilk K"si ancak bu liste siraliysa "en alakalilar"
   anlamina gelir (bkz. replay.retrieve docstring'i).
2. reusable_prefix_depth(), greedy_reorder()'in ICINDE hesapladigi derinligin
   aynisini olcuyor mu -- iki ayri kod yolu, aralarindaki tutarsizlik dogrudan
   yanlis bir "reorder depth" metrigi olarak rapora sizardi.
3. insert-ZAMANLAMASI gercekten fark yaratiyor mu -- completion vs dispatch
   ayrimi ancak agac o an guncellenmemisse anlamli.

Kosum:
    python3 smoke_test_greedy_order.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "bench"))

from cacheweaver_util import CacheWeaverKnowledgeTree  # noqa: E402
from replay import Corpus, retrieve, reusable_prefix_depth  # noqa: E402


def _corpus() -> Corpus:
    """4 chunk, elle secilmis embedding'ler. chunk_id sirasi (c0..c3) ile
    relevance sirasi BILEREK ters -- boylece canonical ve relevance kollari
    ayni ciktiyi verirse test bunu yakalar."""
    emb = np.array([
        [0.1, 0.995],   # c0 -- sorguya en UZAK
        [0.4, 0.917],   # c1
        [0.7, 0.714],   # c2
        [0.99, 0.141],  # c3 -- sorguya en YAKIN
    ], dtype="float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return Corpus(np.array(["c0", "c1", "c2", "c3"]),
                  ["t0", "t1", "t2", "t3"], emb)


def main() -> None:
    corpus = _corpus()
    qvec = np.array([1.0, 0.0], dtype="float32")

    # --- 1. retrieve() kollari ------------------------------------------
    # .tolist() KASITLI: replay.run() de chunk id'leri boyle uretiyor, yani
    # duz str -- np.str_ ile dolasmak id_to_text aramalarini sessizce farkli
    # bir kod yoluna sokardi.
    def ids(order: str) -> list[str]:
        return corpus.chunk_ids[retrieve(corpus, qvec, 4, order)].tolist()

    rel, greedy, canon = ids("relevance"), ids("greedy"), ids("canonical")

    assert rel == ["c3", "c2", "c1", "c0"], f"relevance sirasi beklenmedik: {rel}"
    assert greedy == rel, f"greedy kolu relevance sirasi almali, aldi: {greedy}"
    assert canon == ["c0", "c1", "c2", "c3"], f"canonical sirasi beklenmedik: {canon}"
    assert canon != rel, "test korpusu bozuk: canonical ve relevance ayni cikti"
    print(f"1. retrieve(): relevance={rel}  greedy={greedy}  canonical={canon}  OK")

    # --- 2. reusable_prefix_depth() == greedy_reorder()'in bulduğu derinlik
    tree = CacheWeaverKnowledgeTree(cache_ttl_seconds=30.0)
    tree.insert(["c3", "c2", "c1", "c0"])          # onceki istek bu sirayla servis edildi

    # Ayni kume, karisik sirada gelsin: greedy onbelleklenmis yolu bulmali.
    retrieved = ["c1", "c3", "c0", "c2"]
    ordered = tree.greedy_reorder(list(retrieved))
    depth = reusable_prefix_depth(tree, ordered)
    assert ordered == ["c3", "c2", "c1", "c0"], f"greedy onbellek yolunu bulamadi: {ordered}"
    assert depth == 4, f"derinlik 4 olmali, oldu: {depth}"
    print(f"2a. tam eslesme : ordered={ordered} depth={depth}  OK")

    # Yolda OLMAYAN bir chunk, derinligi tam da ayristigi yerde kesmeli.
    tree2 = CacheWeaverKnowledgeTree(cache_ttl_seconds=30.0)
    tree2.insert(["c3", "c2"])
    ordered2 = tree2.greedy_reorder(["c0", "c2", "c3"])
    depth2 = reusable_prefix_depth(tree2, ordered2)
    assert ordered2[0] == "c3" and depth2 == 2, f"kismi eslesme yanlis: {ordered2} depth={depth2}"
    assert set(ordered2) == {"c0", "c2", "c3"}, "reorder KUMEYI degistirmemeli"
    print(f"2b. kismi eslesme: ordered={ordered2} depth={depth2}  OK")

    # Bos agac: hicbir sey onbellekte degil -> girdi sirasi aynen korunmali,
    # derinlik 0. Bu, greedy kolunun "sogukta relevance koluna esit" olmasini
    # garanti eder -- iki kolun ilk isteklerde ayrismasi bir hata olurdu.
    cold = CacheWeaverKnowledgeTree(cache_ttl_seconds=30.0)
    ordered3 = cold.greedy_reorder(list(rel))
    assert ordered3 == rel, f"soguk agac sirayi degistirmemeli: {ordered3}"
    assert reusable_prefix_depth(cold, ordered3) == 0
    print(f"2c. soguk agac  : ordered={ordered3} depth=0  OK")

    # --- 3. protect_top_k gercekten pinliyor mu -------------------------
    ordered4 = tree.greedy_reorder(list(retrieved), protect_top_k=2)
    assert ordered4[:2] == retrieved[:2], f"ilk 2 chunk sabit kalmaliydi: {ordered4}"
    assert set(ordered4) == set(retrieved), "protect_top_k KUMEYI degistirmemeli"
    print(f"3. protect_top_k=2: ordered={ordered4}  (pinned={retrieved[:2]})  OK")

    # --- 4. insert zamanlamasi fark yaratiyor mu ------------------------
    # Ayni istek iki kez: agaca yazilmadan once derinlik 0, yazildiktan sonra
    # tam. fire_greedy'nin completion/dispatch ayrimi tam olarak bu farka
    # dayaniyor.
    t = CacheWeaverKnowledgeTree(cache_ttl_seconds=30.0)
    first = t.greedy_reorder(list(rel))
    assert reusable_prefix_depth(t, first) == 0, "insert ONCESI derinlik 0 olmali"
    t.insert(first)
    second = t.greedy_reorder(list(rel))
    assert reusable_prefix_depth(t, second) == len(rel), "insert SONRASI tam derinlik olmali"
    print("4. insert zamanlamasi: once depth=0, sonra depth=4  OK")

    print("\nsmoke_test_greedy_order PASSED.")


if __name__ == "__main__":
    main()
