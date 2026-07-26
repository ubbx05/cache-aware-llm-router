"""Grade the router's cache model against what the engine actually reused.

Every cache_gain-driven routing decision rests on one unverified assumption:
that the router's LRU of block hashes approximates the engine's real eviction
well enough to be informative. The router keeps its own bounded LRU; vLLM
evicts on its own schedule, in its own block accounting, with concurrent
requests competing for the same cache. Those can drift apart, and if they do,
cache_gain is noise wearing a useful-looking name -- and every routing result
built on it is unsupported.

So it gets measured. `believed_cached_tokens` comes from the router's response
header; `actual_cached_tokens` from the engine's own
usage.prompt_tokens_details.cached_tokens on the same request.

Read the two headline numbers together:

  * correlation -- does the router rank requests correctly? This is what
    routing needs, because a decision is a comparison between workers, not an
    absolute cost estimate.
  * bias/MAE -- is it right in magnitude? Matters less for ranking, but a large
    systematic bias means alpha is not on the scale it appears to be, so the
    alpha/beta sweep is measuring something other than what it claims.

Usage:
    python validate_tracker.py results.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load(path: Path):
    believed, actual, prompt, rows = [], [], [], 0
    missing_belief = missing_actual = 0

    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("error"):
            continue
        rows += 1
        b, a = r.get("believed_cached_tokens"), r.get("actual_cached_tokens")
        if b is None:
            missing_belief += 1
            continue
        if a is None:
            missing_actual += 1
            continue
        believed.append(b)
        actual.append(a)
        prompt.append(r.get("prompt_tokens") or 0)

    return (np.array(believed, dtype=float), np.array(actual, dtype=float),
            np.array(prompt, dtype=float), rows, missing_belief, missing_actual)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    b, a, p, rows, miss_b, miss_a = load(path)

    if len(b) == 0:
        print(f"{path}: karsilastirilacak veri yok ({rows} basarili istek)")
        if miss_b:
            print(f"  {miss_b} istekte router basligi yok "
                  "-> istek routerdan gecmemis, ya da eski main.py calisiyor")
        if miss_a:
            print(f"  {miss_a} istekte engine usage yok "
                  "-> stream_options.include_usage gonderilmemis (eski replay.py)")
        raise SystemExit(1)

    err = b - a
    denom = np.where(p > 0, p, np.nan)

    # Pearson on the raw token counts. Requires both sides to vary; a run where
    # nothing was ever cached gives a degenerate result rather than a wrong one.
    if b.std() < 1e-9 or a.std() < 1e-9:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(b, a)[0, 1])

    # Directional agreement is the property routing actually depends on: for a
    # pair of requests, does the router order them the same way the engine did?
    # Sampled rather than all pairs, which would be O(n^2).
    rng = np.random.default_rng(0)
    n_pairs = min(20000, len(b) * (len(b) - 1) // 2) if len(b) > 1 else 0
    concordant = 0
    if n_pairs:
        i = rng.integers(0, len(b), n_pairs)
        j = rng.integers(0, len(b), n_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
        same = np.sign(b[i] - b[j]) == np.sign(a[i] - a[j])
        concordant = float(same.mean())

    print(f"{path}")
    print(f"  karsilastirilan istek : {len(b)} / {rows}")
    if miss_b or miss_a:
        print(f"  atlanan               : belief yok {miss_b}, usage yok {miss_a}")
    print()
    print(f"  router inanci   ort : {b.mean():8.1f} token")
    print(f"  engine gercek   ort : {a.mean():8.1f} token")
    print(f"  prompt          ort : {p.mean():8.1f} token")
    print()
    print(f"  korelasyon (Pearson): {corr:.3f}")
    if n_pairs:
        print(f"  siralama uyumu      : {concordant:.1%}  (rastgele cift ustunde)")
    print(f"  ortalama sapma      : {err.mean():+8.1f} token  "
          f"({'router fazla tahmin ediyor' if err.mean() > 0 else 'router az tahmin ediyor'})")
    print(f"  MAE                 : {np.abs(err).mean():8.1f} token")
    print(f"  MAE / prompt        : {np.nanmean(np.abs(err) / denom):8.1%}")
    print()

    over = float((b > a * 1.2).mean())
    under = float((b < a * 0.8).mean())
    close = 1.0 - over - under
    print(f"  %20 icinde tutan    : {close:.1%}")
    print(f"  fazla tahmin        : {over:.1%}")
    print(f"  az tahmin           : {under:.1%}")

    print()
    if np.isnan(corr):
        print("  YORUM: bir taraf hic degismemis -- cache hic devreye girmemis olabilir.")
    elif corr > 0.8:
        print("  YORUM: tracker motoru iyi izliyor. cache_gain guvenilir bir sinyal,")
        print("         routing sonuclari bu varsayim uzerinde saglam duruyor.")
    elif corr > 0.5:
        print("  YORUM: orta duzey uyum. Siralama cogunlukla dogru, mutlak deger degil.")
        print("         Routing icin yeterli olabilir ama alpha'nin olcegi guvenilmez;")
        print("         raporda bu ayrimi yaz.")
    else:
        print("  YORUM: zayif uyum. cache_gain motorun davranisini yansitmiyor --")
        print("         cache_gain'e dayanan tum routing sonuclari sorgulanmali.")
        print("         Ilk bakilacak yer: ROUTER_TOKENIZER=approx mi? Blok sinirlari")
        print("         motorunkiyle hizalanmiyorsa hash'ler bastan tutmaz.")


if __name__ == "__main__":
    main()
