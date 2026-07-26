"""Generate a deterministic request trace from the TQuAD question pool.

The trace is generated once and replayed identically for every routing
strategy. Regenerating it per run would mean comparing strategies across
different workloads, which makes the comparison meaningless.

Two independent knobs control cache locality, and conflating them is the most
common way to produce a workload that quietly proves nothing:

  * WHICH chunks get asked about -- popularity skew (--zipf-s). Uniform
    popularity means nothing is ever reused and the router has nothing to work
    with.
  * WHEN they get asked -- reuse distance. Ten hits on one chunk spread over an
    hour is worthless; the cache empties in between. Session structure
    (--session-len, --think-time) is what puts related requests close together
    in time.

Session structure falls out of TQuAD's own hierarchy: title -> paragraphs ->
questions. A session is several questions drawn from one title, so the
requests in it genuinely touch overlapping chunks. That is real locality, not
locality injected by hand.

Usage:
    python gen_trace.py --corpus ./corpus --n 3000 --out trace.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_questions(corpus_dir: Path) -> dict[int, list[dict]]:
    by_title: dict[int, list[dict]] = defaultdict(list)
    with (corpus_dir / "qa.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_title[row["title_idx"]].append(row)
    return dict(by_title)


def zipf_weights(n: int, s: float) -> np.ndarray:
    """Popularity weights over n items, rank i getting weight 1/(i+1)^s.

    s=0 is uniform (no skew), s=1 is classic Zipf, s>1 is heavier. Sweeping s
    from 0 upwards and showing where the router's advantage appears is a
    stronger result than picking one skew and reporting a win: it answers the
    obvious reviewer question of whether the workload was chosen favourably.
    """
    ranks = np.arange(1, n + 1, dtype=np.float64)
    w = ranks ** (-s)
    return w / w.sum()


def interarrival(rng: np.random.Generator, rate: float, burst: float) -> float:
    """Gap until the next session starts.

    burst=0 gives a Poisson process (exponential gaps). Larger burst values
    draw from a Gamma with shape<1, which clusters arrivals: many near-zero
    gaps punctuated by long quiet stretches, at the same mean rate. That
    clustering is what makes queues actually form -- at a perfectly steady
    arrival rate `num_requests_waiting` stays near zero, the Load(w) term never
    activates, and the load half of the policy goes untested.
    """
    mean = 1.0 / rate
    if burst <= 0:
        return float(rng.exponential(mean))
    shape = 1.0 / (1.0 + burst)
    return float(rng.gamma(shape, mean / shape))


def generate(args) -> None:
    corpus_dir = Path(args.corpus)
    by_title = load_questions(corpus_dir)
    titles = sorted(by_title)
    if not titles:
        raise SystemExit("no questions found -- run build_corpus.py first")

    rng = np.random.default_rng(args.seed)

    # Popularity is assigned to a shuffled permutation of titles so that rank 1
    # is not simply the first article in the file.
    order = rng.permutation(len(titles))
    weights = zipf_weights(len(titles), args.zipf_s)

    records: list[dict] = []
    session_id = 0
    clock = 0.0

    while len(records) < args.n:
        clock += interarrival(rng, args.session_rate, args.burst)

        # Drift: the popularity ranking rotates as the trace progresses, so the
        # hot set at the end differs from the hot set at the start. A router
        # with a static view of locality degrades here; an adaptive one should
        # not. This is the scenario that justifies the adaptive threshold.
        shift = int(args.drift * clock) % len(titles)
        ranked = np.roll(order, shift)

        title_idx = titles[int(rng.choice(ranked, p=weights))]
        pool = by_title[title_idx]

        n_q = max(1, int(rng.poisson(args.session_len - 1)) + 1)
        n_q = min(n_q, len(pool))
        picked = rng.choice(len(pool), size=n_q, replace=False)

        t = clock
        for k, qi in enumerate(picked):
            qa = pool[int(qi)]
            records.append({
                "session_id": session_id,
                "turn": k,
                "arrival_offset_s": round(t, 4),
                "query_text": qa["question"],
                "qa_id": qa["qa_id"],
                "title_idx": title_idx,
                # Ground truth for offline analysis only. Never hand this to the
                # router: leaking it into the routing decision invalidates every
                # cache number the experiment produces.
                "expected_chunk_ids": [qa["chunk_id"]],
                "gold_answer": qa["answer"],
            })
            t += float(rng.exponential(args.think_time))
        session_id += 1

    records.sort(key=lambda r: r["arrival_offset_s"])
    records = records[:args.n]
    for i, r in enumerate(records):
        r["request_id"] = i

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {k: v for k, v in vars(args).items()}
    manifest["n_written"] = len(records)
    manifest["duration_s"] = round(records[-1]["arrival_offset_s"], 2)
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2))

    report(records, args)


def report(records: list[dict], args) -> None:
    duration = records[-1]["arrival_offset_s"]
    chunk_hits: dict[int, int] = defaultdict(int)
    for r in records:
        for c in r["expected_chunk_ids"]:
            chunk_hits[c] += 1

    counts = np.array(sorted(chunk_hits.values(), reverse=True))
    top10 = counts[:max(1, len(counts) // 10)].sum() / counts.sum()
    sessions = len({r["session_id"] for r in records})

    # Reuse distance: how many other requests land between two hits on the same
    # chunk. This, not the hit count, is what decides whether the second hit
    # still finds anything in cache.
    last_seen: dict[int, int] = {}
    distances: list[int] = []
    for r in records:
        for c in r["expected_chunk_ids"]:
            if c in last_seen:
                distances.append(r["request_id"] - last_seen[c])
            last_seen[c] = r["request_id"]

    print(f"requests           : {len(records)}")
    print(f"sessions           : {sessions}")
    print(f"duration           : {duration:.1f}s  ({len(records) / max(duration, 1e-9):.2f} req/s)")
    print(f"distinct chunks    : {len(chunk_hits)}")
    print(f"repeat rate        : {1 - len(chunk_hits) / len(records):.1%}")
    print(f"top-10% chunk share: {top10:.1%}")
    if distances:
        d = np.array(distances)
        print(f"reuse distance     : p50={np.percentile(d, 50):.0f}  "
              f"p90={np.percentile(d, 90):.0f}  (requests apart)")
    else:
        print("reuse distance     : no repeats -- raise --zipf-s or --session-len")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="./corpus")
    p.add_argument("--out", default="trace.jsonl")
    p.add_argument("--n", type=int, default=3000, help="number of requests")
    p.add_argument("--seed", type=int, default=401)
    p.add_argument("--zipf-s", type=float, default=1.0,
                   help="popularity skew over titles; 0 = uniform")
    p.add_argument("--session-len", type=float, default=4.0,
                   help="mean questions per session")
    p.add_argument("--think-time", type=float, default=8.0,
                   help="mean seconds between turns within a session")
    p.add_argument("--session-rate", type=float, default=0.5,
                   help="new sessions per second")
    p.add_argument("--burst", type=float, default=0.0,
                   help="0 = Poisson arrivals; higher clusters them")
    p.add_argument("--drift", type=float, default=0.0,
                   help="popularity ranks shifted per second; 0 = stationary")
    generate(p.parse_args())


if __name__ == "__main__":
    main()
