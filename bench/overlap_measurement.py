"""Measure retrieved-chunk-set overlap between consecutive queries.

CacheWeaver-style overlap: Jaccard(A, B) = |A ∩ B| / |A ∪ B| between the
top-k retrieved chunk sets of two consecutive queries. This is a property of
the corpus + trace + retriever alone -- it does not touch the router or a
live vLLM engine, so it reuses replay.py's Corpus/retrieve rather than firing
requests.

Two notions of "consecutive" are reported separately because they answer
different questions:

  * session-adjacent: turn k -> turn k+1 within one session. This is what a
    prefix cache actually sees if the router keeps a session pinned to one
    worker -- the scenario the caching argument depends on.
  * global-adjacent: request_id i -> i+1 in arrival order, regardless of
    session. This is the raw opportunity the router faces before any
    session-affinity policy is applied; sessions interleave, so it is
    typically much lower than the session-adjacent number.

Usage:
    python overlap_measurement.py --corpus ./corpus --trace trace.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from replay import Corpus, retrieve


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_trace(path: Path, limit: int) -> list[dict]:
    trace = [json.loads(l) for l in path.open(encoding="utf-8")]
    if limit:
        trace = trace[:limit]
    return trace


def retrieved_sets(corpus: Corpus, trace: list[dict], args) -> list[set[int]]:
    from sentence_transformers import SentenceTransformer

    print(f"embedding {len(trace)} queries on {args.device} ...")
    model = SentenceTransformer(args.embed_model, device=args.device)
    qvecs = model.encode(
        [f"query: {r['query_text']}" for r in trace],
        batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    ).astype("float32")

    sets = []
    for qv in qvecs:
        idxs = retrieve(corpus, qv, args.top_k, args.order)
        sets.append(set(corpus.chunk_ids[idxs].tolist()))
    return sets


def session_adjacent_pairs(trace: list[dict], sets: list[set[int]]) -> list[float]:
    by_session: dict[int, list[tuple[int, set[int]]]] = defaultdict(list)
    for rec, s in zip(trace, sets):
        by_session[rec["session_id"]].append((rec["turn"], s))

    scores = []
    for turns in by_session.values():
        turns.sort(key=lambda t: t[0])
        for (_, a), (_, b) in zip(turns, turns[1:]):
            scores.append(jaccard(a, b))
    return scores


def global_adjacent_pairs(trace: list[dict], sets: list[set[int]]) -> list[float]:
    order = sorted(range(len(trace)), key=lambda i: trace[i]["request_id"])
    return [jaccard(sets[order[i]], sets[order[i + 1]]) for i in range(len(order) - 1)]


def report(name: str, scores: list[float]) -> None:
    if not scores:
        print(f"{name:<17}: no pairs")
        return
    s = sorted(scores)
    nonzero = sum(1 for x in s if x > 0) / len(s)
    print(f"{name:<17}: n={len(s)}  mean={statistics.fmean(s):.3f}  "
          f"p50={s[len(s) // 2]:.3f}  p90={s[int(len(s) * 0.9)]:.3f}  "
          f"nonzero={nonzero:.1%}")


def run(args) -> None:
    corpus = Corpus.load(Path(args.corpus))
    trace = load_trace(Path(args.trace), args.limit)
    if not trace:
        raise SystemExit("empty trace")

    sets = retrieved_sets(corpus, trace, args)
    session_scores = session_adjacent_pairs(trace, sets)
    global_scores = global_adjacent_pairs(trace, sets)

    print()
    print(f"queries          : {len(trace)}  (top_k={args.top_k}, order={args.order})")
    report("session-adjacent", session_scores)
    report("global-adjacent", global_scores)

    if args.out:
        out = Path(args.out)
        with out.open("w", encoding="utf-8") as f:
            for label, scores in (("session", session_scores), ("global", global_scores)):
                for score in scores:
                    f.write(json.dumps({"kind": label, "jaccard": score}) + "\n")
        print(f"pairwise scores  : {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="./corpus")
    p.add_argument("--trace", default="trace.jsonl")
    p.add_argument("--out", default=None, help="optional jsonl of every pairwise score")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--order", choices=["canonical", "relevance"], default="canonical")
    p.add_argument("--embed-model", default="intfloat/multilingual-e5-base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0, help="measure only the first N trace rows")
    run(p.parse_args())


if __name__ == "__main__":
    main()
