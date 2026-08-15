"""Repeat the chunk-ordering ablation N times per arm and aggregate.

A single run separated the arms by ~10 points of cache hit rate at top_k=10,
which is far more than score_quality's noise threshold -- but "far more than
the threshold" is a guess until the run-to-run spread is actually measured.
That is all this script does: run each arm N times under an identical
protocol and report mean +/- stdev, so the spread is a number rather than an
assumption.

Two design choices worth stating, because both change what the numbers mean:

1. ARMS ARE INTERLEAVED, NOT BLOCKED. The loop is repeat-major: every arm runs
   once, then every arm runs again. Running one arm three times in a row and
   then the next would let any slow drift in machine state (thermal throttling,
   a background process, GPU clock settling) land entirely on one arm and look
   like an ordering effect. Interleaving spreads it across all of them. The arm
   order is also rotated each repeat so no arm is always first.

2. THE ENGINE RESTART IS MANUAL AND CANNOT BE SKIPPED. vLLM's prefix cache is
   persistent and this deployment has no reset endpoint (checked: 404), so the
   only way to start an arm cold is to restart both workers. The script pauses
   and waits for you to confirm. The router IS restarted automatically, which
   matters for the same reason: its PrefixTracker holds its own belief about
   what is cached, and leaving it up across a wiped engine cache starts the
   next arm with the router confidently wrong.

Usage:
    python sweep_ordering.py --corpus ./corpus --trace runs/trace_hot.jsonl \\
        --worker http://100.89.101.52:8000 --worker http://100.97.250.11:8000 \\
        --arms canonical,relevance,greedy --repeats 3 --top-k 10 --limit 300
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

# Reused rather than reimplemented: the router lifecycle and the results
# reader already exist and are already the ones every other sweep uses. A
# second copy would be a second thing to keep in sync.
from sweep_overlap_load import (  # noqa: E402
    run_replay,
    score_results,
    start_router,
    stop_router,
    wait_healthy,
)
from score_quality import evaluate  # noqa: E402

ROUTER_PORT = 8099  # deliberately not 8080, so a router you left running by
                    # hand cannot silently serve these runs instead


@dataclass
class Run:
    arm: str
    repeat: int
    out_path: Path
    hit_rate: float = float("nan")
    ttft_p50: float = float("nan")
    ttft_p95: float = float("nan")
    load_cv: float = float("nan")
    contains: float = float("nan")
    n_ok: int = 0
    n_failed: int = 0


@dataclass
class ArmSummary:
    arm: str
    runs: list[Run] = field(default_factory=list)

    def _agg(self, attr: str) -> tuple[float, float]:
        vals = [getattr(r, attr) for r in self.runs
                if getattr(r, attr) == getattr(r, attr)]  # drop NaN
        if not vals:
            return float("nan"), float("nan")
        if len(vals) == 1:
            return vals[0], 0.0
        return statistics.fmean(vals), statistics.stdev(vals)


def confirm_restart(args, arm: str, repeat: int, total: int, index: int) -> None:
    """Block until the human confirms both engines are cold again."""
    print()
    print("=" * 68)
    print(f"  [{index}/{total}]  repeat {repeat + 1}  |  arm: {arm}")
    print("=" * 68)
    for w in args.worker:
        print(f"  -> restart vLLM at {w}")
    if args.no_pause:
        print("  (--no-pause: restart NOT confirmed, arms may share a warm cache)")
        return
    input("  Both engines restarted and ready? [Enter] ")


def one_run(args, arm: str, repeat: int) -> Run:
    out_path = Path(args.outdir) / f"ord_{arm}_r{repeat + 1}.jsonl"
    router_url = f"http://127.0.0.1:{ROUTER_PORT}"

    env = {"ROUTER_STRATEGY": args.strategy, "ROUTER_TOKENIZER": args.tokenizer}
    if args.tracker_capacity:
        env["ROUTER_TRACKER_CAPACITY"] = str(args.tracker_capacity)

    proc = start_router(args, env, ROUTER_PORT)
    try:
        asyncio.run(wait_healthy(router_url, timeout_s=args.router_timeout))
        run_replay(args, Path(args.trace), router_url, arm, args.speedup, out_path)
    finally:
        # Always stopped, even if replay raised: a router left listening on
        # ROUTER_PORT would be inherited by the next arm with its tracker
        # already warm, which is the exact confound this script exists to avoid.
        stop_router(proc)

    cell = score_results(out_path)
    run = Run(arm=arm, repeat=repeat, out_path=out_path,
              hit_rate=cell.cache_hit_rate, ttft_p50=cell.ttft_p50_s,
              ttft_p95=cell.ttft_p95_s, load_cv=cell.load_cv,
              n_ok=cell.n_ok, n_failed=cell.n_failed)
    try:
        run.contains = evaluate(out_path)["contains"]
    except SystemExit:
        pass  # no gold answers in this trace; cache metrics still stand
    print(f"  hit={run.hit_rate:.1%}  ttft_p50={run.ttft_p50:.3f}s  "
          f"quality={run.contains:.1%}  ({run.n_ok} ok, {run.n_failed} failed)")
    return run


def report(summaries: list[ArmSummary], repeats: int) -> None:
    print()
    print("=" * 68)
    print(f"  {repeats} run(s) per arm")
    print("=" * 68)
    print(f"{'arm':<16}{'hit rate':>18}{'TTFT p50':>18}{'quality':>16}")
    for s in summaries:
        h, hs = s._agg("hit_rate")
        t, ts = s._agg("ttft_p50")
        q, qs = s._agg("contains")
        print(f"{s.arm:<16}{h:>11.1%} ±{hs:.1%}{t:>12.3f}s ±{ts:.3f}"
              f"{q:>10.1%} ±{qs:.1%}")

    # Does the between-arm gap survive the within-arm spread? Compared against
    # 2x the largest stdev rather than a fixed threshold, so the answer comes
    # from this experiment's own noise instead of a borrowed constant.
    ranked = sorted(summaries, key=lambda s: s._agg("hit_rate")[0], reverse=True)
    if len(ranked) < 2 or repeats < 2:
        print("\n(2+ arms and 2+ repeats needed to judge separation)")
        return
    best, worst = ranked[0], ranked[-1]
    bh, bs = best._agg("hit_rate")
    wh, ws = worst._agg("hit_rate")
    gap = bh - wh
    noise = 2 * max(bs, ws)
    print()
    print(f"best  : {best.arm}  {bh:.1%}")
    print(f"worst : {worst.arm}  {wh:.1%}")
    print(f"gap   : {gap:.1%}   (2x largest stdev = {noise:.1%})")
    if gap > noise:
        print(f"-> {best.arm} separates from {worst.arm} beyond run-to-run spread.")
    else:
        print("-> gap does NOT clear the spread; these arms are not distinguishable")
        print("   at this repeat count. More repeats, or the effect is not there.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="./corpus")
    p.add_argument("--trace", required=True)
    p.add_argument("--outdir", default="runs")
    p.add_argument("--worker", action="append", required=True,
                   help="worker base URL; repeatable (must match your deployment)")
    p.add_argument("--arms", default="canonical,relevance,greedy",
                   help="comma-separated replay.py --order values")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--strategy", default="cache_aware",
                   help="ROUTER_STRATEGY, held CONSTANT across arms -- the "
                        "ablation's one variable is chunk order, not routing")
    p.add_argument("--tokenizer", default="hf")
    p.add_argument("--tracker-capacity", type=int, default=0,
                   help="ROUTER_TRACKER_CAPACITY; measure it for your GPU, do "
                        "not leave it at the 50000 default (see SETUP.md)")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--speedup", type=float, default=5.0)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--router-timeout", type=float, default=120.0)
    p.add_argument("--no-pause", action="store_true",
                   help="skip the restart prompts (dry runs only -- without a "
                        "cold engine per arm the comparison is invalid)")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms:
        raise SystemExit("--arms is empty")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if not args.tracker_capacity:
        print("WARNING: --tracker-capacity not set, router will use the 50000")
        print("         default. That constant is machine-specific and has")
        print("         already inverted one ablation in this project.")

    summaries = {a: ArmSummary(arm=a) for a in arms}
    total = len(arms) * args.repeats
    index = 0

    for rep in range(args.repeats):
        # Rotate so no arm is permanently first in the sequence.
        for arm in arms[rep % len(arms):] + arms[:rep % len(arms)]:
            index += 1
            confirm_restart(args, arm, rep, total, index)
            summaries[arm].runs.append(one_run(args, arm, rep))

    ordered = [summaries[a] for a in arms]
    report(ordered, args.repeats)

    summary_path = Path(args.outdir) / "ordering_sweep.json"
    summary_path.write_text(json.dumps(
        {s.arm: [r.__dict__ | {"out_path": str(r.out_path)} for r in s.runs]
         for s in ordered}, indent=2), encoding="utf-8")
    print(f"\nraw: {summary_path}")


if __name__ == "__main__":
    main()
