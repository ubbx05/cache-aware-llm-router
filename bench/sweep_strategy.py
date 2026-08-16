"""Repeat the ROUTING-STRATEGY comparison N times per arm and aggregate.

The strategy table in the report is currently one run per arm, and one run
cannot tell a real gap from run-to-run spread. The concrete worry is specific:
cache_aware reads 67.7% and cacheweaver_dualmap 66.8% in the single-run table,
but their own 3x repeats (ca_r*/cwdm_r*) both land at 66.x -- so the headline
gap may be noise. This script measures the spread instead of assuming it.

Same design as sweep_ordering.py, with the ablation axis moved: there the arm
was --order and ROUTER_STRATEGY was held constant, here the arm IS
ROUTER_STRATEGY and the chunk order is held constant.

Three things this script does that a bash loop would not:

1. ARMS ARE INTERLEAVED AND ROTATED. Repeat-major, rotating the arm order each
   repeat, so a slow drift in machine state (thermal, background load, GPU
   clocks settling) spreads across all arms instead of landing on whichever
   one ran three times in a row.

2. per_worker_tree GETS ITS OWN --order. That strategy routes via a two-phase
   /router/decide_order call, which replay.py only performs under
   --order per_worker_tree. Passing canonical there would silently measure a
   different thing than the strategy actually does. The mapping is explicit in
   ORDER_FOR_STRATEGY below rather than left to the caller to remember.

3. THE TABLE REPORTS TAIL LATENCY AND BALANCE, not just the mean. The one
   place cache_aware clearly separated from cacheweaver_dualmap in earlier
   runs was TTFT p90/p99, and DualMap's worker split is locked (~538/800
   across three repeats) by hash-ring skew. Both belong in the comparison.

The engine restart is manual and cannot be skipped: vLLM's prefix cache is
persistent, this deployment has no reset endpoint, and a warm cache inherited
from the previous arm credits it to the next one. The router IS restarted
automatically for the same reason -- its PrefixTracker would otherwise start
the next arm confidently believing in blocks the engine no longer holds.

Usage:
    python sweep_strategy.py --corpus ./corpus --trace runs/trace_hot.jsonl \
        --worker http://100.89.101.52:8000 --worker http://100.97.250.11:8000 \
        --arms round_robin,least_loaded,cache_aware,cacheweaver_dualmap,per_worker_tree \
        --repeats 3 --limit 800 --speedup 5 --top-k 3 \
        --tracker-capacity 5840 --outdir runs/strat
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

import os  # noqa: E402
import subprocess  # noqa: E402

from sweep_overlap_load import (  # noqa: E402
    REPO_ROOT,
    run_replay as _run_replay,
    score_results,
    stop_router,
    wait_healthy,
)
from score_quality import evaluate  # noqa: E402

ROUTER_PORT = 8099  # deliberately not 8080, so a router left running by hand
                    # cannot silently serve these runs instead


def start_router(args, env_overrides: dict[str, str], port: int) -> subprocess.Popen:
    """Same as sweep_overlap_load.start_router, but the router's own output
    goes to a FILE instead of an unread pipe.

    The upstream version passes stdout=PIPE/stderr=STDOUT and never reads it.
    uvicorn writes an access-log line per request, so the 64 KB pipe buffer
    fills mid-run, the router blocks forever on write(), stops accepting new
    connections, and the replay hangs with no error at all -- it looks like an
    infinite loop rather than a deadlock. It is size-dependent, which is why it
    stayed hidden for so long: ~800 single-call requests sit right at the edge
    and usually squeak past, while a strategy issuing two calls per request
    (per_worker_tree: decide_order + completion) doubles the log and reliably
    hits it. An earlier "per_worker_tree does not scale under concurrency"
    observation was this bug, not the strategy.

    The log is kept rather than sent to DEVNULL: when a cell does fail, the
    router traceback is the only place the reason exists.
    """
    env = os.environ.copy()
    env.update(env_overrides)
    env["ROUTER_PORT"] = str(port)
    if args.worker:
        env["W1_URL"] = args.worker[0]
        env["W1_ENABLED"] = "true"
        if len(args.worker) > 1:
            env["W2_URL"] = args.worker[1]
            env["W2_ENABLED"] = "true"
    log = open(Path(args.outdir) / f"router_{port}.log", "ab")
    cmd = [sys.executable, "-m", "uvicorn", "main:app",
           "--host", "127.0.0.1", "--port", str(port)]
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=log)
    proc._router_log = log  # keep the handle alive until stop_router runs
    return proc

# Strategies whose routing decision is made through replay.py's two-phase
# /router/decide_order path. Anything not listed uses --default-order.
ORDER_FOR_STRATEGY = {
    "per_worker_tree": "per_worker_tree",
    "semantic_per_worker_tree": "per_worker_tree",
}

LAG_WARN_S = 1.0  # replay.py's own threshold, kept identical on purpose


def run_replay(*a, **kw) -> None:
    """sweep_overlap_load.run_replay, but it does not eat the child's output."""
    try:
        _run_replay(*a, **kw)
    except subprocess.CalledProcessError as exc:
        out = exc.output.decode("utf-8", "replace") if exc.output else "(no output)"
        print("\n--- replay.py failed, its output follows ---")
        print(out.strip()[-4000:])
        print("--- end replay.py output ---\n")
        raise


def schedule_lag_p99(path: Path) -> float:
    """Seconds the client was behind its own schedule at p99.

    If the generator cannot keep up, the arrival pattern it produced is not
    the trace's, and the run describes a load that was never delivered.
    """
    lags = sorted(r["sent_s"] - r["scheduled_s"]
                  for r in map(json.loads, path.open(encoding="utf-8")))
    if not lags:
        return float("nan")
    return lags[min(int(len(lags) * 0.99), len(lags) - 1)]


def ttft_p99(path: Path) -> float:
    """score_results gives p50/p95; the tail argument needs p99 too."""
    vals = sorted(r["ttft_s"] for r in map(json.loads, path.open(encoding="utf-8"))
                  if r.get("error") is None and r.get("ttft_s") is not None)
    if not vals:
        return float("nan")
    return vals[min(int(len(vals) * 0.99), len(vals) - 1)]


def worker_split(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in map(json.loads, path.open(encoding="utf-8")):
        if r.get("error") is not None:
            continue
        w = r.get("worker") or "?"
        counts[w] = counts.get(w, 0) + 1
    return counts


@dataclass
class Run:
    arm: str
    repeat: int
    out_path: Path
    order: str = ""
    hit_rate: float = float("nan")
    ttft_p50: float = float("nan")
    ttft_p95: float = float("nan")
    ttft_p99: float = float("nan")
    load_cv: float = float("nan")
    contains: float = float("nan")
    lag_p99: float = float("nan")
    split: dict = field(default_factory=dict)
    n_ok: int = 0
    n_failed: int = 0


@dataclass
class ArmSummary:
    arm: str
    runs: list[Run] = field(default_factory=list)

    def agg(self, attr: str) -> tuple[float, float]:
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
    print(f"  [{index}/{total}]  repeat {repeat + 1}  |  strategy: {arm}")
    print("=" * 68)
    for w in args.worker:
        print(f"  -> restart vLLM at {w}")
    print("  -> check for zombie VLLM::EngineCore (nvidia-smi), kill -9 if found")
    if args.no_pause:
        print("  (--no-pause: restart NOT confirmed, arms may share a warm cache)")
        return
    input("  Both engines restarted and ready? [Enter] ")


def one_run(args, arm: str, repeat: int) -> Run:
    out_path = Path(args.outdir) / f"strat_{arm}_r{repeat + 1}.jsonl"
    router_url = f"http://127.0.0.1:{ROUTER_PORT}"
    order = ORDER_FOR_STRATEGY.get(arm, args.default_order)

    env = {"ROUTER_STRATEGY": arm, "ROUTER_TOKENIZER": args.tokenizer}
    if args.tracker_capacity:
        env["ROUTER_TRACKER_CAPACITY"] = str(args.tracker_capacity)

    proc = start_router(args, env, ROUTER_PORT)
    try:
        asyncio.run(wait_healthy(router_url, timeout_s=args.router_timeout))
        run_replay(args, Path(args.trace), router_url, order, args.speedup, out_path)
    finally:
        # Always stopped, even if replay raised: a router left listening would
        # be inherited by the next arm with its tracker already warm.
        stop_router(proc)

    cell = score_results(out_path)
    run = Run(arm=arm, repeat=repeat, out_path=out_path, order=order,
              hit_rate=cell.cache_hit_rate, ttft_p50=cell.ttft_p50_s,
              ttft_p95=cell.ttft_p95_s, load_cv=cell.load_cv,
              n_ok=cell.n_ok, n_failed=cell.n_failed)
    run.ttft_p99 = ttft_p99(out_path)
    run.split = worker_split(out_path)
    try:
        run.contains = evaluate(out_path)["contains"]
    except SystemExit:
        pass  # no gold answers in this trace; cache metrics still stand
    run.lag_p99 = schedule_lag_p99(out_path)

    split_s = " ".join(f"{k}:{v}" for k, v in sorted(run.split.items()))
    print(f"  hit={run.hit_rate:.1%}  ttft p50={run.ttft_p50:.3f}s "
          f"p95={run.ttft_p95:.3f}s p99={run.ttft_p99:.3f}s  "
          f"quality={run.contains:.1%}  split[{split_s}]  "
          f"({run.n_ok} ok, {run.n_failed} failed)")
    if run.lag_p99 > LAG_WARN_S:
        print(f"  !! schedule lag p99 = {run.lag_p99:.2f}s -- the client fell behind,")
        print(f"     so this run's arrival pattern is NOT the trace's. Lower --speedup.")
    return run


def judge(summaries: list[ArmSummary], attr: str, label: str,
          lower_is_better: bool = False) -> None:
    """Does the between-arm gap survive the within-arm spread?

    Compared against 2x the largest stdev rather than a fixed threshold, so
    the verdict comes from this experiment's own noise, not a borrowed constant.
    """
    ranked = sorted((s for s in summaries if s.agg(attr)[0] == s.agg(attr)[0]),
                    key=lambda s: s.agg(attr)[0], reverse=not lower_is_better)
    if len(ranked) < 2:
        return
    best, worst = ranked[0], ranked[-1]
    bv, bs = best.agg(attr)
    wv, ws = worst.agg(attr)
    gap = abs(bv - wv)
    noise = 2 * max(bs, ws)
    fmt = (lambda v: f"{v:.3f}s") if "ttft" in attr else (lambda v: f"{v:.1%}")
    print()
    print(f"{label}: best {best.arm} {fmt(bv)}  |  worst {worst.arm} {fmt(wv)}")
    print(f"  gap {fmt(gap)}   (2x largest stdev = {fmt(noise)})")
    if gap > noise:
        print(f"  -> {best.arm} separates from {worst.arm} beyond run-to-run spread.")
    else:
        print(f"  -> gap does NOT clear the spread; not distinguishable at this "
              f"repeat count.")

    # The pairwise question the report actually turns on, when both are present.
    names = {s.arm: s for s in ranked}
    if "cache_aware" in names and "cacheweaver_dualmap" in names:
        a, b = names["cache_aware"], names["cacheweaver_dualmap"]
        av, as_ = a.agg(attr)
        bv2, bs2 = b.agg(attr)
        d = abs(av - bv2)
        n2 = 2 * max(as_, bs2)
        verdict = "SEPARATE" if d > n2 else "NOT separable"
        print(f"  cache_aware {fmt(av)} vs cacheweaver_dualmap {fmt(bv2)} "
              f"-> {verdict} (diff {fmt(d)}, noise {fmt(n2)})")


def report(summaries: list[ArmSummary], repeats: int) -> None:
    print()
    print("=" * 92)
    print(f"  {repeats} run(s) per arm")
    print("=" * 92)
    print(f"{'strategy':<22}{'hit rate':>16}{'TTFT p50':>16}"
          f"{'TTFT p99':>16}{'load CV':>12}{'quality':>14}")
    for s in summaries:
        h, hs = s.agg("hit_rate")
        t, ts = s.agg("ttft_p50")
        t9, t9s = s.agg("ttft_p99")
        c, _ = s.agg("load_cv")
        q, qs = s.agg("contains")
        print(f"{s.arm:<22}{h:>9.1%} ±{hs:.1%}{t:>10.3f}s ±{ts:.3f}"
              f"{t9:>10.3f}s ±{t9s:.3f}{c:>11.3f}{q:>8.1%} ±{qs:.1%}")

    if repeats < 2 or len(summaries) < 2:
        print("\n(2+ arms and 2+ repeats needed to judge separation)")
        return
    judge(summaries, "hit_rate", "CACHE HIT RATE")
    judge(summaries, "ttft_p50", "TTFT p50", lower_is_better=True)
    judge(summaries, "ttft_p99", "TTFT p99 (tail)", lower_is_better=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="./corpus")
    p.add_argument("--trace", required=True)
    p.add_argument("--outdir", default="runs/strat")
    p.add_argument("--worker", action="append", required=True,
                   help="worker base URL; repeatable (must match your deployment)")
    p.add_argument("--arms",
                   default="round_robin,least_loaded,cache_aware,"
                           "cacheweaver_dualmap,per_worker_tree",
                   help="comma-separated ROUTER_STRATEGY values")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--default-order", default="canonical",
                   help="replay.py --order for every arm EXCEPT those in "
                        "ORDER_FOR_STRATEGY; held constant so the one variable "
                        "is the routing strategy, not the chunk order")
    p.add_argument("--tokenizer", default="hf")
    p.add_argument("--tracker-capacity", type=int, default=0,
                   help="ROUTER_TRACKER_CAPACITY = GPU KV cache size / BLOCK_SIZE, "
                        "using the SMALLER of the two workers. Re-measure per "
                        "restart; do not leave it at the 50000 default")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--limit", type=int, default=800)
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

    # Absolute, because replay.py is launched with cwd=BENCH_DIR. A relative
    # path typed at the repo root would resolve against bench/ and fail there,
    # with the real error buried in a captured pipe.
    for attr in ("corpus", "trace"):
        path = Path(getattr(args, attr)).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"--{attr}: not found: {path}")
        setattr(args, attr, str(path))
    args.outdir = Path(args.outdir).expanduser().resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.tracker_capacity:
        print("WARNING: --tracker-capacity not set, router will use the 50000")
        print("         default. That constant is machine-specific and has")
        print("         already inverted one finding in this project.")

    pwt = [a for a in arms if a in ORDER_FOR_STRATEGY]
    if pwt:
        print(f"NOTE: {', '.join(pwt)} will run with --order "
              f"{ORDER_FOR_STRATEGY[pwt[0]]} (two-phase /router/decide_order);")
        print(f"      every other arm uses --order {args.default_order}.")

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

    summary_path = Path(args.outdir) / "strategy_sweep.json"
    # The configuration travels with the numbers, so a later comparison does
    # not depend on a directory name to remember what was varied.
    config = {k: getattr(args, k) for k in
              ("repeats", "speedup", "top_k", "limit", "default_order",
               "tokenizer", "tracker_capacity", "trace", "corpus", "worker")
              if hasattr(args, k)}
    summary_path.write_text(json.dumps(
        {"config": config,
         "arms": {s.arm: [r.__dict__ | {"out_path": str(r.out_path)} for r in s.runs]
                  for s in ordered}}, indent=2), encoding="utf-8")
    print(f"\nraw: {summary_path}")


if __name__ == "__main__":
    main()
