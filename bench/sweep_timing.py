"""Repeat the dispatch-vs-completion bookkeeping ablation and aggregate.

This is the project's most counter-intuitive result and it currently rests on
one run per arm. Recording a block as cached when a request is DISPATCHED is
optimistic -- prefill has not happened yet -- while recording at COMPLETION is
conservative and should therefore be the more accurate of the two. The single
run found the opposite: completion-time bookkeeping was both less correlated
with engine ground truth and far more biased toward underestimating real
cache hits. A finding that surprising should not stand on n=1, which is what
this script fixes.

The metrics are not the ordering sweep's. Throughput and hit rate say nothing
here, because both arms serve identical prompts to identical engines -- the
only thing that changes is what the ROUTER BELIEVES while choosing. So the
measurement is the router's belief against the engine's own report of the
same request, via validate_tracker:

  * correlation and rank concordance -- does the router order requests the way
    the engine did? Routing is a comparison between workers, so this is the
    property that actually decides anything.
  * bias and underestimate rate -- is it right in magnitude, and if wrong, in
    which direction? The hypothesised mechanism is specifically a systematic
    UNDERestimate: at high concurrency, completion-time recording cannot see
    sibling requests that were dispatched to the same prefix but have not
    finished yet. If the mechanism is real, `under` is where it shows up.

Protocol matches sweep_ordering: arms interleaved and rotated so machine drift
cannot land on one arm, the router restarted between every arm (it holds the
tracker whose accuracy is the thing being measured), engines restarted by hand
because vLLM has no cache-reset endpoint here.

Usage:
    python sweep_timing.py --corpus ./corpus --trace runs/trace_hot.jsonl \\
        --worker http://100.89.101.52:8000 --worker http://100.97.250.11:8000 \\
        --repeats 3 --top-k 10 --limit 300
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

# Reused, not recopied: the restart prompt, the error-surfacing replay wrapper
# and the mean/stdev helper are already correct in sweep_ordering and there is
# no reason for a second copy to drift from them.
from sweep_ordering import (  # noqa: E402
    ROUTER_PORT,
    LAG_WARN_S,
    ArmSummary,
    confirm_restart,
    run_replay,
    schedule_lag_p99,
)
from sweep_overlap_load import score_results, start_router, stop_router, wait_healthy  # noqa: E402
from validate_tracker import compute  # noqa: E402

MODES = ["dispatch", "completion"]


@dataclass
class TimingRun:
    """Field names are what ArmSummary._agg aggregates over, so they double as
    the metric keys."""
    arm: str
    repeat: int
    out_path: Path
    corr: float = float("nan")
    concordance: float = float("nan")
    bias: float = float("nan")
    under: float = float("nan")
    over: float = float("nan")
    mae_frac: float = float("nan")
    hit_rate: float = float("nan")
    lag_p99: float = float("nan")
    n: int = 0


def one_run(args, mode: str, repeat: int) -> TimingRun:
    out_path = Path(args.outdir) / f"timing_{mode}_r{repeat + 1}.jsonl"
    router_url = f"http://127.0.0.1:{ROUTER_PORT}"

    env = {
        "ROUTER_STRATEGY": args.strategy,
        "ROUTER_TOKENIZER": args.tokenizer,
        # The one variable. Read once at router startup, which is why the
        # router must be restarted per arm and not merely reconfigured.
        "ROUTER_TRACKER_TIMING": mode,
    }
    if args.tracker_capacity:
        env["ROUTER_TRACKER_CAPACITY"] = str(args.tracker_capacity)

    log_path = Path(args.outdir) / f"router_{ROUTER_PORT}_{mode}_r{repeat + 1}.log"
    proc = start_router(args, env, ROUTER_PORT, log_path)
    try:
        asyncio.run(wait_healthy(router_url, timeout_s=args.router_timeout))
        run_replay(args, Path(args.trace), router_url, args.order, args.speedup, out_path)
    finally:
        stop_router(proc)

    run = TimingRun(arm=mode, repeat=repeat, out_path=out_path)
    expected_workers = [f"w{i + 1}" for i in range(len(args.worker))]
    run.hit_rate = score_results(out_path, expected_workers).cache_hit_rate
    try:
        d = compute(out_path)
    except SystemExit as exc:
        # No belief/usage pairs means the comparison this script exists for is
        # impossible -- loud, because a silently empty arm would aggregate as
        # NaN and quietly halve the experiment.
        raise SystemExit(f"{out_path}: tracker validation impossible.\n{exc}") from None
    run.corr, run.concordance = d["corr"], d["concordance"]
    run.bias, run.under, run.over = d["bias"], d["under"], d["over"]
    run.mae_frac, run.n = d["mae_frac"], d["n"]

    run.lag_p99 = schedule_lag_p99(out_path)
    print(f"  corr={run.corr:.3f}  concordance={run.concordance:.1%}  "
          f"bias={run.bias:+.1f}tok  under={run.under:.1%}  ({run.n} compared)")
    if run.lag_p99 > LAG_WARN_S:
        # Load-dependence is this script's whole claim, so a load axis the
        # client could not actually deliver invalidates the comparison it draws.
        print(f"  !! schedule lag p99 = {run.lag_p99:.2f}s -- the client fell behind.")
        print(f"     The load axis is not trustworthy at this --speedup.")
    return run


def report(summaries: list[ArmSummary], repeats: int) -> None:
    print()
    print("=" * 74)
    print(f"  {repeats} run(s) per arm  |  router belief vs engine truth")
    print("=" * 74)
    print(f"{'recording':<14}{'corr':>14}{'concordance':>17}"
          f"{'bias (tok)':>16}{'underest.':>14}")
    for s in summaries:
        c, cs = s._agg("corr")
        k, ks = s._agg("concordance")
        b, bs = s._agg("bias")
        u, us = s._agg("under")
        print(f"{s.arm:<14}{c:>8.3f} ±{cs:.3f}{k:>11.1%} ±{ks:.1%}"
              f"{b:>10.1f} ±{bs:.1f}{u:>8.1%} ±{us:.1%}")

    if len(summaries) < 2 or repeats < 2:
        print("\n(2 arms and 2+ repeats needed to judge separation)")
        return

    # Correlation is the headline the paper reports, so that is what gets the
    # verdict. Same rule as sweep_ordering: the gap must clear twice the
    # largest within-arm spread, so the threshold comes from this experiment.
    ranked = sorted(summaries, key=lambda s: s._agg("corr")[0], reverse=True)
    best, worst = ranked[0], ranked[-1]
    bc, bs_ = best._agg("corr")
    wc, ws_ = worst._agg("corr")
    gap, noise = bc - wc, 2 * max(bs_, ws_)
    print()
    print(f"correlation: {best.arm} {bc:.3f}  vs  {worst.arm} {wc:.3f}")
    print(f"gap        : {gap:.3f}   (2x largest stdev = {noise:.3f})")
    if gap > noise:
        print(f"-> {best.arm} tracks the engine better, beyond run-to-run spread.")
    else:
        print("-> gap does NOT clear the spread; the two recording points are")
        print("   not distinguishable at this repeat count.")

    # The hypothesised mechanism is a directional one, so it is checked
    # directly rather than inferred from the correlation gap.
    by_name = {s.arm: s for s in summaries}
    if set(MODES) <= set(by_name):
        du, _ = by_name["dispatch"]._agg("under")
        cu, _ = by_name["completion"]._agg("under")
        print()
        print(f"underestimate rate: dispatch {du:.1%}  vs  completion {cu:.1%}")
        if cu > du:
            print("-> consistent with the proposed mechanism: completion-time")
            print("   recording misses in-flight siblings sharing a prefix.")
        else:
            print("-> NOT what the proposed mechanism predicts; completion does not")
            print("   underestimate more. The explanation in the paper needs revisiting.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="./corpus")
    p.add_argument("--trace", required=True)
    p.add_argument("--outdir", default="runs/timing")
    p.add_argument("--worker", action="append", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--strategy", default="cache_aware",
                   help="must be a strategy that USES the tracker; round_robin "
                        "never computes a belief and makes this measurement empty")
    p.add_argument("--order", default="canonical",
                   help="replay.py --order, held constant across both arms")
    p.add_argument("--tokenizer", default="hf")
    p.add_argument("--tracker-capacity", type=int, default=0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--speedup", type=float, default=5.0,
                   help="the mechanism under test is a concurrency effect, so "
                        "this must be high enough for requests to overlap")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--router-timeout", type=float, default=120.0)
    p.add_argument("--no-pause", action="store_true")
    args = p.parse_args()

    for attr in ("corpus", "trace"):
        path = Path(getattr(args, attr)).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"--{attr}: not found: {path}")
        setattr(args, attr, str(path))
    args.outdir = Path(args.outdir).expanduser().resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.tracker_capacity:
        print("WARNING: --tracker-capacity not set (50000 default). Section VI-B")
        print("         attributes much of the ORIGINAL dispatch-vs-completion")
        print("         gap to exactly this constant being uncalibrated, so it")
        print("         is held constant here and the arms remain comparable to")
        print("         each other -- but not to a calibrated run.")

    summaries = {m: ArmSummary(arm=m) for m in MODES}
    total = len(MODES) * args.repeats
    index = 0
    for rep in range(args.repeats):
        for mode in MODES[rep % len(MODES):] + MODES[:rep % len(MODES)]:
            index += 1
            confirm_restart(args, mode, rep, total, index)
            summaries[mode].runs.append(one_run(args, mode, rep))

    ordered = [summaries[m] for m in MODES]
    report(ordered, args.repeats)

    summary_path = Path(args.outdir) / "timing_sweep.json"
    # The run's configuration travels with its numbers. Comparing two sweeps
    # at different --speedup (which is exactly how the load axis gets built)
    # otherwise means trusting a directory name to remember what was varied.
    config = {k: getattr(args, k) for k in
              ("repeats", "speedup", "top_k", "limit", "strategy", "order",
               "tokenizer", "tracker_capacity", "trace", "corpus")
              if hasattr(args, k)}
    summary_path.write_text(json.dumps(
        {"config": config,
         "arms": {s.arm: [r.__dict__ | {"out_path": str(r.out_path)} for r in s.runs]
                  for s in ordered}}, indent=2), encoding="utf-8")
    print(f"\nraw: {summary_path}")


if __name__ == "__main__":
    main()
