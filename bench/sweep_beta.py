"""Re-run the beta x load x locality 2x2 with a CALIBRATED tracker capacity.

The 2x2 is the report's most distinctive figure: the winning corner moves as
both load AND workload locality change, so neither "more load -> raise beta"
nor "more locality -> raise beta" holds on its own. But it was measured at the
50000 default capacity, and an uncalibrated capacity has already inverted one
finding in this project (the 30x/35x crossover reversed direction once the
capacity was fixed). A figure that carries the paper's most original claim
cannot rest on the one constant known to flip conclusions.

This is a replication, not a new experiment: same traces, same loads, same
request count as the original crossover/sens runs, with capacity as the single
changed variable. Keeping everything else identical is the point -- if the
pattern survives, the claim is now standing on calibrated ground; if it does
not, we learn that the original pattern was an artifact, which is equally
worth knowing.

Eight cells: {high, low} locality x {low, high} load x {beta=1, beta=0}.
Engines are restarted manually between every cell for the usual reason (vLLM's
prefix cache is persistent and would credit the previous cell), and the router
is restarted automatically so its PrefixTracker starts cold too.

Usage:
    python sweep_beta.py --corpus ./corpus \
        --trace-high runs/trace_hot.jsonl \
        --trace-low  runs/trace_low_locality.jsonl \
        --worker http://100.89.101.52:8000 --worker http://100.97.250.11:8000 \
        --limit 3000 --top-k 3 --loads 30,35 \
        --tracker-capacity 5840 --outdir runs/beta2x2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path
import sys

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from sweep_overlap_load import (  # noqa: E402
    REPO_ROOT,
    score_results,
    start_router as _start_router_piped,
    stop_router,
    wait_healthy,
)
from sweep_strategy import run_replay, schedule_lag_p99, ttft_p99, worker_split, LAG_WARN_S  # noqa: E402

import os  # noqa: E402
import subprocess  # noqa: E402

ROUTER_PORT = 8098  # not 8080 and not sweep_strategy's 8099, so none of the
                    # three can silently serve each other's runs


def start_router(args, env_overrides: dict[str, str], port: int) -> subprocess.Popen:
    """Same as sweep_overlap_load.start_router, but the router's own output
    goes to a FILE instead of an unread pipe.

    The upstream version passes stdout=PIPE/stderr=STDOUT and never reads it.
    uvicorn writes an access-log line per request, so the 64 KB pipe buffer
    fills mid-run, the router blocks forever on write(), stops accepting
    connections, and the replay hangs with no error -- the failure looks like
    an infinite loop rather than a deadlock. It is size-dependent, which is
    why it stayed hidden: ~800 requests is roughly 72 KB and squeaked past,
    3000 requests is ~270 KB and never does. A strategy that issues two calls
    per request (per_worker_tree: decide_order + completion) doubles the log
    and hits it at 800 too -- which is what the earlier "per_worker_tree does
    not scale" observation actually was.

    Keeping the log rather than sending it to DEVNULL: when a cell does fail,
    the router's traceback is the only place the reason exists.
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
    log_path = Path(args.outdir) / f"router_{port}.log"
    log = open(log_path, "ab")  # append: one file per sweep, not per cell
    cmd = [sys.executable, "-m", "uvicorn", "main:app",
           "--host", "127.0.0.1", "--port", str(port)]
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=log)
    proc._router_log = log  # keep the handle alive until stop_router runs
    return proc


def one_cell(args, loc: str, trace: Path, load: float, beta: float,
             index: int, total: int) -> dict:
    print()
    print("=" * 70)
    print(f"  [{index}/{total}]  locality={loc}  load={load:g}x  beta={beta:g}")
    print("=" * 70)
    for w in args.worker:
        print(f"  -> restart vLLM at {w}")
    print("  -> check for zombie VLLM::EngineCore (nvidia-smi), kill -9 if found")
    if not args.no_pause:
        input("  Both engines restarted and ready? [Enter] ")

    out = Path(args.outdir) / f"beta{beta:g}_{loc}_{load:g}x.jsonl"
    url = f"http://127.0.0.1:{ROUTER_PORT}"
    env = {"ROUTER_STRATEGY": "cache_aware", "ROUTER_TOKENIZER": args.tokenizer,
           "ROUTER_BETA": str(beta)}
    if args.tracker_capacity:
        env["ROUTER_TRACKER_CAPACITY"] = str(args.tracker_capacity)

    proc = start_router(args, env, ROUTER_PORT)
    try:
        asyncio.run(wait_healthy(url, timeout_s=args.router_timeout))
        run_replay(args, trace, url, args.order, load, out)
    finally:
        stop_router(proc)

    c = score_results(out)
    row = {"locality": loc, "load": load, "beta": beta,
           "hit": c.cache_hit_rate, "t50": c.ttft_p50_s, "t95": c.ttft_p95_s,
           "t99": ttft_p99(out), "load_cv": c.load_cv,
           "n_ok": c.n_ok, "n_failed": c.n_failed,
           "lag_p99": schedule_lag_p99(out), "split": worker_split(out),
           "out": str(out)}
    print(f"  hit={row['hit']:.1%}  ttft p50={row['t50']:.3f}s p99={row['t99']:.3f}s  "
          f"({row['n_ok']} ok, {row['n_failed']} failed)")
    if row["lag_p99"] > LAG_WARN_S:
        print(f"  !! schedule lag p99 = {row['lag_p99']:.2f}s -- the client fell behind.")
        print(f"     This cell's arrival pattern is NOT the trace's; the load axis is void here.")
    return row


def report(rows: list[dict]) -> None:
    print()
    print("=" * 78)
    print("  beta x load x locality  (cache_aware, calibrated capacity)")
    print("=" * 78)
    print(f"{'locality':<10}{'load':>7}{'beta=1 p50':>14}{'beta=0 p50':>14}"
          f"{'winner':>10}{'margin':>10}")
    by = {(r["locality"], r["load"], r["beta"]): r for r in rows}
    localities = sorted({r["locality"] for r in rows}, reverse=True)
    loads = sorted({r["load"] for r in rows})
    verdicts = {}
    for loc in localities:
        for ld in loads:
            a, b = by.get((loc, ld, 1.0)), by.get((loc, ld, 0.0))
            if not a or not b:
                continue
            win = "beta=1" if a["t50"] < b["t50"] else "beta=0"
            margin = abs(a["t50"] - b["t50"]) / max(a["t50"], b["t50"]) * 100
            verdicts[(loc, ld)] = win
            print(f"{loc:<10}{ld:>6g}x{a['t50']:>13.3f}s{b['t50']:>13.3f}s"
                  f"{win:>10}{margin:>9.0f}%")

    # The claim the figure makes: the winner is not a function of load alone,
    # nor of locality alone. That is true exactly when the winners do not form
    # a single row/column pattern -- i.e. the corner moves.
    if len(verdicts) == 4:
        print()
        wins = set(verdicts.values())
        if len(wins) == 1:
            print("-> SAME winner in all four cells: the original 2x2 pattern did NOT")
            print("   survive calibration. beta's advantage looks one-directional here.")
        else:
            per_load = {ld: {verdicts[(loc, ld)] for loc in localities} for ld in loads}
            per_loc = {loc: {verdicts[(loc, ld)] for ld in loads} for loc in localities}
            flips_with_locality = any(len(v) > 1 for v in per_load.values())
            flips_with_load = any(len(v) > 1 for v in per_loc.values())
            if flips_with_locality and flips_with_load:
                print("-> The winning corner MOVES along both axes: beta's benefit is a")
                print("   joint function of load and locality, not of either alone.")
                print("   The original finding survives calibration.")
            else:
                axis = "locality" if flips_with_locality else "load"
                print(f"-> The winner flips with {axis} only. Weaker than the original")
                print(f"   claim: state it as a one-axis effect, not a joint one.")

    print("\nNOTE: one run per cell. TTFT p50 run-to-run stdev on this setup has")
    print("      been as large as 0.30s (cacheweaver_dualmap, 3 repeats), so treat")
    print("      any margin under ~25% as provisional until repeated.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="./corpus")
    p.add_argument("--trace-high", required=True, help="high-locality trace")
    p.add_argument("--trace-low", required=True, help="low-locality trace")
    p.add_argument("--worker", action="append", required=True)
    p.add_argument("--loads", default="30,35", help="two speedup values, comma-separated")
    p.add_argument("--betas", default="1,0", help="two ROUTER_BETA values")
    p.add_argument("--order", default="canonical")
    p.add_argument("--tokenizer", default="hf")
    p.add_argument("--tracker-capacity", type=int, default=0)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--router-timeout", type=float, default=120.0)
    p.add_argument("--outdir", default="runs/beta2x2")
    p.add_argument("--no-pause", action="store_true")
    args = p.parse_args()

    for attr in ("corpus", "trace_high", "trace_low"):
        path = Path(getattr(args, attr)).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"--{attr.replace('_', '-')}: not found: {path}")
        setattr(args, attr, str(path))
    args.outdir = Path(args.outdir).expanduser().resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    # run_replay reads args.trace only via the argument we pass, but score/lag
    # helpers expect the attribute to exist on args in sweep_strategy's shape.
    args.trace = args.trace_high

    if not args.tracker_capacity:
        print("WARNING: --tracker-capacity not set. This sweep exists BECAUSE the")
        print("         default flipped a finding once; running it uncalibrated")
        print("         defeats the entire point.")

    loads = [float(x) for x in args.loads.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    traces = [("high", Path(args.trace_high)), ("low", Path(args.trace_low))]

    rows: list[dict] = []
    total = len(traces) * len(loads) * len(betas)
    i = 0
    # Ordered locality-major so the two beta arms of a cell sit next to each
    # other in time: whatever the machine is doing slowly, both arms of the
    # comparison get the same amount of it.
    for loc, trace in traces:
        for ld in loads:
            for b in betas:
                i += 1
                rows.append(one_cell(args, loc, trace, ld, b, i, total))

    report(rows)
    summary = Path(args.outdir) / "beta_2x2.json"
    config = {k: getattr(args, k) for k in
              ("loads", "betas", "top_k", "limit", "order", "tokenizer",
               "tracker_capacity", "trace_high", "trace_low", "corpus", "worker")}
    summary.write_text(json.dumps({"config": config, "cells": rows}, indent=2),
                       encoding="utf-8")
    print(f"\nraw: {summary}")


if __name__ == "__main__":
    main()
