r"""Replicate the beta x load x locality experiment under one fixed policy.

The only experimental variables in this sweep are locality (two supplied
traces), arrival load (``--loads``), and ``ROUTER_BETA`` (``--betas``).  The
router is deliberately fixed to ``cache_aware`` and replay ordering to
``canonical`` so a cell cannot accidentally mix a routing or ordering change
into the beta comparison.

Live runs require an explicitly measured ``--tracker-capacity``.  Every cell
also requires a human confirmation that both vLLM workers were restarted: the
engine cache has no reset endpoint, and carrying a warm cache into the next
cell invalidates the comparison.  Cells are scheduled repeat-major and the
order is rotated on every repeat to spread slow machine drift across them.

``--dry-run`` prints that schedule before any path validation or directory
creation.  It starts no subprocess, performs no network request, and writes no
files, so the orchestration can be inspected on a machine without the corpus,
traces, or workers.

Example:
    python bench/sweep_beta.py --dry-run --repeats 2

    python bench/sweep_beta.py \
        --corpus ./corpus \
        --high-trace runs/trace_hot.jsonl \
        --low-trace runs/trace_low_locality.jsonl \
        --worker http://100.89.101.52:8000 \
        --worker http://100.97.250.11:8000 \
        --tracker-capacity 5840 --repeats 2
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

# A dry-run promises no writes.  Set this before importing local modules so
# Python itself does not create __pycache__ entries as an incidental write.
sys.dont_write_bytecode = True

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from score_quality import evaluate  # noqa: E402
from sweep_overlap_load import (  # noqa: E402
    run_replay,
    score_results,
    start_router,
    stop_router,
    wait_healthy,
)


STRATEGY = "cache_aware"
ORDER = "canonical"


@dataclass(frozen=True)
class Cell:
    locality: str
    trace: str
    speedup: float
    beta: float


@dataclass(frozen=True)
class ScheduledCell:
    repeat: int
    sequence: int
    cell: Cell


@dataclass
class RunResult:
    locality: str
    trace: str
    speedup: float
    beta: float
    repeat: int
    sequence: int
    port: int
    results_path: str
    router_log_path: str
    n_ok: int = 0
    n_failed: int = 0
    cache_hit_rate: float | None = None
    cache_metric: str = "unavailable"
    cached_tokens_total: int = 0
    prompt_tokens_total: int = 0
    ttft_p50_s: float | None = None
    ttft_p99_s: float | None = None
    load_cv: float | None = None
    contains: float | None = None
    phase2_ttft_p50_s: float | None = None
    phase2_ttft_p99_s: float | None = None
    e2e_ttft_p50_s: float | None = None
    e2e_ttft_p99_s: float | None = None
    error: str = ""


METRICS = (
    "cache_hit_rate",
    "ttft_p50_s",
    "ttft_p99_s",
    "load_cv",
    "contains",
    "phase2_ttft_p50_s",
    "phase2_ttft_p99_s",
    "e2e_ttft_p50_s",
    "e2e_ttft_p99_s",
)


def parse_numbers(raw: str, flag: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"{flag}: expected comma-separated numbers, got {raw!r}") from exc
    if not values:
        raise SystemExit(f"{flag}: list is empty")
    if not all(math.isfinite(value) for value in values):
        raise SystemExit(f"{flag}: values must be finite")
    return values


def build_cells(args) -> list[Cell]:
    loads = parse_numbers(args.loads, "--loads")
    betas = parse_numbers(args.betas, "--betas")
    if any(load <= 0 for load in loads):
        raise SystemExit("--loads: every speedup must be > 0")
    if any(beta < 0 for beta in betas):
        raise SystemExit("--betas: every beta must be >= 0")

    traces = (("high", args.high_trace), ("low", args.low_trace))
    return [
        Cell(locality=locality, trace=trace, speedup=load, beta=beta)
        for locality, trace in traces
        for load in loads
        for beta in betas
    ]


def build_schedule(cells: list[Cell], repeats: int) -> list[ScheduledCell]:
    if repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if not cells:
        return []

    schedule: list[ScheduledCell] = []
    sequence = 0
    for repeat in range(repeats):
        offset = repeat % len(cells)
        rotated = cells[offset:] + cells[:offset]
        for cell in rotated:
            sequence += 1
            schedule.append(ScheduledCell(repeat=repeat + 1,
                                           sequence=sequence,
                                           cell=cell))
    return schedule


def print_plan(schedule: list[ScheduledCell], tracker_capacity: int | None) -> None:
    print("Fixed protocol: strategy=cache_aware, order=canonical")
    capacity = str(tracker_capacity) if tracker_capacity else "<required for live run>"
    print(f"Tracker capacity: {capacity}")
    print()
    print(f"{'seq':>4} {'repeat':>6} {'locality':>9} {'load':>8} {'beta':>8}  trace")
    print("-" * 78)
    for item in schedule:
        cell = item.cell
        print(f"{item.sequence:>4} {item.repeat:>6} {cell.locality:>9} "
              f"{cell.speedup:>8g} {cell.beta:>8g}  {cell.trace}")
    print(f"\n{len(schedule)} cells")


def validate_live_args(args, schedule: list[ScheduledCell]) -> None:
    if args.tracker_capacity is None or args.tracker_capacity <= 0:
        raise SystemExit(
            "live sweep requires a measured positive --tracker-capacity "
            "(for example, 5840 on the calibrated deployment)"
        )
    if len(args.worker) != 2:
        raise SystemExit("live beta sweep requires exactly two --worker URLs")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.port < 1 or args.port + len(schedule) - 1 > 65535:
        raise SystemExit("--port range would fall outside 1..65535")

    corpus = Path(args.corpus).expanduser().resolve()
    if not corpus.exists():
        raise SystemExit(f"--corpus: not found: {corpus}")
    args.corpus = str(corpus)

    for attr in ("high_trace", "low_trace"):
        path = Path(getattr(args, attr)).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"--{attr.replace('_', '-')}: not found: {path}")
        setattr(args, attr, str(path))


def confirm_cold(args, item: ScheduledCell, total: int) -> None:
    cell = item.cell
    print()
    print("=" * 78)
    print(f"[{item.sequence}/{total}] repeat={item.repeat} locality={cell.locality} "
          f"load={cell.speedup:g} beta={cell.beta:g}")
    print("=" * 78)
    for worker in args.worker:
        print(f"  restart vLLM cold: {worker}")
    input("Both workers restarted, cache cold, and ready? [Enter] ")


def slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def finite_metric(obj: object, *names: str) -> float | None:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def score_one_run(path: Path, expected_workers: list[str]) -> dict[str, Any]:
    scored = score_results(path, expected_workers)
    try:
        quality = evaluate(path)
        raw_contains = quality.get("contains")
        contains = (float(raw_contains)
                    if isinstance(raw_contains, (int, float))
                    and math.isfinite(float(raw_contains)) else None)
    except SystemExit:
        contains = None

    # In the shared scorer, ttft_* is the phase-2 clock for two-phase rows and
    # e2e_ttft_* is the full clock.  Accept explicit aliases as well so this
    # sweep keeps reporting both if the scorer later names phase 2 directly.
    ttft_p50 = finite_metric(scored, "ttft_p50_s")
    ttft_p99 = finite_metric(scored, "ttft_p99_s")
    phase2_p50 = finite_metric(scored, "phase2_ttft_p50_s")
    phase2_p99 = finite_metric(scored, "phase2_ttft_p99_s")
    if phase2_p50 is None:
        phase2_p50 = ttft_p50
    if phase2_p99 is None:
        phase2_p99 = ttft_p99

    n_ok = int(getattr(scored, "n_ok", 0))
    score_error = str(getattr(scored, "error", ""))
    if not score_error and n_ok == 0:
        score_error = "scorer found no successful requests"

    return {
        "n_ok": n_ok,
        "n_failed": int(getattr(scored, "n_failed", 0)),
        "cache_hit_rate": finite_metric(scored, "cache_hit_rate"),
        "cache_metric": str(getattr(scored, "cache_metric", "unavailable")),
        "cached_tokens_total": int(getattr(scored, "cached_tokens_total", 0)),
        "prompt_tokens_total": int(getattr(scored, "prompt_tokens_total", 0)),
        "ttft_p50_s": ttft_p50,
        "ttft_p99_s": ttft_p99,
        "load_cv": finite_metric(scored, "load_cv"),
        "contains": contains,
        "phase2_ttft_p50_s": phase2_p50,
        "phase2_ttft_p99_s": phase2_p99,
        "e2e_ttft_p50_s": finite_metric(
            scored, "e2e_ttft_p50_s", "e2e_p50_s"
        ),
        "e2e_ttft_p99_s": finite_metric(
            scored, "e2e_ttft_p99_s", "e2e_p99_s"
        ),
        "error": score_error,
    }


def run_cell(args, item: ScheduledCell, outdir: Path) -> RunResult:
    cell = item.cell
    stem = (f"beta_{cell.locality}_l{slug(cell.speedup)}_b{slug(cell.beta)}_"
            f"r{item.repeat}_s{item.sequence}")
    results_path = outdir / f"{stem}.jsonl"
    log_path = outdir / f"router_{stem}.log"
    port = args.port + item.sequence - 1
    result = RunResult(
        locality=cell.locality,
        trace=cell.trace,
        speedup=cell.speedup,
        beta=cell.beta,
        repeat=item.repeat,
        sequence=item.sequence,
        port=port,
        results_path=str(results_path),
        router_log_path=str(log_path),
    )

    env = {
        "ROUTER_STRATEGY": STRATEGY,
        "ROUTER_ALPHA": "1.0",
        "ROUTER_BETA": f"{cell.beta:g}",
        "ROUTER_DELTA0": "0.5",
        "ROUTER_LOAD_REF": "16",
        "ROUTER_TRACKER_TIMING": "dispatch",
        "ROUTER_TOKENIZER": args.tokenizer,
        "ROUTER_TOKENIZER_MODEL": args.model,
        "ROUTER_TRACKER_CAPACITY": str(args.tracker_capacity),
    }
    proc = None
    try:
        proc = start_router(args, env, port, log_path)
        router_url = f"http://127.0.0.1:{port}"
        asyncio.run(wait_healthy(router_url, args.startup_timeout))
        run_replay(args, Path(cell.trace), router_url, ORDER,
                   cell.speedup, results_path)
        expected_workers = [f"w{i + 1}" for i in range(len(args.worker))]
        for key, value in score_one_run(results_path, expected_workers).items():
            setattr(result, key, value)
    except subprocess.CalledProcessError as exc:
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        else:
            output = str(output)
        result.error = f"replay failed ({exc.returncode}): {output[-2000:].strip()}"
    except Exception as exc:  # keep a long sweep moving; the log identifies the cell
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None:
            stop_router(proc)
    return result


def aggregate(runs: Iterable[RunResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, float], list[RunResult]] = {}
    for run in runs:
        groups.setdefault((run.locality, run.speedup, run.beta), []).append(run)

    rows: list[dict[str, Any]] = []
    for (locality, speedup, beta), group in groups.items():
        row: dict[str, Any] = {
            "locality": locality,
            "speedup": speedup,
            "beta": beta,
            "n_runs": len(group),
            "n_successful_runs": sum(not run.error for run in group),
            "n_ok_total": sum(run.n_ok for run in group),
            "n_failed_total": sum(run.n_failed for run in group),
            "cache_metric": ",".join(sorted({run.cache_metric for run in group})),
        }
        for metric in METRICS:
            values = [getattr(run, metric) for run in group]
            finite = [float(value) for value in values
                      if value is not None and math.isfinite(float(value))]
            row[f"{metric}_mean"] = statistics.fmean(finite) if finite else None
            row[f"{metric}_stdev"] = statistics.stdev(finite) if len(finite) > 1 else (
                0.0 if finite else None
            )
        rows.append(row)
    return rows


def fmt(value: float | None, *, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if pct else f"{value:.3f}"


def report(rows: list[dict[str, Any]]) -> None:
    print()
    print("Neutral cell summary (mean +/- sample stdev; no winner rule applied)")
    print(f"{'locality':<9} {'load':>5} {'beta':>5} {'runs':>5} "
          f"{'hit rate':>17} {'TTFT p50':>19} {'TTFT p99':>19} "
          f"{'load CV':>17} {'contains':>17}")
    for row in rows:
        def pair(metric: str, percent: bool = False) -> str:
            mean = fmt(row[f"{metric}_mean"], pct=percent)
            stdev = fmt(row[f"{metric}_stdev"], pct=percent)
            return f"{mean} +/- {stdev}"

        print(f"{row['locality']:<9} {row['speedup']:>5g} {row['beta']:>5g} "
              f"{row['n_successful_runs']:>5} {pair('cache_hit_rate', True):>17} "
              f"{pair('ttft_p50_s'):>19} {pair('ttft_p99_s'):>19} "
              f"{pair('load_cv'):>17} {pair('contains', True):>17}")

    if any(row["e2e_ttft_p50_s_mean"] is not None for row in rows):
        print()
        print("Latency clocks exposed by the scorer")
        print(f"{'locality':<9} {'load':>5} {'beta':>5} "
              f"{'phase2 p50':>19} {'phase2 p99':>19} "
              f"{'e2e p50':>19} {'e2e p99':>19}")
        for row in rows:
            def pair(metric: str) -> str:
                return (f"{fmt(row[f'{metric}_mean'])} +/- "
                        f"{fmt(row[f'{metric}_stdev'])}")

            print(f"{row['locality']:<9} {row['speedup']:>5g} {row['beta']:>5g} "
                  f"{pair('phase2_ttft_p50_s'):>19} "
                  f"{pair('phase2_ttft_p99_s'):>19} "
                  f"{pair('e2e_ttft_p50_s'):>19} "
                  f"{pair('e2e_ttft_p99_s'):>19}")


def write_summaries(args, schedule: list[ScheduledCell], runs: list[RunResult],
                    rows: list[dict[str, Any]], outdir: Path) -> None:
    json_path = outdir / "beta_sweep_summary.json"
    csv_path = outdir / "beta_sweep_summary.csv"
    payload = {
        "config": {
            "strategy": STRATEGY,
            "order": ORDER,
            "tracker_capacity": args.tracker_capacity,
            "repeats": args.repeats,
            "loads": parse_numbers(args.loads, "--loads"),
            "betas": parse_numbers(args.betas, "--betas"),
            "high_trace": args.high_trace,
            "low_trace": args.low_trace,
            "corpus": args.corpus,
            "model": args.model,
            "tokenizer": args.tokenizer,
            "top_k": args.top_k,
            "limit": args.limit,
            "workers": args.worker,
            "base_port": args.port,
        },
        "schedule": [
            {"repeat": item.repeat, "sequence": item.sequence, **asdict(item.cell)}
            for item in schedule
        ],
        "runs": [asdict(run) for run in runs],
        "cells": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")

    fieldnames = list(rows[0]) if rows else [
        "locality", "speedup", "beta", "n_runs", "n_successful_runs",
        "n_ok_total", "n_failed_total", "cache_metric",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nJSON summary: {json_path}")
    print(f"CSV summary : {csv_path}")


def run(args) -> None:
    cells = build_cells(args)
    schedule = build_schedule(cells, args.repeats)
    if args.dry_run:
        print_plan(schedule, args.tracker_capacity)
        print("dry-run: no paths checked, subprocesses started, network calls made, "
              "or files written")
        return

    validate_live_args(args, schedule)
    # Rebuild after live-path normalisation so the recorded trace paths are
    # absolute and exactly match the files handed to replay.py.
    schedule = build_schedule(build_cells(args), args.repeats)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for item in schedule:
        confirm_cold(args, item, len(schedule))
        result = run_cell(args, item, outdir)
        results.append(result)
        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  hit={fmt(result.cache_hit_rate, pct=True)} "
                  f"TTFT p50={fmt(result.ttft_p50_s)}s "
                  f"p99={fmt(result.ttft_p99_s)}s "
                  f"CV={fmt(result.load_cv)} contains={fmt(result.contains, pct=True)}")

    rows = aggregate(results)
    report(rows)
    write_summaries(args, schedule, results, rows, outdir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", default="./corpus")
    parser.add_argument("--high-trace", "--trace-high", dest="high_trace",
                        default="runs/trace_hot.jsonl")
    parser.add_argument("--low-trace", "--trace-low", dest="low_trace",
                        default="runs/trace_low_locality.jsonl")
    parser.add_argument("--loads", default="30,35",
                        help="comma-separated replay speedups (default: 30,35)")
    parser.add_argument("--betas", default="1,0",
                        help="comma-separated ROUTER_BETA values (default: 1,0)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--worker", action="append", default=[],
                        help="vLLM worker URL; repeat exactly twice for a live run")
    parser.add_argument("--tracker-capacity", type=int, default=None,
                        help="measured cache capacity in tracker blocks; required live")
    parser.add_argument("--tokenizer", default="hf")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--outdir", default="runs/beta_sweep")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
