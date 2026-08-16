"""Repeat the main routing-strategy comparison under one calibrated protocol.

The sweep is deliberately repeat-major and rotated.  Every strategy runs once
before any strategy gets its second run, and the first strategy changes on
each repeat.  This spreads slow machine drift across arms instead of turning
it into an apparent strategy effect.

Every live run also starts from a cold engine cache.  vLLM has no cache-reset
endpoint in this deployment, so the script asks the operator to restart both
workers before it starts a fresh router process.  Router stdout/stderr goes to
a per-run file through ``sweep_overlap_load.start_router``; leaving uvicorn on
an unread PIPE can fill the pipe buffer and deadlock a long replay.

The default table contains the five primary strategies.  ``--all-strategies``
adds the adaptive and semantic variants.  A measured ``--tracker-capacity`` is
mandatory for live runs, because the uncalibrated default has already changed
the outcome of an ablation in this project.

Examples:
    # Validate the 15-run plan without touching files or starting processes.
    python sweep_strategy.py --dry-run

    # Main 5-strategy table: 800 requests, k=10, speedup=8, three repeats.
    python sweep_strategy.py --corpus ./corpus --trace runs/trace_hot.jsonl \\
        --worker http://100.89.101.52:8000 \\
        --worker http://100.97.250.11:8000 \\
        --tracker-capacity 5840

    # Include adaptive_cache_aware and semantic_per_worker_tree too.
    python sweep_strategy.py --all-strategies ...
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

BENCH_DIR = Path(__file__).resolve().parent
# ``--dry-run`` promises no writes.  Set this before importing local modules so
# Python cannot create bench/__pycache__ as a side effect of merely showing the
# plan on a clean checkout.
sys.dont_write_bytecode = True
sys.path.insert(0, str(BENCH_DIR))

from score_quality import evaluate  # noqa: E402
from sweep_overlap_load import (  # noqa: E402
    run_replay,
    score_results,
    start_router,
    stop_router,
    wait_healthy,
)


@dataclass(frozen=True)
class StrategyConfig:
    label: str
    env: dict[str, str]
    order: str


CORE_STRATEGIES: tuple[StrategyConfig, ...] = (
    StrategyConfig("round_robin", {"ROUTER_STRATEGY": "round_robin"}, "canonical"),
    StrategyConfig("least_loaded", {"ROUTER_STRATEGY": "least_loaded"}, "canonical"),
    StrategyConfig(
        "cacheweaver_dualmap",
        {"ROUTER_STRATEGY": "cacheweaver_dualmap"},
        "canonical",
    ),
    StrategyConfig("cache_aware", {"ROUTER_STRATEGY": "cache_aware"}, "canonical"),
    StrategyConfig(
        "per_worker_tree",
        {
            "ROUTER_STRATEGY": "per_worker_tree",
            "ROUTER_OVERLAP_ADAPTIVE_MODE": "off",
        },
        "per_worker_tree",
    ),
)

EXTRA_STRATEGIES: tuple[StrategyConfig, ...] = (
    StrategyConfig(
        "adaptive_cache_aware",
        {"ROUTER_STRATEGY": "adaptive_cache_aware"},
        "canonical",
    ),
    StrategyConfig(
        "semantic_per_worker_tree",
        {"ROUTER_STRATEGY": "semantic_per_worker_tree"},
        "per_worker_tree",
    ),
)

ALL_STRATEGIES: tuple[StrategyConfig, ...] = CORE_STRATEGIES + EXTRA_STRATEGIES
BY_LABEL = {strategy.label: strategy for strategy in ALL_STRATEGIES}


@dataclass(frozen=True)
class RunSpec:
    sequence: int
    repeat: int
    strategy: StrategyConfig


@dataclass
class RunResult:
    strategy: str
    repeat: int
    sequence: int
    order: str
    port: int
    results_path: str
    router_log_path: str
    n_ok: int = 0
    n_failed: int = 0
    cache_hit_rate: float = float("nan")
    cache_metric: str = "unavailable"
    cached_tokens_total: int = 0
    prompt_tokens_total: int = 0
    # replay.py's ttft_s is phase 2 for the two-phase per-worker protocol and
    # the ordinary TTFT for one-phase strategies.  Keep the label explicit in
    # the output so it cannot be mistaken for end-to-end latency.
    phase2_ttft_p50_s: float = float("nan")
    phase2_ttft_p99_s: float = float("nan")
    e2e_ttft_p50_s: float = float("nan")
    e2e_ttft_p99_s: float = float("nan")
    e2e_total_p50_s: float = float("nan")
    e2e_total_p99_s: float = float("nan")
    throughput_req_s: float = float("nan")
    load_cv: float = float("nan")
    contains_gold: float = float("nan")
    quality_n: int = 0
    error: str = ""


METRICS: tuple[str, ...] = (
    "cache_hit_rate",
    "phase2_ttft_p50_s",
    "phase2_ttft_p99_s",
    "e2e_ttft_p50_s",
    "e2e_ttft_p99_s",
    "e2e_total_p50_s",
    "e2e_total_p99_s",
    "throughput_req_s",
    "load_cv",
    "contains_gold",
)


@dataclass
class StrategySummary:
    strategy: str
    completed_runs: int
    failed_runs: int
    metrics: dict[str, dict[str, float | int | None]] = field(default_factory=dict)


def percentile(values: Iterable[float], p: int) -> float:
    vals = sorted(values)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    return statistics.quantiles(vals, n=100, method="inclusive")[p - 1]


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def json_safe(value: Any) -> Any:
    """Replace non-finite floats recursively; strict JSON has no NaN value."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def select_strategies(args: argparse.Namespace) -> list[StrategyConfig]:
    if args.strategies:
        wanted = [name.strip() for name in args.strategies.split(",") if name.strip()]
        if not wanted:
            raise SystemExit("--strategies is empty")
        if len(set(wanted)) != len(wanted):
            raise SystemExit("--strategies contains a duplicate label")
        unknown = [name for name in wanted if name not in BY_LABEL]
        if unknown:
            raise SystemExit(
                f"unknown strategy label(s): {unknown}; valid: {sorted(BY_LABEL)}"
            )
        return [BY_LABEL[name] for name in wanted]
    return list(ALL_STRATEGIES if args.all_strategies else CORE_STRATEGIES)


def build_plan(strategies: list[StrategyConfig], repeats: int) -> list[RunSpec]:
    """Build a repeat-major schedule whose starting arm rotates each repeat."""
    plan: list[RunSpec] = []
    sequence = 0
    for repeat in range(repeats):
        offset = repeat % len(strategies)
        rotated = strategies[offset:] + strategies[:offset]
        for strategy in rotated:
            sequence += 1
            plan.append(RunSpec(sequence, repeat, strategy))
    return plan


def show_plan(args: argparse.Namespace, plan: list[RunSpec]) -> None:
    print(
        f"strategy sweep: repeats={args.repeats}, top_k={args.top_k}, "
        f"limit={args.limit}, speedup={args.speedup:g}, tokenizer={args.tokenizer}, "
        f"semantic_top_k={args.semantic_top_k}"
    )
    print(f"{'run':>4} {'repeat':>6} {'strategy':<28} order")
    print("-" * 68)
    for spec in plan:
        print(
            f"{spec.sequence:>4} {spec.repeat + 1:>6} "
            f"{spec.strategy.label:<28} {spec.strategy.order}"
        )
    print(f"\n{len(plan)} runs (dry-run: no files written, no processes started)")


def validate_live_args(args: argparse.Namespace) -> None:
    if args.tracker_capacity is None or args.tracker_capacity <= 0:
        raise SystemExit(
            "a positive, measured --tracker-capacity is required for live runs "
            "(--dry-run is the only exception)"
        )
    if len(args.worker) != 2:
        raise SystemExit("exactly two --worker URLs are required for a live run")
    if not args.trace:
        raise SystemExit("--trace is required for a live run")

    for attr in ("corpus", "trace"):
        path = Path(getattr(args, attr)).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"--{attr}: not found: {path}")
        setattr(args, attr, str(path))


def confirm_cold(args: argparse.Namespace, spec: RunSpec, total: int) -> None:
    print()
    print("=" * 76)
    print(
        f"  [{spec.sequence}/{total}] repeat {spec.repeat + 1} | "
        f"strategy: {spec.strategy.label}"
    )
    print("=" * 76)
    print("  Cold-start requirement: restart every vLLM worker before this run:")
    for worker in args.worker:
        print(f"    -> {worker}")
    input("  Both workers are restarted, healthy, and cold? [Enter] ")


def read_e2e_totals(path: Path) -> tuple[float, float]:
    """Read optional end-to-end totals added by the two-phase replay path."""
    values: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("error") is not None:
                continue
            value = row.get("e2e_total_s")
            if finite(value):
                values.append(float(value))
    return percentile(values, 50), percentile(values, 99)


def score_run(
    spec: RunSpec,
    port: int,
    results_path: Path,
    router_log_path: Path,
    workers: list[str],
) -> RunResult:
    expected_workers = [f"w{i + 1}" for i in range(len(workers))]
    cell = score_results(results_path, expected_workers)
    result = RunResult(
        strategy=spec.strategy.label,
        repeat=spec.repeat + 1,
        sequence=spec.sequence,
        order=spec.strategy.order,
        port=port,
        results_path=str(results_path),
        router_log_path=str(router_log_path),
        n_ok=cell.n_ok,
        n_failed=cell.n_failed,
        cache_hit_rate=cell.cache_hit_rate,
        cache_metric=getattr(cell, "cache_metric", "unavailable"),
        cached_tokens_total=getattr(cell, "cached_tokens_total", 0),
        prompt_tokens_total=getattr(cell, "prompt_tokens_total", 0),
        phase2_ttft_p50_s=cell.ttft_p50_s,
        phase2_ttft_p99_s=getattr(cell, "ttft_p99_s", float("nan")),
        e2e_ttft_p50_s=getattr(cell, "e2e_ttft_p50_s", float("nan")),
        e2e_ttft_p99_s=getattr(cell, "e2e_ttft_p99_s", float("nan")),
        throughput_req_s=cell.throughput_req_s,
        load_cv=cell.load_cv,
        error=cell.error,
    )
    result.e2e_total_p50_s, result.e2e_total_p99_s = read_e2e_totals(results_path)
    try:
        quality = evaluate(results_path)
    except SystemExit as exc:
        print(f"  quality unavailable: {exc}")
    else:
        result.contains_gold = quality["contains"]
        result.quality_n = quality["n"]
    return result


def run_one(
    args: argparse.Namespace,
    spec: RunSpec,
    port: int,
    outdir: Path,
) -> RunResult:
    stem = f"strategy_{spec.strategy.label}_r{spec.repeat + 1}"
    results_path = outdir / f"{stem}.jsonl"
    router_log_path = outdir / f"router_{port}_{spec.strategy.label}_r{spec.repeat + 1}.log"
    router_url = f"http://127.0.0.1:{port}"

    # Pin the shared policy knobs that define this protocol instead of
    # inheriting a stale shell override from an earlier beta/timing experiment.
    env = {
        "ROUTER_ALPHA": "1.0",
        "ROUTER_BETA": "1.0",
        "ROUTER_DELTA0": "0.5",
        "ROUTER_LOAD_REF": "16",
        "ROUTER_TRACKER_TIMING": "dispatch",
        "ROUTER_PROTECT_TOP_K": "0",
        "ROUTER_TOKENIZER": args.tokenizer,
        "ROUTER_TOKENIZER_MODEL": args.model,
        "ROUTER_TRACKER_CAPACITY": str(args.tracker_capacity),
        **spec.strategy.env,
    }
    if spec.strategy.label == "adaptive_cache_aware":
        env.update({
            "ROUTER_D_TARGET": "0.322",
            "ROUTER_DRIFT_LAM": "0.1",
            "ROUTER_CUSUM_K": "0.03",
            "ROUTER_CUSUM_H": "0.30",
        })
    if spec.strategy.label == "semantic_per_worker_tree":
        env["ROUTER_SEMANTIC_TOP_K"] = str(args.semantic_top_k)

    proc = start_router(args, env, port, router_log_path)
    try:
        asyncio.run(wait_healthy(router_url, timeout_s=args.router_timeout))
        run_replay(
            args,
            Path(args.trace),
            router_url,
            spec.strategy.order,
            args.speedup,
            results_path,
        )
    finally:
        stop_router(proc)

    result = score_run(spec, port, results_path, router_log_path, args.worker)
    print(
        f"  hit={result.cache_hit_rate:.1%}  "
        f"phase2 TTFT p50/p99={result.phase2_ttft_p50_s:.3f}/"
        f"{result.phase2_ttft_p99_s:.3f}s  "
        f"e2e TTFT p50/p99={result.e2e_ttft_p50_s:.3f}/"
        f"{result.e2e_ttft_p99_s:.3f}s  load CV={result.load_cv:.3f}  "
        f"contains={result.contains_gold:.1%}"
    )
    return result


def failed_result(spec: RunSpec, port: int, outdir: Path, exc: Exception) -> RunResult:
    stem = f"strategy_{spec.strategy.label}_r{spec.repeat + 1}"
    return RunResult(
        strategy=spec.strategy.label,
        repeat=spec.repeat + 1,
        sequence=spec.sequence,
        order=spec.strategy.order,
        port=port,
        results_path=str(outdir / f"{stem}.jsonl"),
        router_log_path=str(
            outdir / f"router_{port}_{spec.strategy.label}_r{spec.repeat + 1}.log"
        ),
        error=f"{type(exc).__name__}: {exc}",
    )


def summarise(results: list[RunResult], strategies: list[StrategyConfig]) -> list[StrategySummary]:
    summaries: list[StrategySummary] = []
    for strategy in strategies:
        runs = [run for run in results if run.strategy == strategy.label]
        completed = [run for run in runs if not run.error]
        metric_summary: dict[str, dict[str, float | int | None]] = {}
        for metric in METRICS:
            values = [
                float(getattr(run, metric))
                for run in completed
                if finite(getattr(run, metric))
            ]
            if not values:
                metric_summary[metric] = {"mean": None, "stdev": None, "n": 0}
            else:
                metric_summary[metric] = {
                    "mean": statistics.fmean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
        summaries.append(
            StrategySummary(
                strategy=strategy.label,
                completed_runs=len(completed),
                failed_runs=len(runs) - len(completed),
                metrics=metric_summary,
            )
        )
    return summaries


def print_summary(summaries: list[StrategySummary]) -> None:
    print()
    print("=" * 118)
    print("Repeated strategy summary (mean +/- sample stdev)")
    print("=" * 118)
    print(
        f"{'strategy':<28}{'cache hit':>18}{'phase2 TTFT p50':>22}"
        f"{'phase2 TTFT p99':>22}{'load CV':>14}{'contains':>14}"
    )
    for summary in summaries:
        def pair(metric: str) -> tuple[float, float]:
            row = summary.metrics[metric]
            return float(row["mean"] or 0.0), float(row["stdev"] or 0.0)

        hit, hit_sd = pair("cache_hit_rate")
        p50, p50_sd = pair("phase2_ttft_p50_s")
        p99, p99_sd = pair("phase2_ttft_p99_s")
        cv, cv_sd = pair("load_cv")
        quality, quality_sd = pair("contains_gold")
        print(
            f"{summary.strategy:<28}{hit:>10.1%} +/-{hit_sd:.1%}"
            f"{p50:>13.3f}s +/-{p50_sd:.3f}"
            f"{p99:>13.3f}s +/-{p99_sd:.3f}"
            f"{cv:>8.3f} +/-{cv_sd:.3f}"
            f"{quality:>8.1%} +/-{quality_sd:.1%}"
        )


def separation_verdict(summaries: list[StrategySummary]) -> dict[str, Any] | None:
    """Predeclared comparison: cache_aware against CacheWeaver DualMap."""
    by_name = {summary.strategy: summary for summary in summaries}
    names = ("cache_aware", "cacheweaver_dualmap")
    if not all(name in by_name for name in names):
        return None

    verdict: dict[str, Any] = {"arms": list(names), "rule": "gap > 2*max(stdev)"}
    verdict["metrics"] = {}
    for metric in ("cache_hit_rate", "phase2_ttft_p50_s", "phase2_ttft_p99_s"):
        left = by_name[names[0]].metrics[metric]
        right = by_name[names[1]].metrics[metric]
        if left["mean"] is None or right["mean"] is None:
            verdict["metrics"][metric] = {"separated": None}
            continue
        gap = abs(float(left["mean"]) - float(right["mean"]))
        noise = 2.0 * max(float(left["stdev"] or 0.0), float(right["stdev"] or 0.0))
        verdict["metrics"][metric] = {
            "absolute_gap": gap,
            "noise_threshold": noise,
            "separated": gap > noise,
        }
    return verdict


def write_outputs(
    args: argparse.Namespace,
    strategies: list[StrategyConfig],
    results: list[RunResult],
    summaries: list[StrategySummary],
    outdir: Path,
) -> None:
    csv_path = outdir / "strategy_sweep.csv"
    metric_columns = [
        column
        for metric in METRICS
        for column in (f"{metric}_mean", f"{metric}_stdev", f"{metric}_n")
    ]
    csv_fields = ["strategy", "completed_runs", "failed_runs", *metric_columns]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for summary in summaries:
            row: dict[str, Any] = {
                "strategy": summary.strategy,
                "completed_runs": summary.completed_runs,
                "failed_runs": summary.failed_runs,
            }
            for metric, values in summary.metrics.items():
                row[f"{metric}_mean"] = values["mean"]
                row[f"{metric}_stdev"] = values["stdev"]
                row[f"{metric}_n"] = values["n"]
            writer.writerow(row)

    config = {
        "strategies": [strategy.label for strategy in strategies],
        "repeats": args.repeats,
        "top_k": args.top_k,
        "limit": args.limit,
        "speedup": args.speedup,
        "tokenizer": args.tokenizer,
        "semantic_top_k": args.semantic_top_k,
        "tracker_capacity": args.tracker_capacity,
        "model": args.model,
        "corpus": args.corpus,
        "trace": args.trace,
        "workers": args.worker,
    }
    json_path = outdir / "strategy_sweep.json"
    payload = {
        "config": config,
        "runs": [asdict(result) for result in results],
        "summaries": [asdict(summary) for summary in summaries],
        "cache_aware_vs_cacheweaver_dualmap": separation_verdict(summaries),
    }
    json_path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nJSON summary: {json_path}")
    print(f"CSV summary : {csv_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", default="./corpus")
    parser.add_argument("--trace", default="runs/trace_hot.jsonl")
    parser.add_argument("--outdir", default="runs/strategy_sweep")
    parser.add_argument("--worker", action="append", default=[])
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--tokenizer", default="hf")
    parser.add_argument(
        "--semantic-top-k",
        type=int,
        default=1,
        help="workers retained by semantic pre-filter (semantic arm only; pool size is 2)",
    )
    parser.add_argument("--tracker-capacity", type=int, default=None)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--router-timeout", type=float, default=120.0)
    parser.add_argument(
        "--all-strategies",
        "--all",
        action="store_true",
        help="add adaptive_cache_aware and semantic_per_worker_tree",
    )
    parser.add_argument(
        "--strategies",
        help="explicit comma-separated subset/order; overrides the default/all selection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the rotated plan; do not validate paths, write files, or start processes",
    )
    args = parser.parse_args(argv)

    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.speedup <= 0:
        parser.error("--speedup must be positive")
    if not (1 <= args.port <= 65535):
        parser.error("--port must be between 1 and 65535")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    strategies = select_strategies(args)
    if any(s.label == "semantic_per_worker_tree" for s in strategies):
        if not 1 <= args.semantic_top_k < 2:
            raise SystemExit(
                "--semantic-top-k must satisfy 1 <= k < 2 when the semantic "
                "strategy is selected; k=2 cannot prune a two-worker pool"
            )
    plan = build_plan(strategies, args.repeats)

    # This return is intentionally before path checks and mkdir: ``--dry-run``
    # must be usable on a laptop with no corpus, trace, workers, or GPU.
    if args.dry_run:
        show_plan(args, plan)
        return

    validate_live_args(args)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for offset, spec in enumerate(plan):
        port = args.port + offset
        if port > 65535:
            raise SystemExit("router port range exceeded 65535; choose a lower --port")
        confirm_cold(args, spec, len(plan))
        try:
            result = run_one(args, spec, port, outdir)
        except Exception as exc:  # noqa: BLE001 -- preserve the remaining sweep cells
            result = failed_result(spec, port, outdir, exc)
            print(f"  FAILED: {result.error}")
            print(f"  router log: {result.router_log_path}")
        results.append(result)

    summaries = summarise(results, strategies)
    print_summary(summaries)
    write_outputs(args, strategies, results, summaries, outdir)


if __name__ == "__main__":
    main()
