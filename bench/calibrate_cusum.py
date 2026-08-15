"""Calibrate the drift detector's CUSUM_K / CUSUM_H / DRIFT_LAM on real traces.

These three constants are currently defaults validated only against the
synthetic step-change in adaptive_drift_model.__main__. That test proves the
mechanism fires; it says nothing about whether these particular values are
right for this workload, which is the same gap TRACKER_CAPACITY had before
Section VI-B -- and adaptive_drift_model's own docstring already flags it
("gercek trafikte kalibre edilmeli ... recipe, sayi degil").

No GPU and no router: the detector is a pure function of the overlap sequence,
so calibration only needs the sequence real traffic would have produced.

THE SEQUENCE MATTERS AND IS EASY TO GET WRONG. The router observes, for each
request in ARRIVAL order, the Jaccard against the previous request OF THE SAME
SESSION (strategies.py, AdaptiveCacheAware.select). That is not what
overlap_measurement.session_adjacent_pairs returns -- it groups by session and
emits each session's pairs together, which is the right shape for a summary
statistic and the wrong one for a detector whose whole behaviour is sequential.
Calibrating on the grouped order would produce numbers that look fine and
transfer to nothing. This script rebuilds the arrival-order sequence.

What is measured, for each (lam, k, h):

  * false alarms on a STABLE trace -- the binding constraint. Every alarm
    re-baselines d_ref and rescales beta, so a detector that cries drift on
    steady traffic makes routing worse than not adapting at all.
  * alarms on a DRIFTING trace -- the detector has to actually fire, or the
    adaptivity is decoration.

A parameter set is only useful if it separates the two. The recommendation at
the end applies one stated rule rather than eyeballing the grid; the full grid
is printed so a different rule can be applied to the same numbers.

Usage:
    python calibrate_cusum.py --corpus ./corpus \\
        --trace stable:runs/trace.jsonl --trace drift:runs/trace_drift01.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

from adaptive_drift_model import CusumDriftDetector, OnlineDriftEstimator  # noqa: E402
from overlap_measurement import jaccard, load_trace, retrieved_sets  # noqa: E402


def arrival_order_observations(trace: list[dict], sets: list[set]) -> list[float]:
    """The exact sequence AdaptiveCacheAware.select() sees.

    Mirrors strategies.py: keep the last retrieved set per session; on each
    arrival, if that session has been seen before, emit Jaccard(current, prev)
    and then replace. A session's first turn produces no observation, because
    the router has nothing to compare against yet -- feeding it a 0.0 there
    would look like total locality loss and is exactly the kind of phantom
    alarm this script exists to tune away.
    """
    order = sorted(range(len(trace)), key=lambda i: trace[i].get("request_id", i))
    last: dict = {}
    obs: list[float] = []
    for i in order:
        sid = trace[i].get("session_id")
        cur = sets[i]
        prev = last.get(sid)
        if prev is not None:
            obs.append(jaccard(cur, prev))
        last[sid] = cur
    return obs


def run_detector(obs: list[float], d_ref: float, lam: float, k: float, h: float,
                 feed_ewma: bool = False) -> dict:
    """Replay one observation sequence through one parameter set."""
    est = OnlineDriftEstimator(lam=lam)
    det = CusumDriftDetector(d_ref=d_ref, k=k, h=h)
    alarms_at: list[int] = []
    for i, x in enumerate(obs):
        d_t = est.update(x)
        if det.update(d_t if feed_ewma else x, current_ewma_estimate=d_t):
            alarms_at.append(i)
    return {
        "alarms": len(alarms_at),
        "per_1000": 1000.0 * len(alarms_at) / len(obs) if obs else float("nan"),
        "first_alarm": alarms_at[0] if alarms_at else None,
        "final_ewma": est.current,
    }


def cache_path(label: str, args, cache_dir: Path) -> Path:
    """Keyed by everything that changes the sequence, so a --top-k or --order
    change cannot silently reuse the wrong cached observations."""
    return cache_dir / f"obs_{label}_k{args.top_k}_{args.order}.json"


def observations_for(label: str, path: Path, args, cache_dir: Path) -> list[float]:
    """Embedding thousands of queries takes minutes and the grid sweep is
    instant, so the sequence is cached. Deleting the cache dir forces a
    recompute."""
    cache = cache_path(label, args, cache_dir)
    if cache.exists() and not args.no_cache:
        print(f"  {label}: cached ({cache})")
        return json.loads(cache.read_text(encoding="utf-8"))

    from replay import Corpus
    corpus = Corpus.load(Path(args.corpus))
    trace = load_trace(path, args.limit)
    if not trace:
        raise SystemExit(f"{path}: empty trace")
    sets = retrieved_sets(corpus, trace, args)
    obs = arrival_order_observations(trace, sets)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(obs), encoding="utf-8")
    print(f"  {label}: {len(obs)} observations -> cached")
    return obs


def separation(res: dict, stable: str, other: list[str]) -> float:
    """How many times more often the detector fires on drifting traffic than
    on steady traffic. This, not the raw alarm count, is what decides whether
    the detector is informative: a setting can be admirably quiet on the
    stable trace and equally quiet on the drifting one, which is silence, not
    detection. 1.0 means the two regimes are indistinguishable.
    """
    s_rate = res[stable]["per_1000"]
    if not other or s_rate <= 0:
        return float("nan")
    return statistics.fmean([res[l]["per_1000"] for l in other]) / s_rate


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="./corpus")
    p.add_argument("--trace", action="append", required=True,
                   help="label:path -- label containing 'stable' is the "
                        "false-alarm reference; repeatable")
    p.add_argument("--cache-dir", default="runs/cusum_cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--lam", default="0.05,0.1,0.2")
    p.add_argument("--k", default="0.02,0.03,0.05,0.08")
    p.add_argument("--h", default="0.10,0.15,0.20,0.30")
    p.add_argument("--max-false-per-1000", type=float, default=1.0,
                   help="lowest false-alarm budget to report in the trade-off "
                        "table; higher budgets are always shown alongside it")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--order", choices=["canonical", "relevance"], default="canonical")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--embed-model", default="intfloat/multilingual-e5-base")
    p.add_argument("--feed-ewma", action="store_true",
                   help="feed the CUSUM the smoothed EWMA instead of the raw "
                        "Jaccard. Tested and rejected: it silences the detector "
                        "(1-4 alarms per 2000 observations) on both traces "
                        "rather than sharpening it. Kept so that result is "
                        "reproducible, not because it should be used")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    specs = []
    for t in args.trace:
        if ":" not in t:
            raise SystemExit(f"--trace must be label:path, got {t!r}")
        label, _, path = t.partition(":")
        specs.append((label, Path(path).expanduser().resolve()))

    # Validated BEFORE any observation is built. Everything below this point
    # costs minutes of CPU embedding per trace, and being told about a
    # mislabelled argument after that wait is the kind of error that gets
    # discovered twice.
    stable_labels = [l for l, _ in specs if "stable" in l.lower()]
    if not stable_labels:
        raise SystemExit(
            f"no trace label contains 'stable' (got: {[l for l, _ in specs]}).\n"
            "One is required as the false-alarm reference -- the whole "
            "calibration is 'quiet here, loud there', and without a 'here' "
            "there is nothing to calibrate against.")
    stable = stable_labels[0]

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    # Only traces that still have to be embedded need to exist. A cached
    # sequence is the point of the cache: re-sweeping the grid must not
    # require the corpus and trace to still be sitting where they were.
    missing = [f"{label}: {path}" for label, path in specs
               if not path.exists()
               and not cache_path(label, args, cache_dir).exists()]
    if missing:
        raise SystemExit("trace file(s) not found and not cached:\n  "
                         + "\n  ".join(missing))
    print("building observation sequences (arrival order, session-adjacent):")
    obs_by_label = {label: observations_for(label, path, args, cache_dir)
                    for label, path in specs}

    # d_ref is not swept: it is a MEASURED property of the workload, the same
    # discipline D_TARGET is documented under in config.py. Calibrating it as a
    # free parameter would let a detector look quiet by simply expecting less.
    d_ref = statistics.fmean(obs_by_label[stable])
    print()
    print(f"stable trace       : {stable}  ({len(obs_by_label[stable])} observations)")
    print(f"measured d_ref     : {d_ref:.3f}   <- this is your ROUTER_D_TARGET")
    for label, obs in obs_by_label.items():
        if label != stable:
            print(f"{label:<19}: mean={statistics.fmean(obs):.3f}  n={len(obs)}")

    other = [l for l in obs_by_label if l != stable]
    rows = []
    for lam in parse_floats(args.lam):
        for k in parse_floats(args.k):
            for h in parse_floats(args.h):
                res = {l: run_detector(o, d_ref, lam, k, h, args.feed_ewma)
                       for l, o in obs_by_label.items()}
                rows.append((lam, k, h, res))

    print()
    header = f"{'lam':>6}{'k':>7}{'h':>7}{stable + ' /1k':>16}"
    for l in other:
        header += f"{l + ' /1k':>16}"
    header += f"{'sep':>10}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for lam, k, h, res in rows:
        line = f"{lam:>6.2f}{k:>7.2f}{h:>7.2f}{res[stable]['per_1000']:>16.2f}"
        for l in other:
            line += f"{res[l]['per_1000']:>16.2f}"
        line += f"{separation(res, stable, other):>9.2f}x"
        print(line)

    # A single recommendation hid the actual shape of this problem: the
    # quietest admissible setting is often quiet on BOTH traces, i.e. mute.
    # What matters is the trade-off, so it gets printed as one -- for each
    # false-alarm budget, the best separation available at that budget.
    if not other:
        print()
        print("Only a stable trace was given, so detection power is unmeasured.")
        print("Generate a drift trace (gen_trace.py --drift 0.1) and re-run;")
        print("without it these numbers only prove the detector stays quiet.")
        return

    print()
    print("trade-off (best separation available at each false-alarm budget):")
    print(f"{'budget /1k':>12}{'best sep':>11}{'lam':>7}{'k':>7}{'h':>7}"
          f"{'stable /1k':>13}")
    budgets = [b for b in (1, 5, 10, 25, 50, 100, 1e9)
               if b >= args.max_false_per_1000 or b == 1e9]
    seen = set()
    best_overall = None
    for budget in budgets:
        pool = [r for r in rows if r[3][stable]["per_1000"] <= budget]
        if not pool:
            continue
        best = max(pool, key=lambda r: separation(r[3], stable, other))
        lam, k, h, res = best
        sep = separation(res, stable, other)
        if (lam, k, h) in seen:
            continue
        seen.add((lam, k, h))
        label = "any" if budget > 1e8 else f"{budget:g}"
        print(f"{label:>12}{sep:>10.2f}x{lam:>7.2f}{k:>7.2f}{h:>7.2f}"
              f"{res[stable]['per_1000']:>13.2f}")
        if best_overall is None or sep > separation(best_overall[3], stable, other):
            best_overall = best

    print()
    print(f"measured ROUTER_D_TARGET = {d_ref:.3f}  "
          f"(top_k={args.top_k}, order={args.order})")
    print("This is top_k-dependent -- calibrate it at the top_k you actually")
    print("deploy, or the detector starts from the wrong reference.")

    if best_overall is not None:
        sep = separation(best_overall[3], stable, other)
        if sep <= 1.05:
            print()
            print("WARNING: no setting fires meaningfully more on drift than on")
            print("steady traffic. The detector is not separating the regimes on")
            print("this workload, so adaptivity built on it is not doing what it")
            print("claims -- a parameter choice cannot fix that.")


if __name__ == "__main__":
    main()
