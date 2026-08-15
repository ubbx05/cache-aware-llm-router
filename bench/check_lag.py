"""Did the load generator keep up with its own schedule?

replay.py fires each request at a scheduled offset. When the client cannot
keep up, the arrival pattern it actually produced is no longer the trace's,
and every claim about how a result varies WITH LOAD is then describing a load
that was never delivered. replay.py prints a warning about this, but the
sweeps capture its output, and older runs predate the sweeps recording it --
so this recomputes it from the results rows, which is where the evidence
actually lives.

Reported per file: p50/p99 of (sent_s - scheduled_s), against replay.py's own
1s p99 threshold. Files with no timing fields (manifests, sweep summaries) are
skipped rather than reported as broken.

Usage:
    python check_lag.py                 # everything under ./runs
    python check_lag.py runs/timing_s15 # one directory
    python check_lag.py a.jsonl b.jsonl # specific files
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

WARN_P99_S = 1.0  # identical to replay.py's own threshold


def lag_percentiles(path: Path) -> tuple[int, float, float] | None:
    lags = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                return None  # not a results file (e.g. a sweep summary)
            if not isinstance(r, dict) or "sent_s" not in r or "scheduled_s" not in r:
                return None
            lags.append(r["sent_s"] - r["scheduled_s"])
    if not lags:
        return None
    lags.sort()
    p50 = lags[min(int(len(lags) * 0.50), len(lags) - 1)]
    p99 = lags[min(int(len(lags) * 0.99), len(lags) - 1)]
    return len(lags), p50, p99


def collect(argv: list[str]) -> list[Path]:
    if not argv:
        argv = ["runs"]
    out: list[Path] = []
    for a in argv:
        p = Path(a)
        if p.is_dir():
            out += sorted(p.rglob("*.jsonl"))
        elif p.exists():
            out.append(p)
        else:
            print(f"skip (not found): {p}")
    return out


def main() -> None:
    paths = collect(sys.argv[1:])
    if not paths:
        raise SystemExit("no .jsonl files found -- check the path you passed")

    print(f"{len(paths)} .jsonl file(s) to inspect\n")
    print(f"{'file':<52}{'n':>7}{'p50':>10}{'p99':>10}")
    checked = behind = 0
    for p in paths:
        res = lag_percentiles(p)
        if res is None:
            continue  # no timing fields: not a replay results file
        n, p50, p99 = res
        checked += 1
        flag = ""
        if p99 > WARN_P99_S:
            behind += 1
            flag = "  <-- BEHIND"
        print(f"{str(p):<52}{n:>7}{p50 * 1000:>9.0f}ms{p99 * 1000:>9.0f}ms{flag}")

    print()
    if not checked:
        print("None of these files carry sent_s/scheduled_s -- nothing to judge.")
    elif behind:
        print(f"{behind}/{checked} run(s) fell behind their schedule at p99.")
        print("Those runs' arrival patterns are not the trace's, so any")
        print("load-axis comparison drawn across them is not supported.")
        print("Re-run them at a lower --speedup.")
    else:
        print(f"All {checked} run(s) kept their schedule (p99 < {WARN_P99_S:.0f}s).")
        print("The load axis is trustworthy.")


if __name__ == "__main__":
    main()
