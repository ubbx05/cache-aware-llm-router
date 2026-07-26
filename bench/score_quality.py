"""Score answer quality against TQuAD gold spans.

This exists because "canonical ordering raised the cache hit rate" is only half
a result. Canonical ordering gives up putting the most relevant chunk first,
and models are known to attend unevenly across a long context, so the ordering
could plausibly cost accuracy. Without measuring that, the cache win cannot be
claimed as free.

Metric choice matters here. TQuAD gold answers are short extracted spans
("Afrikalı Leo") while the model produces a sentence, so classic SQuAD F1
punishes verbosity rather than error and will look artificially bad. The
primary metric is therefore containment -- did the model's answer contain the
gold span -- with token recall and F1 reported alongside. Report all three in
the paper and say which one is primary; picking whichever looks best after the
fact is the thing a reviewer will catch.

Usage:
    python score_quality.py r_canon.jsonl
    python score_quality.py r_canon.jsonl r_relevance.jsonl   # A/B
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# str.lower() mishandles Turkish: "İ" becomes "i" plus a combining dot, and "I"
# should fold to "ı" rather than "i". Mapping first avoids both problems.
_TR_LOWER = str.maketrans({"İ": "i", "I": "ı", "Ş": "ş", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ç": "ç"})
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(_TR_LOWER).lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    return normalise(text).split()


def score_one(pred: str, gold: str) -> dict[str, float]:
    p, g = tokens(pred), tokens(gold)
    if not g:
        return {}
    if not p:
        return {"em": 0.0, "f1": 0.0, "recall": 0.0, "contains": 0.0}

    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    precision = overlap / len(p)
    recall = overlap / len(g)
    f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0

    return {
        "em": float(normalise(pred) == normalise(gold)),
        "f1": f1,
        "recall": recall,
        "contains": float(normalise(gold) in normalise(pred)),
    }


def evaluate(path: Path) -> dict:
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    scored = []
    skipped = 0
    for r in rows:
        if r.get("error") or not r.get("gold_answer") or not r.get("output_text"):
            skipped += 1
            continue
        scored.append(score_one(r["output_text"], r["gold_answer"]))

    if not scored:
        raise SystemExit(
            f"{path}: nothing scoreable. Re-run replay.py after the update that "
            "records output_text, otherwise there are no answers to grade."
        )

    keys = ("contains", "recall", "f1", "em")
    out = {k: sum(s[k] for s in scored) / len(scored) for k in keys}
    out["n"] = len(scored)
    out["skipped"] = skipped
    out["mean_output_tokens"] = sum(
        r.get("output_tokens", 0) for r in rows if not r.get("error")
    ) / max(1, sum(1 for r in rows if not r.get("error")))
    return out


def show(name: str, s: dict) -> None:
    print(f"{name}")
    print(f"  scored           : {s['n']}" + (f"  (skipped {s['skipped']})" if s["skipped"] else ""))
    print(f"  contains gold    : {s['contains']:.1%}   <- primary")
    print(f"  token recall     : {s['recall']:.1%}")
    print(f"  F1               : {s['f1']:.1%}")
    print(f"  exact match      : {s['em']:.1%}")
    print(f"  mean out tokens  : {s['mean_output_tokens']:.0f}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    a = evaluate(Path(sys.argv[1]))
    show(sys.argv[1], a)

    if len(sys.argv) > 2:
        b = evaluate(Path(sys.argv[2]))
        print()
        show(sys.argv[2], b)
        print()
        d = a["contains"] - b["contains"]
        print(f"delta (contains) : {d:+.1%}")
        # A single run cannot separate a small effect from run-to-run noise.
        # Three runs per arm is the minimum before this difference means anything.
        if abs(d) < 0.03:
            print("  within noise for a single run -- repeat both arms ~3x before reporting")


if __name__ == "__main__":
    main()
