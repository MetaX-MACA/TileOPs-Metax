#!/usr/bin/env python3
"""Compare TileOPs benchmark JSON records by workload and shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEYS = ("workload", "op", "name", "shape", "dtype")
METRICS = ("latency_ms", "time_ms", "median_ms")


def _load(path: Path) -> dict[tuple[tuple[str, str], ...], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("results", data) if isinstance(data, dict) else data
    out = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        key = tuple((field, str(record[field])) for field in KEYS if field in record)
        if key:
            out[key] = record
    return out


def _metric(record: dict[str, Any]) -> float:
    for metric in METRICS:
        if metric in record:
            return float(record[metric])
    raise ValueError(f"no latency metric in record: {record}")


def compare(base: Path, candidate: Path, threshold: float) -> dict[str, Any]:
    base_records = _load(base)
    candidate_records = _load(candidate)
    comparisons = []
    regressions = []
    for key in sorted(base_records.keys() & candidate_records.keys()):
        before = _metric(base_records[key])
        after = _metric(candidate_records[key])
        change = ((after - before) / before * 100.0) if before else 0.0
        row = {"key": dict(key), "base_ms": before, "candidate_ms": after, "change_percent": round(change, 4)}
        comparisons.append(row)
        if change > threshold:
            regressions.append(row)
    return {"matched": len(comparisons), "threshold_percent": threshold, "regressions": regressions, "comparisons": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = compare(args.base, args.candidate, args.threshold)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 1 if report["regressions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
