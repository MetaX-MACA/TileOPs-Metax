#!/usr/bin/env python3
"""Summarize pytest JUnit XML into compact JSON."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> dict[str, object]:
    tree = ET.parse(path)
    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    for case in tree.iter("testcase"):
        if case.find("skipped") is not None:
            outcome = "skipped"
        elif case.find("failure") is not None or case.find("error") is not None:
            outcome = "failed"
        else:
            outcome = "passed"
        counts[outcome] += 1
        if outcome == "failed":
            failures.append(
                {
                    "classname": case.attrib.get("classname", ""),
                    "name": case.attrib.get("name", ""),
                }
            )
    return {
        "total": sum(counts.values()),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize JUnit XML as JSON.")
    parser.add_argument("xml", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(summarize(args.xml), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
