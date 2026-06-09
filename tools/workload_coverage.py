#!/usr/bin/env python3
"""Report TileOPs workload modules and matching operator implementations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _modules(root: Path, directory: str) -> set[str]:
    return {
        path.stem
        for path in (root / directory).rglob("*.py")
        if path.name != "__init__.py" and not path.stem.startswith("_")
    }


def coverage(root: Path) -> dict[str, object]:
    workloads = _modules(root, "workloads")
    ops = _modules(root, "tileops/ops")
    matched = sorted(workloads & ops)
    return {
        "workload_count": len(workloads),
        "operator_count": len(ops),
        "matched_count": len(matched),
        "matched": matched,
        "workloads_without_operator": sorted(workloads - ops),
        "operators_without_workload": sorted(ops - workloads),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(coverage(args.root), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
