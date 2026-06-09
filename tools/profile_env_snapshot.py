#!/usr/bin/env python3
"""Collect environment metadata for TileOPs profiling runs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def snapshot() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "torch": _version("torch"),
            "tilelang": _version("tilelang"),
            "tileops": _version("tileops"),
        },
        "commands": {
            "python": shutil.which("python") or "",
            "mxcc": shutil.which("mxcc") or "",
            "cmake_maca": shutil.which("cmake_maca") or "",
        },
        "environment": {
            "MACA_PATH": os.environ.get("MACA_PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(snapshot(), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
