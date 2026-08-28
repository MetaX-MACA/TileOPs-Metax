#!/usr/bin/env python3
"""Collect TileOPs runtime environment metadata as JSON."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def pkg(name: str) -> dict[str, object]:
    if importlib.util.find_spec(name) is None:
        return {"installed": False}
    try:
        return {"installed": True, "version": version(name)}
    except PackageNotFoundError:
        return {"installed": True, "version": None}


def collect() -> dict[str, object]:
    payload = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "environment": {
            "MACA_PATH": os.environ.get("MACA_PATH"),
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        "packages": {
            "torch": pkg("torch"),
            "tilelang": pkg("tilelang"),
            "tileops": pkg("tileops"),
            "pytest": pkg("pytest"),
        },
    }
    try:
        import torch

        payload["torch_runtime"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda": getattr(torch.version, "cuda", None),
        }
    except Exception as exc:
        payload["torch_runtime"] = {"error": str(exc)}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect TileOPs environment JSON.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(collect(), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
