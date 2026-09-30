#!/usr/bin/env python3
"""List tests that are explicitly marked as smoke tests."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def _has_smoke_mark(node: ast.AST) -> bool:
    text = ast.unparse(node) if hasattr(ast, "unparse") else ""
    return "pytest.mark.smoke" in text or "mark.smoke" in text


def collect_smoke_tests(root: Path) -> list[str]:
    tests: list[str] = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        module_smoke = any(
            isinstance(node, ast.Assign) and "pytest.mark.smoke" in ast.unparse(node)
            for node in tree.body
            if hasattr(ast, "unparse")
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                is_smoke = module_smoke or any(_has_smoke_mark(dec) for dec in node.decorator_list)
                if is_smoke and node.name.startswith("test_"):
                    tests.append(f"{path.relative_to(root).as_posix()}::{node.name}")
    return tests


def main() -> int:
    parser = argparse.ArgumentParser(description="List TileOPs smoke tests.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    tests = collect_smoke_tests(args.root.resolve())
    if args.json:
        print(json.dumps({"count": len(tests), "tests": tests}, indent=2))
    else:
        print("\n".join(tests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
