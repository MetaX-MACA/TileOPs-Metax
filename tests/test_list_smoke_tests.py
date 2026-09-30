from pathlib import Path

from scripts.list_smoke_tests import collect_smoke_tests


def test_collect_smoke_tests_detects_decorator(tmp_path: Path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_demo.py").write_text(
        "import pytest\n@pytest.mark.smoke\ndef test_demo():\n    pass\n",
        encoding="utf-8",
    )

    assert collect_smoke_tests(tmp_path) == ["tests/test_demo.py::test_demo"]
