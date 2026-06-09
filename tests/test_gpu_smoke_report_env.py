from scripts import gpu_smoke_report


def test_generate_report_includes_environment_section(monkeypatch):
    monkeypatch.setattr(
        gpu_smoke_report,
        "_get_runtime_env",
        lambda: {"python": "3.12.0", "cuda_available": "False"},
    )

    report = gpu_smoke_report.generate_report([], "quick")
    assert "## Environment" in report
    assert "| `python` | `3.12.0` |" in report
    assert "| `cuda_available` | `False` |" in report
