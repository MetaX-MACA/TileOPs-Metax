import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import gpu_smoke_report


class GpuSmokeReportJsonTest(unittest.TestCase):
    def test_summarize_results_counts_outcomes(self):
        original_get_git_commit = gpu_smoke_report._get_git_commit
        gpu_smoke_report._get_git_commit = lambda: "abc123"
        try:
            results = [
                {"outcome": "passed", "op": "softmax"},
                {"outcome": "skipped", "op": "rms_norm"},
                {"outcome": "failed", "op": "softmax"},
                {"outcome": "failed", "op": "rms_norm"},
                {"outcome": "failed", "op": "softmax"},
            ]

            summary = gpu_smoke_report.summarize_results(results, "smoke")
        finally:
            gpu_smoke_report._get_git_commit = original_get_git_commit

        self.assertEqual(
            summary,
            {
                "target": "smoke",
                "git_commit": "abc123",
                "total_cases": 5,
                "passed_count": 1,
                "skipped_count": 1,
                "failed_count": 3,
                "failed_ops": ["softmax", "rms_norm"],
            },
        )

    def test_generate_report_can_be_written_as_utf8(self):
        report = gpu_smoke_report.generate_report(
            [{"outcome": "failed", "op": "softmax", "name": "case", "nodeid": "node", "failure_reason": "bad"}],
            "smoke",
        )

        self.assertIn("TileOPs GPU Smoke Report", report)
        report.encode("utf-8")


if __name__ == "__main__":
    unittest.main()
