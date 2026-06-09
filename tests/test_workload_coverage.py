import tempfile
import unittest
from pathlib import Path

from tools.workload_coverage import coverage


class WorkloadCoverageTest(unittest.TestCase):
    def test_matches_workload_and_operator_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "workloads").mkdir()
            (root / "tileops" / "ops").mkdir(parents=True)
            (root / "workloads" / "gemm.py").write_text("", encoding="utf-8")
            (root / "tileops" / "ops" / "gemm.py").write_text("", encoding="utf-8")

            report = coverage(root)

        self.assertEqual(report["matched"], ["gemm"])


if __name__ == "__main__":
    unittest.main()
