import json
import tempfile
import unittest
from pathlib import Path

from tools.compare_benchmark_json import compare


class CompareBenchmarkJsonTest(unittest.TestCase):
    def test_flags_latency_regression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.json"
            candidate = root / "candidate.json"
            base.write_text(json.dumps([{"workload": "gemm", "shape": "m1", "latency_ms": 10.0}]), encoding="utf-8")
            candidate.write_text(json.dumps([{"workload": "gemm", "shape": "m1", "latency_ms": 11.0}]), encoding="utf-8")

            report = compare(base, candidate, threshold=5.0)

        self.assertEqual(len(report["regressions"]), 1)


if __name__ == "__main__":
    unittest.main()
