import unittest

from tools.profile_env_snapshot import snapshot


class ProfileEnvSnapshotTest(unittest.TestCase):
    def test_snapshot_has_reproducibility_sections(self):
        report = snapshot()

        self.assertIn("packages", report)
        self.assertIn("environment", report)
        self.assertIn("commands", report)


if __name__ == "__main__":
    unittest.main()
