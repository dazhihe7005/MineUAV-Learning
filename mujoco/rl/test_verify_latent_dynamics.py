import json
import tempfile
import unittest
from pathlib import Path

try:
    from verify_latent_dynamics_one_step import verify_report
except ImportError:
    verify_report = None


class VerificationTests(unittest.TestCase):
    def test_missing_or_corrupted_report_artifact_fails_verification(self):
        self.assertIsNotNone(verify_report, 'artifact verifier is not implemented')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'report.json'
            path.write_text(json.dumps(dict(models={'markov': {'artifact': {
                'path': str(Path(td) / 'missing.pt'), 'sha256': 'invalid'}}})))
            with self.assertRaises((FileNotFoundError, AssertionError, ValueError)):
                verify_report(path)


if __name__ == '__main__':
    unittest.main()
