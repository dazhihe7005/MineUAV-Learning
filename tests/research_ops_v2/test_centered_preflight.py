"""R13 fixed no-training preflight boundaries; standard library only."""
import importlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from research_ops.storage import OpsError
from research_regression.authorized_rl.budget import validate_experiment


class CenteredPreflightTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.centered_preflight'),
                             'missing R13 independent preflight protocol')
        return importlib.import_module('research_regression.authorized_rl.centered_preflight')

    def test_fixed_no_training_scope_cannot_be_reserved_as_a_training_profile(self):
        m = self.module()
        p = m.configuration()
        self.assertEqual(p['experiment_id'], 'AUTORL_R13_CENTERED_ACTOR_PREFLIGHT_ONLY')
        self.assertEqual((p['training_timesteps'], p['optimizer_updates'], p['physics_steps'], p['test_episodes']), (0, 0, 0, 0))
        with self.assertRaises(OpsError):
            validate_experiment(p)

    def test_missing_published_history_cannot_be_created_as_replacement(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OpsError):
                m.references(Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_stale_or_missing_admission_cannot_launch(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OpsError):
                m.verify_checks({'source_identity': {}}, Path(directory))

    def test_cli_rejects_arbitrary_script_training_or_checkpoint(self):
        self.module()
        for flag in ('--checkpoint', '--script', '--timesteps', '--seed'):
            p = subprocess.run([sys.executable, '-m', 'research_regression.authorized_rl.centered_preflight_run', flag, 'x'],
                               capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('unrecognized arguments', p.stderr)


if __name__ == '__main__':
    unittest.main()
