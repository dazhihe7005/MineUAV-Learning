import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from support import project


def api():
    assert importlib.util.find_spec('research_ops.identity'), 'identity audit missing'
    from research_ops import identity
    return identity


class IdentityTests(unittest.TestCase):
    def test_baseline_detects_modified_or_missing_history_without_restoring_it(self):
        m = api()
        with tempfile.TemporaryDirectory() as d:
            root = project(Path(d) / 'repo')
            config = json.loads((root/'research_ops/operations_config.json').read_text())
            baseline = m.capture_baseline(root, config)
            self.assertTrue(m.verify_baseline(root, baseline)['unchanged'])
            (root/'history/failure.json').write_text('{"result":"PASS"}')
            with self.assertRaises(m.OpsError): m.verify_baseline(root, baseline)
            self.assertEqual(json.loads((root/'history/failure.json').read_text())['result'], 'PASS')

    def test_untracked_identity_compares_names_and_contents_not_just_count(self):
        m = api()
        with tempfile.TemporaryDirectory() as d:
            root = project(Path(d)/'repo')
            (root/'a.data').write_bytes(b'a')
            config = json.loads((root/'research_ops/operations_config.json').read_text())
            config['history_worktrees'] = [str(root)]
            baseline = m.capture_baseline(root, config)
            (root/'a.data').unlink(); (root/'b.data').write_bytes(b'a')
            with self.assertRaises(m.OpsError): m.verify_baseline(root, baseline)

    def test_source_change_or_symlink_path_is_rejected(self):
        m = api()
        with tempfile.TemporaryDirectory() as d:
            root = project(Path(d)/'repo')
            initial = m.source_identity(root, ['runtime.py'])
            (root/'runtime.py').write_text('changed')
            self.assertNotEqual(initial, m.source_identity(root, ['runtime.py']))
            (root/'escape').symlink_to(Path(d))
            with self.assertRaises(m.OpsError): m.safe_path(root, 'escape/private')

    def test_environment_rejects_missing_python_and_wrong_package_version(self):
        assert importlib.util.find_spec('research_ops.environment'), 'environment checker missing'
        from research_ops.environment import inspect_environment
        with tempfile.TemporaryDirectory() as d:
            root = project(Path(d)/'repo')
            lock = json.loads((root/'research_ops/environment_lock.json').read_text())
            lock['interpreter'] = str(root/'missing-python')
            with self.assertRaises(Exception): inspect_environment(root, lock)
            import sys
            lock['interpreter'] = sys.executable
            lock['packages'] = {'definitely-not-installed-ops-package': '0.0.1'}
            with self.assertRaises(Exception): inspect_environment(root, lock)

    def test_ignored_historical_dataset_is_protected_in_other_worktree(self):
        m=api()
        with tempfile.TemporaryDirectory() as d:
            root=project(Path(d)/'repo');other=project(Path(d)/'other')
            (other/'.gitignore').write_text('*.npz\n.research_local/\n')
            (other/'datasets').mkdir();p=other/'datasets/raw.npz';p.write_bytes(b'original')
            config=json.loads((root/'research_ops/operations_config.json').read_text())
            config['historical_ignored_paths']={str(other):['datasets']}
            baseline=m.capture_baseline(root,config);p.write_bytes(b'changed')
            with self.assertRaises(m.OpsError):m.verify_baseline(root,baseline)


if __name__ == '__main__': unittest.main()
