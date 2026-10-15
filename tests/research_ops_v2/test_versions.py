import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from support_v2 import fixture,proposal,review,git


def manager(root):
    assert importlib.util.find_spec('research_ops_v2'), 'versioned operations missing'
    from research_ops_v2.manager import OperationsV2
    return OperationsV2(root)


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=fixture(Path(self.temp.name)/'repo')
    def tearDown(self):self.temp.cleanup()

    def test_migration_preserves_exact_v1_files_and_uses_new_identity(self):
        m=manager(self.root);reg=proposal(self.root)
        paths=[self.root/'.research_local'/p for p in ('research_state.json','protocol.json','budget_ledger.json')]
        before=[p.read_bytes() for p in paths]
        result=m.register(reg,review(reg),migrating=True)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual([p.read_bytes() for p in paths],before)
        self.assertEqual(m.verify()['status'],'PASS')
        self.assertEqual(m.status()['state']['phase_id'],'OPS-02')
        self.assertEqual(m.status()['state']['training_budget'],0)
        with self.assertRaises(Exception):m.register(reg,review(reg),migrating=True)

    def test_bad_review_and_training_tasks_do_not_activate(self):
        m=manager(self.root);reg=proposal(self.root);bad=review(reg);bad['verdict']='NO_GO'
        with self.assertRaises(Exception):m.register(reg,bad,migrating=True)
        self.assertFalse((m.home/'active.json').exists())
        for task_type in ('ppo-pilot','formal-training','shell','python-script'):
            with self.assertRaises(Exception):proposal(self.root,tasks=[{'task_id':'x','task_type':task_type,'parameters':{},'wall_time_s':60,'workers':1}])

    def test_unknown_parameters_and_budget_escalation_are_rejected(self):
        m=manager(self.root)
        for parameters in ({'scope':'history','command':'touch marker'},{'scope':'history','script':'train.py'}):
            with self.assertRaises(Exception):proposal(self.root,tasks=[{'task_id':'x','task_type':'offline-audit','parameters':parameters,'wall_time_s':60,'workers':1}])
        reg=proposal(self.root);reg['training_budget']=4096
        with self.assertRaises(Exception):m.register(reg,review(reg),migrating=True)

    def test_actual_executor_and_tests_cannot_be_omitted_from_source_identity(self):
        from research_ops.storage import OpsError
        m=manager(self.root);reg=proposal(self.root);reg['source_paths']=[];reg['source_identity']={}
        with self.assertRaisesRegex(OpsError,'mandatory'):m.register(reg,review(reg),migrating=True)

    def test_source_changed_after_review_refuses_migration(self):
        m=manager(self.root);reg=proposal(self.root)
        (self.root/'research_ops_v2/core.py').write_text('# changed\n')
        with self.assertRaises(Exception):m.register(reg,review(reg),migrating=True)

    def test_new_phase_registration_preserves_previous_version(self):
        m=manager(self.root);a=proposal(self.root);m.register(a,review(a),migrating=True)
        old=m.status()['version_id'];old_bytes=(m.home/'versions'/old/'protocol.json').read_bytes()
        b=proposal(self.root,phase='DIAG-01');m.register(b,review(b))
        self.assertNotEqual(m.status()['version_id'],old)
        self.assertEqual((m.home/'versions'/old/'protocol.json').read_bytes(),old_bytes)
        self.assertEqual(m.status()['state']['training_budget'],0)
        p=m.home/'versions'/old/'protocol.json';p.write_bytes(old_bytes+b' ')
        from research_ops.storage import OpsError
        with self.assertRaisesRegex(OpsError,'previous operations version'):m.verify()

    def test_new_delivery_rules_reject_raw_history_and_escape(self):
        for path in ('../outside.md','mujoco/rl/models/model.pt','mujoco/reports/raw.npz','docs/token.key','/etc/passwd'):
            with self.assertRaises(Exception):proposal(self.root,sync_files=[path])

    def test_unknown_or_corrupt_v1_state_cannot_be_reinitialized(self):
        m=manager(self.root);reg=proposal(self.root)
        p=self.root/'.research_local/research_state.json';p.write_text('{broken')
        with self.assertRaises(Exception):m.register(reg,review(reg),migrating=True)
        self.assertEqual(p.read_text(),'{broken')

    def test_failed_atomic_migration_never_publishes_active_version(self):
        m=manager(self.root);reg=proposal(self.root)
        from unittest.mock import patch
        from research_ops_v2 import manager as module
        original=module.atomic_json
        def fail(path,value,**kwargs):
            if Path(path).name=='state.json':raise OSError('FORCED_TEST_STATE disk failure')
            return original(path,value,**kwargs)
        with patch.object(module,'atomic_json',side_effect=fail):
            with self.assertRaises(OSError):m.register(reg,review(reg),migrating=True)
        self.assertFalse((m.home/'active.json').exists())
        self.assertTrue((self.root/'.research_local/protocol.json').is_file())


if __name__=='__main__':unittest.main()
