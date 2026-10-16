"""R18 fixed read-only evaluation permissions, no ML import."""
import importlib,importlib.util,tempfile,unittest
from pathlib import Path
from research_ops.storage import OpsError


class StoppingDynamicsProfileTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.stopping_dynamics'),
                             'missing R18 fixed stopping diagnostic')
        return importlib.import_module('research_regression.authorized_rl.stopping_dynamics')

    def test_three_fixed_policies_one_val_no_training_or_test(self):
        c=self.module().configuration()
        self.assertEqual((c['model_count'],c['episodes'],c['max_policy_steps_per_model']),(3,3,375))
        self.assertEqual((c['training_timesteps'],c['optimizer_updates'],c['test_episodes']),(0,0,0))
        self.assertEqual(c['validation_identity'],'fe85ac443a0dc1bd2c676b6455f28ef0a4d15ea046287b692cc7eb3f6eefe06f')

    def test_missing_history_refuses_without_creating_attempt(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.references(Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_missing_current_checks_refuses(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_checks({'source_identity':{}},Path(d))

    def test_stop_or_unknown_training_status_blocks_new_diagnostic(self):
        self.module()
        from research_regression.authorized_rl.feedback_followup import eligible_budget
        for b in ({'stop_requested':True,'attempts':[]},{'stop_requested':False,'attempts':[{'status':'UNKNOWN','experiment_id':'x'}]}):
            with self.assertRaises(OpsError):eligible_budget(b)
