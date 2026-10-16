"""Fixed no-training feedback audit boundary; no ML imports."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from research_ops.storage import OpsError


class FeedbackSpecTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.feedback_spec'),'missing fixed feedback specification')
        from research_regression.authorized_rl import feedback_spec
        return feedback_spec

    def test_fixed_cases_separate_goal_velocity_and_yaw_without_privileged_input(self):
        cases=self.module().probe_cases()
        self.assertEqual(len(cases),62)
        self.assertEqual(len({c['case_id'] for c in cases}),62)
        self.assertEqual(cases[0]['state'],[-2.,0.,0.,-1.,0.,0.,0.])
        self.assertTrue(all(c['provenance']=='OFFLINE_COUNTERFACTUAL_INPUT' for c in cases))
        for c in cases:self.module().validate_case(c)

    def test_nonfinite_dimensions_and_privileged_fields_fail_closed(self):
        module=self.module();base=module.probe_cases()[0]
        for key,value in [('state',[0]*6),('state',[float('nan')]+[0]*6),('state',[float('inf')]+[0]*6),('obstacle_map',[]),('provenance','PHYSICAL_TRAJECTORY')]:
            c=copy.deepcopy(base);c[key]=value
            with self.subTest(key=key),self.assertRaises(OpsError):module.validate_case(c)

    def test_yaw_wrap_domain_and_boolean_state_are_rejected(self):
        module=self.module();c=module.probe_cases()[-1];c['state'][-1]=4.
        with self.assertRaises(OpsError):module.validate_case(c)
        c=module.probe_cases()[0];c['state'][0]=True
        with self.assertRaises(OpsError):module.validate_case(c)

    def test_fixed_complete_checkpoint_coverage_has_no_test_or_warm_start(self):
        module=self.module();refs=module.checkpoint_references()
        self.assertEqual(len(refs),11)
        self.assertEqual(sorted({r['actual_step'] for r in refs}),[0,32768,65536])
        self.assertEqual(sum(r['experiment_id']=='ARL_R1_UNIFORM_SEED0' for r in refs),2)
        self.assertTrue(all(r['usage']=='READ_ONLY_INFERENCE_NOT_RESUME' for r in refs))

    def test_missing_history_is_not_replaced_by_a_new_model(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):module.checkpoint_references(Path(d))

    def test_modified_zip_is_rejected_before_any_model_import(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'changed.zip';p.write_bytes(b'SYNTHETIC_NOT_A_MODEL')
            with self.assertRaises(OpsError):module.verify_file(p,'0'*64)

    def test_missing_current_no_update_admission_refuses_execution(self):
        module=self.module()
        self.assertTrue(callable(getattr(module,'verify_probe_checks',None)),'missing no-update admission')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):module.verify_probe_checks({'source_identity':{}},Path(d))


if __name__=='__main__':unittest.main()
