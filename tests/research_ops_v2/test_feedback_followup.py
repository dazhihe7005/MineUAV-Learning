"""R12 current-model, fixed-background diagnostic boundaries; no ML imports."""
import importlib,importlib.util,subprocess,sys,tempfile,unittest
from pathlib import Path
from research_ops.storage import OpsError,atomic_json,sha256,read_json
from research_regression.authorized_rl.budget import validate_experiment


class FeedbackFollowupTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.feedback_followup'),'missing independent R12 diagnostic')
        return importlib.import_module('research_regression.authorized_rl.feedback_followup')

    def test_fixed_scope_is_not_a_trainable_budget_profile(self):
        m=self.module();p=m.configuration()
        self.assertEqual(p['experiment_id'],'AUTORL_R12_LOW_KL_FEEDBACK_ONLY')
        self.assertEqual((p['model_count'],p['case_count'],p['optimizer_updates'],p['new_physics_steps']),(4,62,0,0))
        with self.assertRaises(OpsError):validate_experiment(p)

    def test_only_closed_known_failures_are_eligible(self):
        m=self.module();good={'stop_requested':False,'attempts':[{'experiment_id':'ARL_R6_DISTANCE_SEED0','status':'FAIL'},{'experiment_id':'COMPLETE','status':'COMPLETE'}]}
        m.eligible_budget(good)
        for entry in ({'experiment_id':'ARL_R6_DISTANCE_SEED0','status':'RESERVED_OR_INTERRUPTED'},{'experiment_id':'UNKNOWN','status':'FAIL'}):
            with self.assertRaises(OpsError):m.eligible_budget({'stop_requested':False,'attempts':[entry]})
        with self.assertRaises(OpsError):m.eligible_budget({**good,'stop_requested':True})

    def test_missing_historical_background_not_replaced_with_new_scan(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.references(Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_missing_current_checks_never_launches(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_checks({'source_identity':{}},Path(d))

    def test_only_four_preselected_new_checkpoints_in_references(self):
        m=self.module();self.assertEqual(set(m.SUMMARIES),{'ARL_R10_PRIOR_DISTANCE_SEED0','ARL_R11_PRIOR_LOWER_LR_SEED0'})
        self.assertEqual(m.configuration()['model_steps'],[32768,65536])
        self.assertEqual(m.configuration()['reference_usage'],'R4_R3_PRIOR_STEP0_REUSED_NO_NEW_INFERENCE')

    def test_no_arbitrary_checkpoint_script_or_budget_cli(self):
        self.module()
        for opt in ('--checkpoint','--script','--seed','--timesteps'):
            p=subprocess.run([sys.executable,'-m','research_regression.authorized_rl.feedback_followup_run',opt,'x'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('unrecognized arguments',p.stderr)

    def test_feedback_retention_is_not_navigation_success(self):
        m=self.module();p=m.configuration()
        self.assertEqual(p['retention']['zero_velocity_norm_max_m_s'],.05)
        self.assertEqual(p['retention']['max_action_difference_to_reference'],.05)
        self.assertEqual(p['new_navigation_episodes'],0)

    def test_noncheckpoint_raw_evidence_corruption_is_not_blessed_as_new_baseline(self):
        m=self.module();self.assertTrue(hasattr(m,'verify_released_raw'),'missing released-raw manifest gate')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f=root/'raw.json';atomic_json(f,{'source':'FORCED_TEST_FIXTURE'},exclusive=True)
            report={'raw_files':[{'path':'raw.json','bytes':f.stat().st_size,'sha256':sha256(f)}]}
            m.verify_released_raw(report,root)
            atomic_json(f,{'source':'CORRUPTED_TEST_FIXTURE'})
            with self.assertRaises(OpsError):m.verify_released_raw(report,root)
            report['raw_files'][0]['path']='../outside.json'
            with self.assertRaises(OpsError):m.verify_released_raw(report,root)

    def test_prior_retention_does_not_demand_velocity_feedback_absent_from_original(self):
        m=self.module();t=m.configuration()['retention']
        self.assertEqual(t.get('velocity_response_difference_error_max_m_s'),.05,'retention must accept original zero velocity sensitivity')
        self.assertEqual(t['all_velocity_response_differences'],'WITHIN_REFERENCE_ZERO_TOLERANCE')
        old=read_json(Path(__file__).resolve().parents[2]/'.research_local/autorl_r4_feedback/run_01/result.json')
        prior=next(v for v in old['models'] if v['reference']['experiment_id']=='ARL_R3_PRIOR_STD03_SEED0' and v['reference']['actual_step']==0)
        for axis,name in enumerate(('x','y','z')):
            chosen=[v for v in prior['rows'] if v['axis']==name and v['state'][axis]==.1]
            self.assertEqual(len({v['normalized_action'][axis] for v in chosen}),1)


if __name__=='__main__':unittest.main()
