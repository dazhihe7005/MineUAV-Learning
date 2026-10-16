"""R15 fixed-reference guards; no framework import or optimizer."""
import importlib,importlib.util,subprocess,sys,tempfile,unittest
from pathlib import Path
from research_ops.storage import OpsError,atomic_json,sha256
from research_regression.authorized_rl.budget import validate_experiment


class CenteredFeedbackTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.centered_feedback'),
                             'missing independent R15 centered checkpoint diagnostic')
        return importlib.import_module('research_regression.authorized_rl.centered_feedback')

    def test_inference_profile_cannot_be_used_as_trainable_budget(self):
        m=self.module();p=m.configuration()
        self.assertEqual((p['model_count'],p['case_count'],p['new_inference_rows']),(2,62,124))
        self.assertEqual((p['optimizer_updates'],p['training_timesteps'],p['physics_steps'],p['test_episodes']),(0,0,0,0))
        with self.assertRaises(OpsError):validate_experiment(p)

    def test_only_completed_known_history_can_be_used(self):
        m=self.module();good={'stop_requested':False,'attempts':[{'experiment_id':'ARL_R6_DISTANCE_SEED0','status':'FAIL'},
            {'experiment_id':'ARL_R14_CENTERED_ACTOR_SEED0','status':'COMPLETE'}]}
        m.eligible_budget(good)
        for status in ('RESERVED_OR_INTERRUPTED','FAIL'):
            with self.assertRaises(OpsError):m.eligible_budget({'stop_requested':False,'attempts':[{'experiment_id':'ARL_R14_CENTERED_ACTOR_SEED0','status':status}]})
        with self.assertRaises(OpsError):m.eligible_budget({**good,'stop_requested':True})

    def test_missing_history_is_not_replaced_with_new_data(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.references(Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_missing_current_admission_refuses_execution(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_checks({'source_identity':{}},Path(d))

    def test_non_checkpoint_released_raw_corruption_is_rejected(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);f=r/'raw.json';atomic_json(f,{'source':'FORCED_TEST_FIXTURE'},exclusive=True)
            report={'raw_files':[{'path':'raw.json','bytes':f.stat().st_size,'sha256':sha256(f)}]}
            m.verify_released_raw(report,r);atomic_json(f,{'source':'CORRUPTED_FORCED_TEST_FIXTURE'})
            with self.assertRaises(OpsError):m.verify_released_raw(report,r)
            report['raw_files'][0]['path']='../other.json'
            with self.assertRaises(OpsError):m.verify_released_raw(report,r)

    def test_retention_does_not_invent_initial_velocity_braking(self):
        m=self.module();t=m.configuration()['retention']
        self.assertEqual(t['all_velocity_response_differences'],'WITHIN_REFERENCE_ZERO_TOLERANCE')
        self.assertEqual(t['velocity_response_difference_error_max_m_s'],.05)
        self.assertIn('not physical',m.configuration()['claim_limit'])

    def test_no_arbitrary_checkpoint_code_or_training_cli(self):
        self.module()
        for opt in ('--checkpoint','--script','--seed','--timesteps'):
            p=subprocess.run([sys.executable,'-m','research_regression.authorized_rl.centered_feedback_run',opt,'x'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('unrecognized arguments',p.stderr)


if __name__=='__main__':unittest.main()
