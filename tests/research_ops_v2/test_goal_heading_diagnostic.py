"""R27 fixed task-heading contrast, never a training admission."""
import copy,importlib,importlib.util,tempfile,unittest
from pathlib import Path
from research_ops.storage import OpsError,digest
from _historical_budget_fixture import r28_budget_reader

class GoalHeadingDiagnosticTests(unittest.TestCase):
    def module(self):
        name='research_regression.authorized_rl.goal_heading_diagnostic'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing bounded R27 goal heading contract')
        return importlib.import_module(name)
    def test_no_training_or_repeat_control_episodes(self):
        c=self.module().configuration()
        self.assertEqual((c['new_episodes'],c['reused_control_episodes'],c['optimizer_updates'],c['training_timesteps'],c['test_episodes']),(2,2,0,0,0))
        self.assertEqual(c['max_policy_steps_per_episode'],375)
        self.assertEqual(c['goal_yaw_mode'],'HOLD_INITIAL_ENU_YAW')
    def test_task_identity_changes_only_declared_heading_not_base_geometry(self):
        base={'identity_sha256':'a'*64,'start_yaw_rad':-2.8,'start_xyz':[0,0,1],'target_xyz':[2,0,1],'obstacle_configuration_hash':'b'*64}
        before=copy.deepcopy(base);task=self.module().heading_task(base)
        self.assertEqual(base,before);self.assertEqual(task['scenario'],base)
        self.assertEqual(task['goal_yaw_rad'],-2.8);self.assertNotEqual(task['identity_sha256'],base['identity_sha256'])
        forged=copy.deepcopy(base);forged['target_xyz'][0]=3
        self.assertNotEqual(self.module().heading_task(forged)['identity_sha256'],task['identity_sha256'])
    def test_missing_evidence_refuses_without_writes(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):self.module().references(Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])
    def test_no_admission_refuses_before_any_model_or_output(self):
        name='research_regression.authorized_rl.goal_heading_run'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing fixed R27 admission')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):importlib.import_module(name).verify_checks({'source_identity':{}},Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])
    def test_actual_sources_are_same_R26_original_validation_and_prior_only(self):
        root=Path(__file__).resolve().parents[2]
        with r28_budget_reader(root):
            s=self.module().references(root)
        self.assertEqual(s['model']['step'],0);self.assertEqual(len(s['cases']),2)
        self.assertTrue(all(v['scenario']['layout']=='empty' and v['scenario']['split']=='validation' for v in s['cases']))
        self.assertTrue(all(v['outcome']=='time_limit' for v in s['controls']))
    def fixture(self):
        cases=[self.module().heading_task({'identity_sha256':str(i)*64,'start_yaw_rad':yaw}) for i,yaw in [(1,-2.8),(2,-2.5)]]
        spec={'cases':cases,'controls':[{'outcome':'time_limit'},{'outcome':'time_limit'}],'model':{'step':0}}
        rows=[{'task_identity_sha256':c['identity_sha256'],'scenario_identity_sha256':c['scenario']['identity_sha256'],
            'goal_yaw_rad':c['goal_yaw_rad'],'outcome':'success','policy_steps':25,'physics_ticks':500,
            'optimizer_unchanged':True,'training_timesteps_before_after':0,'policy_sha256_before_after':'a'*64,
            'stop_metrics':{'max_consecutive_joint_samples':5}} for c in cases]
        return rows,spec
    def test_threshold_predefined_and_partial_success_not_full_pass(self):
        m=self.module();rows,s=self.fixture();self.assertEqual(m.evaluate(rows,s)['heading_contrast'],'PASS')
        rows[1].update(outcome='time_limit',policy_steps=375,physics_ticks=7500)
        self.assertEqual(m.evaluate(rows,s)['heading_contrast'],'FAIL')
    def test_future_wrong_identity_false_success_or_changed_model_refused(self):
        m=self.module()
        for change in ({'task_identity_sha256':'f'*64},{'goal_yaw_rad':0.},{'optimizer_unchanged':False},{'training_timesteps_before_after':True},
                {'physics_ticks':480},{'stop_metrics':{'max_consecutive_joint_samples':4}},{'outcome':'physical_failure'}):
            rows,s=self.fixture();rows[0].update(change)
            with self.assertRaises(ValueError):m.evaluate(rows,s)
        rows,s=self.fixture();rows[1]['policy_sha256_before_after']='b'*64
        with self.assertRaises(ValueError):m.evaluate(rows,s)

if __name__=='__main__':unittest.main()
