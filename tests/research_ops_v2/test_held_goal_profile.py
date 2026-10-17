"""R28 fixed changed-task profile, not a generic parameter override API."""
import copy,importlib,importlib.util,tempfile,unittest
from pathlib import Path
from research_ops.storage import read_json,OpsError,digest

class HeldGoalProfileTests(unittest.TestCase):
    def module(self):
        name='research_regression.authorized_rl.held_goal_profile'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing fixed R28 profile')
        return importlib.import_module(name)
    def prereg(self):return read_json(Path(__file__).resolve().parents[2]/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
    def test_algorithm_and_budget_frozen_no_checkpoint_init(self):
        p=self.module().held_goal_experiment()
        self.assertEqual((p['schema'],p['seed'],p['workers'],p['timesteps'],p['wall_seconds']),(28,0,1,65536,3600))
        self.assertEqual((p['gamma'],p['gae_lambda'],p['learning_rate'],p['initial_std']),(.99,.95,3e-5,.1))
        self.assertEqual(p['reward_version'],'NAV_V2_COLLISION_PREFLIGHT_V1');self.assertEqual(p['validation_steps'],[0,32768,65536])
        self.assertEqual(p['expected_initial_policy_sha256'],'650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4')
        self.assertEqual(p['test_episode_count'],0);self.assertNotIn('expected_prefix_policy_sha256',p)
        self.assertEqual(p['goal_yaw_mode'],'HOLD_INITIAL_ENU_YAW')
    def test_profile_rejects_unknown_and_core_parameter_changes(self):
        m=self.module();p=m.held_goal_experiment();m.validate_held_goal_profile(p)
        for key,value in [('seed',1),('workers',2),('timesteps',65664),('learning_rate',3e-4),('reward_version','other'),('checkpoint_path','old.zip')]:
            altered=copy.deepcopy(p);altered[key]=value
            with self.assertRaises(OpsError):m.validate_held_goal_profile(altered)
    def test_manifest_geometry_and_targets_unchanged_no_test_tasks(self):
        m=self.module();p=self.prereg();manifest=m.held_goal_manifest(p)
        self.assertEqual(len(manifest['train']['SHORT']),12);self.assertEqual(len(manifest['train']['ORIGINAL']),12)
        self.assertEqual(len(manifest['validation']['SHORT']),6);self.assertEqual(len(manifest['validation']['ORIGINAL']),6)
        self.assertEqual(manifest['test'],'SEALED_NOT_INCLUDED')
        for row,task in zip(p['scenario_manifest']['train'],manifest['train']['ORIGINAL']):
            self.assertEqual(task['scenario'],row);self.assertAlmostEqual(task['goal_yaw_rad'],row['start_yaw_rad'],places=14)
            self.assertNotEqual(task['identity_sha256'],row['identity_sha256'])
        all_ids=[t['identity_sha256'] for group in ('train','validation') for stage in ('SHORT','ORIGINAL') for t in manifest[group][stage]]
        self.assertEqual(len(set(all_ids)),36)
    def test_manifest_mutation_or_wrong_base_is_not_accepted(self):
        m=self.module();p=self.prereg();manifest=m.held_goal_manifest(p);m.verify_held_goal_manifest(manifest,p)
        altered=copy.deepcopy(manifest);altered['validation']['SHORT'][0]['goal_yaw_rad']=0.
        with self.assertRaises(ValueError):m.verify_held_goal_manifest(altered,p)
        altered=copy.deepcopy(p);altered['scenario_manifest']['train'][0]['start_xyz'][0]+=.1
        with self.assertRaises(ValueError):m.held_goal_manifest(altered)
    def test_missing_published_heading_evidence_blocks_without_writes(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_held_goal_proof(m.held_goal_experiment(),Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])
    def test_budget_accepts_fixed28_only_after_explicit_validator_dispatch(self):
        p=self.module().held_goal_experiment()
        from research_regression.authorized_rl.budget import validate_experiment
        validate_experiment(p)
        p['timesteps']+=128
        with self.assertRaises(OpsError):validate_experiment(p)

if __name__=='__main__':unittest.main()
