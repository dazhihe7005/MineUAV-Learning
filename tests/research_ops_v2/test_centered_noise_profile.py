"""R16 frozen single-noise-variable authorization; no ML or real ledger writes."""
import copy,importlib,importlib.util,subprocess,sys,tempfile,unittest
from pathlib import Path
from research_ops.storage import OpsError,digest
from research_regression.authorized_rl.budget import validate_experiment,grant,BudgetStore


class CenteredNoiseProfileTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.centered_noise_profile'),
                             'missing independently frozen R16 noise profile')
        return importlib.import_module('research_regression.authorized_rl.centered_noise_profile')

    def test_only_initial_noise_changes_algorithm_vs_centered_r14(self):
        from research_regression.authorized_rl.centered_training_profile import centered_experiment
        p=self.module().noise_experiment();old=centered_experiment()
        self.assertEqual((p['schema'],p['experiment_id'],p['initial_std']),(16,'ARL_R16_CENTERED_STD01_SEED0',.1))
        for k in ('learning_rate','timesteps','wall_seconds','seed','curriculum_rule','manifest_sha256',
                  'short_manifest_sha256','validation_steps','reward_version','n_steps','actor_parameterization'):
            self.assertEqual(p[k],old[k])
        validate_experiment(p)
        self.assertEqual(digest(grant()),'1897b889fd3fea3f269abed9533104c123ef4d81fd9843d1cbe0d448db1cb7fc')

    def test_seed_budget_reward_old_checkpoint_or_test_overrides_fail_closed(self):
        p=self.module().noise_experiment()
        for k,v in (('seed',1),('timesteps',131072),('workers',2),('learning_rate',3e-4),('initial_std',.3),
                    ('checkpoint','old.zip'),('test_episode_count',1),('reward_version','NEW'),
                    ('expected_initial_policy_sha256','0'*64),('actor_parameterization','UNCENTERED')):
            q=copy.deepcopy(p);q[k]=v
            with self.subTest(k=k),self.assertRaises(OpsError):validate_experiment(q)

    def test_missing_released_r14_r15_prevents_new_attempt(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_noise_proof(m.noise_experiment(),Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_missing_current_admission_blocks_before_reservation(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_noise_checks(m.noise_experiment(),Path(d))

    def test_cli_has_no_arbitrary_seed_checkpoint_script_budget_or_noise(self):
        self.module()
        for flag in ('--seed','--checkpoint','--script','--timesteps','--initial-std'):
            p=subprocess.run([sys.executable,'-m','research_regression.authorized_rl','prepare-centered-noise',flag,'x'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('unrecognized arguments',p.stderr)

    def test_mechanism_success_is_not_navigation_or_changed_threshold(self):
        p=self.module().noise_experiment();v=p['evaluation']
        self.assertEqual(v['short_candidate_success_min'],4);self.assertEqual(v['short_candidate_collision_max'],1)
        self.assertEqual(v['control_original_final_success'],1);self.assertEqual(v['control_original_final_collision_max'],3)
        self.assertEqual(v['short_train_success_ratio_control'],[34,108]);self.assertEqual(v['near_goal_fast_ratio_control'],[1867,2077])
        self.assertEqual(v['actual_new_validation_episodes'],36);self.assertEqual(v['reused_original_step0_episodes'],0)

    def test_same_cumulative_grant_charges_once_and_replay_remains_refused(self):
        p=self.module().noise_experiment()
        with tempfile.TemporaryDirectory() as d:
            store=BudgetStore(Path(d)/'FORCED_BUDGET_FIXTURE_ONLY');store.initialize(expected_grant_sha256=digest(grant()))
            store.reserve(p);self.assertEqual(store.status()['remaining_timesteps'],1934464)
            self.assertEqual(store.status()['remaining_wall_seconds'],82800)
            store.fail(p['experiment_id'],'FORCED_TEST_NO_TRAINING')
            with self.assertRaises(OpsError):BudgetStore(store.home).reserve(p)


if __name__=='__main__':unittest.main()
