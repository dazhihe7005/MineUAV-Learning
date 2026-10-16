"""Fixed R14 profile, no ML imports or actual budget changes."""
import copy
import importlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from research_ops.storage import OpsError, digest
from research_regression.authorized_rl.budget import validate_experiment, grant, BudgetStore


class CenteredTrainingProfileTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.centered_training_profile'),
                             'missing independently authorized R14 fixed profile')
        return importlib.import_module('research_regression.authorized_rl.centered_training_profile')

    def test_only_actor_function_changes_algorithm_against_lower_lr_control(self):
        from research_regression.authorized_rl.lower_lr_profile import lower_lr_experiment
        p = self.module().centered_experiment()
        old = lower_lr_experiment()
        self.assertEqual((p['schema'], p['experiment_id']), (14, 'ARL_R14_CENTERED_ACTOR_SEED0'))
        for k in ('learning_rate','timesteps','wall_seconds','seed','initial_std','curriculum_rule',
                  'manifest_sha256','short_manifest_sha256','validation_steps','reward_version','n_steps'):
            self.assertEqual(p[k], old[k])
        self.assertEqual(p['evaluation']['actual_new_validation_episodes'],36)
        self.assertEqual(p['evaluation']['reused_original_step0_episodes'],0)
        self.assertEqual(p['expected_initial_policy_sha256'],'20f5c226abc762e5aa4a9046ba400b2b0f7012b0da6715d14204b097cae06001')
        validate_experiment(p)
        self.assertEqual(digest(grant()),'1897b889fd3fea3f269abed9533104c123ef4d81fd9843d1cbe0d448db1cb7fc')

    def test_budget_seed_policy_checkpoint_and_test_overrides_fail_closed(self):
        p = self.module().centered_experiment()
        for k,v in (('seed',1),('timesteps',131072),('workers',2),('learning_rate',3e-4),
                    ('initial_std',.05),('checkpoint','old.zip'),('test_episode_count',1),
                    ('expected_initial_policy_sha256','0'*64),('actor_parameterization','UNCENTERED')):
            q = copy.deepcopy(p);q[k]=v
            with self.subTest(k=k),self.assertRaises(OpsError):
                validate_experiment(q)

    def test_missing_r13_release_blocks_without_creating_a_new_attempt(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):
                m.verify_centered_proof(m.centered_experiment(),Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_missing_current_admission_blocks_before_reservation(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):
                m.verify_centered_checks(m.centered_experiment(),Path(d))

    def test_explicit_cli_rejects_script_checkpoint_seed_and_budget_flags(self):
        self.module()
        for flag in ('--script','--checkpoint','--seed','--timesteps','--learning-rate'):
            p = subprocess.run([sys.executable,'-m','research_regression.authorized_rl','prepare-centered',flag,'x'],
                               capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('unrecognized arguments',p.stderr)

    def test_structural_zero_is_not_frozen_navigation_success(self):
        p = self.module().centered_experiment()
        self.assertEqual(p['evaluation']['short_candidate_success_min'],4)
        self.assertEqual(p['evaluation']['short_candidate_collision_max'],1)
        self.assertNotIn('first32_approx_kl_mean_max',p['evaluation'])
        self.assertIn('NOT_LEARNED_SUCCESS',p['zero_state_claim'])

    def test_same_grant_charges_new_profile_and_permanently_refuses_replay(self):
        p=self.module().centered_experiment()
        with tempfile.TemporaryDirectory() as d:
            home=Path(d)/'FORCED_BUDGET_TEST_FIXTURE'
            store=BudgetStore(home);store.initialize(expected_grant_sha256=digest(grant()))
            store.reserve(p)
            self.assertEqual(store.status()['reserved_timesteps'],65536)
            self.assertEqual(store.status()['remaining_timesteps'],1934464)
            self.assertEqual(store.status()['remaining_wall_seconds'],82800)
            store.fail(p['experiment_id'],'FORCED_TEST_FIXTURE_NO_ACTUAL_TRAINING')
            restored=BudgetStore(home)
            self.assertEqual(restored.status()['reserved_timesteps'],65536)
            with self.assertRaises(OpsError):
                restored.reserve(p)


if __name__ == '__main__':
    unittest.main()
