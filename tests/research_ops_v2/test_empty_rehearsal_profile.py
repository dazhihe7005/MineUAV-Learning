"""R17 fixed sampling contrast, no ML and no real ledger writes."""
import copy,importlib,importlib.util,subprocess,sys,tempfile,unittest
from pathlib import Path
from research_ops.storage import OpsError,digest
from research_regression.authorized_rl.budget import validate_experiment,grant,BudgetStore


class EmptyRehearsalProfileTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.empty_rehearsal_profile'),
                             'missing frozen R17 reset-sampling profile')
        return importlib.import_module('research_regression.authorized_rl.empty_rehearsal_profile')

    def test_only_reset_sampling_changes_algorithm_from_r16(self):
        from research_regression.authorized_rl.centered_noise_profile import noise_experiment
        p=self.module().rehearsal_experiment();old=noise_experiment();validate_experiment(p)
        self.assertEqual((p['schema'],p['experiment_id']),(17,'ARL_R17_EMPTY_REHEARSAL_SEED0'))
        for k in ('initial_std','expected_initial_policy_sha256','learning_rate','timesteps','wall_seconds','seed',
                  'manifest_sha256','short_manifest_sha256','validation_steps','reward_version','n_steps','actor_parameterization'):
            self.assertEqual(p[k],old[k])
        self.assertEqual(p['original_empty_probability'],.5)

    def test_changed_weights_reward_budget_seed_and_test_access_refused(self):
        p=self.module().rehearsal_experiment()
        for k,v in (('original_empty_probability',.75),('seed',1),('timesteps',131072),('workers',2),
                    ('initial_std',.3),('checkpoint','old.zip'),('test_episode_count',1),('reward_version','NEW'),
                    ('expected_prefix_policy_sha256','0'*64)):
            q=copy.deepcopy(p);q[k]=v
            with self.subTest(k=k),self.assertRaises(OpsError):validate_experiment(q)

    def test_missing_control_evidence_fails_before_attempt(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_rehearsal_proof(m.rehearsal_experiment(),Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_missing_current_checks_blocks(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_rehearsal_checks(m.rehearsal_experiment(),Path(d))

    def test_cli_cannot_supply_arbitrary_weights_or_initial_checkpoint(self):
        self.module()
        for flag in ('--probability','--seed','--script','--checkpoint','--timesteps'):
            p=subprocess.run([sys.executable,'-m','research_regression.authorized_rl','prepare-empty-rehearsal',flag,'x'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('unrecognized arguments',p.stderr)

    def test_no_lowered_navigation_criteria_or_old_evaluation_reuse(self):
        p=self.module().rehearsal_experiment();e=p['evaluation']
        self.assertEqual((e['short_candidate_success_min'],e['short_candidate_collision_max']),(4,1))
        self.assertEqual((e['control_original_final_success'],e['control_original_final_collision_max']),(0,3))
        self.assertEqual((e['actual_new_validation_episodes'],e['reused_original_step0_episodes']),(36,0))

    def test_same_grant_charged_once_no_retry_after_failure(self):
        p=self.module().rehearsal_experiment()
        with tempfile.TemporaryDirectory() as d:
            s=BudgetStore(Path(d)/'FORCED_TEST_ONLY');s.initialize(expected_grant_sha256=digest(grant()))
            s.reserve(p);self.assertEqual(s.status()['remaining_timesteps'],1934464)
            s.fail(p['experiment_id'],'FORCED_TEST_NOT_TRAINING')
            with self.assertRaises(OpsError):s.reserve(p)

    def test_exact_prefix_guard_rejects_different_weights_at_32k(self):
        m=self.module();self.assertTrue(hasattr(m,'verify_rehearsal_prefix'),'missing 32k prefix equivalence guard')
        m.verify_rehearsal_prefix(32768,'33bf6699158a35842fc05c3d4ca97f391f778db60c759fdad2564a6e94c87f60')
        with self.assertRaises(OpsError):m.verify_rehearsal_prefix(32768,'0'*64)
