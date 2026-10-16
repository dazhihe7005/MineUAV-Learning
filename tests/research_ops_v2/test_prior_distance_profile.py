"""Fixed R10 grant scope, prior proof and diagnostic boundaries; no ML imports."""
import copy
import importlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from research_ops.storage import OpsError,digest
from research_regression.authorized_rl.budget import validate_experiment,grant


class PriorDistanceProfileTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.prior_distance_profile'),'missing independent prior-distance profile')
        return importlib.import_module('research_regression.authorized_rl.prior_distance_profile')

    def test_only_initial_actor_changes_scientific_control(self):
        from research_regression.authorized_rl.repaired_distance_profile import repaired_distance_experiment
        m=self.module();p=m.prior_distance_experiment();old=repaired_distance_experiment()
        self.assertEqual((p['schema'],p['experiment_id']),(10,'ARL_R10_PRIOR_DISTANCE_SEED0'))
        for key in ('timesteps','wall_seconds','seed','initial_std','curriculum_rule','manifest_sha256','short_manifest_sha256','validation_steps','reward_version'):
            self.assertEqual(p[key],old[key])
        self.assertNotEqual(p['initialization'],old['initialization']);validate_experiment(p);validate_experiment(old)
        self.assertEqual(digest(grant()),'1897b889fd3fea3f269abed9533104c123ef4d81fd9843d1cbe0d448db1cb7fc')

    def test_budget_seed_checkpoint_test_or_initial_policy_override_refused(self):
        p=self.module().prior_distance_experiment()
        for key,value in [('seed',1),('timesteps',131072),('workers',2),('initial_std',.05),('checkpoint','old.zip'),('test_episode_count',1),('expected_initial_policy_sha256','0'*64),('control_summary_sha256','0'*64)]:
            q=copy.deepcopy(p);q[key]=value
            with self.subTest(key=key),self.assertRaises(OpsError):validate_experiment(q)

    def test_missing_or_changed_proof_cannot_authorize_reservation(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_prior_distance_proof(m.prior_distance_experiment(),Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_missing_current_admission_cannot_launch(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_prior_distance_checks(m.prior_distance_experiment(),Path(d))

    def test_cli_does_not_accept_script_checkpoint_or_budget_overrides(self):
        for option in ('--checkpoint','--script','--seed','--timesteps'):
            p=subprocess.run([sys.executable,'-m','research_regression.authorized_rl','prepare-prior-distance',option,'x'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('unrecognized arguments',p.stderr)

    def test_goal_diagnostics_separate_speed_from_contiguous_joint_samples(self):
        self.module()
        from research_regression.authorized_rl.prior_distance_training import GoalStopDiagnostics
        a=GoalStopDiagnostics()
        for distance,speed in [(.2,.1),(.09,.2),(.08,.1),(.07,.1),(.11,.1),(.09,.1)]:a.observe(distance,speed)
        self.assertEqual(a.record(),{'near_goal_samples':4,'near_goal_fast_samples':1,'joint_samples':3,'max_consecutive_joint_samples':2,'last_consecutive_joint_samples':1})
        for distance,speed in [(float('nan'),.1),(.1,-1)]:
            with self.assertRaises(ValueError):a.observe(distance,speed)

    def test_short_initial_evaluation_is_prior_not_random_or_learned(self):
        self.module()
        from research_regression.authorized_rl import prior_distance_training as training
        self.assertTrue(hasattr(training,'annotate_prior_validation'),'missing accurate R10 prior evaluation label')
        original={'training_timesteps_before_after':0,'policy_sha256_before_after':'6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74','model_status':'UNTRAINED_RANDOM_INITIALIZATION'}
        actual=training.annotate_prior_validation(original)
        self.assertEqual(actual['model_status'],'ANALYTIC_HAND_WRITTEN_PRIOR_NOT_LEARNED')
        self.assertEqual(original['model_status'],'UNTRAINED_RANDOM_INITIALIZATION')
        wrong={**original,'policy_sha256_before_after':'db33ba4df5f7aaf76f98b7915675cbfc80dbe52273045ed86811b5cc074dfa06'}
        with self.assertRaises(ValueError):training.annotate_prior_validation(wrong)


if __name__=='__main__':unittest.main()
