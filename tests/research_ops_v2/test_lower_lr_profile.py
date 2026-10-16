"""R11 fixed independent experiment and inherited authorization; zero ML imports."""
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


class LowerLRProfileTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.lower_lr_profile'),'missing independent R11 LR profile')
        return importlib.import_module('research_regression.authorized_rl.lower_lr_profile')

    def test_only_learning_rate_changes_optimizer_scientific_variable(self):
        from research_regression.authorized_rl.prior_distance_profile import prior_distance_experiment
        p=self.module().lower_lr_experiment();old=prior_distance_experiment()
        self.assertEqual((p['schema'],p['experiment_id'],p['learning_rate']),(11,'ARL_R11_PRIOR_LOWER_LR_SEED0',3e-5))
        for key in ('timesteps','wall_seconds','seed','initial_std','curriculum_rule','manifest_sha256','short_manifest_sha256','validation_steps','reward_version','initialization','expected_initial_policy_sha256'):
            self.assertEqual(p[key],old[key])
        validate_experiment(p);validate_experiment(old)
        self.assertEqual(digest(grant()),'1897b889fd3fea3f269abed9533104c123ef4d81fd9843d1cbe0d448db1cb7fc')

    def test_all_overrides_including_lr_fail_closed(self):
        p=self.module().lower_lr_experiment()
        for key,value in [('learning_rate',3e-4),('seed',1),('workers',2),('timesteps',131072),('checkpoint','old.zip'),('test_episode_count',1),('initial_std',.05),('control_summary_sha256','0'*64)]:
            q=copy.deepcopy(p);q[key]=value
            with self.subTest(key=key),self.assertRaises(OpsError):validate_experiment(q)

    def test_missing_frozen_r10_result_never_creates_attempt(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_lower_lr_proof(m.lower_lr_experiment(),Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_missing_current_admission_rejected(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_lower_lr_checks(m.lower_lr_experiment(),Path(d))

    def test_cli_no_checkpoint_seed_script_or_budget_override(self):
        for opt in ('--checkpoint','--seed','--script','--timesteps','--learning-rate'):
            p=subprocess.run([sys.executable,'-m','research_regression.authorized_rl','prepare-lower-lr',opt,'x'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('unrecognized arguments',p.stderr)

    def test_mechanism_and_learning_acceptance_are_not_conflated(self):
        p=self.module().lower_lr_experiment()
        self.assertEqual(p['evaluation']['first32_approx_kl_mean_max'],.07136825448833406*.5)
        self.assertEqual(p['evaluation']['control_original_final_collision_max'],1)
        self.assertEqual(p['evaluation']['short_candidate_success_min'],4)


if __name__=='__main__':unittest.main()
