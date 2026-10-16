"""No ML imports: new fixed profiles must preserve old authority/history."""
import tempfile
import unittest
from pathlib import Path
from research_ops.storage import OpsError,digest
from research_regression.authorized_rl.budget import BudgetStore,grant,experiment,validate_experiment


class StdProfileTests(unittest.TestCase):
    def profile(self,variant='RANDOM'):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.std_profile'), 'missing fixed std profile')
        from research_regression.authorized_rl import std_profile
        self.assertTrue(callable(getattr(std_profile,'std_experiment',None)), 'missing fixed std profile')
        return std_profile.std_experiment(variant)

    def test_fixed_arms_use_original_cumulative_budget_not_new_authority(self):
        a=self.profile();b=self.profile('PRIOR')
        validate_experiment(a);validate_experiment(b)
        self.assertEqual(a['experiment_id'],'ARL_R3_RANDOM_STD03_SEED0')
        self.assertEqual(b['experiment_id'],'ARL_R3_PRIOR_STD03_SEED0')
        self.assertEqual(a['initial_std'],.3)
        self.assertEqual(a['validation_steps'],[0,32768,65536])
        self.assertEqual(digest(grant()),'1897b889fd3fea3f269abed9533104c123ef4d81fd9843d1cbe0d448db1cb7fc')

    def test_unreviewed_budget_std_checkpoint_and_evaluation_cannot_execute(self):
        for k,v in [('initial_std',.1),('timesteps',131072),('seed',1),('checkpoint','L9.zip'),
                    ('experiment_id','ARL_R3_RETRY'),('variant','OTHER'),('evaluation',{}),('schema',4)]:
            p=self.profile();p[k]=v
            with self.subTest(key=k),self.assertRaises(OpsError):validate_experiment(p)

    def test_three_prior_events_and_both_new_arms_share_one_ledger(self):
        from research_regression.authorized_rl.prior_profile import prior_experiment
        with tempfile.TemporaryDirectory() as d:
            s=BudgetStore(Path(d)/'ledger');s.initialize(expected_grant_sha256=digest(grant()))
            for p in (experiment('UNIFORM',65536,3600),experiment('CURRICULUM',65536,3600),prior_experiment()):
                s.reserve(p);s.fail(p['experiment_id'],'SYNTHETIC_NO_TRAINING')
            old={p.name:p.read_bytes() for p in (s.home/'events').iterdir()}
            for arm in ('RANDOM','PRIOR'):
                p=self.profile(arm);s.reserve(p);s.fail(p['experiment_id'],'SYNTHETIC_NO_TRAINING')
                with self.assertRaises(OpsError):s.reserve(p)
            self.assertEqual(s.status()['reserved_timesteps'],327680)
            self.assertEqual(s.status()['remaining_wall_seconds'],68400)
            self.assertTrue(all((s.home/'events'/n).read_bytes()==data for n,data in old.items()))

    def test_unknown_variant_is_rejected_not_coerced_to_prior(self):
        self.profile()
        from research_regression.authorized_rl.std_profile import std_experiment
        with self.assertRaises(OpsError):std_experiment('other')

    def test_missing_current_checks_refuses_before_reservation(self):
        p=self.profile();p.update(source_identity_sha256='a'*64,registration_sha256='b'*64)
        from research_regression.authorized_rl.std_profile import verify_std_checks
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):verify_std_checks(p,Path(d))

    def test_explicit_cli_has_no_checkpoint_or_script_capability(self):
        import subprocess,sys
        p=subprocess.run([sys.executable,'-m','research_regression.authorized_rl','--help'],text=True,capture_output=True,timeout=10)
        self.assertEqual(p.returncode,0)
        self.assertIn('prepare-std',p.stdout)
        self.assertIn('start-std',p.stdout)
        self.assertNotIn('--checkpoint',p.stdout)
        self.assertNotIn('--script',p.stdout)


if __name__=='__main__':unittest.main()
