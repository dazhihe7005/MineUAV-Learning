"""Schema2 is a fixed execution profile, never a new or reset authorization."""
import tempfile
import unittest
from pathlib import Path
from research_ops.storage import OpsError,digest
from research_regression.authorized_rl.budget import BudgetStore,grant,experiment,validate_experiment


class PriorProfileTests(unittest.TestCase):
    def profile(self):
        from research_regression.authorized_rl.prior_profile import prior_experiment
        return prior_experiment()

    def test_new_profile_uses_same_immutable_grant(self):
        p=self.profile();validate_experiment(p)
        self.assertEqual(p['experiment_id'],'ARL_R2_PRIOR_SEED0')
        self.assertEqual(p['schema'],2)
        self.assertEqual(p['validation_steps'],[0,32768,65536])
        self.assertEqual(digest(grant()),'1897b889fd3fea3f269abed9533104c123ef4d81fd9843d1cbe0d448db1cb7fc')

    def test_any_unreviewed_profile_or_budget_change_is_rejected(self):
        for k,v in [('initialization','LOAD_OLD'),('timesteps',131072),('wall_seconds',7200),
                    ('experiment_id','ARL_R2_RETRY'),('seed',1),('validation_steps',[65536]),
                    ('prior_report_sha256','0'*64),('control_summary_sha256','1'*64),
                    ('evaluation',{'candidate_success_min':0}),
                    ('unreviewed_parameter',True)]:
            p=self.profile();p[k]=v
            with self.subTest(k=k),self.assertRaises(OpsError):validate_experiment(p)

    def test_old_events_are_retained_with_new_conservative_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            s=BudgetStore(Path(d)/'budget');s.initialize(expected_grant_sha256=digest(grant()))
            old=experiment('UNIFORM',65536,3600);s.reserve(old);s.fail(old['experiment_id'],'SYNTHETIC_TEST_NO_OPTIMIZER')
            oldbytes=(s.home/'events/000001.json').read_bytes()
            s.reserve(self.profile())
            status=s.status()
            self.assertEqual(status['reserved_timesteps'],131072)
            self.assertEqual(status['remaining_wall_seconds'],79200)
            self.assertEqual(status['attempts'][0]['protocol'],old)
            self.assertEqual((s.home/'events/000001.json').read_bytes(),oldbytes)
            with self.assertRaises(OpsError):s.reserve(self.profile())

    def test_unknown_schema_and_checkpoint_argument_are_not_execution_capabilities(self):
        p=self.profile();p['schema']=3
        with self.assertRaises(OpsError):validate_experiment(p)
        p=self.profile();p['checkpoint']='L9.zip'
        with self.assertRaises(OpsError):validate_experiment(p)

    def test_prior_proof_cannot_be_substituted_by_current_source_or_path(self):
        from research_regression.authorized_rl.prior_profile import verify_prior_proof
        p=self.profile();p['prior_report_sha256']='0'*64
        with self.assertRaises(OpsError):verify_prior_proof(p)

    def test_cli_has_explicit_prior_modes_without_arbitrary_script_parameter(self):
        import subprocess,sys
        p=subprocess.run([sys.executable,'-m','research_regression.authorized_rl','--help'],capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,0)
        self.assertIn('prepare-prior',p.stdout)
        self.assertIn('start-prior',p.stdout)
        self.assertNotIn('--script',p.stdout)

    def test_no_current_preflight_admission_means_no_training(self):
        from research_regression.authorized_rl.prior_profile import verify_training_checks
        p=self.profile();p.update(source_identity_sha256='a'*64,registration_sha256='b'*64)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):verify_training_checks(p,Path(d))


if __name__=='__main__':unittest.main()
