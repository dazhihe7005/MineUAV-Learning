"""R20 fixed budget/profile/entry contracts, no ML imports or optimizer."""
import copy,importlib,importlib.util,tempfile,unittest
from pathlib import Path
from research_ops.storage import OpsError,digest


class TemporalCreditTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.temporal_credit_profile'),'missing R20 fixed profile')
        return importlib.import_module('research_regression.authorized_rl.temporal_credit_profile')

    def test_fixed_single_gamma_and_prior_algorithm_budget(self):
        p=self.module().temporal_experiment()
        self.assertEqual((p['schema'],p['gamma'],p['gae_lambda']),(20,.995,.95))
        self.assertEqual((p['timesteps'],p['wall_seconds'],p['workers'],p['seed']),(65536,3600,1,0))
        self.assertEqual(p['original_empty_probability'],.5);self.assertNotIn('expected_prefix_policy_sha256',p)

    def test_only_gamma_algorithm_change_from_sealed_rehearsal(self):
        from research_regression.authorized_rl.empty_rehearsal_profile import rehearsal_experiment
        p=self.module().temporal_experiment();old=rehearsal_experiment()
        meta={'schema','experiment_id','execution_profile','control_summary_sha256','control_report_sha256','single_variable','expected_prefix_policy_sha256'}
        for key in set(old)-meta:self.assertEqual(p[key],old[key],key)

    def test_profile_changes_including_unapproved_seed_reward_and_budget_fail_closed(self):
        m=self.module()
        for key,value in (('gamma',.99),('gae_lambda',.9),('seed',1),('timesteps',131072),('reward_version','OTHER'),('workers',2),('initialization','CHECKPOINT')):
            p=m.temporal_experiment();p[key]=value
            with self.assertRaises(OpsError):m.validate_temporal_profile(p)

    def test_proof_missing_cannot_create_new_output(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_temporal_proof(m.temporal_experiment(),Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_new_schema_valid_but_extra_shell_path_or_identity_field_rejected(self):
        from research_regression.authorized_rl.budget import validate_experiment
        m=self.module();p=m.temporal_experiment();validate_experiment(p)
        for key,value in (('command','python train.py'),('checkpoint_path','old.zip'),('source_identity_sha256','bad')):
            bad=copy.deepcopy(p);bad[key]=value
            with self.assertRaises(OpsError):validate_experiment(bad)

    def test_existing_grant_and_old_profile_not_mutated(self):
        from research_regression.authorized_rl.budget import grant
        from research_regression.authorized_rl.empty_rehearsal_profile import rehearsal_experiment
        self.module().temporal_experiment()
        self.assertEqual(digest(grant()),'1897b889fd3fea3f269abed9533104c123ef4d81fd9843d1cbe0d448db1cb7fc')
        self.assertEqual(rehearsal_experiment()['expected_prefix_policy_sha256'],'33bf6699158a35842fc05c3d4ca97f391f778db60c759fdad2564a6e94c87f60')
