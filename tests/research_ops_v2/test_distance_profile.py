"""New distance profile must not change old grant or replay completed training."""
import copy
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from research_ops.storage import OpsError,digest
from research_regression.authorized_rl.budget import BudgetStore,grant,experiment,validate_experiment


class DistanceProfileTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.distance_profile'),'missing fixed distance profile')
        from research_regression.authorized_rl import distance_profile
        return distance_profile

    def test_new_profile_charges_same_cumulative_ledger_and_preserves_five_old_events(self):
        from research_regression.authorized_rl.prior_profile import prior_experiment
        from research_regression.authorized_rl.std_profile import std_experiment
        p=self.module().distance_experiment();validate_experiment(p)
        self.assertEqual(p['experiment_id'],'ARL_R6_DISTANCE_SEED0');self.assertEqual(p['timesteps'],65536)
        self.assertEqual(digest(grant()),'1897b889fd3fea3f269abed9533104c123ef4d81fd9843d1cbe0d448db1cb7fc')
        with tempfile.TemporaryDirectory() as d:
            s=BudgetStore(Path(d)/'synthetic');s.initialize(expected_grant_sha256=digest(grant()))
            for old in [experiment('UNIFORM',65536,3600),experiment('CURRICULUM',65536,3600),prior_experiment(),std_experiment('RANDOM'),std_experiment('PRIOR')]:
                s.reserve(old);s.fail(old['experiment_id'],'SYNTHETIC_NO_OPTIMIZER')
            before={f.name:f.read_bytes() for f in (s.home/'events').iterdir()}
            s.reserve(p);self.assertEqual(s.status()['reserved_timesteps'],393216)
            self.assertEqual(s.status()['remaining_wall_seconds'],64800)
            self.assertTrue(all((s.home/'events'/k).read_bytes()==v for k,v in before.items()))
            s.fail(p['experiment_id'],'SYNTHETIC_NO_OPTIMIZER')
            with self.assertRaises(OpsError):s.reserve(p)

    def test_changed_seed_budget_algorithm_std_manifest_and_initialization_refused(self):
        p=self.module().distance_experiment()
        for k,v in [('seed',1),('timesteps',131072),('initial_std',1.),('initialization','L9.zip'),
                    ('short_manifest_sha256','0'*64),('evaluation',{}),('schema',4),('checkpoint','old.zip')]:
            q=copy.deepcopy(p);q[k]=v
            with self.subTest(key=k),self.assertRaises(OpsError):validate_experiment(q)

    def test_real_released_proof_and_historical_control_required(self):
        m=self.module();p=m.distance_experiment();self.assertEqual(m.verify_distance_proof(p)['status'],'PASS')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_distance_proof(p,Path(d))
        p['feasibility_result_sha256']='0'*64
        with self.assertRaises(OpsError):m.verify_distance_proof(p)

    def test_current_admission_is_required_before_consuming_budget(self):
        m=self.module();p=m.distance_experiment();p.update(source_identity_sha256='a'*64,registration_sha256='b'*64)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_distance_checks(p,Path(d))

    def test_explicit_cli_has_no_initial_checkpoint_or_arbitrary_code_option(self):
        p=subprocess.run([sys.executable,'-m','research_regression.authorized_rl','--help'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0);self.assertIn('prepare-distance',p.stdout);self.assertIn('start-distance',p.stdout)
        self.assertNotIn('--checkpoint',p.stdout);self.assertNotIn('--script',p.stdout)

    def test_schema_is_not_authority_to_increase_resources_or_use_test(self):
        p=self.module().distance_experiment()
        for k,v in [('workers',2),('device','cuda'),('test_episode_count',1),('wall_seconds',7200),('experiment_id','ARL_R6_RETRY')]:
            q=copy.deepcopy(p);q[k]=v
            with self.subTest(key=k),self.assertRaises(OpsError):validate_experiment(q)


if __name__=='__main__':unittest.main()
