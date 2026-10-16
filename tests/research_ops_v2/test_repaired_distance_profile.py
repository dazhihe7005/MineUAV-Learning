"""R8 fixed profile must not relax old failed-attempt or budget constraints."""
import copy
import importlib.util
import unittest
from pathlib import Path
from research_ops.storage import OpsError,digest
from research_regression.authorized_rl.budget import validate_experiment


class RepairedDistanceProfileTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.repaired_distance_profile'),'missing independent R8 profile')
        from research_regression.authorized_rl import repaired_distance_profile
        return repaired_distance_profile

    def test_new_profile_is_independent_and_same_scientific_algorithm(self):
        from research_regression.authorized_rl.distance_profile import distance_experiment
        m=self.module();p=m.repaired_distance_experiment();old=distance_experiment()
        self.assertEqual(p['schema'],8);self.assertEqual(p['experiment_id'],'ARL_R8_REPAIRED_DISTANCE_SEED0')
        self.assertEqual(p['failed_predecessor_experiment_id'],'ARL_R6_DISTANCE_SEED0')
        for k in ('timesteps','wall_seconds','seed','initial_std','curriculum_rule','evaluation','manifest_sha256','short_manifest_sha256','initialization'):
            self.assertEqual(p[k],old[k])
        validate_experiment(p);validate_experiment(old)

    def test_modified_budget_seed_algorithm_or_proof_is_rejected(self):
        m=self.module();p=m.repaired_distance_experiment()
        for k,value in [('timesteps',131072),('seed',1),('schema',6),('repair_report_sha256','0'*64),('initial_std',.1),('workers',2),('test_episode_count',1)]:
            q=copy.deepcopy(p);q[k]=value
            with self.subTest(k=k),self.assertRaises(OpsError):validate_experiment(q)

    def test_missing_repaired_admission_refuses_before_reservation(self):
        import tempfile
        m=self.module();p=m.repaired_distance_experiment()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_repaired_checks(p,Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_wrong_repair_release_cannot_be_self_rehashed_into_proof(self):
        import tempfile
        m=self.module();p=m.repaired_distance_experiment()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_repaired_proof(p,Path(d))

    def test_cli_refuses_checkpoint_and_script_instead_of_warm_start(self):
        import subprocess,sys
        for option in ('--checkpoint','--script','--seed','--timesteps'):
            p=subprocess.run([sys.executable,'-m','research_regression.authorized_rl','prepare-repaired-distance',option,'x'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('unrecognized arguments',p.stderr)


if __name__=='__main__':unittest.main()
