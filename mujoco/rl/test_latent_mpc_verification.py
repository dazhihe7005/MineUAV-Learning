"""Recorded-result integrity guards, without replaying training or tuning."""
import unittest
import json
from pathlib import Path
from latent_mpc_verification import verify_episode_cohort, verify


class RecordedIntegrityTests(unittest.TestCase):
    def test_verifier_rejects_missing_baselines_splits_and_episodes(self):
        root=Path(__file__).resolve().parents[2]
        path=root/'mujoco/reports/latent_random_shooting_mpc_seed0.json'
        report=json.loads(path.read_text())
        for mutation in ('zero','scripted','holdout','episode'):
            bad=json.loads(json.dumps(report))
            if mutation in ('zero','scripted'): del bad['results'][mutation]
            elif mutation=='holdout': del bad['results']['mpc']['holdout']
            else: bad['results']['mpc']['benchmark']['episodes'].pop()
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError): verify(root,bad)

    def test_cache_identity_binds_environment_and_original_hardware(self):
        from run_latent_random_shooting_mpc import condition_identity
        identity={'checkpoint':'abc','n':512,'h':10}
        source={'PI':'gain-original','physics':'original'}
        hardware={'device':'CPU','torch_version':'same','cpu':'original'}
        first=condition_identity(identity,source,hardware)
        self.assertNotEqual(first,condition_identity(identity,dict(source,PI='changed'),hardware))
        self.assertNotEqual(first,condition_identity(identity,source,dict(hardware,cpu='other')))
        self.assertNotEqual(first,condition_identity(identity,source,dict(hardware,torch_version='other')))
        self.assertEqual(first['execution_hardware'],hardware)

    def test_cohort_rejects_lost_duplicate_or_changed_targets(self):
        rows=[{'env_seed':1,'target':[1,0,1],'episode_steps':5,'termination_reason':'success','success':True,'episode_duration_s':.2}]
        targets=[(1,[1,0,1])]
        verify_episode_cohort(rows,targets)
        with self.assertRaises(AssertionError): verify_episode_cohort([],targets)
        with self.assertRaises(AssertionError): verify_episode_cohort(rows*2,targets)
        with self.assertRaises(AssertionError): verify_episode_cohort([dict(rows[0],target=[0,0,1])],targets)

    def test_reason_and_simulated_time_cannot_claim_success_for_timeout(self):
        target=[(1,[1,0,1])]
        row={'env_seed':1,'target':[1,0,1],'episode_steps':375,'termination_reason':'time_limit','success':False,'episode_duration_s':15.}
        verify_episode_cohort([row],target)
        with self.assertRaises(AssertionError): verify_episode_cohort([dict(row,success=True)],target)
        with self.assertRaises(AssertionError): verify_episode_cohort([dict(row,episode_duration_s=14.)],target)


if __name__=='__main__': unittest.main()
