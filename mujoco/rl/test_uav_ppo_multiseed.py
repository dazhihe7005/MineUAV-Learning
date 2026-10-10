"""Tests pin experiment identity, actual target isolation and safe completion."""
import importlib.util
import unittest
import tempfile
from pathlib import Path
import numpy as np

class ProtocolTests(unittest.TestCase):
    def test_locked_runtime_config_rejects_post_registration_tuning(self):
        from unittest.mock import patch
        from uav_ppo_multiseed import read_manifest,config
        changed=dict(config(),learning_rate=.001)
        with patch('uav_ppo_multiseed.config',return_value=changed):
            with self.assertRaises(ValueError):read_manifest()

    def test_config_hash_rejects_tuning_and_target_tampering(self):
        from uav_ppo_multiseed import seal, verify_seal
        m=seal(dict(lr=.0003,targets=[[1,2,1]]));verify_seal(m)
        m['lr']=.001
        with self.assertRaises(ValueError):verify_seal(m)

    def test_train_reset_and_step_are_passive_and_identity_recorded(self):
        from uav_ppo_multiseed import AuditedPIEnv
        from ppo_pi_env import MineUAVPIEnv
        a=AuditedPIEnv(rank=3,forbidden=[[2.,2.,1.5]]);b=MineUAVPIEnv(reward_version='v2',target_distribution='full')
        try:
            np.testing.assert_array_equal(a.reset(seed=73)[0],b.reset(seed=73)[0])
            for _ in range(3):
                x=a.step(np.zeros(4));y=b.step(np.zeros(4))
                np.testing.assert_array_equal(x[0],y[0]);self.assertEqual(x[1:4],y[1:4])
            self.assertEqual(len(a.visits),1);self.assertEqual(a.visits[0]['rank'],3)
            self.assertEqual(a.visits[0]['target'],b.target_position.tolist())
        finally:a.close();b.close()

    def test_train_overlap_fails_without_resampling(self):
        from uav_ppo_multiseed import AuditedPIEnv
        a=AuditedPIEnv(rank=0,forbidden=[[1.,0.,1.]])
        try:
            with self.assertRaises(ValueError):a.reset(seed=0,options={'target_position':[1.,0.,1.]})
            self.assertEqual(len(a.visits),0)
        finally:a.close()

    def test_split_isolation_detects_coordinates_ids_and_reset_seeds(self):
        from uav_bc_yaw_data import assert_isolated
        old=[dict(target_id='a',env_seed=1,target=[0.,0.,1.])]
        for x in [dict(target_id='b',env_seed=2,target=[0.,0.,1.]),dict(target_id='a',env_seed=3,target=[1.,0.,1.]),dict(target_id='c',env_seed=1,target=[1.,0.,1.])]:
            with self.assertRaises(ValueError):assert_isolated([x],old)

    def test_validation_restores_training_rng_and_policy_mode(self):
        import random
        import torch
        from uav_ppo_multiseed_train import preserve_rng
        module=torch.nn.Linear(7,4);module.train()
        random.seed(9);np.random.seed(9);torch.manual_seed(9)
        expected=(random.random(),float(np.random.rand()),torch.rand(2))
        random.seed(9);np.random.seed(9);torch.manual_seed(9)
        with preserve_rng(module):
            random.random();np.random.rand();torch.rand(17);module.eval()
        self.assertTrue(module.training)
        self.assertEqual(random.random(),expected[0]);self.assertEqual(float(np.random.rand()),expected[1])
        torch.testing.assert_close(torch.rand(2),expected[2],rtol=0,atol=0)

    def test_atomic_checkpoint_reload_and_independent_seed_initialization(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from ppo_pi_env import MineUAVPIEnv
        from train_ppo_waypoint import make_ppo
        from uav_bc_policy import parameter_hash
        from uav_ppo_multiseed_train import atomic_model
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            env=DummyVecEnv([lambda:MineUAVPIEnv(reward_version='v2')])
            try:
                hashes=[]
                for seed in [0,0,1]:
                    model=make_ppo(env,Path(d),log_std_init=-2.,seed=seed)
                    hashes.append(parameter_hash(model.policy))
                self.assertEqual(hashes[0],hashes[1]);self.assertNotEqual(hashes[0],hashes[2])
                path=Path(d)/'atomic.zip';atomic_model(model,path)
                loaded=PPO.load(path,device='cpu');self.assertEqual(parameter_hash(loaded.policy),hashes[-1])
            finally:env.close()

    def test_incomplete_seed_cannot_resume_as_complete(self):
        from uav_ppo_multiseed import seal,verify_completion
        m=seal(dict(config='fixture'))
        record=seal(dict(status='complete',seed=2,manifest_sha256=m['sha256'],actual_timesteps=51200,
            optimization_epochs=250,history=[{}]*25))
        with self.assertRaises(ValueError):verify_completion(record,m,2)
        record=seal(dict(status='partial',seed=2,manifest_sha256=m['sha256']))
        with self.assertRaises(ValueError):verify_completion(record,m,2)

    def test_protocol_factory_rejects_changed_observation_budget_or_optimizer(self):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from ppo_pi_env import MineUAVPIEnv
        from train_ppo_waypoint import make_ppo
        from uav_ppo_multiseed import assert_model_protocol
        env=DummyVecEnv([lambda:MineUAVPIEnv(reward_version='v2') for _ in range(8)])
        try:
            with tempfile.TemporaryDirectory() as d:
                model=make_ppo(env,Path(d),log_std_init=-2.,seed=0);assert_model_protocol(model)
                model.learning_rate=.001
                with self.assertRaises(ValueError):assert_model_protocol(model)
        finally:env.close()

    def test_preregistered_split_hash_detects_test_changes(self):
        from uav_ppo_multiseed import seal,verify_seal
        m=seal(dict(splits={'validation':[[0,0,1]],'final':[[1,0,1]]}))
        m['splits']['final'][0][0]=2
        with self.assertRaises(ValueError):verify_seal(m)

    def test_seed_statistics_use_five_runs_sample_sd_not_targets(self):
        from uav_ppo_multiseed_analysis import seed_statistics
        s=seed_statistics([1.,2.,3.,4.,5.])
        self.assertEqual(s['n_seeds'],5);self.assertEqual(s['mean'],3.)
        self.assertAlmostEqual(s['sample_sd'],1.5811388300841898)
        self.assertEqual(s['median'],3.);self.assertEqual(s['min'],1.);self.assertEqual(s['max'],5.)
        with self.assertRaises(ValueError):seed_statistics([1]*200)

    def test_failure_diagnosis_keeps_timeout_and_no_success_completion_missing(self):
        from uav_ppo_multiseed_analysis import failure_diagnosis
        rows=[dict(success=False,timeout=True,physical_failure=False,reached_target_region=True,
                   ever_joint_threshold=True,maximum_success_streak=4,final_distance_m=.09,final_speed_m_s=.2,crossings=1),
              dict(success=True,timeout=False,physical_failure=False,reached_target_region=True,
                   ever_joint_threshold=True,maximum_success_streak=5,final_distance_m=.08,final_speed_m_s=.1,crossings=0)]
        s=failure_diagnosis(rows)
        self.assertEqual(s['timeouts'],1);self.assertEqual(s['failures_ever_joint_threshold'],1)
        self.assertEqual(s['failed_max_streak4'],1);self.assertEqual(s['failed_final_distance_pass_only'],1)
        self.assertEqual(s['failed_final_speed_pass_only'],0)

    def test_completed_training_cache_identity_and_budget_match(self):
        import json
        from uav_ppo_multiseed import ROOT,PARTS,read_manifest,verify_completion,seal
        path=ROOT/'mujoco/reports'/PARTS/'seed0.json'
        if not path.exists():self.skipTest('local completed training cache unavailable')
        record=json.loads(path.read_text());m=read_manifest();verify_completion(record,m,0)
        with self.assertRaises(ValueError):verify_completion(record,m,1)
        bad=dict(record);bad.pop('sha256');bad['checkpoint_sha256']='0'*64;bad=seal(bad)
        with self.assertRaises(ValueError):verify_completion(bad,m,0)

    def test_final_resume_never_executes_a_duplicate_episode(self):
        import json
        from unittest.mock import patch
        from uav_ppo_multiseed import ROOT,PARTS,read_manifest
        from uav_ppo_bc_nominal_run import evaluate
        path=ROOT/'mujoco/reports'/PARTS/'evaluation.json'
        if not path.exists():self.skipTest('local final evaluation cache not yet available')
        e=json.loads(path.read_text());m=read_manifest();row=e['records'][0]
        identity=dict(manifest_sha256=m['sha256'],controller=row['controller'],model_identity=e['identities'][row['controller']])
        state=next(s for s in m['states'] if s['key']==row['key'])
        cache=path.parent/f"eval_{state['index']:03d}_{row['controller']}.json"
        with patch('uav_ppo_bc_nominal_run.rollout',side_effect=AssertionError('duplicate flight')):
            result,new=evaluate(None,None,None,state,row['controller'],cache,identity)
        self.assertFalse(new);self.assertEqual(result,row)

    def test_train_audit_final_targets_are_all_separated(self):
        import json
        from uav_ppo_multiseed import ROOT,PARTS,read_manifest
        m=read_manifest();forbidden=np.array([t['target'] for v in m['splits'].values() for t in v])
        files=list((ROOT/'mujoco/reports'/PARTS).glob('seed[0-4].json'))
        if not files:self.skipTest('no completed local training cache')
        for p in files:
            visits=json.loads(p.read_text())['actual_train_targets']
            self.assertEqual(len(visits),len({(v['rank'],v['episode_index']) for v in visits}))
            for v in visits:self.assertTrue(np.all(np.linalg.norm(forbidden-v['target'],axis=1)>=1e-8))

    def test_all_final_checkpoint_optimizer_steps_and_seeds_are_real(self):
        import json
        from stable_baselines3 import PPO
        from uav_ppo_multiseed import ROOT,PARTS
        from uav_bc_policy import parameter_hash
        files=list((ROOT/'mujoco/reports'/PARTS).glob('seed[0-4].json'))
        if len(files)!=5:self.skipTest('all five training artifacts not yet present')
        initials=[]
        for f in files:
            r=json.loads(f.read_text());m=PPO.load(r['checkpoint_path'],device='cpu')
            self.assertEqual(m.seed,r['seed']);self.assertEqual(m.num_timesteps,100352)
            self.assertEqual(m._n_updates,490)
            self.assertEqual({int(v['step'].item()) for v in m.policy.optimizer.state.values()},{3920})
            self.assertEqual(parameter_hash(m.policy),r['final_parameter_sha256']);initials.append(r['initial_parameter_sha256'])
        self.assertEqual(len(set(initials)),5)

if __name__ == '__main__': unittest.main()
