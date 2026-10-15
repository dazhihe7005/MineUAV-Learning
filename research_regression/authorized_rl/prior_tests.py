"""Explicit no-optimizer unit suite; not an automatically discovered RL run."""
import copy
import unittest
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from unittest.mock import patch
from lidar_integration.ppo_preflight import make_ppo


class ObservationOnly(gym.Env):
    observation_space=spaces.Dict({
        'lidar':spaces.Box(0,1,shape=(3,12,24),dtype=np.float32),
        'state':spaces.Box(-np.inf,np.inf,shape=(7,),dtype=np.float32),
        'heading':spaces.Box(-1,1,shape=(2,),dtype=np.float32),
        'lidar_age':spaces.Box(0,1,shape=(1,),dtype=np.float32)})
    action_space=spaces.Box(-1,1,shape=(4,),dtype=np.float32)
    def reset(self,*,seed=None,options=None):raise AssertionError('unit test must not step or reset physics')
    def step(self,action):raise AssertionError('unit test must not step physics')


class ActorPriorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*args,**kwargs):raise AssertionError('optimizer/learning forbidden')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)
        self.model=make_ppo(ObservationOnly(),seed=0)

    def initializer(self):
        from research_regression.authorized_rl import actor_prior
        return actor_prior.initialize_actor_prior

    def observation(self,state):
        n=len(state)
        return {'lidar':np.zeros((n,3,12,24),dtype=np.float32),'state':state.astype(np.float32),
                'heading':np.tile([0,1],(n,1)).astype(np.float32),'lidar_age':np.zeros((n,1),dtype=np.float32)}

    def test_missing_prior_is_implemented_and_only_actor_mean_changes(self):
        before=copy.deepcopy(self.model.policy.state_dict())
        result=self.initializer()(self.model)
        changed=[]
        for k,v in self.model.policy.state_dict().items():
            if not torch.equal(before[k],v):changed.append(k)
        self.assertTrue(changed)
        self.assertTrue(all(k.startswith(('mlp_extractor.policy_net.','action_net.')) for k in changed))
        self.assertEqual(result['optimizer_updates'],0)
        self.assertFalse(self.model.policy.optimizer.state)
        self.assertTrue(torch.equal(before['log_std'],self.model.policy.log_std))

    def test_clipped_action_approximates_scripted_on_dense_and_extreme_state(self):
        self.initializer()(self.model)
        y=np.concatenate([np.linspace(-2,2,4001),[-1e6,1e6]])
        state=np.zeros((len(y),7),dtype=np.float32)
        state[:,0]=4.5*y;state[:,1]=4.5*y;state[:,2]=3*y;state[:,6]=y
        actual=self.model.predict(self.observation(state),deterministic=True)[0]
        target=np.tile(np.clip(y,-1,1)[:,None],(1,4))
        self.assertLessEqual(float(np.max(np.abs(actual-target))),.02)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.all(np.abs(actual)<=1))

    def test_initial_mean_ignores_lidar_velocity_heading_and_age_not_future(self):
        self.initializer()(self.model)
        a=self.observation(np.zeros((4,7)));a['state'][:,:3]=[1,-1,.5];a['state'][:,6]=.3
        original=self.model.predict(a,deterministic=True)[0]
        a['lidar'][:]=1;a['state'][:,3:6]=[20,-20,5]
        a['heading'][:]=[1,0];a['lidar_age'][:]=.14
        self.assertTrue(np.array_equal(original,self.model.predict(a,deterministic=True)[0]))

    def test_updated_model_is_refused_without_mutation(self):
        self.model.num_timesteps=128  # FORCED_TEST_STATE only; no training.
        before=copy.deepcopy(self.model.policy.state_dict())
        with self.assertRaises(ValueError):self.initializer()(self.model)
        self.assertTrue(all(torch.equal(v,self.model.policy.state_dict()[k]) for k,v in before.items()))

    def test_unknown_activation_is_refused_before_assignment(self):
        self.model.policy.mlp_extractor.policy_net[1]=torch.nn.ReLU()
        before=copy.deepcopy(self.model.policy.state_dict())
        with self.assertRaises(ValueError):self.initializer()(self.model)
        self.assertTrue(all(torch.equal(v,self.model.policy.state_dict()[k]) for k,v in before.items()))

    def test_nonisolated_probe_is_refused_before_creating_attempt(self):
        import tempfile
        from pathlib import Path
        from research_regression.authorized_rl import prior_preflight
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'unused'
            with patch.object(prior_preflight,'OUT',target),patch('os.getpgrp',return_value=-1):
                with self.assertRaisesRegex(Exception,'isolated process group'):prior_preflight.run()
            self.assertFalse(target.exists())


if __name__=='__main__':unittest.main()
