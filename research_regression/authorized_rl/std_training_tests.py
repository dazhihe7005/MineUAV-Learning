"""R3 initialization/snapshot tests: real CPU model/physics, learning forbidden."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from lidar_integration.ppo_preflight import make_ppo
from lidar_integration.ppo_formal_scheduler import load_l6_prereg
from lidar_integration.ppo_formal_checkpoint import CheckpointManager
from lidar_integration.ppo_training_runner import _weights_sha
from .runtime import ROOT
from .curriculum import make_scheduler


class InitialStdTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_INITIAL_STD_TEST')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)
        self.prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        self.env=make_scheduler(self.prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR','UNIFORM',0)
        self.addCleanup(self.env.close)
        self.model=make_ppo(self.env,seed=0)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.manager=CheckpointManager(Path(self.tmp.name),self.prereg,execution_protocol_sha256='a'*64)

    def initialize(self,variant):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.std_training'),'missing bounded std initialization')
        from .std_training import initialize_std_training
        return initialize_std_training(self.model,self.env,self.manager,identity='UNTRAINED_STD_SNAPSHOT_TEST',seed=0,variant=variant)

    def test_random_std_only_changes_log_std_and_snapshot_reloads_without_new_reset(self):
        before=copy.deepcopy(self.model.policy.state_dict());rng=torch.get_rng_state().clone()
        info,event,probe=self.initialize('RANDOM')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in before.items():
            if k!='log_std':self.assertTrue(torch.equal(v,self.model.policy.state_dict()[k]),k)
        self.assertTrue(torch.allclose(self.model.policy.log_std.exp(),torch.full((4,),.3),atol=1e-8))
        self.assertTrue(self.model.policy.log_std.requires_grad)
        state=copy.deepcopy(self.env.scheduler_state())
        obs={k:v.copy() for k,v in self.model._last_obs.items()}
        self.model._setup_learn(128,reset_num_timesteps=False)
        self.assertEqual(self.env.scheduler_state(),state)
        self.assertEqual(len(self.env.selected_identities),1)
        restored=self.manager.load_verified(None,run_id=event['checkpoint']['run_id'])
        self.assertEqual(restored.num_timesteps,0);self.assertFalse(restored.policy.optimizer.state)
        self.assertEqual(_weights_sha(restored.policy),info['assigned_policy_sha256'])
        self.assertTrue(np.array_equal(probe[1],restored.predict(obs,deterministic=True)[0]))

    def test_prior_std_preserves_prior_mean_cnn_critic_and_rng(self):
        from .prior_tests import ObservationOnly
        from .actor_prior import initialize_actor_prior
        reference=make_ppo(ObservationOnly(),seed=0);initialize_actor_prior(reference)
        before=copy.deepcopy(reference.policy.state_dict());rng=torch.get_rng_state().clone()
        info,event,probe=self.initialize('PRIOR')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in before.items():
            if k!='log_std':self.assertTrue(torch.equal(v,self.model.policy.state_dict()[k]),k)
        self.assertTrue(np.array_equal(reference.predict(probe[0],deterministic=True)[0],probe[1]))
        self.assertTrue(torch.allclose(self.model.policy.log_std.exp(),torch.full((4,),.3),atol=1e-8))
        self.assertEqual(info['base_mean_policy_sha256'],'1f3e9a0a6779d9b7290ba138bda5e6f916e5d73380b291cc9a97d1650c5e835c')

    def test_mutated_or_reset_model_refused_before_assigning_std(self):
        with torch.no_grad():self.model.policy.action_net.weight[0,0]+=.1
        before=_weights_sha(self.model.policy)
        with self.assertRaises(ValueError):self.initialize('RANDOM')
        self.assertEqual(_weights_sha(self.model.policy),before)
        self.assertEqual(self.env.selected_identities,[])
        self.model=make_ppo(self.env,seed=0);self.model._last_obs=self.model.get_env().reset()
        before=_weights_sha(self.model.policy)
        with self.assertRaises(ValueError):self.initialize('PRIOR')
        self.assertEqual(_weights_sha(self.model.policy),before)

    def test_unknown_variant_does_not_mutate_or_reset(self):
        before=_weights_sha(self.model.policy)
        from research_ops.storage import OpsError
        with self.assertRaises(OpsError):self.initialize('WARM_START')
        self.assertEqual(_weights_sha(self.model.policy),before)
        self.assertEqual(self.env.selected_identities,[])


if __name__=='__main__':unittest.main()
