"""Actual fresh PRIOR×distance worker hook, checkpoint and step; zero updates."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from lidar_integration.ppo_formal_scheduler import load_l6_prereg
from lidar_integration.ppo_formal_seed0_health import FormalHealthMetricsEnv
from lidar_integration.ppo_formal_checkpoint import CheckpointManager
from lidar_integration.ppo_preflight import make_ppo
from lidar_integration.ppo_training_runner import _weights_sha
from .runtime import ROOT


class PriorDistanceTests(unittest.TestCase):
    def test_actual_hook_prior_checkpoint_short_identity_step_and_correct_step0_reuse(self):
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_R10_INTERFACE')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)
        torch.set_num_threads(1)
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.prior_distance_training'),'missing R10 actual hook')
        from .prior_distance_training import initialize_prior_distance,reuse_prior_original_step0
        from .repaired_distance_training import make_repaired_scheduler
        prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        scheduler=make_repaired_scheduler(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0);self.addCleanup(scheduler.close)
        with tempfile.TemporaryDirectory() as d:
            env=FormalHealthMetricsEnv(scheduler,diagnostics_dir=Path(d));self.addCleanup(env.close)
            model=make_ppo(env,seed=0);manager=CheckpointManager(Path(d),prereg,execution_protocol_sha256='c'*64)
            init,event,probe=initialize_prior_distance(model,scheduler,manager,identity='R10_NO_UPDATE_ONLY',seed=0)
            self.assertEqual(init['execution_profile'],'PRIOR_DISTANCE_R10_V1')
            self.assertEqual(_weights_sha(model.policy),'6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74')
            self.assertEqual(len(scheduler.selected_identities),1)
            model._setup_learn(128,reset_num_timesteps=False)
            self.assertEqual(len(scheduler.selected_identities),1)
            _,_,_,_,info=env.step(np.zeros(4,dtype=np.float32))
            self.assertEqual((env.policy_steps,env.physics_ticks),(1,20));self.assertEqual(info['distance_stage'],'SHORT')
            self.assertEqual(event['checkpoint']['scheduler_state']['current_identity'],scheduler.current_identity)
            self.assertEqual(info['scenario_identity_sha256'],scheduler.current_identity)
            old=reuse_prior_original_step0(_weights_sha(model.policy))
            self.assertTrue(old['reused_historical_evidence']);self.assertEqual(old['newly_executed_episodes'],0)
            self.assertEqual(old['overall']['navigation_success_rate'],1/6)
            self.assertEqual(old['overall']['collision_rate'],3/6)
            with self.assertRaises(ValueError):reuse_prior_original_step0('db33ba4df5f7aaf76f98b7915675cbfc80dbe52273045ed86811b5cc074dfa06')
            self.assertFalse(model.policy.optimizer.state);self.assertEqual(model._n_updates,0)


if __name__=='__main__':unittest.main()
