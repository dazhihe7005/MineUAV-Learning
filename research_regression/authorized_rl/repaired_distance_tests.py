"""Actual R8 hook, accounting and fresh snapshot; optimizer strictly forbidden."""
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


class RepairedDistanceTests(unittest.TestCase):
    def test_worker_hook_initializes_true_short_task_and_accounts_real_step_without_update(self):
        torch.set_num_threads(1)
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_R8_INTERFACE')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.repaired_distance_training'),'missing R8 actual hook')
        from .repaired_distance_training import make_repaired_scheduler,initialize_repaired_training
        prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        scheduler=make_repaired_scheduler(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0)
        self.addCleanup(scheduler.close)
        with tempfile.TemporaryDirectory() as d:
            measured=FormalHealthMetricsEnv(scheduler,diagnostics_dir=Path(d));self.addCleanup(measured.close)
            model=make_ppo(measured,seed=0);manager=CheckpointManager(Path(d),prereg,execution_protocol_sha256='c'*64)
            initialization,event,probe=initialize_repaired_training(model,scheduler,manager,identity='R8_NO_UPDATE_ONLY',seed=0)
            self.assertEqual(initialization['execution_profile'],'DISTANCE_CURRICULUM_R8_REPAIRED_V1')
            self.assertEqual(_weights_sha(model.policy),'db33ba4df5f7aaf76f98b7915675cbfc80dbe52273045ed86811b5cc074dfa06')
            self.assertEqual(len(scheduler.selected_identities),1);self.assertEqual(scheduler.current_identity,scheduler.env.active_task['identity_sha256'])
            model._setup_learn(128,reset_num_timesteps=False)
            self.assertEqual(len(scheduler.selected_identities),1)
            obs,reward,t,tr,info=measured.step(np.zeros(4,dtype=np.float32))
            self.assertEqual(measured.policy_steps,1);self.assertEqual(measured.physics_ticks,20)
            self.assertEqual(info['scenario_identity_sha256'],scheduler.current_identity)
            self.assertNotEqual(scheduler.current_identity,scheduler.base_current_identity)
            self.assertFalse(model.policy.optimizer.state);self.assertEqual(model._n_updates,0)
            self.assertEqual(event['checkpoint']['scheduler_state']['current_identity'],scheduler.current_identity)


if __name__=='__main__':unittest.main()
