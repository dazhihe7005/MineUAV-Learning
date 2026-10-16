"""Actual R11 fresh initialization, LR schedule and checkpoint; no optimizer."""
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


class LowerLRTests(unittest.TestCase):
    def test_actual_lower_lr_before_step0_save_schedule_reload_and_env_step(self):
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_R11_INTERFACE')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)
        torch.set_num_threads(1)
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.lower_lr_training'),'missing R11 LR hook')
        from .lower_lr_training import initialize_lower_lr,verify_effective_lr
        from .repaired_distance_training import make_repaired_scheduler
        from .prior_distance_training import reuse_prior_original_step0
        prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        scheduler=make_repaired_scheduler(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0);self.addCleanup(scheduler.close)
        with tempfile.TemporaryDirectory() as d:
            env=FormalHealthMetricsEnv(scheduler,diagnostics_dir=Path(d));self.addCleanup(env.close)
            model=make_ppo(env,seed=0);manager=CheckpointManager(Path(d),prereg,execution_protocol_sha256='d'*64)
            init,event,probe=initialize_lower_lr(model,scheduler,manager,identity='R11_NO_UPDATE_ONLY',seed=0)
            self.assertEqual(init['effective_learning_rate'],3e-5)
            self.assertEqual(_weights_sha(model.policy),'6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74')
            restored=manager.load_verified(None,run_id='R11_NO_UPDATE_ONLY_STEP0')
            for instance in (model,restored):
                self.assertEqual(verify_effective_lr(instance)['learning_rate'],3e-5)
                for progress in (0.,.5,1.):self.assertEqual(instance.lr_schedule(progress),3e-5)
                self.assertTrue(all(g['lr']==3e-5 for g in instance.policy.optimizer.param_groups))
                self.assertFalse(instance.policy.optimizer.state);self.assertEqual(instance.num_timesteps,0)
            self.assertTrue(np.array_equal(restored.predict(probe[0],deterministic=True)[0],probe[1]))
            self.assertEqual(reuse_prior_original_step0(_weights_sha(restored.policy))['overall']['navigation_success_rate'],1/6)
            model._setup_learn(128,reset_num_timesteps=False)
            _,_,_,_,info=env.step(np.zeros(4,dtype=np.float32))
            self.assertEqual((env.policy_steps,env.physics_ticks),(1,20));self.assertEqual(info['distance_stage'],'SHORT')
            self.assertEqual(info['scenario_identity_sha256'],scheduler.current_identity)
            self.assertEqual(event['checkpoint']['scheduler_state']['current_identity'],scheduler.current_identity)
            with self.assertRaises(ValueError):initialize_lower_lr(model,scheduler,manager,identity='AGAIN',seed=0)
            model.policy.optimizer.param_groups[0]['lr']=3e-4
            with self.assertRaises(ValueError):verify_effective_lr(model)


if __name__=='__main__':unittest.main()
