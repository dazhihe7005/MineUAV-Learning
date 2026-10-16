"""R14 actual fresh init/save/load/one-step chain; learning is forbidden."""
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
from lidar_integration.ppo_training_runner import _weights_sha
from .runtime import ROOT


class CenteredTrainingTests(unittest.TestCase):
    def test_actual_new_policy_step0_checkpoint_reload_and_one_physics_step_no_optimizer(self):
        def forbidden(*a,**kw):
            raise AssertionError('NO_OPTIMIZER_R14_ADMISSION')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            guard=patch.object(cls,name,forbidden);guard.start();self.addCleanup(guard.stop)
        torch.set_num_threads(1)
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.centered_training'),
                             'missing R14 integration hook')
        from .centered_training import initialize_centered_training, annotate_centered_validation
        from .state_centered_policy import make_centered_ppo,verify_centered_policy
        from .repaired_distance_training import make_repaired_scheduler
        prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        scheduler=make_repaired_scheduler(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0)
        self.addCleanup(scheduler.close)
        with tempfile.TemporaryDirectory() as d:
            env=FormalHealthMetricsEnv(scheduler,diagnostics_dir=Path(d));self.addCleanup(env.close)
            model=make_centered_ppo(env,seed=0)
            manager=CheckpointManager(Path(d),prereg,execution_protocol_sha256='e'*64)
            init,event,probe=initialize_centered_training(model,scheduler,manager,identity='R14_NO_UPDATE_ONLY',seed=0)
            self.assertEqual(init['assigned_policy_sha256'],'20f5c226abc762e5aa4a9046ba400b2b0f7012b0da6715d14204b097cae06001')
            restored=manager.load_verified(None,run_id='R14_NO_UPDATE_ONLY_STEP0')
            verify_centered_policy(restored)
            self.assertTrue(np.array_equal(restored.predict(probe[0],deterministic=True)[0],probe[1]))
            self.assertEqual(event['checkpoint']['scheduler_state']['current_identity'],scheduler.current_identity)
            self.assertEqual(restored.num_timesteps,0)
            self.assertFalse(restored.policy.optimizer.state)
            result={'training_timesteps_before_after':0,'policy_sha256_before_after':_weights_sha(restored.policy),
                    'model_status':'UNTRAINED_RANDOM_MEAN'}
            labelled=annotate_centered_validation(result)
            self.assertEqual(labelled['model_status'],'ANALYTIC_HAND_WRITTEN_PRIOR_NOT_LEARNED')
            self.assertFalse(labelled['reused_historical_evidence'])
            self.assertEqual(result['model_status'],'UNTRAINED_RANDOM_MEAN')
            with self.assertRaises(ValueError):
                annotate_centered_validation({**result,'policy_sha256_before_after':'6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74'})
            model._setup_learn(128,reset_num_timesteps=False)
            _,_,_,_,info=env.step(np.zeros(4,dtype=np.float32))
            self.assertEqual((env.policy_steps,env.physics_ticks),(1,20))
            self.assertEqual(info['distance_stage'],'SHORT')
            self.assertEqual(info['scenario_identity_sha256'],scheduler.current_identity)
            with self.assertRaises(ValueError):
                initialize_centered_training(model,scheduler,manager,identity='AGAIN',seed=0)


if __name__=='__main__':
    unittest.main()
