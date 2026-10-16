"""Real fresh Actor / true short task step, zero optimizer, not an episode study."""
import importlib.util
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from lidar_integration.ppo_formal_scheduler import load_l6_prereg
from lidar_integration.ppo_formal_seed0_health import FormalHealthMetricsEnv
from lidar_integration.ppo_preflight import make_ppo
from lidar_integration.ppo_training_runner import _weights_sha
from .repaired_distance_training import make_repaired_scheduler
from .runtime import ROOT


class ShortPriorRealTests(unittest.TestCase):
    def test_fresh_prior_real_task_step_keeps_weights_and_accounting(self):
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_R9_INTERFACE')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)
        torch.set_num_threads(1)
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.short_prior'),'missing R9 initialization')
        from .short_prior import initialize_short_prior
        prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        scheduler=make_repaired_scheduler(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0)
        self.addCleanup(scheduler.close)
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        env=FormalHealthMetricsEnv(scheduler,diagnostics_dir=Path(directory.name));self.addCleanup(env.close)
        model=make_ppo(env,seed=0);record=initialize_short_prior(model)
        self.assertEqual(record['policy_sha256'],'6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74')
        self.assertEqual(scheduler.selected_identities,[])
        obs,info=env.reset(options={'scenario_identity_sha256':scheduler.env.rows[0]['identity_sha256']})
        action=model.predict(obs,deterministic=True)[0]
        # Original task yaw can request a large yaw-rate even at a short XYZ goal.
        # Check the independently specified analytic map, not a made-up 4D norm.
        expected=np.clip(10*np.tanh(.5*np.tanh(.2*obs['state'][[0,1,2,6]]*np.array([1/4.5,1/4.5,1/3,1]))),-1,1)
        np.testing.assert_allclose(action,expected,atol=2e-7)
        self.assertTrue(np.isfinite(action).all());self.assertLess(float(np.linalg.norm(action[:3])),.05)
        _,_,_,_,info=env.step(action)
        self.assertEqual((env.policy_steps,env.physics_ticks),(1,20))
        self.assertEqual(info['distance_stage'],'SHORT');self.assertEqual(info['scenario_identity_sha256'],scheduler.current_identity)
        self.assertEqual(_weights_sha(model.policy),record['policy_sha256'])
        self.assertFalse(model.policy.optimizer.state);self.assertEqual(model._n_updates,0)
        with self.assertRaises(ValueError):initialize_short_prior(model)


if __name__=='__main__':unittest.main()
