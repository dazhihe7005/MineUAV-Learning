"""R20 real tensor/checkpoint/GAE contracts, no learn/train/Adam.step."""
import importlib.util,tempfile,unittest,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from research_ops.storage import OpsError
from .runtime import ROOT


class TemporalCreditTensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*a,**k):raise AssertionError('NO_OPTIMIZER_R20_TEST')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)

    def setup_model(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.temporal_credit_training'),'missing R20 temporal adapter')
        from lidar_integration.ppo_formal_scheduler import load_l6_prereg
        from .empty_rehearsal import make_rehearsal_scheduler
        from .state_centered_policy import make_centered_ppo
        from lidar_integration.ppo_formal_seed0_health import FormalHealthMetricsEnv
        self.prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        diagnostic=tempfile.TemporaryDirectory();self.addCleanup(diagnostic.cleanup)
        s=make_rehearsal_scheduler(self.prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0);self.addCleanup(s.close)
        env=FormalHealthMetricsEnv(s,diagnostics_dir=Path(diagnostic.name));self.addCleanup(env.close)
        m=make_centered_ppo(env,seed=0);return m,s,env

    def test_gamma_and_buffer_change_but_weights_predictions_optimizer_objects_do_not(self):
        m,s,e=self.setup_model()
        from .temporal_credit_training import configure_temporal_credit,verify_temporal_configuration
        from lidar_integration.ppo_training_runner import _weights_sha
        obs,_=e.reset();before=_weights_sha(m.policy);action=m.predict(obs,deterministic=True)[0].copy()
        value=m.policy.predict_values(m.policy.obs_to_tensor(obs)[0]).detach().clone();params=list(map(id,m.policy.parameters()))
        configured=configure_temporal_credit(m);self.assertEqual((m.gamma,m.rollout_buffer.gamma),(.995,.995))
        self.assertEqual((m.gae_lambda,m.rollout_buffer.gae_lambda),(.95,.95));self.assertEqual(before,_weights_sha(m.policy))
        np.testing.assert_array_equal(m.predict(obs,deterministic=True)[0],action)
        torch.testing.assert_close(m.policy.predict_values(m.policy.obs_to_tensor(obs)[0]),value,rtol=0,atol=0)
        self.assertEqual(params,list(map(id,m.policy.parameters())));self.assertFalse(m.policy.optimizer.state)
        self.assertEqual(verify_temporal_configuration(m),configured)
        with self.assertRaises(OpsError):configure_temporal_credit(m)
        m.rollout_buffer.gamma=.99
        with self.assertRaises(OpsError):verify_temporal_configuration(m)

    def test_actual_sb3_gae_return_recursion_uses_new_gamma_not_old(self):
        m,s,e=self.setup_model()
        from .temporal_credit_training import configure_temporal_credit
        configure_temporal_credit(m);b=m.rollout_buffer;obs,_=e.reset()
        for i in range(128):
            b.add({k:np.expand_dims(v,0) for k,v in obs.items()},np.zeros((1,4),np.float32),
                np.array([1. if i==127 else 0.]),np.array([i==0]),torch.zeros(1),torch.zeros(1))
        b.compute_returns_and_advantage(last_values=torch.zeros(1),dones=np.ones(1))
        self.assertTrue(np.isfinite(b.advantages).all());self.assertAlmostEqual(float(b.advantages[-2,0]),.995*.95,places=6)
        self.assertNotAlmostEqual(float(b.advantages[-2,0]),.99*.95,places=4)
        self.assertAlmostEqual(float(b.advantages[0,0]),(.995*.95)**127,places=6)
        self.assertEqual(m.num_timesteps,0);self.assertFalse(m.policy.optimizer.state)

    def test_real_zero_update_checkpoint_preserves_gamma_initial_policy_and_scheduler(self):
        m,s,e=self.setup_model()
        from .temporal_credit_training import configure_temporal_credit,verify_temporal_configuration
        from .centered_noise_training import initialize_noise_training
        from lidar_integration.ppo_formal_checkpoint import CheckpointManager
        configure_temporal_credit(m)
        with tempfile.TemporaryDirectory() as d:
            cm=CheckpointManager(Path(d),self.prereg,execution_protocol_sha256='a'*64)
            init,event,probe=initialize_noise_training(m,s,cm,identity='R20_NO_UPDATE_ONLY',seed=0)
            self.assertEqual(init['assigned_policy_sha256'],'650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4')
            loaded=cm.load_verified(None,run_id='R20_NO_UPDATE_ONLY_STEP0');verify_temporal_configuration(loaded)
            self.assertFalse(loaded.policy.optimizer.state);self.assertEqual(loaded.num_timesteps,0)
            np.testing.assert_array_equal(loaded.predict(probe[0],deterministic=True)[0],probe[1])
            _,_,_,_,info=e.step(np.zeros(4,dtype=np.float32));self.assertEqual((e.policy_steps,e.physics_ticks),(1,20))
            self.assertEqual(info['distance_stage'],'SHORT')
