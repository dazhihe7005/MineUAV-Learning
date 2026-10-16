"""Actual low-level diagnostic chain and causal counters, no optimizer updates."""
import copy,importlib.util,unittest
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from .runtime import ROOT


class StoppingDynamicsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_R18_TEST')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.stopping_trace'),
                             'missing R18 actual dynamic collector')
        from . import stopping_trace
        return stopping_trace

    def test_actual_physics_trace_causal_times_command_and_policy_unchanged(self):
        m=self.module()
        from .prior_tests import ObservationOnly
        from .state_centered_policy import make_centered_ppo
        from .centered_noise_training import assign_noise_prior
        from lidar_integration.ppo_formal_scheduler import load_l6_prereg
        p=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        model=make_centered_ppo(ObservationOnly(),seed=0);assign_noise_prior(model)
        before=copy.deepcopy(model.policy.state_dict());row=p['scenario_manifest']['validation'][0]
        emitted=[]
        result=m.collect_trace(model,row,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',max_steps=3,on_record=emitted.append)
        self.assertEqual(emitted,result['trace'])
        self.assertEqual((result['policy_steps'],result['physics_ticks']),(3,60));self.assertEqual(result['outcome'],'probe_cutoff')
        for i,t in enumerate(result['trace']):
            self.assertAlmostEqual(t['pre_time_s'],.04*i);self.assertAlmostEqual(t['post_time_s'],.04*(i+1))
            self.assertLessEqual(t['lidar_source_time_s'],t['post_time_s']);self.assertGreaterEqual(t['lidar_age_pre_s'],0)
            np.testing.assert_allclose(t['velocity_command_enu_m_s'],np.array(t['action_clipped'][:3])*[1.5,1.5,1.0],atol=1e-12)
            self.assertEqual(t['physics_steps'],20)
        self.assertFalse(model.policy.optimizer.state);self.assertEqual(model.num_timesteps,0)
        self.assertTrue(all(torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items()))

    def test_counter_uses_post_state_five_consecutive_joint_not_close_distance_alone(self):
        m=self.module();rr=[]
        for i in range(8):rr.append({'distance_m':.09,'speed_m_s':.20 if i<3 else .10,'post_time_s':.04*(i+1)})
        v=m.stop_metrics(rr)
        self.assertEqual((v['near_goal_samples'],v['near_goal_fast_samples'],v['joint_samples'],v['max_consecutive_joint_samples']),(8,3,5,5))
        self.assertAlmostEqual(v['first_five_joint_time_s'],.32)

    def test_invalid_or_noncausal_trace_cannot_be_reported_as_valid(self):
        m=self.module()
        for rows in ([{'distance_m':float('nan'),'speed_m_s':0,'post_time_s':.04}],
                     [{'distance_m':.1,'speed_m_s':0,'post_time_s':.04}]*2):
            with self.assertRaises(ValueError):m.stop_metrics(rows)
