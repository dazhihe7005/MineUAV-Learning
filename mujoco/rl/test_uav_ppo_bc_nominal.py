"""Evaluation-only tests; never invoke PPO.learn or expert data training."""
from pathlib import Path
import numpy as np
import unittest
import tempfile
import json
from uav_ppo_bc_nominal import fresh_targets, frozen_controller, compatibility, initial_snapshot
from uav_bc_yaw_data import assert_isolated
from uav_bc_policy import parameter_hash
from ppo_pi_env import MineUAVPIEnv
from decision_fidelity_snapshot import capture, restore, fingerprint

ROOT=Path(__file__).resolve().parents[2]

def targets_deterministic_and_isolated():
    a=fresh_targets([])
    assert len(a)==200 and a==fresh_targets([])
    assert len({t['env_seed'] for t in a})==200
    assert_isolated(a,[])
    with unittest.TestCase().assertRaises(ValueError):fresh_targets([a[0]])

def ppo_7d_current_pi_compatibility_and_frozen():
    env=MineUAVPIEnv(reward_version='v2')
    try:
        controller,module,meta=frozen_controller(ROOT,'ppo7d')
        assert compatibility(env,module)['compatible']
        assert meta['training_seed']==0 and meta['actual_training_timesteps']==100352
        assert not module.training and all(not p.requires_grad for p in module.parameters())
        obs,_=env.reset(seed=3)
        before=parameter_hash(module)
        a=controller(obs);b=controller(obs)
        np.testing.assert_array_equal(a,b)
        assert a.shape==(4,) and np.isfinite(a).all() and (np.abs(a)<=1).all()
        assert parameter_hash(module)==before
    finally:env.close()

def paired_initial_state_and_observation_semantics():
    env=MineUAVPIEnv(reward_version='v2');target=fresh_targets([])[0]
    try:
        snap,meta=initial_snapshot(env,target)
        assert env._get_obs().shape==(7,)
        np.testing.assert_allclose(env._get_obs()[:3],np.asarray(target['target'])-[0,0,1],atol=1e-7)
        np.testing.assert_array_equal(env._get_obs()[3:],np.zeros(4))
        env.step(np.ones(4,dtype=np.float32)*.2)
        previous=restore(env,snap)
        assert fingerprint(capture(env,previous))==meta['snapshot_sha256']
        assert env.physics_dt==.002 and env.control_dt==.01 and env.policy_dt==.04
        assert env.max_episode_steps==375
    finally:env.close()

def bc_independent_frozen_inference(name):
    predict,module,_=frozen_controller(ROOT,name)
    before=parameter_hash(module)
    a=predict(np.array([.2,-.5,.1,.1,.2,0.,.03],np.float32))
    assert a.shape==(4,) and (np.abs(a)<=1).all()
    assert all(not p.requires_grad for p in module.parameters())
    assert parameter_hash(module)==before

class NominalTests(unittest.TestCase):
    test_targets=staticmethod(targets_deterministic_and_isolated)
    test_ppo_compatibility=staticmethod(ppo_7d_current_pi_compatibility_and_frozen)
    test_initial_state=staticmethod(paired_initial_state_and_observation_semantics)
    def test_original_bc(self):bc_independent_frozen_inference('original_bc')
    def test_yaw_bc(self):bc_independent_frozen_inference('yaw_bc')
    def test_motion_metrics(self):
        from uav_ppo_bc_nominal_run import motion_metrics
        trace=dict(positions=np.array([[0,0,1],[.3,.4,1],[.6,.8,1.]]),
            observations=np.array([[1,0,0,0,0,0,0],[.8,0,0,.3,.4,0,0],[.2,0,0,0,0,0,0]]),
            physical_state_valid=np.ones(3,bool))
        m=motion_metrics(trace)
        self.assertAlmostEqual(m['trajectory_length_m'],1.)
        self.assertEqual(m['final_velocity_xyz_m_s'],[0.,0.,0.])
        self.assertFalse(m['ever_joint_threshold'])
    def test_resume_duplicate_missing_and_corruption(self):
        from uav_bc_robustness import validate_cohort
        with self.assertRaises(ValueError):validate_cohort([{'task_key':'a'},{'task_key':'a'}],['a','b'])
        with self.assertRaises(ValueError):validate_cohort([{'task_key':'a'}],['a','b'])
        from uav_ppo_bc_nominal_run import validate_execution
        with self.assertRaises(ValueError):validate_execution({'kind':'repeat'},'expected')
    def test_incompatible_10d_rejected(self):
        from types import SimpleNamespace
        import gymnasium as gym
        env=MineUAVPIEnv(reward_version='v2')
        try:
            policy=SimpleNamespace(observation_space=gym.spaces.Box(-100,100,(10,),dtype=np.float32),action_space=env.action_space)
            with self.assertRaises(ValueError):compatibility(env,policy)
        finally:env.close()
    def test_no_teacher_in_bc_or_ppo(self):
        from unittest.mock import patch
        for name in ['original_bc','yaw_bc','ppo7d']:
            predict,_,_=frozen_controller(ROOT,name)
            with patch('test_env_scripted_policy.scripted_action',side_effect=AssertionError('teacher leak')):
                self.assertEqual(predict(np.zeros(7,np.float32)).shape,(4,))
    def test_all_failures_not_completion_and_denominator(self):
        from uav_ppo_bc_nominal_analysis import summary
        from types import SimpleNamespace
        # A representative cached row is not needed: minimal real metric schema.
        row=dict(termination_reason='time_limit',steps=375,near_steps=0,success=False,
            completion_time_s=None,simulated_seconds=15.,final_distance_m=1.,final_speed_m_s=.5,
            near_actual_sum=0.,near_command_sum=0.,action_saturation_elements=0,crossings=1,
            initial_distance_m=2.,distance_reduction_m=1.,max_position_excursion_m=2.,maximum_speed_m_s=.8,
            final_abs_yaw_error_rad=.1,maximum_abs_yaw_error_rad=.2,velocity_settled_below_015_s=None,
            yaw_settled_below_2deg_s=None,reached_target_region=True,trajectory_length_m=3.,
            ever_joint_threshold=False,maximum_success_streak=0)
        s=summary([row,row.copy()])
        self.assertEqual(s['episodes'],2);self.assertEqual(s['timeouts'],2)
        self.assertEqual(s['successes'],0);self.assertIsNone(s['completion_time_s'])
        self.assertEqual(s['failed_target_entry_without_success'],2)
    def test_paired_wrong_targets_and_success_censoring(self):
        from uav_ppo_bc_nominal_analysis import paired
        a=[dict(key='fresh/0',target=[1,0,1],initial_snapshot_sha256='a',success=False,
                termination_reason='time_limit',final_distance_m=1.,completion_time_s=None,
                trajectory_length_m=3.,maximum_speed_m_s=1.,final_speed_m_s=.4)]
        b=[dict(a[0],success=True,termination_reason='success',completion_time_s=7.,final_distance_m=.08)]
        p=paired(a,b)
        self.assertEqual(p['discordance']['alternative_only_success'],1)
        self.assertIsNone(p['completion_time_delta_s'])
        with self.assertRaises(ValueError):paired(a,[dict(b[0],target=[2,0,1])])
    def test_sb3_actual_actor_critic_and_distribution_shapes(self):
        import torch
        _,policy,_=frozen_controller(ROOT,'ppo7d')
        with torch.inference_mode():
            actions,values,logs=policy(torch.zeros((3,7)),deterministic=True)
            v,lp,entropy=policy.evaluate_actions(torch.zeros((3,7)),actions)
        self.assertEqual(tuple(actions.shape),(3,4));self.assertEqual(tuple(values.shape),(3,1))
        self.assertEqual(tuple(logs.shape),(3,));self.assertEqual(tuple(lp.shape),(3,))
        self.assertEqual(tuple(entropy.shape),(3,));self.assertEqual(tuple(policy.log_std.shape),(4,))
        self.assertEqual(sum(p.numel() for p in policy.parameters()),9673)
    def test_actual_reward_v2_formula_without_training(self):
        env=MineUAVPIEnv(reward_version='v2')
        try:
            env.reset(seed=2);env.data.qvel[:3]=[.2,-.1,.3]
            action=np.array([.2,.1,-.1,.3])
            reward,_=env._compute_reward(.4,.3,action,True,False)
            expected=10*.1-.005*(action@action)-.5*.5*(.04+.01+.09)+10
            self.assertAlmostEqual(reward,expected)
        finally:env.close()
    def test_resume_full_current_cache_no_new_simulation(self):
        from uav_ppo_bc_nominal import MANIFEST,PARTS
        from uav_ppo_bc_nominal_run import evaluate
        from unittest.mock import patch
        manifest_path=ROOT/'mujoco/reports'/MANIFEST
        data_path=ROOT/'mujoco/reports'/PARTS/'evaluation.json'
        if not data_path.exists():self.skipTest('local-only raw evaluation cache unavailable')
        m=json.loads(manifest_path.read_text());data=json.loads(data_path.read_text());row=data['records'][0]
        identity=dict(manifest_sha256=m['sha256'],model_identity=m['identity']['models'])
        state=next(s for s in m['states'] if s['key']==row['key'])
        path=data_path.parent/f"{state['split']}_{state['index']:03d}_{row['controller']}.json"
        with patch('uav_ppo_bc_nominal_run.rollout',side_effect=AssertionError('resume re-simulation')):
            result,new=evaluate(None,None,None,state,row['controller'],path,identity)
        self.assertFalse(new);self.assertEqual(result,row)
        with self.assertRaises(ValueError):evaluate(None,None,None,state,row['controller'],path,{'wrong':True})
    def test_gae_actual_buffer_numerical_formula(self):
        import torch
        from gymnasium.spaces import Box
        from stable_baselines3.common.buffers import RolloutBuffer
        b=RolloutBuffer(2,Box(-100,100,(7,),dtype=np.float32),Box(-1,1,(4,),dtype=np.float32),
            device='cpu',gamma=.99,gae_lambda=.95,n_envs=1)
        b.rewards[:]=[[1.],[2.]];b.values[:]=[[.5],[.7]];b.episode_starts[:]=[[True],[False]]
        b.compute_returns_and_advantage(torch.tensor([[.8]]),np.array([False]))
        a1=2+.99*.8-.7;a0=1+.99*.7-.5+.99*.95*a1
        np.testing.assert_allclose(b.advantages[:,0],[a0,a1],rtol=1e-6)
        np.testing.assert_allclose(b.returns[:,0],[a0+.5,a1+.7],rtol=1e-6)

if __name__=='__main__':unittest.main()
