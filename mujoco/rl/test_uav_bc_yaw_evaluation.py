"""Inference and paired reporting must not smuggle labels or exclude failures."""
import importlib,importlib.util,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

class YawEvaluationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_bc_yaw_evaluation'),'yaw evaluator missing')
        return importlib.import_module('uav_bc_yaw_evaluation')
    def test_independent_policy_keeps_snapshot_and_frozen_hash(self):
        a=self.api();from uav_bc_policy import load,parameter_hash
        from ppo_pi_env import MineUAVPIEnv
        from uav_bc_robustness import prepare_snapshot
        root=Path(__file__).resolve().parents[2];p,_=load(root/'mujoco/rl/models/uav_bc_mlp_seed0.pt');h=parameter_hash(p.actor)
        env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.08)
        try:
            t=dict(key='yaw_large/final/000',condition='yaw_large',split='final',target_id='new',env_seed=7,target=[1.,0,1.])
            perturb=dict(position_m=[.1,0,0],velocity_m_s=[.15,0,0],yaw_rad=np.pi/6)
            s,t['initial_state']=prepare_snapshot(env,t,perturb);expected=p.predict(env._get_obs())
            with tempfile.TemporaryDirectory() as d:
                with patch('test_env_scripted_policy.scripted_action',side_effect=AssertionError('teacher leakage')):
                    row=a.audit_task(env,s,p.predict,t,'original',Path(d)/'a.json',dict(model='fixed'))
                with np.load(row['trace_path']) as z:np.testing.assert_array_equal(z['actions'][0],expected)
                cached=a.audit_task(env,s,lambda o:self.fail('duplicate execution'),t,'original',Path(d)/'a.json',dict(model='fixed'))
                self.assertEqual(cached,row);self.assertFalse(row['success']);self.assertTrue(row['timeout'])
                self.assertIsNone(row['completion_time_s']);self.assertEqual(parameter_hash(p.actor),h)
        finally:env.close()
    def test_overshoot_does_not_use_invalid_terminal_zero(self):
        a=self.api();z=dict(observations=np.array([[1,0,0,.8,0,0,0],[.3,0,0,.2,0,0,0],[.5,0,0,.1,0,0,0],[0,0,0,0,0,0,0]]),physical_state_valid=np.array([1,1,1,0],bool))
        r=a.extra_metrics(z);self.assertAlmostEqual(r['distance_rebound_overshoot_m'],.2)
        self.assertIsNone(r['post_peak_velocity_recovery_s'])
    def test_final_conditions_target_reuse_and_yaw_balance(self):
        a=self.api();from ppo_pi_env import MineUAVPIEnv
        env=MineUAVPIEnv(reward_version='v2')
        try:
            targets=[dict(target_id=f'new{i}',env_seed=900+i,target=[.5+.001*i,.2,1.]) for i in range(100)]
            states=a.final_states(env,targets)
            self.assertEqual(len(states),900);self.assertEqual(len({s['key'] for s in states}),900)
            for c in ('yaw_small','yaw_large','combined_small','combined_large'):
                rows=[r for r in states if r['condition']==c]
                self.assertEqual(sum(r['perturbation']['yaw_rad']>0 for r in rows),50)
            for s in states:self.assertEqual(s['target'],targets[s['index']]['target'])
        finally:env.close()
    def test_all_controller_pairs_keep_failure_denominators(self):
        a=self.api();from uav_bc_robustness import paired_summary
        x=[dict(key='a',success=True,completion_time_s=1.,final_distance_m=.05,termination_reason='success'),
           dict(key='b',success=False,completion_time_s=None,final_distance_m=.5,termination_reason='time_limit')]
        y=[dict(x[0],success=False,completion_time_s=None,termination_reason='time_limit'),dict(x[1])]
        r=paired_summary(x,y);self.assertEqual(r['episodes'],2);self.assertEqual(r['success_delta'],-.5);self.assertEqual(r['both_success_pairs'],0)
        self.assertIsNone(r['completion_time_delta_s'])
    def test_offline_region_group_targets_are_not_inference_inputs(self):
        a=self.api();from uav_bc_evaluation import action_metrics
        o=np.zeros((2,7),np.float32);o[:,0]=[1,.1];acts=np.zeros((2,4),np.float32);acts[1,0]=.1
        r=a.offline_arrays(o,acts,np.array([30,30]),np.array([0,20]),acts,acts,np.ones(4))
        self.assertEqual(r['by_initial_yaw']['30']['raw']['rmse'],0)
        self.assertEqual(r['early_large_yaw']['samples'],1);self.assertEqual(r['near_braking']['samples'],1)
        acts[0,1]=.2;r=a.offline_arrays(o,np.zeros_like(acts),np.array([30,30]),np.array([0,20]),acts,acts,np.ones(4))
        self.assertGreater(r['early_large_yaw']['rmse'],0)

if __name__=='__main__':unittest.main()
