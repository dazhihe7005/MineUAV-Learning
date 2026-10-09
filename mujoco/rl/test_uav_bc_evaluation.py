"""Closed-loop independence, accounting, and bounded offline diagnostics."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch


class EvaluationTests(unittest.TestCase):
    def test_published_outcomes_never_invent_success_or_failure(self):
        from run_uav_behavior_cloning import outcome_prose
        def evaluated(successes):
            return {'bc':{'benchmark':dict(episodes=100,successes=successes),
                          'holdout':dict(episodes=100,successes=successes)},
                    'scripted':{'benchmark':dict(episodes=100,successes=100),
                                'holdout':dict(episodes=100,successes=100)}}
        good=outcome_prose(evaluated(100))
        self.assertIn('200/200',good['conclusion']);self.assertIn('No failed BC',good['failure_note'])
        bad=outcome_prose(evaluated(0))
        self.assertIn('0/200',bad['conclusion']);self.assertNotIn('200/200 successes',bad['conclusion'])
        self.assertNotIn('No failed BC',bad['failure_note'])
        mixed=outcome_prose(evaluated(50))
        self.assertIn('100/200',mixed['conclusion']);self.assertIn('100',mixed['failure_note'])

    def test_missing_trajectory_caption_distinguishes_cohorts(self):
        from uav_bc_plots import missing_trajectory_notice
        self.assertIn('No BC success',missing_trajectory_notice(True,200))
        self.assertNotIn('No BC failure',missing_trajectory_notice(True,200))
        self.assertIn('No BC failure',missing_trajectory_notice(False,200))

    def test_offline_error_and_train_regions(self):
        from uav_bc_evaluation import action_metrics, region_metrics
        y=np.array([[0,1,2,3],[1,2,3,4]],float)
        r=action_metrics(y+2,y,[1,2,4,8])
        self.assertEqual(r['rmse'],2.)
        self.assertEqual(r['mae'],2.)
        self.assertAlmostEqual(r['normalized_mse'],np.mean((2/np.array([1,2,4,8]))**2))
        self.assertEqual(r['per_dimension']['rmse'],[2.]*4)
        np.testing.assert_allclose(r['per_dimension']['correlation'],[1.]*4,atol=1e-14)
        x=np.zeros((2,7)); x[:,0]=[.05,.7]
        groups=region_metrics(y+2,y,x,[1,2,4,8],dict(speed=[.1,.2],action_magnitude=[1.,2.]))
        self.assertEqual(groups['distance']['<0.1']['samples'],1)
        self.assertEqual(groups['distance']['>=0.5']['samples'],1)
        self.assertIsNone(groups['distance']['0.1-0.2'])

    def test_failure_accounting_all_episodes(self):
        from uav_bc_evaluation import trace_metrics, summarize
        rows=[]
        for reason in ['success','time_limit','ground_collision']:
            row=dict(steps=2,success=reason=='success',termination_reason=reason,
                simulated_seconds=.08,final_distance_m=.3,final_speed_m_s=.4)
            arrays=dict(observations=np.zeros((3,7)),actions=np.ones((2,4)))
            row.update(trace_metrics(arrays))
            rows.append(row)
        r=summarize(rows)
        self.assertEqual(r['successes'],1);self.assertEqual(r['physical_failures'],1)
        self.assertEqual(r['timeouts'],1);self.assertEqual(r['episodes'],3)
        self.assertEqual(r['action_saturation_fraction'],1.)

    def test_bc_execution_does_not_call_teacher(self):
        from uav_bc_data import collect_episode
        from uav_bc_policy import BCActor,BCPolicy,fit_stats
        from ppo_pi_env import MineUAVPIEnv
        torch.manual_seed(0)
        actor=BCActor()
        for p in actor.parameters():p.data.zero_()
        policy=BCPolicy(actor,fit_stats(np.zeros((2,7)),np.zeros((2,4))))
        env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.08)
        try:
            with patch('uav_bc_data.scripted_action',side_effect=AssertionError('teacher leakage')):
                row,a=collect_episode(env,dict(target_id=1,env_seed=22,target=[.3,0.,1.]),policy.predict)
            self.assertEqual(row['termination_reason'],'time_limit')
            np.testing.assert_array_equal(a['actions'],np.zeros((2,4)))
        finally:env.close()

    def test_closed_loop_cache_identity_and_reproduction(self):
        from uav_bc_evaluation import closed_loop_identity, evaluate_one
        from uav_bc_policy import BCActor,BCPolicy,fit_stats
        from ppo_pi_env import MineUAVPIEnv
        torch.manual_seed(0)
        policy=BCPolicy(BCActor(),fit_stats(np.zeros((2,7)),np.zeros((2,4))))
        root=Path(__file__).resolve().parents[2]
        identity=closed_loop_identity(root,'literal-model-hash')
        t=dict(target_id=0,env_seed=33,target=[.3,0.,1.])
        env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.08)
        try:
            with tempfile.TemporaryDirectory() as d:
                r1=evaluate_one(env,t,policy.predict,Path(d)/'one.json',identity,'bc','benchmark')
                r2=evaluate_one(env,t,policy.predict,Path(d)/'two.json',identity,'bc','benchmark')
                with np.load(r1['path']) as a,np.load(r2['path']) as b:
                    np.testing.assert_array_equal(a['observations'],b['observations'])
                    np.testing.assert_array_equal(a['actions'],b['actions'])
                cached=evaluate_one(env,t,lambda _:self.fail('repeated cached episode'),Path(d)/'one.json',identity,'bc','benchmark')
                self.assertEqual(cached,r1)
                with self.assertRaises(ValueError):evaluate_one(env,t,policy.predict,Path(d)/'one.json',dict(identity,bad=True),'bc','benchmark')
        finally:env.close()


if __name__=='__main__':unittest.main()
