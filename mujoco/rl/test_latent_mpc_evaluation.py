import unittest
from pathlib import Path
import numpy as np
import torch
from joint_autonomous_consistency_training import load_autonomous
from latent_mpc_core import RandomShootingMPC
from latent_mpc_evaluation import (load_targets, action_ood, timing_summary,
    crossings, evaluate_episode, summarize_episodes, realized_prefix_check)
from ppo_pi_env import MineUAVPIEnv


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root=Path(__file__).resolve().parents[2]
        cls.model,cls.stats,_=load_autonomous(Path(__file__).parent/'models/joint_latent_world_model_v3_autonomous_consistency.pt')

    def test_target_sets_are_archived_100_each_with_prior_hash(self):
        targets,meta=load_targets(self.root)
        self.assertEqual([len(targets[k]) for k in ('benchmark','holdout')],[100,100])
        self.assertEqual(meta['benchmark']['target_sha256'],'f6b18c73b9834bd0ad121f32304d2dbf01f934fa1a1c1cdd1f211ecff08c3283')
        self.assertEqual(meta['holdout']['target_sha256'],'df8b5995b7f1a4ab423c9d4e8aba95336085a8ca601e9e5d8db053f0bbb795fc')

    def test_action_ood_and_timing_use_every_selected_step(self):
        stats={'mean':[0]*4,'std':[1]*4,'q025':[-.5]*4,'q975':[.5]*4}
        out=action_ood(np.array([[0,0,0,0],[1,0,0,0]]),stats)
        self.assertEqual(out['outside_train_95_any_dimension_fraction'],.5)
        self.assertEqual(out['outside_train_95_element_fraction'],.125)
        self.assertEqual(out['standardized_norm']['mean'],.5)
        t=timing_summary([.01,.02,.03,.04])
        self.assertAlmostEqual(t['mean_ms'],25)
        self.assertAlmostEqual(t['median_ms'],25)
        self.assertAlmostEqual(t['p95_ms'],38.5)
        self.assertEqual(t['decision_count'],4)

    def test_crossing_uses_first_near_axis_and_two_cm_hysteresis(self):
        obs=np.zeros((5,7)); obs[:,0]=[.5,.15,-.01,-.03,.04]
        self.assertEqual(crossings(obs),2)
        self.assertEqual(crossings(np.array([[.5,0,0,0,0,0,0]])),0)

    def test_real_pi_receding_horizon_runs_one_replan_per_real_step_and_resets(self):
        env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.12)
        planner=RandomShootingMPC(self.model,self.stats,env.action_space.low,env.action_space.high,0)
        try:
            row,trace=evaluate_episode(env,'mpc',7,[1.,0.,1.],planner,19)
            self.assertEqual(row['episode_steps'],3)
            self.assertEqual(planner.encoding_count,3)
            self.assertEqual(planner.planning_count,3)
            self.assertEqual(len(trace['actions']),3)
            self.assertEqual(len(trace['observations']),4)
            np.testing.assert_array_equal(trace['previous_actions'][0],np.zeros(4))
            np.testing.assert_array_equal(trace['previous_actions'][1:],trace['actions'][:-1])
            row2,trace2=evaluate_episode(env,'mpc',7,[1.,0.,1.],planner,19)
            np.testing.assert_array_equal(trace['actions'],trace2['actions'])
            np.testing.assert_array_equal(trace['observations'],trace2['observations'])
            self.assertEqual(row['termination_reason'],'time_limit')
            s=summarize_episodes([row,row2])
            self.assertEqual(s['timeouts'],2)
            self.assertIsNone(s['near_target_actual_speed_m_s'])
            self.assertIsNone(s['mean_completion_time_s'])
        finally: env.close()

    def test_realized_prefix_predictions_do_not_consume_future_observations(self):
        env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.52)
        planner=RandomShootingMPC(self.model,self.stats,env.action_space.low,env.action_space.high,0)
        try:
            _,trace=evaluate_episode(env,'mpc',0,[.6,0.,1.],planner,9)
            result=realized_prefix_check(self.model,self.stats,trace,stride=1)
            altered=dict(trace,observations=trace['observations'].copy())
            altered['observations'][1:]+=100
            changed=realized_prefix_check(self.model,self.stats,altered,stride=1)
            np.testing.assert_array_equal(result['predicted'],changed['predicted'])
            self.assertFalse(np.array_equal(result['actual'],changed['actual']))
            self.assertEqual(result['predicted'].shape[1:],(2,7))
        finally: env.close()


if __name__=='__main__': unittest.main()
