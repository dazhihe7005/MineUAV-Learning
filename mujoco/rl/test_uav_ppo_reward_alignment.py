"""Pure offline fixtures: no simulator construction, reset or step."""
import importlib
import hashlib
import json
import unittest
import numpy as np

try:
    audit = importlib.import_module('uav_ppo_reward_alignment')
except ModuleNotFoundError:
    audit = None


def trace():
    d=np.array([.4,.3,.09,.08,.09,.08,.07])
    v=np.array([0,.2,.3,.1,.1,.1,.1])
    o=np.zeros((7,7),np.float32);o[:,0]=d;o[:,3]=v
    p=np.zeros((7,3));p[:,0]=-d
    a=np.zeros((6,4),np.float32);a[:,0]=.2
    return dict(observations=o,positions=p,actions=a,
        previous_actions=np.r_[np.zeros((1,4),np.float32),a[:-1]],
        step_index=np.arange(6),success_streak=np.array([0,0,0,1,2,3,4]),
        physical_state_valid=np.ones(7,bool))


def training_fixture():
    return dict(rank=0,episode_index=0,target=[1.,0.,1.],steps=375,
        termination_reason='time_limit',reward_components=dict(
            progress=8.,action=-.2,brake=-.3,success=0.,failure=0.,total=7.5)),dict(
            r='7.500000',l='375',distance_m='.2',termination_reason='time_limit')


class RewardAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit,'offline reward-audit implementation missing')

    def test_missing_reward_is_not_estimated_from_state(self):
        x=audit.reward_availability(trace())
        self.assertFalse(x['recorded_per_step_reward'])
        self.assertIsNone(x['final_episode_return'])
        self.assertIn('not observable',x['status'])

    def test_recorded_reward_presence_requires_alignment(self):
        a=trace();a['rewards']=np.zeros(5)
        with self.assertRaises(ValueError):audit.reward_availability(a)

    def test_state_metrics_use_post_transition_streak_and_pre_action_near_region(self):
        x=audit.state_metrics(trace(),[0,0,0])
        self.assertEqual(x['steps'],6)
        self.assertEqual(x['joint_post_samples'],4)
        self.assertEqual(x['maximum_streak'],4)
        self.assertEqual(x['near_pre_samples'],4)
        self.assertAlmostEqual(x['near_speed_mean_m_s'],.15,places=6)
        self.assertAlmostEqual(x['near_command_speed_mean_m_s'],.3,places=6)
        self.assertAlmostEqual(x['last3s_distance_reduction_m'],.33)
        self.assertIsNone(x['first2s_distance_reduction_m'])

    def test_action_history_corruption_is_rejected(self):
        a=trace();a['previous_actions'][2,0]=-.8
        with self.assertRaises(ValueError):audit.state_metrics(a,[0,0,0])

    def test_nonfinite_placeholder_is_not_real_task_state(self):
        a=trace();a['physical_state_valid'][-1]=False
        with self.assertRaises(ValueError):audit.state_metrics(a,[0,0,0])

    def test_wrong_target_or_float_observation_frame_is_rejected(self):
        with self.assertRaises(ValueError):audit.state_metrics(trace(),[1,0,0])

    def test_recorded_training_components_match_monitor_and_distance(self):
        r,m=training_fixture();x=audit.verify_training_episode(r,m)
        self.assertEqual(x['recorded_return'],7.5)
        self.assertEqual(x['difficulty_adjusted_return'], -2.5)
        self.assertFalse(x['success'])

    def test_monitor_total_disagreement_is_rejected(self):
        r,m=training_fixture();m['r']='7.6'
        with self.assertRaises(ValueError):audit.verify_training_episode(r,m)

    def test_component_progress_must_match_independent_distance(self):
        r,m=training_fixture();m['distance_m']='.3'
        with self.assertRaises(ValueError):audit.verify_training_episode(r,m)

    def test_component_sum_and_terminal_bonus_consistency(self):
        r,m=training_fixture();r['reward_components']['success']=10
        with self.assertRaises(ValueError):audit.verify_training_episode(r,m)

    def test_empty_statistics_are_missing_not_zero(self):
        self.assertIsNone(audit.distribution([]))

    def test_correlations_are_missing_for_constant_or_too_few_samples(self):
        self.assertIsNone(audit.correlations([1,1],[0,1])['spearman'])
        x=audit.correlations([1,2,3],[3,2,1])
        self.assertAlmostEqual(x['spearman'],-1.)

    def test_actual_reward_method_has_no_stationary_distance_penalty(self):
        self.assertTrue(hasattr(audit,'source_examples'),'source probes missing')
        cases=audit.source_examples()
        self.assertEqual(cases['stopped_at_020m']['total'],0.)
        self.assertAlmostEqual(cases['approaching_009m_at_020m_s']['total'],.08)
        self.assertEqual(set(cases['stopped_at_020m']),{'progress','action','brake','success','failure','total'})

    def test_discounted_bonus_uses_step_minus_one(self):
        self.assertTrue(hasattr(audit,'discounted_bonus'),'discount diagnostic missing')
        self.assertEqual(audit.discounted_bonus(1),10.)
        self.assertAlmostEqual(audit.discounted_bonus(2),9.9)
        self.assertAlmostEqual(audit.discounted_bonus(3),9.801)

    def test_empty_matched_strata_preserve_null_values(self):
        self.assertTrue(hasattr(audit,'stratify'),'strata implementation missing')
        r=dict(initial_distance_bin='<1m',duration_bin='>12s',success=False,recorded_return=2.)
        x=audit.stratify([r],'recorded_return')
        self.assertEqual(x['<1m/>12s/failure']['n'],1)
        self.assertIsNone(x['>=2m/<=8s/success']['value'])

    def test_cohort_rejects_duplicate_or_omitted_tasks(self):
        self.assertTrue(hasattr(audit,'validate_cohort'),'cohort guard missing')
        with self.assertRaises(ValueError):audit.validate_cohort(['a','a'],['a','b'])

    def test_tampered_or_wrong_identity_is_rejected(self):
        self.assertTrue(hasattr(audit,'verify_record_identity'),'record guard missing')
        row={'task_key':'a/ppo_seed0','identity':{'manifest_sha256':'m','controller':'ppo_seed0','model_identity':{'sha':'c'}}}
        row['record_sha256']=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        audit.verify_record_identity(row,'m','ppo_seed0',{'sha':'c'})
        row['task_key']='b/ppo_seed0'
        with self.assertRaises(ValueError):audit.verify_record_identity(row,'m','ppo_seed0',{'sha':'c'})

    def test_source_bootstrap_characterization_without_environment(self):
        from stable_baselines3.common.buffers import RolloutBuffer
        from gymnasium.spaces import Box
        import torch
        buffer=RolloutBuffer(1,Box(-1,1,(7,)),Box(-1,1,(4,)),gamma=.99,gae_lambda=.95)
        buffer.rewards[0,0]=1.98  # env reward0 plus .99*terminal_value2
        buffer.values[0,0]=3
        buffer.compute_returns_and_advantage(torch.tensor([999.]),np.array([True]))
        self.assertAlmostEqual(float(buffer.returns[0,0]),1.98,places=6)

    def test_training_summary_serializes_count_as_python_scalar(self):
        from uav_ppo_reward_alignment_analysis import training_summary
        r,m=training_fixture();failed=audit.verify_training_episode(r,m)
        r.update(steps=100,termination_reason='success')
        r['reward_components']=dict(progress=9.5,action=-.2,brake=-13.3,success=10.,failure=0.,total=6.)
        m.update(r='6',l='100',distance_m='.05',termination_reason='success')
        success=audit.verify_training_episode(r,m)
        result=training_summary([success,failed])
        self.assertEqual(result['failed_above_success_q25'],1)
        self.assertIsInstance(result['failed_above_success_q25'],int)
        json.dumps(result,allow_nan=False)

    def test_common_time_prefix_uses_same_target_and_reports_censoring(self):
        self.assertTrue(hasattr(audit,'paired_early_states'),'common-time target pairing missing')
        rows=[dict(target_id='a',seed=0,success=True,first2s_distance_reduction_m=.4,first2s_speed_mean_m_s=.2),
              dict(target_id='a',seed=1,success=False,first2s_distance_reduction_m=.1,first2s_speed_mean_m_s=.3),
              dict(target_id='b',seed=0,success=True,first2s_distance_reduction_m=None,first2s_speed_mean_m_s=None),
              dict(target_id='b',seed=1,success=False,first2s_distance_reduction_m=.1,first2s_speed_mean_m_s=.3)]
        x=audit.paired_early_states(rows)['0_vs_1']
        self.assertEqual(x['discordant_pairs'],2)
        self.assertEqual(x['observed_2s_pairs'],1)
        self.assertEqual(x['censored_before_2s'],1)
        self.assertAlmostEqual(x['failed_minus_success_distance_reduction_m']['mean'],-.3)
        self.assertAlmostEqual(x['failed_minus_success_speed_m_s']['mean'],.1)

    def test_first2s_uses_fifty_post_transition_samples(self):
        d=np.linspace(3.,1.,51);o=np.zeros((51,7),np.float32);o[:,0]=d;o[:,3]=.3
        p=np.zeros((51,3));p[:,0]=-d;a=np.zeros((50,4),np.float32)
        x=audit.state_metrics(dict(observations=o,positions=p,actions=a,
            previous_actions=a.copy(),step_index=np.arange(50),
            success_streak=np.zeros(51,int),physical_state_valid=np.ones(51,bool)),[0,0,0])
        self.assertAlmostEqual(x['first2s_distance_reduction_m'],2.)
        self.assertAlmostEqual(x['first2s_speed_mean_m_s'],.3,places=6)


if __name__=='__main__':unittest.main()
