"""Catches reset-after-perturb, teacher mixing, bad outcome denominators and stale resume."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ppo_pi_env import MineUAVPIEnv
from decision_fidelity_snapshot import fingerprint,restore,capture
import uav_bc_robustness as api


class EvaluationRobustnessTests(unittest.TestCase):
    def require(self,name):
        self.assertTrue(hasattr(api,name),f'{name} evaluation behavior missing')
        return getattr(api,name)

    def test_rollout_keeps_perturbed_state_and_independent_bc_actions(self):
        run=self.require('rollout')
        from uav_bc_policy import load,parameter_hash
        root=Path(__file__).resolve().parents[2]
        policy,_=load(root/'mujoco/rl/models'/api.MODEL);before=parameter_hash(policy.actor)
        env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.08)
        try:
            t=dict(target_id=0,env_seed=18,target=[.3,.2,1.])
            s,m=api.prepare_snapshot(env,t,dict(position_m=[.1,0,0],velocity_m_s=[.15,0,0],yaw_rad=.2))
            predicted=policy.predict(env._get_obs())
            with patch('test_env_scripted_policy.scripted_action',side_effect=AssertionError('teacher leak')):
                row,a=run(env,s,policy.predict)
            np.testing.assert_array_equal(a['observations'][0],m['observation'])
            np.testing.assert_array_equal(a['positions'][0],[.1,0,1])
            np.testing.assert_array_equal(a['actions'][0],predicted)
            np.testing.assert_array_equal(a['previous_actions'][0],[0,0,0,0])
            np.testing.assert_array_equal(a['previous_actions'][1:],a['actions'][:-1])
            self.assertEqual(row['initial_snapshot_sha256'],m['snapshot_sha256'])
            self.assertEqual(row['termination_reason'],'time_limit')
            self.assertIsNone(row['completion_time_s'])
            self.assertEqual(parameter_hash(policy.actor),before)
            self.assertTrue(all(not p.requires_grad for p in policy.actor.parameters()))
        finally:env.close()

    def test_completed_cache_reused_without_simulation_and_corruption_rejected(self):
        evaluate=self.require('evaluate_task')
        env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.08)
        try:
            target=dict(target_id=0,env_seed=1,target=[.3,0,1])
            s,meta=api.prepare_snapshot(env,target,api.sample_perturbation('velocity_small','benchmark',0))
            state=dict(key='velocity_small/benchmark/000',condition='velocity_small',split='benchmark',index=0,
                **target,perturbation=api.sample_perturbation('velocity_small','benchmark',0),initial_state=meta)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'a.json';identity=dict(manifest='fixed')
                r=evaluate(env,s,lambda o:np.zeros(4,np.float32),state,'bc',path,identity)
                cached=evaluate(env,s,lambda o:self.fail('duplicated cached episode'),state,'bc',path,identity)
                self.assertEqual(r,cached)
                with self.assertRaises(ValueError):evaluate(env,s,None,state,'bc',path,dict(manifest='other'))
                raw=Path(r['trace_path']);raw.write_bytes(b'corrupt')
                with self.assertRaises(ValueError):evaluate(env,s,None,state,'bc',path,identity)
        finally:env.close()

    def test_paired_denominators_discordance_and_success_only_time(self):
        summary=self.require('paired_summary')
        # Same literal paired cohort: both success, only BC success, neither success.
        scripted=[dict(key='a',success=True,completion_time_s=3.,final_distance_m=.05,termination_reason='success'),
                  dict(key='b',success=False,completion_time_s=None,final_distance_m=2.,termination_reason='time_limit'),
                  dict(key='c',success=False,completion_time_s=None,final_distance_m=3.,termination_reason='excessive_tilt')]
        bc=[dict(key='a',success=True,completion_time_s=4.,final_distance_m=.08,termination_reason='success'),
            dict(key='b',success=True,completion_time_s=5.,final_distance_m=.09,termination_reason='success'),
            dict(key='c',success=False,completion_time_s=None,final_distance_m=4.,termination_reason='time_limit')]
        r=summary(scripted,bc)
        self.assertEqual(r['episodes'],3);self.assertAlmostEqual(r['success_delta'],1/3)
        self.assertEqual(r['discordance'],dict(both_success=1,bc_only_success=1,scripted_only_success=0,both_failed=1))
        self.assertEqual(r['both_success_pairs'],1);self.assertEqual(r['completion_time_delta_s']['mean'],1.)
        self.assertAlmostEqual(r['final_distance_delta_m']['mean'],(.03-1.91+1)/3)
        self.assertEqual(r,summary(scripted,bc))
        with self.assertRaises(ValueError):summary(scripted,bc[:2])

    def test_no_duplicated_or_omitted_task_keys(self):
        check=self.require('validate_cohort')
        keys=['a/bc','a/scripted']
        rows=[dict(task_key=k) for k in keys]
        check(rows,keys)
        for bad in [rows[:1],rows+rows[:1],[dict(task_key='bad'),rows[1]]]:
            with self.assertRaises(ValueError):check(bad,keys)

    def test_wilson_interval_and_all_failure_accounting(self):
        wilson=self.require('wilson')
        lo,hi=wilson(100,100)
        self.assertAlmostEqual(lo,.9630065,places=6);self.assertAlmostEqual(hi,1.)
        self.assertGreater(wilson(0,100)[1],0)
        run=self.require('rollout');env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.08)
        try:
            s,_=api.prepare_snapshot(env,dict(target_id=0,env_seed=1,target=[0,0,1]),api.sample_perturbation('nominal','benchmark',0))
            s.data.qpos[1]=7. # Deliberate bad snapshot after valid preparation tests physical failure accounting.
            row,_=run(env,s,lambda o:np.zeros(4,np.float32))
            self.assertFalse(row['success']);self.assertTrue(row['physical_failure'])
            self.assertEqual(row['termination_reason'],'outside_flight_area');self.assertIsNone(row['completion_time_s'])
        finally:env.close()

    def test_nonfinite_failure_serializes_without_fake_target_recovery(self):
        # Inject at the simulator boundary; env itself detects nonfinite state.
        from uav_bc_safety import atomic_json
        from run_uav_bc_robustness import aggregate
        env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.08)
        try:
            t=dict(target_id=0,env_seed=1,target=[1.,0,1.])
            p=api.sample_perturbation('nominal','benchmark',0);s,m=api.prepare_snapshot(env,t,p)
            state=dict(key='nominal/benchmark/000',condition='nominal',split='benchmark',index=0,**t,perturbation=p,initial_state=m)
            def broken_physics(obs):
                env.data.qpos[0]=np.nan
                return np.zeros(4,np.float32)
            with tempfile.TemporaryDirectory() as d:
                r=api.evaluate_task(env,s,broken_physics,state,'bc',Path(d)/'nonfinite.json',dict(manifest='test'))
                self.assertEqual(r['termination_reason'],'nonfinite_state')
                self.assertFalse(r['reached_target_region'])
                self.assertIsNone(r['final_distance_m']);self.assertIsNone(r['final_abs_yaw_error_rad'])
                self.assertIsNone(r['distance_reduction_m']);self.assertEqual(r['minimum_distance_m'],1.)
                self.assertEqual(r['max_position_excursion_m'],0.)
                self.assertEqual(r['physical_valid_observation_count'],1)
                with np.load(r['trace_path']) as a:np.testing.assert_array_equal(a['physical_state_valid'],[True,False])
                other=dict(r,controller='scripted',task_key=state['key']+'/scripted')
                stats=aggregate([r,other],[state])['nominal']['pooled']
                self.assertEqual(stats['bc']['physical_failures'],1)
                self.assertIsNone(stats['bc']['final_distance_m'])
                self.assertEqual(stats['paired']['finite_final_distance_pairs'],0)
                self.assertIsNone(stats['paired']['final_distance_delta_m'])
                atomic_json(Path(d)/'summary.json',stats)
        finally:env.close()


if __name__=='__main__':unittest.main()
