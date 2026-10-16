"""Full real-physics distance accounting, zero optimizer updates allowed."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from lidar_integration.ppo_formal_scheduler import load_l6_prereg
from lidar_integration.ppo_formal_seed0_health import FormalHealthMetricsEnv
from lidar_integration.failure_evidence import FailureEvidenceRecorder
from .runtime import ROOT
from .short_goal_env import make_short_scheduler


class TaskAccountingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_R7_TEST')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)
        self.prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        self.upstream=ROOT/'.research_local/dependencies/MuJoCo-LiDAR'
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.output=Path(self.temp.name)

    def chain(self,*,adapt=True,seed=0):
        scheduler=make_short_scheduler(self.prereg,self.upstream,seed=seed)
        self.addCleanup(scheduler.close)
        if adapt:
            from .task_accounting import DistanceTaskAccountingAdapter
            scheduler=DistanceTaskAccountingAdapter(scheduler,self.prereg)
        recorder=FailureEvidenceRecorder(self.output/'failures',run_id='R7_FORCED_TEST',attempt_id='R7_FORCED_TEST',origin='FORCED_TEST_STATE')
        metrics=FormalHealthMetricsEnv(scheduler,failure_recorder=recorder,diagnostics_dir=self.output)
        self.addCleanup(metrics.close);return scheduler,metrics

    def test_first_real_short_transition_is_accounted_with_true_task_identity(self):
        scheduler,m=self.chain();obs,info=m.reset()
        self.assertNotEqual(info['scenario_identity_sha256'],info['base_scenario_identity_sha256'])
        obs,r,t,tr,info=m.step(np.zeros(4,dtype=np.float32))
        self.assertEqual(m.policy_steps,1);self.assertEqual(m.physics_ticks,20)
        self.assertEqual(scheduler.completed_policy_steps,1);self.assertFalse(t or tr)
        self.assertEqual(info['terminal_class'],None);self.assertAlmostEqual(float(obs['lidar_age'][0]),.04)
        self.assertAlmostEqual(float(scheduler.active.unwrapped.data.time),.04)
        self.assertEqual(list((self.output/'failures').glob('*.json')),[])

    def test_old_chain_reproduces_rejection_and_latches_without_changing_r6(self):
        s,m=self.chain(adapt=False);m.reset()
        with self.assertRaisesRegex(RuntimeError,'unregistered'):m.step(np.zeros(4,dtype=np.float32))
        self.assertEqual(m.policy_steps,0);self.assertAlmostEqual(float(s.active.unwrapped.data.time),.04)
        self.assertEqual(len(list((self.output/'failures').glob('*.json'))),1)
        with self.assertRaisesRegex(RuntimeError,'latched'):m.step(np.zeros(4,dtype=np.float32))

    def test_all_twelve_train_tasks_pass_full_chain_and_reset_state_is_clean(self):
        s,m=self.chain();self.assertEqual(len(s.by_identity),24)
        for b in self.prereg['scenario_manifest']['train']:
            obs,info=m.reset(options={'scenario_identity_sha256':b['identity_sha256']})
            p=s.active.unwrapped
            np.testing.assert_array_equal(p.data.qvel,np.zeros(6))
            np.testing.assert_array_equal(p.velocity_controller.integral_error,np.zeros(3))
            self.assertEqual(p.lidar_frame_count,1);self.assertEqual(float(obs['lidar_age'][0]),0)
            obs,r,t,tr,info=m.step(np.zeros(4,dtype=np.float32))
            self.assertEqual(info['base_scenario_identity_sha256'],b['identity_sha256'])
            self.assertNotEqual(info['scenario_identity_sha256'],b['identity_sha256'])
            self.assertEqual(s.active_environment_count,1)
        self.assertEqual(m.policy_steps,12);self.assertEqual(m.physics_ticks,240)
        self.assertEqual(sum(m.terminal_counts.values()),0)

    def test_adapter_does_not_change_trajectory_reward_obs_or_train_rng(self):
        results=[]
        for adapted in (False,True):
            s=make_short_scheduler(self.prereg,self.upstream,seed=0)
            if adapted:
                from .task_accounting import DistanceTaskAccountingAdapter
                s=DistanceTaskAccountingAdapter(s,self.prereg)
            try:
                s.reset();transitions=[]
                for _ in range(3):
                    transitions.append(s.step(np.array([.1,-.2,.1,.05],dtype=np.float32)))
                results.append((transitions,s.scheduler_state()['rng_state'],s.active.unwrapped.data.qpos.copy()))
            finally:s.close()
        self.assertEqual(results[0][1],results[1][1]);np.testing.assert_array_equal(results[0][2],results[1][2])
        for left,right in zip(results[0][0],results[1][0]):
            self.assertEqual(left[1:4],right[1:4])
            self.assertEqual(left[4]['reward_breakdown'],right[4]['reward_breakdown'])
            for k in left[0]:np.testing.assert_array_equal(left[0][k],right[0][k])

    def test_forced_threshold_changes_only_on_reset_and_checkpoint_retains_both_id_scopes(self):
        from lidar_integration.ppo_preflight import make_ppo
        from lidar_integration.ppo_formal_checkpoint import CheckpointManager
        from lidar_integration.ppo_training_runner import _weights_sha
        from .worker import save_checkpoint_probe
        s,m=self.chain();obs,info=m.reset();short=info['scenario_identity_sha256'];goal=s.active.unwrapped.target_position.copy()
        s.env.completed_policy_steps=32767  # FORCED_TEST_COUNTER, not learned progress.
        m.step(np.zeros(4,dtype=np.float32));np.testing.assert_array_equal(s.active.unwrapped.target_position,goal)
        rng=copy.deepcopy(s.scheduler_state()['rng_state']);base=self.prereg['scenario_manifest']['train'][0]
        obs,info=m.reset(options={'scenario_identity_sha256':base['identity_sha256']})
        self.assertEqual(info['scenario_identity_sha256'],base['identity_sha256']);self.assertEqual(info['distance_stage'],'ORIGINAL')
        self.assertEqual(s.scheduler_state()['rng_state'],rng);self.assertEqual(s.phase_switch_step,32768)
        m.step(np.zeros(4,dtype=np.float32));self.assertEqual(m.policy_steps,2)
        model=make_ppo(m,seed=0);model._last_obs=model.get_env().reset()
        manager=CheckpointManager(self.output/'checkpoint',self.prereg,execution_protocol_sha256='b'*64)
        record,probe=save_checkpoint_probe(model,s,manager,run_id='R7_UNTRAINED_ONLY',seed=0)
        state=record['scheduler_state'];self.assertEqual(len(state['accounting_train_task_identities']),24)
        self.assertEqual(len(state['visit_counts']),12);self.assertIn(short,state['accounting_train_task_identities'])
        restored=manager.load_verified(None,run_id='R7_UNTRAINED_ONLY')
        self.assertEqual(_weights_sha(restored.policy),_weights_sha(model.policy))
        self.assertFalse(restored.policy.optimizer.state)

    def test_forged_task_metadata_is_rejected_and_latched(self):
        for changes in ({'scenario_identity_sha256':'0'*64},{'scenario_split':'validation'},
                        {'base_scenario_identity_sha256':'0'*64},{'distance_stage':'ORIGINAL'},
                        {'task_target_xyz':[9.,9.,9.]},
                        {'scenario_identity_sha256':self.prereg['scenario_manifest']['validation'][0]['identity_sha256']}):
            with self.subTest(changes=changes):
                s,m=self.chain();m.reset();original=s.env._info
                def corrupt(info):return {**original(info),**changes}
                with patch.object(s.env,'_info',corrupt):
                    with self.assertRaises(RuntimeError):m.step(np.zeros(4,dtype=np.float32))
                with self.assertRaises(RuntimeError):m.step(np.zeros(4,dtype=np.float32))
                with self.assertRaises(RuntimeError):m.reset()
                self.assertEqual(m.policy_steps,0);m.close()

    def test_validation_test_unknown_and_short_ids_cannot_force_reset(self):
        for bad in [self.prereg['scenario_manifest'][k][0]['identity_sha256'] for k in ('validation','test')]+['0'*64]:
            s,m=self.chain()
            with self.assertRaises(ValueError):m.reset(options={'scenario_identity_sha256':bad})
            m.close()
        s,m=self.chain();_,info=m.reset()
        with self.assertRaises(ValueError):m.reset(options={'scenario_identity_sha256':info['scenario_identity_sha256']})

    def test_actual_target_tamper_is_refused_before_physics_and_membership_is_readonly(self):
        s,m=self.chain();m.reset();p=s.active.unwrapped
        with self.assertRaises(TypeError):s.by_identity['0'*64]='0'*64
        p.target_position[0]+=.01
        with self.assertRaisesRegex(RuntimeError,'identity mismatch'):m.step(np.zeros(4,dtype=np.float32))
        self.assertEqual(float(p.data.time),0);self.assertEqual(m.policy_steps,0)
        with self.assertRaises(RuntimeError):m.reset()

    def test_corrupt_return_is_not_invented_as_complete_transition_evidence(self):
        from research_ops.storage import read_json
        s,m=self.chain();_,reset_info=m.reset();original=s.env._info
        def corrupt_only_actual_return(info):
            result=original(info)
            if 'physics_steps' in info:result['scenario_identity_sha256']='0'*64
            return result
        with patch.object(s.env,'_info',corrupt_only_actual_return):
            with self.assertRaisesRegex(RuntimeError,'identity mismatch'):m.step(np.zeros(4,dtype=np.float32))
        self.assertAlmostEqual(float(s.active.unwrapped.data.time),.04);self.assertEqual(m.policy_steps,0)
        paths=list((self.output/'failures').glob('*.json'));self.assertEqual(len(paths),1)
        row=read_json(paths[0]);self.assertEqual(row['context_kind'],'LAST_KNOWN_CONTEXT')
        self.assertEqual(row['scenario_identity_sha256'],reset_info['scenario_identity_sha256'])
        self.assertEqual(s.base_current_identity,reset_info['base_scenario_identity_sha256'])
        self.assertNotEqual(row['scenario_identity_sha256'],s.base_current_identity)
        self.assertEqual(row['returned_scenario_identity_sha256'],'NOT_AVAILABLE')
        self.assertEqual(row['returned_reward'],'NOT_AVAILABLE');self.assertEqual(row['last_known_reward'],'NOT_AVAILABLE')
        self.assertAlmostEqual(row['simulation_time'],.04)

    def test_task_identity_is_unavailable_before_reset_and_after_close(self):
        s,m=self.chain();self.assertIsNone(s.current_identity)
        _,info=m.reset();self.assertEqual(s.current_identity,info['scenario_identity_sha256'])
        state=s.scheduler_state()
        self.assertEqual(state['current_identity'],info['scenario_identity_sha256'])
        self.assertEqual(state['base_current_identity'],info['base_scenario_identity_sha256'])
        m.close();self.assertIsNone(s.current_identity)

    def test_five_terminal_classes_and_four_physical_causes_preserve_raw_reward(self):
        from lidar_integration.test_termination_repair import _force_state
        expected={'success':'success','collision':'collision','time_limit':'time_limit','sensor_failure':'sensor_failure',
                  'z_below_zero':'physical_failure','outside_flight_area':'physical_failure',
                  'excessive_tilt':'physical_failure','nonfinite_state':'physical_failure'}
        for raw,category in expected.items():
            results=[]
            for measured in (False,True):
                with self.subTest(raw=raw,measured=measured):
                    if measured:s,e=self.chain()
                    else:s=make_short_scheduler(self.prereg,self.upstream,seed=0);e=s
                    try:
                        e.reset();_force_state(s.active.unwrapped,raw)  # FORCED_TEST_STATE only.
                        for _ in range(5 if raw=='success' else 1):result=e.step(np.zeros(4,dtype=np.float32))
                        results.append(result)
                        if measured:
                            self.assertEqual(result[4]['terminal_class'],category)
                            self.assertEqual(result[4]['raw_termination_reason'],raw)
                            self.assertEqual(e.terminal_counts[category],1)
                            self.assertEqual(sum(e.terminal_counts.values()),1)
                            self.assertEqual(e.episode_diagnostics[-1]['raw_termination_reason'],raw)
                    finally:e.close()
            self.assertEqual(results[0][1:4],results[1][1:4]);self.assertEqual(results[0][4]['reward_breakdown'],results[1][4]['reward_breakdown'])


if __name__=='__main__':unittest.main()
