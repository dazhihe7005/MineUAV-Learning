"""No-optimizer controlled interface tests, not scientific training/evaluation."""
import importlib,importlib.util,math,copy,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from research_ops.storage import read_json,digest
from lidar_integration.ppo_training_runner import _weights_sha

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'.research_local/dependencies/MuJoCo-LiDAR'

class HeldGoalTests(unittest.TestCase):
    def module(self):
        name='research_regression.authorized_rl.held_goal_env'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing explicit held-goal scheduler')
        return importlib.import_module(name)
    def prereg(self):return read_json(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
    def test_checkpoint_membership_hash_and_contract_describe_actual_held_tasks(self):
        from research_regression.authorized_rl.held_goal_profile import held_goal_manifest
        p=self.prereg();scheduler=self.module().make_held_goal_scheduler(p,UPSTREAM,seed=0)
        try:
            state=scheduler.scheduler_state();manifest=held_goal_manifest(p)
            expected=sorted(v['identity_sha256'] for stage in ('SHORT','ORIGINAL') for v in manifest['train'][stage])
            self.assertEqual(state['accounting_train_task_identities'],expected)
            self.assertEqual(state['accounting_train_task_identities_sha256'],digest(expected))
            self.assertEqual(state['accounting_contract'],'HELD_INITIAL_GOAL_YAW_TASK_MEMBERSHIP_R28_V1')
            previous=state['base_distance_accounting']
            self.assertEqual(previous['accounting_train_task_identities_sha256'],digest(previous['accounting_train_task_identities']))
            self.assertNotEqual(previous['accounting_train_task_identities'],expected)
        finally:scheduler.close()
    def test_deterministic_24task_coverage_physics_reward_and_reset_isolation(self):
        from research_regression.authorized_rl.held_goal_profile import held_goal_manifest
        from lidar_integration.navigation_env import navigation_reward
        p=self.prereg();manifest=held_goal_manifest(p);scheduler=self.module().make_held_goal_scheduler(p,UPSTREAM,seed=0)
        visited=[]
        try:
            for stage in ('SHORT','ORIGINAL'):
                if stage=='ORIGINAL':
                    # FORCED_TEST_SCHEDULER_COUNTER, not actual32768 training steps.
                    scheduler.env.env.completed_policy_steps=32768
                for base,task in zip(p['scenario_manifest']['train'],manifest['train'][stage]):
                    obs,info=scheduler.reset(options={'scenario_identity_sha256':base['identity_sha256']});plant=scheduler.active.unwrapped
                    self.assertEqual(info['scenario_identity_sha256'],task['identity_sha256']);visited.append(task['identity_sha256'])
                    self.assertEqual(info['base_scenario_identity_sha256'],base['identity_sha256']);self.assertEqual(info['distance_stage'],stage)
                    self.assertIn(info['scenario_identity_sha256'],scheduler.by_identity);self.assertEqual(info['goal_yaw_mode'],'HOLD_INITIAL_ENU_YAW')
                    self.assertAlmostEqual(plant.target_yaw,task['goal_yaw_rad']);self.assertAlmostEqual(float(obs['state'][6]),0.)
                    np.testing.assert_array_equal(plant.data.qpos[:3],np.asarray(base['start_xyz']));np.testing.assert_array_equal(plant.data.qvel,np.zeros(6))
                    np.testing.assert_array_equal(plant.velocity_controller.integral_error,np.zeros(3));self.assertEqual(plant.lidar_frame_count,1)
                    self.assertAlmostEqual(plant.data.time,0.);previous=float(np.linalg.norm(plant.target_position-plant.data.qpos[:3]));action=np.zeros(4,dtype=np.float32)
                    obs,reward,t,tr,actual=scheduler.step(action);self.assertFalse(t or tr);self.assertEqual(actual['physics_steps'],20)
                    expected,parts=navigation_reward(previous,actual['distance_m'],actual['speed_m_s'],action,'running',1)
                    self.assertAlmostEqual(reward,expected,places=6);self.assertAlmostEqual(plant.data.time,.04);self.assertEqual(plant.lidar_frame_count,1)
                    self.assertEqual(actual['scenario_identity_sha256'],task['identity_sha256']);self.assertAlmostEqual(actual['task_goal_yaw_rad'],task['goal_yaw_rad'])
            state=scheduler.scheduler_state();self.assertEqual(state['selected_tasks'],visited);self.assertEqual(len(set(visited)),24)
            self.assertEqual(sum(state['task_visit_counts'].values()),24)
        finally:scheduler.close()
    def test_seeded_base_sampling_unchanged_and_no_extra_rng_draw(self):
        p=self.prereg();scheduler=self.module().make_held_goal_scheduler(p,UPSTREAM,seed=0);rng=np.random.default_rng(0)
        expected=[p['scenario_manifest']['train'][int(rng.integers(12))]['identity_sha256'] for _ in range(6)]
        actual=[]
        try:
            for _ in range(6):actual.append(scheduler.reset()[1]['base_scenario_identity_sha256'])
            self.assertEqual(actual,expected)
        finally:scheduler.close()
    def test_changed_task_heading_is_rejected_before_next_physics_and_latches(self):
        p=self.prereg();scheduler=self.module().make_held_goal_scheduler(p,UPSTREAM,seed=0)
        try:
            scheduler.reset();plant=scheduler.active.unwrapped;before=float(plant.data.time)
            plant.target_yaw+=.1 # FORCED_TEST_TASK_METADATA, never a real training event.
            with self.assertRaises(RuntimeError):scheduler.step(np.zeros(4,dtype=np.float32))
            self.assertEqual(plant.data.time,before)
            with self.assertRaises(RuntimeError):scheduler.reset()
        finally:scheduler.close()
    def test_fresh_prior_is_exact_and_no_optimizer_runs(self):
        from research_regression.authorized_rl.state_centered_policy import make_centered_ppo
        from research_regression.authorized_rl.centered_noise_training import assign_noise_prior
        p=self.prereg();scheduler=self.module().make_held_goal_scheduler(p,UPSTREAM,seed=0)
        def forbidden(*args,**kwargs):raise AssertionError('FORBIDDEN_PPO_OPTIMIZER_TEST')
        try:
            with patch.object(PPO,'learn',forbidden),patch.object(PPO,'train',forbidden),patch.object(torch.optim.Adam,'step',forbidden):
                model=make_centered_ppo(scheduler,seed=0);assign_noise_prior(model)
                self.assertEqual(_weights_sha(model.policy),'650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4')
                self.assertFalse(scheduler.selected_identities);self.assertEqual(model.num_timesteps,0);self.assertFalse(model.policy.optimizer.state)
                self.assertEqual((model.gamma,model.gae_lambda,model.rollout_buffer.gamma,model.rollout_buffer.gae_lambda),(.99,.95,.99,.95))
                obs,_=scheduler.reset()
                with torch.inference_mode():action,_=model.predict(obs,deterministic=True)
                self.assertEqual(action.shape,(4,));self.assertTrue(np.isfinite(action).all());self.assertTrue(all(v.grad is None for v in model.policy.parameters()))
                self.assertFalse(model.policy.optimizer.state)
        finally:scheduler.close()
    def test_independent_validation_probe_has_correct_new_tasks_and_no_training_changes(self):
        name='research_regression.authorized_rl.held_goal_validation'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing separate held-goal Validation')
        from research_regression.authorized_rl.state_centered_policy import make_centered_ppo
        from research_regression.authorized_rl.centered_noise_training import assign_noise_prior
        from research_regression.authorized_rl.held_goal_profile import held_goal_manifest
        p=self.prereg();scheduler=self.module().make_held_goal_scheduler(p,UPSTREAM,seed=0)
        def forbidden(*args,**kwargs):raise AssertionError('FORBIDDEN_VALIDATION_OPTIMIZER')
        try:
            with patch.object(PPO,'learn',forbidden),patch.object(PPO,'train',forbidden),patch.object(torch.optim.Adam,'step',forbidden):
                model=make_centered_ppo(scheduler,seed=0);assign_noise_prior(model);before=digest(scheduler.scheduler_state());fingerprint=_weights_sha(model.policy)
                pair=importlib.import_module(name).evaluate_held_validation(model,p,UPSTREAM,max_policy_steps=1)
                for stage in ('ORIGINAL','SHORT'):
                    result=pair[stage];expected=held_goal_manifest(p)['validation'][stage]
                    self.assertEqual([v['scenario_identity_sha256'] for v in result['episodes']],[v['identity_sha256'] for v in expected])
                    self.assertTrue(result['probe_only']);self.assertEqual(result['overall']['probe_cutoff'],6)
                    self.assertEqual(result['goal_yaw_mode'],'HOLD_INITIAL_ENU_YAW');self.assertIsNone(result['overall']['navigation_success_rate'])
                    self.assertEqual(result['model_status'],'ANALYTIC_HAND_WRITTEN_PRIOR_NOT_LEARNED')
                self.assertEqual(digest(scheduler.scheduler_state()),before);self.assertEqual(_weights_sha(model.policy),fingerprint)
                self.assertEqual(model.num_timesteps,0);self.assertFalse(model.policy.optimizer.state)
        finally:scheduler.close()

    def test_actual_worker_dispatch_and_step0_identity_before_any_sampling(self):
        import os,tempfile
        from . import worker
        from .budget import BudgetStore,grant
        from .held_goal_profile import held_goal_experiment
        from .yaw_coverage import verify_yaw_temporal_configuration
        from research_ops.storage import process_identity
        class SamplingForbidden(Exception):pass
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory)/'grant';store=BudgetStore(home);store.initialize(expected_grant_sha256=digest(grant()))
            p=held_goal_experiment();p.update(registration_sha256='a'*64,source_identity_sha256='b'*64);store.reserve(p)
            output=Path(directory)/'out';output.mkdir();identity=p['experiment_id']
            permit={'experiment_id':identity,'parent_pid':os.getppid(),'parent_identity':process_identity(os.getppid()),'protocol_sha256':digest(p)}
            observed=[]
            def boundary(model,steps,**kwargs):
                verify_yaw_temporal_configuration(model)
                self.assertEqual((model.num_timesteps,model._n_updates,steps),(0,0,32768));self.assertFalse(model.policy.optimizer.state)
                self.assertEqual(_weights_sha(model.policy),'650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4')
                state=read_json(output/f'{identity}_STEP0.json')['scheduler_state']
                self.assertEqual(state.get('accounting_contract'),'HELD_INITIAL_GOAL_YAW_TASK_MEMBERSHIP_R28_V1','actual worker still dispatches old task')
                self.assertEqual(state['held_goal_manifest_sha256'],p['held_goal_manifest_sha256'])
                self.assertEqual(state['accounting_train_task_identities_sha256'],digest(state['accounting_train_task_identities']))
                observed.append('STOP_BEFORE_SAMPLING');raise SamplingForbidden()
            def forbidden(*args,**kwargs):raise AssertionError('NO_PPO_OR_ADAM_UPDATE_IN_DISPATCH_TEST')
            with patch.object(worker,'HOME',home),patch.object(PPO,'learn',boundary),patch.object(PPO,'train',forbidden),patch.object(torch.optim.Adam,'step',forbidden):
                try:worker.train_worker(output,permit,p,{},identity)
                except Exception as error:self.assertIsInstance(error,SamplingForbidden,'wrong schema28 worker dispatch: '+repr(error))
            self.assertEqual(observed,['STOP_BEFORE_SAMPLING'])
            self.assertFalse((output/'ppo_updates.jsonl').exists());self.assertFalse((output/'summary.json').exists())

if __name__=='__main__':unittest.main()
