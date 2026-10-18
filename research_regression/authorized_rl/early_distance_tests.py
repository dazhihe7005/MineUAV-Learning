"""Bounded new reset/worker contracts; optimizer calls forbidden,not training."""
import importlib,importlib.util,copy,math,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from research_ops.storage import read_json,digest,process_identity
from lidar_integration.ppo_training_runner import _weights_sha
ROOT=Path(__file__).resolve().parents[2];UPSTREAM=ROOT/'.research_local/dependencies/MuJoCo-LiDAR'
def forbidden(*a,**k):raise AssertionError('R33_NO_OPTIMIZER_IN_PREFLIGHT')
class EarlyDistanceTests(unittest.TestCase):
 def module(self,name='early_distance_env'):
  path='research_regression.authorized_rl.'+name
  self.assertIsNotNone(importlib.util.find_spec(path),'missing explicit early-distance runtime')
  return importlib.import_module(path)
 def prereg(self):return read_json(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
 def test_24_tasks_have_actual_pose_goal_physics_and_clean_reset(self):
  m=self.module();p=self.prereg();env=m.make_early_mix_scheduler(p,UPSTREAM,seed=0);seen=set()
  try:
   for base in p['scenario_manifest']['train']:
    stages=set()
    for _ in range(16): # Fixed bounded deterministic coverage,not scientific episodes.
     obs,info=env.reset(options={'scenario_identity_sha256':base['identity_sha256']});plant=env.active.unwrapped
     self.assertEqual(info['base_scenario_identity_sha256'],base['identity_sha256']);self.assertEqual(info['goal_yaw_mode'],'HOLD_INITIAL_ENU_YAW')
     np.testing.assert_array_equal(plant.data.qpos[:3],base['start_xyz']);np.testing.assert_array_equal(plant.data.qvel,np.zeros(6))
     np.testing.assert_array_equal(plant.velocity_controller.integral_error,np.zeros(3));self.assertAlmostEqual(plant.data.time,0.)
     self.assertAlmostEqual(float(obs['state'][6]),0.);self.assertEqual(plant.lidar_frame_count,1)
     self.assertEqual(info['obstacle_configuration_hash'],base['obstacle_configuration_hash']);self.assertIn(info['scenario_identity_sha256'],env.by_identity)
     self.assertEqual(env.active_environment_count,1);self.assertEqual(obs['lidar'].shape,(3,12,24))
     stages.add(info['distance_stage']);seen.add(info['scenario_identity_sha256'])
     _,_,t,tr,post=env.step(np.zeros(4,dtype=np.float32));self.assertFalse(t or tr);self.assertEqual(post['physics_steps'],20)
     self.assertAlmostEqual(plant.data.time,.04);self.assertLessEqual(post['lidar_source_time_s'],plant.data.time)
     if len(stages)==2:break
    self.assertEqual(stages,{'SHORT','ORIGINAL'})
   self.assertEqual(len(seen),24);self.assertEqual(env.phase_switch_step,None,'earlyORIGINAL is not latephase switch')
  finally:env.close()
 def test_base_draws_reproducible_and_late_phase_does_not_consume_distance_rng(self):
  m=self.module();p=self.prereg();env=m.make_early_mix_scheduler(p,UPSTREAM,seed=0)
  base=np.random.default_rng(0);distance=np.random.default_rng(330000)
  try:
   for _ in range(6):
    expected=p['scenario_manifest']['train'][int(base.integers(12))]['identity_sha256'];stage='ORIGINAL' if distance.random()<.5 else 'SHORT'
    _,info=env.reset();self.assertEqual(info['base_scenario_identity_sha256'],expected);self.assertEqual(info['distance_stage'],stage)
   self.assertEqual(env.scheduler_state()['early_distance_rng_state'],distance.bit_generator.state)
   # FORCED_TEST_SCHEDULER_COUNTER,not actual32768 physical/training steps.
   env.env.env.completed_policy_steps=32768;before=copy.deepcopy(env.scheduler_state()['early_distance_rng_state'])
   _,info=env.reset();self.assertEqual(info['distance_stage'],'ORIGINAL');self.assertEqual(env.phase_switch_step,32768)
   self.assertEqual(env.scheduler_state()['early_distance_rng_state'],before)
   expected=p['scenario_manifest']['train'][int(base.choice(12,p=[1/6 if v['layout']=='empty' else 1/18 for v in p['scenario_manifest']['train']]))]['identity_sha256']
   self.assertEqual(info['base_scenario_identity_sha256'],expected)
  finally:env.close()
 def test_fresh_prior_and_separate_validation_probe_do_not_change_train_state(self):
  m=self.module();p=self.prereg();env=m.make_early_mix_scheduler(p,UPSTREAM,seed=0)
  from .state_centered_policy import make_centered_ppo
  from .centered_noise_training import assign_noise_prior
  from .held_goal_validation import evaluate_held_validation
  try:
   with patch.object(PPO,'learn',forbidden),patch.object(PPO,'train',forbidden),patch.object(torch.optim.Adam,'step',forbidden):
    model=make_centered_ppo(env,seed=0);assign_noise_prior(model)
    self.assertEqual(_weights_sha(model.policy),'650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4')
    before=digest(env.scheduler_state());pair=evaluate_held_validation(model,p,UPSTREAM,max_policy_steps=1)
    self.assertEqual(digest(env.scheduler_state()),before)
    for stage in ('SHORT','ORIGINAL'):
     self.assertEqual(pair[stage]['overall']['probe_cutoff'],6);self.assertIsNone(pair[stage]['overall']['navigation_success_rate'])
    self.assertEqual(model.num_timesteps,0);self.assertFalse(model.policy.optimizer.state)
    self.assertTrue(all(v.grad is None for v in model.policy.parameters()))
  finally:env.close()
 def test_actual_worker_dispatch_stops_before_sampling_or_optimizer(self):
  self.module();profile=self.module('early_distance_profile')
  from . import worker
  from .budget import BudgetStore,grant
  class SamplingForbidden(Exception):pass
  with tempfile.TemporaryDirectory() as directory:
   home=Path(directory)/'grant';store=BudgetStore(home);store.initialize(expected_grant_sha256=digest(grant()))
   p=profile.early_mix_experiment();p.update(registration_sha256='a'*64,source_identity_sha256='b'*64);store.reserve(p)
   output=Path(directory)/'out';output.mkdir();identity=p['experiment_id']
   permit={'experiment_id':identity,'parent_pid':os.getppid(),'parent_identity':process_identity(os.getppid()),'protocol_sha256':digest(p)};observed=[]
   def boundary(model,steps,**kwargs):
    self.assertEqual((model.num_timesteps,model._n_updates,steps),(0,0,32768));self.assertFalse(model.policy.optimizer.state)
    self.assertEqual(_weights_sha(model.policy),'650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4')
    state=read_json(output/f'{identity}_STEP0.json')['scheduler_state']
    self.assertEqual(state.get('early_distance_mix_configuration',{}).get('version'),'EARLY_DISTANCE_RESET_MIX_R33_V1','worker dispatched an old curriculum')
    self.assertEqual(len(state['accounting_train_task_identities']),24);observed.append('STOP_BEFORE_SAMPLING');raise SamplingForbidden()
   with patch.object(worker,'HOME',home),patch.object(PPO,'learn',boundary),patch.object(PPO,'train',forbidden),patch.object(torch.optim.Adam,'step',forbidden):
    try:worker.train_worker(output,permit,p,{},identity)
    except Exception as e:self.assertIsInstance(e,SamplingForbidden,'wrong schema33 worker dispatch: '+repr(e))
   self.assertEqual(observed,['STOP_BEFORE_SAMPLING']);self.assertFalse((output/'ppo_updates.jsonl').exists());self.assertFalse((output/'summary.json').exists())
if __name__=='__main__':unittest.main()
