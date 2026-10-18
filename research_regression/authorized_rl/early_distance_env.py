"""Explicit new reset-only distance distribution; unchanged physical/held-task APIs."""
import copy
from types import MappingProxyType
import gymnasium as gym
import numpy as np
from lidar_integration.ppo_formal_scheduler import MultiScenarioTrainingEnv,require_memory
from research_ops.storage import digest
from .short_goal import short_manifest
from .task_accounting import DistanceTaskAccountingAdapter
from .held_goal_env import HeldGoalScheduler
from .held_goal_profile import held_goal_manifest
from .early_distance_mix import configuration,stage_at_reset,distance_seed,base_weights

def _make_distance_scheduler(prereg,upstream,*,seed):
 manifest=short_manifest(prereg)
 class EarlyDistanceScheduler(MultiScenarioTrainingEnv):
  def __init__(self):
   self.completed_policy_steps=0;self.phase_switch_step=None;self.active_task=None
   self.selected_tasks=[];self.task_visit_counts={};self.reset_samples=[]
   self.short_by_base={r['base_identity_sha256']:r for r in manifest['train']}
   self._distance_rng=np.random.default_rng(distance_seed(seed))
   super().__init__(prereg,upstream,seed=seed)
  def reset(self,*,seed=None,options=None):
   require_memory()
   if seed is not None:
    distance_seed(seed) # Validate before touching either stream.
    self._rng=np.random.default_rng(seed);self._distance_rng=np.random.default_rng(distance_seed(seed))
   step=self.completed_policy_steps;layouts=[r['layout'] for r in self.rows]
   weights=base_weights(step,layouts)
   if options is None:
    index=int(self._rng.integers(12)) if step<32768 else int(self._rng.choice(12,p=weights))
    base=self.rows[index];mode='UNIFORM_EARLY_DISTANCE_MIX' if step<32768 else 'WEIGHTED_LATE_ORIGINAL'
   else:
    if (not isinstance(options,dict) or set(options)!={'scenario_identity_sha256'} or options['scenario_identity_sha256'] not in self.by_identity):
     raise ValueError('only frozen base Train coverage identity allowed')
    base=self.by_identity[options['scenario_identity_sha256']];mode='EXPLICIT_TRAIN_COVERAGE'
   draw=float(self._distance_rng.random()) if step<32768 else None
   stage=stage_at_reset(step,draw);row=self.short_by_base[base['identity_sha256']] if stage=='SHORT' else base
   if self.active is None or self.current_identity!=base['identity_sha256']:self._open(base)
   obs,info=self.active.reset(seed=base['scene_seed'],options={
    'start_seed':base['start_seed'],'target_seed':base['target_seed'],'start_position':base['start_xyz'],
    'target_position':row['target_xyz'],'start_yaw_rad':base['start_yaw_rad']})
   if info['obstacle_configuration_hash']!=base['obstacle_configuration_hash']:raise ValueError('earlymix geometry changed')
   self.active_task=copy.deepcopy(row);self.active_stage=stage
   self.visit_counts[base['identity_sha256']]+=1;self.selected_identities.append(base['identity_sha256'])
   self.selected_tasks.append(row['identity_sha256']);self.task_visit_counts[row['identity_sha256']]=self.task_visit_counts.get(row['identity_sha256'],0)+1
   if step>=32768 and self.phase_switch_step is None:self.phase_switch_step=step
   self.reset_samples.append({'completed_policy_steps':step,'mode':mode,'distance_stage':stage,'distance_uniform_draw':draw,
    'base_identity_sha256':base['identity_sha256'],'task_identity_sha256':row['identity_sha256']})
   return obs,self._info(info)
  def _info(self,info):
   if self.active_task is None:raise RuntimeError('earlymix task not reset')
   return {**info,'scenario_identity_sha256':self.active_task['identity_sha256'],'base_scenario_identity_sha256':self.current_identity,
    'scenario_split':'train','distance_stage':self.active_stage,'task_target_xyz':list(self.active_task['target_xyz'])}
  def step(self,action):
   obs,r,t,tr,info=super().step(action);self.completed_policy_steps+=1
   return obs,r,t,tr,self._info(info)
  def scheduler_state(self):
   return {**super().scheduler_state(),'completed_policy_steps':self.completed_policy_steps,'phase_switch_step':self.phase_switch_step,
    'short_manifest_sha256':digest(manifest),'selected_tasks':list(self.selected_tasks),'task_visit_counts':dict(self.task_visit_counts),
    'active_task':copy.deepcopy(self.active_task),'early_distance_mix_configuration':configuration(),
    'early_distance_rng_state':copy.deepcopy(self._distance_rng.bit_generator.state),'early_distance_reset_samples':copy.deepcopy(self.reset_samples),
    'switch_semantics':'first reset at/after32768,not first earlyORIGINAL;never changes active episode target'}
 return EarlyDistanceScheduler()

class EarlyMixedHeldGoalScheduler(HeldGoalScheduler):
 def __init__(self,prereg,upstream,*,seed):
  # Explicit new constructor composition. Reuse frozen HeldGoalScheduler methods,
  # not its old scheduler factory; no global stage function or method monkeypatch.
  self._prereg=copy.deepcopy(prereg);self.manifest=held_goal_manifest(prereg);self._manifest_sha=digest(self.manifest)
  rows=[v for stage in ('SHORT','ORIGINAL') for v in self.manifest['train'][stage]]
  self._membership=MappingProxyType({v['identity_sha256']:v['identity_sha256'] for v in rows})
  self._by_distance={(stage,v['scenario']['identity_sha256']):v for stage in ('SHORT','ORIGINAL') for v in self.manifest['train'][stage]}
  self._task=None;self._failed=False;self._closed=False;self.selected_tasks=[];self.task_visit_counts={};self._samples=[]
  inner=_make_distance_scheduler(prereg,upstream,seed=seed)
  try:accounted=DistanceTaskAccountingAdapter(inner,prereg)
  except BaseException:inner.close();raise
  gym.Wrapper.__init__(self,accounted)

def make_early_mix_scheduler(prereg,upstream,*,seed):return EarlyMixedHeldGoalScheduler(prereg,upstream,seed=seed)
