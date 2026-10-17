"""R28 task-yaw wrapper around unchanged R17 reset sampling and physics."""
import copy,math
from types import MappingProxyType
import gymnasium as gym
import numpy as np
from research_ops.storage import digest
from .empty_rehearsal import make_rehearsal_scheduler
from .held_goal_profile import held_goal_manifest,verify_held_goal_manifest

class HeldGoalScheduler(gym.Wrapper):
    def __init__(self,prereg,upstream,*,seed):
        self._prereg=copy.deepcopy(prereg);self.manifest=held_goal_manifest(prereg);self._manifest_sha=digest(self.manifest)
        rows=[v for stage in ('SHORT','ORIGINAL') for v in self.manifest['train'][stage]]
        self._membership=MappingProxyType({v['identity_sha256']:v['identity_sha256'] for v in rows})
        self._by_distance={(stage,v['scenario']['identity_sha256']):v for stage in ('SHORT','ORIGINAL') for v in self.manifest['train'][stage]}
        self._task=None;self._failed=False;self._closed=False;self.selected_tasks=[];self.task_visit_counts={};self._samples=[]
        super().__init__(make_rehearsal_scheduler(prereg,upstream,seed=seed))
    @property
    def by_identity(self):return self._membership
    @property
    def active(self):return self.env.active
    @property
    def active_environment_count(self):return self.env.active_environment_count
    @property
    def completed_policy_steps(self):return self.env.completed_policy_steps
    @property
    def phase_switch_step(self):return self.env.phase_switch_step
    @property
    def selected_identities(self):return list(self.env.selected_identities)
    @property
    def base_current_identity(self):return self.env.base_current_identity
    @property
    def current_identity(self):return self._task['identity_sha256'] if self.active is not None and self._task is not None else None
    def _require_live(self):
        if self._failed or self._closed:raise RuntimeError('R28 scheduler closed or failure latched')
    def _check_task(self):
        task=self._task;plant=self.active.unwrapped if self.active is not None else None
        if task is None or plant is None:raise RuntimeError('R28 no current physical task')
        row=task['scenario'];base=row.get('base_identity_sha256',row['identity_sha256'])
        if (task['identity_sha256'] not in self.by_identity or digest({k:v for k,v in task.items() if k!='identity_sha256'})!=task['identity_sha256']
                or self.env.current_identity!=row['identity_sha256'] or self.base_current_identity!=base
                or plant.scene.geometry_hash!=row['obstacle_configuration_hash'] or not np.array_equal(plant.target_position,np.asarray(row['target_xyz']))
                or abs(plant.target_yaw-task['goal_yaw_rad'])>1e-12
                or not np.isfinite(plant.data.qpos).all() or not np.isfinite(plant.data.qvel).all() or not np.isfinite(plant.data.ctrl).all()):
            raise RuntimeError('R28 actual task/source/heading/finite-state identity mismatch')
    def _info(self,info):
        t=self._task
        return {**info,'scenario_identity_sha256':t['identity_sha256'],'distance_task_identity_sha256':t['scenario']['identity_sha256'],
            'task_goal_yaw_rad':t['goal_yaw_rad'],'target_yaw_rad':t['goal_yaw_rad'],'goal_yaw_mode':'HOLD_INITIAL_ENU_YAW',
            'held_goal_manifest_sha256':self._manifest_sha}
    def _check_inner(self,info):
        row=self._task['scenario']
        if (info['scenario_identity_sha256']!=row['identity_sha256'] or info['base_scenario_identity_sha256']!=self.base_current_identity
                or info['task_target_xyz']!=row['target_xyz'] or (info['distance_stage'],row['identity_sha256']) not in self._by_distance):
            raise RuntimeError('R28 inner physical transition task metadata differs')
    def reset(self,*,seed=None,options=None):
        self._require_live()
        try:
            verify_held_goal_manifest(self.manifest,self._prereg);obs,info=self.env.reset(seed=seed,options=options)
            task=self._by_distance[(info['distance_stage'],info['scenario_identity_sha256'])];plant=self.active.unwrapped
            self._task=copy.deepcopy(task);plant.target_yaw=task['goal_yaw_rad']
            obs=dict(obs);obs['state']=plant._get_obs();self._check_task();self._check_inner(info)
            if abs(float(obs['state'][6]))>1e-6:raise RuntimeError('R28 current reset yaw does not match initial task heading')
            self.selected_tasks.append(task['identity_sha256']);self.task_visit_counts[task['identity_sha256']]=self.task_visit_counts.get(task['identity_sha256'],0)+1
            self._samples.append({'completed_policy_steps':self.completed_policy_steps,'distance_stage':info['distance_stage'],
                'base_identity_sha256':info['base_scenario_identity_sha256'],'distance_task_identity_sha256':info['scenario_identity_sha256'],
                'task_identity_sha256':task['identity_sha256'],'task_goal_yaw_rad':task['goal_yaw_rad'],
                'measured_reset_quaternion_wxyz':plant.data.qpos[3:7].tolist(),'goal_yaw_mode':'HOLD_INITIAL_ENU_YAW'})
            return obs,self._info(info)
        except BaseException:self._failed=True;raise
    def step(self,action):
        self._require_live()
        try:
            self._check_task();obs,r,t,tr,info=self.env.step(action);self._check_task();self._check_inner(info)
            return obs,r,t,tr,self._info(info)
        except BaseException:self._failed=True;raise
    def scheduler_state(self):
        verify_held_goal_manifest(self.manifest,self._prereg);inner=self.env.scheduler_state()
        previous_accounting={k:v for k,v in inner.items() if k.startswith('accounting_')}
        return {**{k:v for k,v in inner.items() if not k.startswith('accounting_')},
            'base_distance_accounting':previous_accounting,
            'base_distance_selected_tasks':inner['selected_tasks'],'base_distance_task_visit_counts':inner['task_visit_counts'],
            'current_identity':self.current_identity,'active_task':copy.deepcopy(self._task),'selected_tasks':list(self.selected_tasks),
            'task_visit_counts':dict(self.task_visit_counts),'held_goal_reset_samples':copy.deepcopy(self._samples),
            'held_goal_manifest_sha256':self._manifest_sha,'goal_yaw_mode':'HOLD_INITIAL_ENU_YAW',
            'accounting_train_task_identities':sorted(self.by_identity),
            'accounting_train_task_identities_sha256':digest(sorted(self.by_identity)),
            'accounting_contract':'HELD_INITIAL_GOAL_YAW_TASK_MEMBERSHIP_R28_V1','accounting_failure_latched':self._failed,
            'identity_scope':'current_identity/selected_tasks/visits are24heldgoal Train tasks;base_distance_* preserve old distance accounting separately',
            'base_sampling_scope':'original R17 RNG/distance/rehearsal sampling unchanged;inner counts retain base distance identities',
            'resume':'RESTART_REQUIRED'}
    def close(self):
        self.env.close();self._task=None;self._closed=True

def make_held_goal_scheduler(prereg,upstream,*,seed):return HeldGoalScheduler(prereg,upstream,seed=seed)
