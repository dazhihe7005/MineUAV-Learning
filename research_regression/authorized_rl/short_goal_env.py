"""Reset-only distance curriculum; reuses the unchanged 7kg plant and control."""
import copy
import numpy as np
from lidar_integration.ppo_formal_scheduler import MultiScenarioTrainingEnv,require_memory
from .short_goal import short_manifest,stage_for_steps
from research_ops.storage import digest


def make_short_scheduler(prereg,upstream,*,seed):
    manifest=short_manifest(prereg)

    class DistanceScheduler(MultiScenarioTrainingEnv):
        def __init__(self):
            self.completed_policy_steps=0;self.phase_switch_step=None
            self.active_task=None;self.selected_tasks=[];self.task_visit_counts={}
            self.short_by_base={r['base_identity_sha256']:r for r in manifest['train']}
            super().__init__(prereg,upstream,seed=seed)

        def reset(self,*,seed=None,options=None):
            require_memory()
            if seed is not None:
                if type(seed) is not int or seed<0:raise ValueError('invalid scheduler seed')
                self._rng=np.random.default_rng(seed)
            if options is None:base=self.rows[int(self._rng.integers(len(self.rows)))]
            else:
                if (not isinstance(options,dict) or set(options)!={'scenario_identity_sha256'}
                    or options['scenario_identity_sha256'] not in self.by_identity):
                    raise ValueError('only frozen base Train coverage identity allowed')
                base=self.by_identity[options['scenario_identity_sha256']]
            stage=stage_for_steps(self.completed_policy_steps)
            row=self.short_by_base[base['identity_sha256']] if stage=='SHORT' else base
            if self.active is None or self.current_identity!=base['identity_sha256']:self._open(base)
            obs,info=self.active.reset(seed=base['scene_seed'],options={
                'start_seed':base['start_seed'],'target_seed':base['target_seed'],
                'start_position':base['start_xyz'],'target_position':row['target_xyz'],
                'start_yaw_rad':base['start_yaw_rad']})
            if info['obstacle_configuration_hash']!=base['obstacle_configuration_hash']:raise ValueError('distance task geometry changed')
            self.active_task=copy.deepcopy(row);self.active_stage=stage
            self.visit_counts[base['identity_sha256']]+=1;self.selected_identities.append(base['identity_sha256'])
            self.selected_tasks.append(row['identity_sha256']);self.task_visit_counts[row['identity_sha256']]=self.task_visit_counts.get(row['identity_sha256'],0)+1
            if stage=='ORIGINAL' and self.phase_switch_step is None:self.phase_switch_step=self.completed_policy_steps
            return obs,self._info(info)

        def _info(self,info):
            if self.active_task is None:raise RuntimeError('distance task has not reset')
            return {**info,'scenario_identity_sha256':self.active_task['identity_sha256'],
                'base_scenario_identity_sha256':self.current_identity,'scenario_split':'train',
                'distance_stage':self.active_stage,'task_target_xyz':list(self.active_task['target_xyz'])}

        def step(self,action):
            # Parent step runs actual dynamics. Never replace a target mid-episode.
            obs,r,t,tr,info=super().step(action)
            self.completed_policy_steps+=1
            return obs,r,t,tr,self._info(info)

        def scheduler_state(self):
            return {**super().scheduler_state(),'completed_policy_steps':self.completed_policy_steps,
                'phase_switch_step':self.phase_switch_step,'short_manifest_sha256':digest(manifest),
                'selected_tasks':list(self.selected_tasks),'task_visit_counts':dict(self.task_visit_counts),
                'active_task':copy.deepcopy(self.active_task),'switch_semantics':'next reset at/after32768; no mid-episode goal change'}

    return DistanceScheduler()
