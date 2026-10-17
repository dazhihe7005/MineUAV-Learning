"""R24 reset-only yaw intervention around the unchanged L6 physical scheduler."""
import copy
import math
import random
from types import MappingProxyType
import gymnasium as gym
import numpy as np
from lidar_integration.ppo_formal_scheduler import MultiScenarioTrainingEnv,require_memory
from research_ops.storage import digest
from .short_goal import stage_for_steps
from .empty_rehearsal import rehearsal_weights
from .yaw_coverage import yaw_manifest,verify_yaw_manifest,resolve_yaw_task

YAW_RNG_SEED_DOMAIN=0x52323400


class YawCoverageScheduler(gym.Wrapper):
    def __init__(self,prereg,upstream,*,seed):
        if type(seed) is not int or seed<0:raise ValueError('nonnegative independent R24 seed required')
        self._prereg=copy.deepcopy(prereg);self.manifest=yaw_manifest(prereg)
        self._manifest_sha=digest(self.manifest);self._weights=rehearsal_weights(prereg)
        self._tasks=MappingProxyType({v['identity_sha256']:v['identity_sha256']
            for stage in ('SHORT','ORIGINAL') for v in self.manifest[stage]})
        self._bases={v['identity_sha256']:copy.deepcopy(v) for v in prereg['scenario_manifest']['train']}
        self._yaw_rng=random.Random(YAW_RNG_SEED_DOMAIN+seed)
        self.completed_policy_steps=0;self.phase_switch_step=None
        self.active_task=None;self.active_stage=None;self._failed=False;self._closed=False
        self.selected_tasks=[];self.task_visit_counts={};self._samples=[]
        self._mode_counts={'UNIFORM_SHORT':0,'WEIGHTED_ORIGINAL':0,'EXPLICIT_TRAIN_COVERAGE':0}
        super().__init__(MultiScenarioTrainingEnv(prereg,upstream,seed=seed))

    @property
    def by_identity(self):return self._tasks

    @property
    def active(self):return self.env.active

    @property
    def active_environment_count(self):return self.env.active_environment_count

    @property
    def base_current_identity(self):return self.env.current_identity

    @property
    def selected_identities(self):return list(self.env.selected_identities)

    @property
    def current_identity(self):
        return self.active_task['identity_sha256'] if self.active is not None and self.active_task is not None else None

    def _require_live(self):
        if self._closed or self._failed:raise RuntimeError('R24 scheduler closed or failure latched')

    def _info(self,info):
        if self.active_task is None:raise RuntimeError('R24 has no actual reset task')
        row=self.active_task
        return {**info,'scenario_identity_sha256':row['identity_sha256'],'scenario_split':'train',
            'base_scenario_identity_sha256':row['base_identity_sha256'],'distance_stage':row['distance_stage'],
            'task_target_xyz':list(row['target_xyz']),'task_start_yaw_rad':row['start_yaw_rad'],
            'base_start_yaw_rad':row['base_start_yaw_rad'],'yaw_option':row['yaw_option'],
            'distance_task_identity_sha256':row['distance_task_identity_sha256'],'yaw_manifest_sha256':self._manifest_sha}

    def _check_info(self,info):
        row=self.active_task;plant=self.active.unwrapped if self.active is not None else None
        if (row is None or plant is None or row['identity_sha256'] not in self.by_identity
                or digest({k:v for k,v in row.items() if k!='identity_sha256'})!=row['identity_sha256']
                or self.base_current_identity!=row['base_identity_sha256'] or self.active_stage!=row['distance_stage']
                or not np.array_equal(plant.target_position,np.asarray(row['target_xyz']))
                or plant.scene.geometry_hash!=row['obstacle_configuration_hash']
                or any(info.get(k)!=v for k,v in self._info({}).items())
                or ('obstacle_configuration_hash' in info and info['obstacle_configuration_hash']!=plant.scene.geometry_hash)):
            raise RuntimeError('R24 actual task/target/geometry/yaw identity mismatch')

    def reset(self,*,seed=None,options=None):
        self._require_live()
        try:
            require_memory();verify_yaw_manifest(self.manifest,self._prereg)
            if seed is not None:
                if type(seed) is not int or seed<0:raise ValueError('invalid R24 reset seed')
                self.env._rng=np.random.default_rng(seed);self._yaw_rng=random.Random(YAW_RNG_SEED_DOMAIN+seed)
            stage=stage_for_steps(self.completed_policy_steps)
            if options is None:
                if stage=='SHORT':index=int(self.env._rng.integers(12));mode='UNIFORM_SHORT'
                else:index=int(self.env._rng.choice(12,p=self._weights));mode='WEIGHTED_ORIGINAL'
                base=self.env.rows[index];option=self._yaw_rng.randrange(5)
                row=resolve_yaw_task(self.manifest,stage,base['identity_sha256'],option)
            else:
                if (not isinstance(options,dict) or set(options)!={'scenario_identity_sha256'}
                        or options['scenario_identity_sha256'] not in self.by_identity):
                    raise ValueError('only explicit R24 Train task coverage identities allowed')
                identity=options['scenario_identity_sha256']
                matches=[v for v in self.manifest[stage] if v['identity_sha256']==identity]
                if len(matches)!=1:raise ValueError('coverage task must match current R24 distance stage')
                row=copy.deepcopy(matches[0]);base=self._bases[row['base_identity_sha256']]
                option=row['yaw_option'];mode='EXPLICIT_TRAIN_COVERAGE'
            if self.active is None or self.base_current_identity!=base['identity_sha256']:self.env._open(base)
            obs,info=self.active.reset(seed=base['scene_seed'],options={'start_seed':base['start_seed'],
                'target_seed':base['target_seed'],'start_position':row['start_xyz'],
                'target_position':row['target_xyz'],'start_yaw_rad':row['start_yaw_rad']})
            self.active_task=row;self.active_stage=stage;self._check_info(self._info(info))
            quat=self.active.unwrapped.data.qpos[3:7]
            yaw=math.atan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
            error=yaw-row['start_yaw_rad']
            if abs(math.atan2(math.sin(error),math.cos(error)))>1e-7:raise RuntimeError('actual reset quaternion does not match R24 yaw')
            self.env.visit_counts[base['identity_sha256']]+=1;self.env.selected_identities.append(base['identity_sha256'])
            self.selected_tasks.append(row['identity_sha256']);self.task_visit_counts[row['identity_sha256']]=self.task_visit_counts.get(row['identity_sha256'],0)+1
            self._mode_counts[mode]+=1
            self._samples.append({'completed_policy_steps':self.completed_policy_steps,'mode':mode,'distance_stage':stage,
                'base_identity_sha256':base['identity_sha256'],'task_identity_sha256':row['identity_sha256'],
                'yaw_option':option,'task_start_yaw_rad':row['start_yaw_rad'],'measured_reset_yaw_rad':yaw})
            if stage=='ORIGINAL' and self.phase_switch_step is None:self.phase_switch_step=self.completed_policy_steps
            return obs,self._info(info)
        except BaseException:
            self._failed=True;raise

    def step(self,action):
        self._require_live()
        try:
            self._check_info(self._info({}));obs,r,t,tr,info=self.env.step(action)
            self.completed_policy_steps+=1;actual=self._info(info);self._check_info(actual)
            return obs,r,t,tr,actual
        except BaseException:
            self._failed=True;raise

    def scheduler_state(self):
        verify_yaw_manifest(self.manifest,self._prereg)
        return {**self.env.scheduler_state(),'completed_policy_steps':self.completed_policy_steps,
            'phase_switch_step':self.phase_switch_step,'current_identity':self.current_identity,
            'base_current_identity':self.base_current_identity,'active_task':copy.deepcopy(self.active_task),
            'selected_tasks':list(self.selected_tasks),'task_visit_counts':dict(self.task_visit_counts),
            'short_manifest_sha256':self.manifest['short_manifest_sha256'],'yaw_manifest_sha256':self._manifest_sha,
            'yaw_task_count':len(self.by_identity),'yaw_rng_seed_domain':YAW_RNG_SEED_DOMAIN,
            'yaw_rng_state':self._yaw_rng.getstate(),'reset_sampling_profile':'R24_INDEPENDENT_FIVE_YAW_OPTIONS',
            'original_train_weights':list(self._weights),'rehearsal_sample_counts':dict(self._mode_counts),
            'rehearsal_samples':copy.deepcopy(self._samples),'accounting_failure_latched':self._failed,
            'accounting_train_task_identities':sorted(self.by_identity),'resume':'RESTART_REQUIRED',
            'switch_semantics':'next natural reset at/after32768; no mid-episode yaw/target changes'}

    def close(self):
        self.env.close();self.active_task=None;self.active_stage=None;self._closed=True


def make_yaw_scheduler(prereg,upstream,*,seed):
    return YawCoverageScheduler(prereg,upstream,seed=seed)
