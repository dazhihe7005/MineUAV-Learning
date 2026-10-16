"""Explicit Train task identity boundary; no dynamics/reward/control changes."""
import copy
from types import MappingProxyType
import gymnasium as gym
import numpy as np
from research_ops.storage import digest
from .short_goal import short_manifest,BASE_MANIFEST_SHA


class DistanceTaskAccountingAdapter(gym.Wrapper):
    """Expose task membership to old metrics, retain base-only reset sampling.

    This does not authorize training or change the failed R6 execution path.
    by_identity is an immutable membership view, not the inner reset row table.
    """
    def __init__(self,env,prereg):
        if digest(prereg['scenario_manifest'])!=BASE_MANIFEST_SHA:
            raise ValueError('distance accounting requires frozen L6 manifest')
        super().__init__(env)
        bases=prereg['scenario_manifest']['train'];short=short_manifest(prereg)['train']
        self._bases={r['identity_sha256']:copy.deepcopy(r) for r in bases}
        self._short={r['base_identity_sha256']:copy.deepcopy(r) for r in short}
        identities=[r['identity_sha256'] for r in [*bases,*short]]
        if len(set(identities))!=24 or env.rows!=tuple(bases) or set(env.by_identity)!=set(self._bases):
            raise ValueError('distance scheduler does not match frozen Train base identities')
        self._task_membership=MappingProxyType({i:i for i in identities})
        self._manifest_sha=digest(short_manifest(prereg));self._failed=False

    @property
    def by_identity(self):return self._task_membership

    @property
    def active(self):return self.env.active

    @property
    def active_environment_count(self):return self.env.active_environment_count

    @property
    def base_current_identity(self):return self.env.current_identity

    @property
    def current_identity(self):
        # Outer consumers (failure evidence, traces) observe the true task.
        # Never expose an initialization-only base or a closed task as active.
        task=self.env.active_task
        if self.active is None or task is None:return None
        return task['identity_sha256']

    @property
    def completed_policy_steps(self):return self.env.completed_policy_steps

    @property
    def phase_switch_step(self):return self.env.phase_switch_step

    @property
    def selected_identities(self):return list(self.env.selected_identities)

    def _require_live(self):
        if self._failed:raise RuntimeError('distance accounting failure latched; new experiment required')

    def _check_info(self,info):
        base=self.base_current_identity;stage=self.env.active_stage
        if base not in self._bases or stage not in ('SHORT','ORIGINAL'):
            raise RuntimeError('unregistered base or distance stage')
        row=self._short[base] if stage=='SHORT' else self._bases[base]
        plant=self.active.unwrapped
        if (self.env.active_task!=row or not isinstance(info,dict)
            or info.get('scenario_split')!='train'
            or info.get('scenario_identity_sha256')!=row['identity_sha256']
            or info.get('base_scenario_identity_sha256')!=base
            or info.get('distance_stage')!=stage
            or info.get('task_target_xyz')!=row['target_xyz']
            or not np.array_equal(plant.target_position,np.asarray(row['target_xyz']))
            or plant.scene.geometry_hash!=self._bases[base]['obstacle_configuration_hash']
            or ('obstacle_configuration_hash' in info and info['obstacle_configuration_hash']!=plant.scene.geometry_hash)):
            raise RuntimeError('distance task/base/stage/target/geometry identity mismatch')

    def reset(self,*,seed=None,options=None):
        self._require_live()
        try:
            if options is not None and (not isinstance(options,dict)
                or set(options)!={'scenario_identity_sha256'}
                or options['scenario_identity_sha256'] not in self._bases):
                raise ValueError('coverage reset accepts only twelve frozen base Train identities')
            result=self.env.reset(seed=seed,options=options);self._check_info(result[1])
            return result
        except BaseException:
            self._failed=True
            raise

    def step(self,action):
        self._require_live()
        try:
            # Validate the actual active task before advancing physics as well as
            # the returned metadata afterwards. Do not manufacture transition info.
            self._check_info(self.env._info({}))
            result=self.env.step(action);self._check_info(result[4])
            return result
        except BaseException:
            self._failed=True
            raise

    def scheduler_state(self):
        return {**self.env.scheduler_state(),
                'current_identity':self.current_identity,
                'base_current_identity':self.base_current_identity,
                'accounting_contract':'DISTANCE_TASK_MEMBERSHIP_V1',
                'accounting_train_task_identities':sorted(self.by_identity),
                'accounting_train_task_identities_sha256':digest(sorted(self.by_identity)),
                'accounting_short_manifest_sha256':self._manifest_sha,
                'identity_scope':'current_identity is task; base_current_identity and visit_counts are base geometry; 24 Train tasks for accounting',
                'accounting_failure_latched':self._failed}
