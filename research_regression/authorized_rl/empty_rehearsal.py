"""R17 reset-only rehearsal; existing physical/accounting paths remain intact."""
import numpy as np
from research_ops.storage import digest
from .short_goal import BASE_MANIFEST_SHA
from .short_goal_env import make_short_scheduler
from .task_accounting import DistanceTaskAccountingAdapter


def rehearsal_weights(prereg):
    if digest(prereg['scenario_manifest'])!=BASE_MANIFEST_SHA:
        raise ValueError('rehearsal requires exact frozen L6 manifest')
    rows=prereg['scenario_manifest']['train']
    if len(rows)!=12 or sum(r['layout']=='empty' for r in rows)!=3:
        raise ValueError('rehearsal requires three empty and nine nonempty Train bases')
    return tuple(1/6 if r['layout']=='empty' else 1/18 for r in rows)


def draw_rehearsal_identity(prereg,rng):
    weights=rehearsal_weights(prereg)
    return prereg['scenario_manifest']['train'][int(rng.choice(12,p=weights))]['identity_sha256']


class EmptyRehearsalAdapter(DistanceTaskAccountingAdapter):
    def __init__(self,env,prereg):
        super().__init__(env,prereg)
        self._rehearsal_prereg=prereg;self._weights=rehearsal_weights(prereg)
        self._sampling=[];self._counts={'UNIFORM_SHORT':0,'WEIGHTED_ORIGINAL':0,'EXPLICIT_TRAIN_COVERAGE':0}

    def reset(self,*,seed=None,options=None):
        self._require_live()
        try:
            mode='EXPLICIT_TRAIN_COVERAGE' if options is not None else 'UNIFORM_SHORT'
            if self.completed_policy_steps>=32768 and options is None:
                if seed is not None:
                    if type(seed) is not int or seed<0:raise ValueError('invalid rehearsal seed')
                    self.env._rng=np.random.default_rng(seed)
                    seed=None  # Do not reseed again after consuming this draw.
                identity=draw_rehearsal_identity(self._rehearsal_prereg,self.env._rng)
                options={'scenario_identity_sha256':identity};mode='WEIGHTED_ORIGINAL'
            result=super().reset(seed=seed,options=options)
            self._counts[mode]+=1
            self._sampling.append({'completed_policy_steps':self.completed_policy_steps,'mode':mode,
                'base_identity_sha256':self.base_current_identity,'task_identity_sha256':self.current_identity})
            return result
        except BaseException:
            self._failed=True
            raise

    def scheduler_state(self):
        return {**super().scheduler_state(),'reset_sampling_profile':'R17_ORIGINAL_EMPTY_PROBABILITY_HALF',
            'original_train_weights':list(self._weights),'rehearsal_sample_counts':dict(self._counts),
            'rehearsal_samples':list(self._sampling)}


def make_rehearsal_scheduler(prereg,upstream,*,seed):
    return EmptyRehearsalAdapter(make_short_scheduler(prereg,upstream,seed=seed),prereg)
