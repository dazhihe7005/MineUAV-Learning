"""Pure Train eligibility; dynamic environment subclass constructed only by worker."""
from research_ops.storage import OpsError


def eligible_rows(rows,arm,steps):
    if not rows or any(r.get('split')!='train' for r in rows):raise OpsError('non-Train curriculum input')
    if arm not in ('UNIFORM','CURRICULUM') or type(steps) is not int or steps<0:raise OpsError('invalid curriculum state')
    result=[r for r in rows if arm=='UNIFORM' or steps>=32768 or r['layout']=='empty']
    if not result:raise OpsError('no eligible training scenes')
    return result


def make_scheduler(prereg,upstream,arm,seed):
    from lidar_integration.ppo_formal_scheduler import MultiScenarioTrainingEnv
    class Scheduler(MultiScenarioTrainingEnv):
        def __init__(self):
            self.completed_policy_steps=0
            self.phase_switch_step=None
            super().__init__(prereg,upstream,seed=seed)

        def reset(self,*,seed=None,options=None):
            if seed is not None:
                import numpy as np
                self._rng=np.random.default_rng(seed)
            if options is None:
                candidates=eligible_rows(self.rows,arm,self.completed_policy_steps)
                row=candidates[int(self._rng.integers(len(candidates)))]
                options={'scenario_identity_sha256':row['identity_sha256']}
                if arm=='CURRICULUM' and self.completed_policy_steps>=32768 and self.phase_switch_step is None:
                    self.phase_switch_step=self.completed_policy_steps
            return super().reset(seed=None,options=options)

        def step(self,action):
            result=super().step(action)
            self.completed_policy_steps+=1
            return result

        def scheduler_state(self):
            return {**super().scheduler_state(),'arm':arm,'completed_policy_steps':self.completed_policy_steps,
                'phase_switch_step':self.phase_switch_step,'switch_semantics':'next natural reset at/after32768'}
    return Scheduler()
