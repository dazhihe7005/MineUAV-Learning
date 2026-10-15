"""One fixed no-optimizer physical prior probe; no training or arbitrary script."""
import os
import time
from research_ops.storage import (FileLock,OpsError,atomic_json,digest,sha256,
                                  read_json,process_identity)
from research_ops.environment import enforce_resources
from .runtime import ROOT,HOME,verify_current
from .budget import BudgetStore

OUT=ROOT/'.research_local/autorl_r2_prior/run_01'


def run():
    if os.getpgrp()!=os.getpid():raise OpsError('probe requires an isolated process group')
    from research_ops_v2.manager import OperationsV2
    m=OperationsV2(ROOT)
    with FileLock(m.lock_path):
        _,reg=verify_current()
        if reg['phase_id']!='AUTORL-R2-PRIOR-PREFLIGHT':raise OpsError('wrong registered preflight phase')
        enforce_resources(ROOT)
        budget_before=BudgetStore(HOME).status()
        if budget_before['stop_requested']:raise OpsError('stop requested')
        OUT.mkdir(parents=True,exist_ok=False)
        atomic_json(OUT/'attempt.json',{'kind':'NO_OPTIMIZER_PRIOR_PHYSICS_PROBE',
            'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity'])},exclusive=True)
        from .lifetime import start_guard,stop_guard
        watcher=start_guard(process_identity(os.getppid()),time.monotonic()+120,ROOT)
        try:return probe(reg,budget_before)
        finally:stop_guard(watcher)


def probe(reg,budget_before):
    import numpy as np
    import torch
    import resource
    from stable_baselines3 import PPO
    from lidar_integration.ppo_preflight import make_ppo
    from lidar_integration.ppo_formal_scheduler import load_l6_prereg,MultiScenarioTrainingEnv
    from lidar_integration.terminal_accounting import classify_terminal
    from lidar_integration.ppo_training_runner import _weights_sha
    from test_env_scripted_policy import scripted_action
    from research_ops.identity import source_identity
    from .worker import runtime_sources
    from .actor_prior import initialize_actor_prior,prior_configuration
    def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_PRIOR_PREFLIGHT')
    PPO.learn=forbidden;PPO.train=forbidden;torch.optim.Adam.step=forbidden
    torch.set_num_threads(1)
    prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
    frozen=runtime_sources(prereg);started=time.monotonic();rows=[];min_mem=float('inf')
    scheduler=MultiScenarioTrainingEnv(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0)
    try:
        model=make_ppo(scheduler,seed=0)
        initial=_weights_sha(model.policy)
        if initial!='bc1adbd197b52b4bfd0bb759a749c16e351092960557f2d341ba3321876b39ba':
            raise OpsError('fresh seed0 initialization differs from R1')
        before={k:v.clone() for k,v in model.policy.state_dict().items()}
        initialize_actor_prior(model)
        assigned=_weights_sha(model.policy)
        changed=[k for k,v in model.policy.state_dict().items() if not torch.equal(v,before[k])]
        if not changed or any(not k.startswith(('mlp_extractor.policy_net.','action_net.')) for k in changed):
            raise OpsError('prior altered more than Actor mean')
        for row in prereg['scenario_manifest']['train']:
            if row['layout']!='empty':continue
            obs,info=scheduler.reset(options={'scenario_identity_sha256':row['identity_sha256']})
            trace=[];total=0.
            for step in range(1,376):
                r=enforce_resources(ROOT);min_mem=min(min_mem,r['mem_available_mib'])
                if BudgetStore(HOME).status()['stop_requested']:raise OpsError('prior probe stopped')
                if time.monotonic()-started>=110:raise OpsError('prior probe bounded wall reached')
                with torch.inference_mode():action=model.predict(obs,deterministic=True)[0]
                reference=scripted_action(obs['state'])
                if np.max(np.abs(action-reference))>.02:raise OpsError('prior action approximation bound violated')
                input_state=obs['state'].tolist()
                obs,reward,t,tr,info=scheduler.step(action);total+=reward
                trace.append({'policy_step':step,'sim_time':float(scheduler.active.unwrapped.data.time),
                    'input_state':input_state,'action':action.tolist(),'reference_action':reference.tolist(),
                    'position':scheduler.active.unwrapped.data.qpos[:3].tolist(),
                    'distance_m':float(info['distance_m']),'speed_m_s':float(info['speed_m_s']),
                    'lidar_age_s':float(info['lidar_age_s'])})
                if t or tr:
                    terminal=classify_terminal(info,t,tr)
                    rows.append({'scenario_identity_sha256':row['identity_sha256'],'steps':step,
                        'terminal_class':terminal.terminal_class,'raw_termination_reason':terminal.raw_termination_reason,
                        'episode_return':float(total),'trace':trace})
                    break
            else:raise OpsError('prior probe exceeded unchanged episode limit')
        if _weights_sha(model.policy)!=assigned or model.num_timesteps or model.policy.optimizer.state:
            raise OpsError('no-optimizer probe modified model/optimizer')
        if source_identity(ROOT,reg['source_paths'])!=reg['source_identity'] or runtime_sources(prereg)!=frozen:
            raise OpsError('probe source changed')
        after=BudgetStore(HOME).status()
        if after!=budget_before:raise OpsError('budget changed during no-learning probe')
        success=len(rows)==3 and all(r['terminal_class']=='success' and len(r['trace'])>=5
            and all(s['distance_m']<.1 and s['speed_m_s']<.15 for s in r['trace'][-5:]) for r in rows)
        result={'status':'PASS' if success else 'FAIL','all_success':success,'rows':rows,
            'classification':'ANALYTIC_PRIOR_NOT_TRAINED_POLICY','optimizer_updates':0,'training_timesteps':0,
            'test_episodes':0,'validation_episodes':0,'configuration':prior_configuration(),
            'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity']),
            'initial_fresh_policy_sha256':initial,'assigned_policy_sha256':assigned,'changed_tensor_names':changed,
            'budget_unchanged':after==budget_before,'runtime_sources':frozen,
            'wall_seconds':time.monotonic()-started,'rss_peak_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'min_mem_available_mib':min_mem}
        atomic_json(OUT/'result.json',result,exclusive=True)
        atomic_json(OUT/'complete.json',{'result_sha256':sha256(OUT/'result.json')},exclusive=True)
        return {'status':result['status'],'result_sha256':sha256(OUT/'result.json'),'optimizer_updates':0}
    finally:scheduler.close()


if __name__=='__main__':print(run())
