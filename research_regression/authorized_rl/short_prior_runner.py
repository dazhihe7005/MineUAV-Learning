"""One fixed zero-optimizer R9 physical feasibility attempt. No training permit."""
import argparse
import gc
import os
import time
import traceback
from research_ops.storage import FileLock,OpsError,atomic_json,read_json,digest,sha256,process_identity
from research_ops.environment import enforce_resources
from .runtime import ROOT,HOME,verify_current
from .budget import BudgetStore
from .short_prior import IDENTITY,short_prior_spec,verify_short_prior_admission,initialize_short_prior,validate_short_prior_result,require_safe_coverage_terminal

OUT=ROOT/'.research_local/autorl_r9_short_prior/run_01'


def run():
    parent=process_identity(os.getppid())
    if os.getpgrp()!=os.getpid():raise OpsError('R9 requires isolated process group')
    from research_ops_v2.manager import OperationsV2
    manager=OperationsV2(ROOT)
    with FileLock(manager.lock_path):
        _,reg=verify_current()
        if reg['phase_id']!='AUTORL-R9-SHORT-PRIOR':raise OpsError('wrong R9 task registration')
        verify_short_prior_admission(reg);enforce_resources(ROOT)
        budget=BudgetStore(HOME).status()
        if budget['stop_requested'] or any(a['status']!='COMPLETE' and not (a['experiment_id']=='ARL_R6_DISTANCE_SEED0' and a['status']=='FAIL') for a in budget['attempts']):
            raise OpsError('R9 stop or unknown/unresolved grant attempt')
        spec=read_json(ROOT/'mujoco/reports/autonomous_rl_short_prior_r9_prereg.json')
        if (spec!={**short_prior_spec(),'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity'])}):
            raise OpsError('R9 frozen execution spec differs')
        from .lifetime import start_guard,stop_guard
        watcher=start_guard(parent,time.monotonic()+180,ROOT)
        try:
            OUT.mkdir(parents=True,exist_ok=False)
            atomic_json(OUT/'attempt.json',{'experiment_id':IDENTITY,'protocol_sha256':digest(spec),
                'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity']),
                'owner':process_identity(),'parent':parent},exclusive=True)
            try:return produce(reg,budget,spec)
            except BaseException as error:
                atomic_json(OUT/'failure.json',{'status':'FAIL','exception_type':type(error).__name__,
                    'message':str(error)[:2000],'traceback':traceback.format_exc()[-12000:],
                    'context':'partial episode context in files only; missing transition NOT_AVAILABLE'},exclusive=True)
                raise
        finally:stop_guard(watcher)


def produce(reg,budget,spec):
    import resource
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from lidar_integration.ppo_formal_scheduler import load_l6_prereg
    from lidar_integration.ppo_formal_seed0_health import FormalHealthMetricsEnv
    from lidar_integration.terminal_accounting import classify_terminal
    from lidar_integration.failure_evidence import json_safe
    from lidar_integration.ppo_preflight import make_ppo
    from lidar_integration.ppo_training_runner import _stats,_weights_sha
    from research_ops.identity import source_identity
    from .repaired_distance_training import make_repaired_scheduler
    from .worker import runtime_sources
    def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_R9_FEASIBILITY')
    PPO.learn=forbidden;PPO.train=forbidden;torch.optim.Adam.step=forbidden;torch.optim.AdamW.step=forbidden
    torch.set_num_threads(1);begun=time.monotonic();cpu=time.process_time()
    prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
    sources=runtime_sources(prereg);budget_files={str(p.relative_to(HOME)):sha256(p) for p in HOME.rglob('*') if p.is_file()}
    scheduler=make_repaired_scheduler(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0)
    env=FormalHealthMetricsEnv(scheduler,diagnostics_dir=OUT/'health')
    rows=[];minimum=float('inf');step_ms=[];scan_ms=[];encode_ms=[];trace=[];last_returned=None
    try:
        model=make_ppo(env,seed=0);initial=initialize_short_prior(model)
        for i,task in enumerate(spec['short_manifest']['train']):
            obs,info=env.reset(options={'scenario_identity_sha256':task['base_identity_sha256']})
            plant=scheduler.active.unwrapped;trace=[];total=0.
            if info['scenario_identity_sha256']!=task['identity_sha256']:raise OpsError('R9 reset task differs')
            previous_frames=plant.lidar_frame_count if hasattr(plant,'lidar_frame_count') else info['lidar_frame_count']
            for step in range(1,376):
                minimum=min(minimum,enforce_resources(ROOT)['mem_available_mib'])
                if BudgetStore(HOME).status()['stop_requested'] or time.monotonic()-begun>170:raise OpsError('R9 stopped')
                if source_identity(ROOT,reg['source_paths'])!=reg['source_identity'] or runtime_sources(prereg)!=sources:
                    raise OpsError('R9 frozen runtime changed')
                with torch.inference_mode():action=model.predict(obs,deterministic=True)[0]
                if action.shape!=(4,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):raise OpsError('R9 invalid action')
                tick=round(float(plant.data.time)/.002);started=time.perf_counter()
                obs,r,t,tr,info=env.step(action);step_ms.append((time.perf_counter()-started)*1000);total+=float(r)
                classification=classify_terminal(info,t,tr)
                last_returned={'terminal_class':classification.terminal_class,'raw_termination_reason':classification.raw_termination_reason,
                               'terminated':t,'truncated':tr,'reward':r,'simulation_time':float(plant.data.time),
                               'scenario_identity_sha256':info['scenario_identity_sha256']}
                if (info['lidar_status']!='FRESH' or not 0<=float(info['lidar_age_s'])<=.15
                    or info['lidar_source_time_s']>float(plant.data.time)+1e-9
                    or info['scenario_identity_sha256']!=task['identity_sha256']):raise OpsError('R9 invalid identity or sensor')
                if info['lidar_frame_count']>previous_frames:
                    scan_ms.append(float(plant.last_scan_ms));encode_ms.append(float(plant.last_encode_ms))
                previous_frames=info['lidar_frame_count']
                trace.append({'sim_time':float(plant.data.time),'distance_m':float(info['distance_m']),
                    'speed_m_s':float(info['speed_m_s']),'position':plant.data.qpos[:3].tolist(),'velocity':plant.data.qvel[:3].tolist(),
                    'action':action.tolist(),'reward':float(r),'physics_ticks':round(float(plant.data.time)/.002)-tick,
                    'lidar_source_time_s':info['lidar_source_time_s'],'lidar_age_s':info['lidar_age_s'],
                    'lidar_frame_count':info['lidar_frame_count']})
                if _weights_sha(model.policy)!=initial['policy_sha256'] or model.policy.optimizer.state or model._n_updates or model.num_timesteps:
                    raise OpsError('R9 weights or optimizer state mutated')
                if t or tr:
                    require_safe_coverage_terminal(classification.terminal_class)
                    record={'task_identity_sha256':task['identity_sha256'],'base_identity_sha256':task['base_identity_sha256'],
                        'layout':task['layout'],'start_xyz':task['start_xyz'],'target_xyz':task['target_xyz'],
                        'terminal_class':classification.terminal_class,'raw_termination_reason':classification.raw_termination_reason,
                        'contact_pairs':info['contact_pairs'],
                        'steps':step,'episode_return':total,'trace':trace}
                    rows.append(record);atomic_json(OUT/f'episode_{i:02d}.json',record,exclusive=True);break
            else:raise OpsError('R9 missing terminal by original375 limit')
    except BaseException:
        atomic_json(OUT/'partial_context.json',json_safe({'scope':'LAST_KNOWN_CONTEXT','trace':trace,
                    'last_returned_transition':last_returned if last_returned is not None else 'NOT_AVAILABLE'}),exclusive=True);raise
    finally:env.close()
    del env,scheduler,model;gc.collect()
    if (BudgetStore(HOME).status()!=budget or {str(p.relative_to(HOME)):sha256(p) for p in HOME.rglob('*') if p.is_file()}!=budget_files
        or source_identity(ROOT,reg['source_paths'])!=reg['source_identity'] or runtime_sources(prereg)!=sources):raise OpsError('R9 budget/source changed')
    result={'status':'PASS' if len(rows)==12 and all(r['terminal_class']=='success' for r in rows) else 'FAIL',
        'experiment_id':IDENTITY,'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity']),
        'protocol_sha256':digest(spec),'short_manifest_sha256':digest(spec['short_manifest']),'initialization':initial,
        'runtime_sources':sources,'rows':rows,'optimizer_updates':0,'training_timesteps':0,'validation_episodes':0,'test_episodes':0,
        'budget_unchanged':True,'budget_files_sha256':budget_files,'wall_seconds':time.monotonic()-begun,'cpu_seconds':time.process_time()-cpu,
        'worker_ru_maxrss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'min_sampled_mem_available_mib':minimum,
        'physics_ticks':sum(v['physics_ticks'] for row in rows for v in row['trace']),
        'environment_step_ms':_stats(step_ms),'scan_ms_new_frame_excluding_reset':_stats(scan_ms),
        'encoding_ms_new_frame_excluding_reset':_stats(encode_ms),'claim':spec['claim']}
    if result['status']=='PASS':validate_short_prior_result(result,spec['short_manifest'])
    atomic_json(OUT/'result.json',result,exclusive=True)
    atomic_json(OUT/'complete.json',{'result_sha256':sha256(OUT/'result.json'),'status':result['status']},exclusive=True)
    return {'status':result['status'],'result_sha256':sha256(OUT/'result.json'),'optimizer_updates':0}


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    import json
    print(json.dumps(run(),allow_nan=False))
