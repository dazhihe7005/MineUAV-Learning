"""One bounded non-learning physical feasibility run. No training CLI capability."""
import argparse
import gc
import os
import time
import traceback
from research_ops.storage import FileLock,OpsError,atomic_json,read_json,digest,sha256,process_identity
from research_ops.environment import enforce_resources
from .runtime import ROOT,HOME,verify_current
from .budget import BudgetStore
from .short_goal import short_manifest,verify_short_checks,validate_feasibility

OUT=ROOT/'.research_local/autorl_r5_short_goal/run_01'


def run():
    parent=process_identity(os.getppid())
    if os.getpgrp()!=os.getpid():raise OpsError('short feasibility requires isolated process group')
    from research_ops_v2.manager import OperationsV2
    m=OperationsV2(ROOT)
    with FileLock(m.lock_path):
        _,reg=verify_current()
        if reg['phase_id']!='AUTORL-R5-SHORT':raise OpsError('wrong short task registration')
        verify_short_checks(reg);enforce_resources(ROOT)
        budget=BudgetStore(HOME).status()
        if budget['stop_requested'] or any(a['status']!='COMPLETE' for a in budget['attempts']):raise OpsError('stop/unresolved attempt')
        spec=read_json(ROOT/'mujoco/reports/autonomous_rl_short_goal_r5_prereg.json')
        if spec['registration_sha256']!=digest(reg) or spec['source_identity_sha256']!=digest(reg['source_identity']):raise OpsError('short task spec identity differs')
        from .lifetime import start_guard,stop_guard
        watcher=start_guard(parent,time.monotonic()+180,ROOT)
        try:
            OUT.mkdir(parents=True,exist_ok=False)
            atomic_json(OUT/'attempt.json',{'experiment_id':'AUTORL_R5_SHORT_FEASIBILITY_ONLY',
                'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity']),
                'protocol_sha256':digest(spec),'owner':process_identity(),'parent':parent},exclusive=True)
            try:return produce(reg,budget,spec)
            except BaseException as e:
                atomic_json(OUT/'failure.json',{'status':'FAIL','exception_type':type(e).__name__,
                    'message':str(e)[:2000],'traceback':traceback.format_exc()[-12000:],
                    'context':'see per-episode result/trace files; no complete transition assumed'},exclusive=True)
                raise
        finally:stop_guard(watcher)


def produce(reg,budget,spec):
    import resource
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from lidar_integration.ppo_formal_scheduler import load_l6_prereg
    from lidar_integration.terminal_accounting import classify_terminal
    from lidar_integration.ppo_training_runner import _stats
    from test_env_scripted_policy import scripted_action
    from research_ops.identity import source_identity
    from .short_goal_env import make_short_scheduler
    def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_SHORT_FEASIBILITY')
    PPO.learn=forbidden;PPO.train=forbidden;torch.optim.Adam.step=forbidden;torch.set_num_threads(1)
    begun=time.monotonic();cpu=time.process_time()
    prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json');manifest=short_manifest(prereg)
    if spec['short_manifest']!=manifest:raise OpsError('frozen short task list changed')
    paths=[HOME/'grant.json',HOME/'head.json',*sorted((HOME/'events').iterdir())]
    before={str(p.relative_to(HOME)):sha256(p) for p in paths}
    env=make_short_scheduler(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0)
    rows=[];minimum=float('inf');step_ms=[];scan_ms=[];encode_ms=[]
    try:
        for i,task in enumerate(manifest['train']):
            obs,info=env.reset(options={'scenario_identity_sha256':task['base_identity_sha256']})
            plant=env.active.unwrapped;trace=[];total=0.
            if info['scenario_identity_sha256']!=task['identity_sha256']:raise OpsError('wrong reset task')
            for step in range(1,376):
                minimum=min(minimum,enforce_resources(ROOT)['mem_available_mib'])
                if BudgetStore(HOME).status()['stop_requested'] or time.monotonic()-begun>170:raise OpsError('short feasibility stopped')
                if source_identity(ROOT,reg['source_paths'])!=reg['source_identity']:raise OpsError('short feasibility runtime changed')
                action=scripted_action(obs['state']);tick=round(float(plant.data.time)/.002);started=time.perf_counter()
                obs,r,t,tr,info=env.step(action);step_ms.append((time.perf_counter()-started)*1000);total+=float(r)
                classification=classify_terminal(info,t,tr)
                if info['lidar_status']!='FRESH' or not 0<=float(info['lidar_age_s'])<=.15:raise OpsError('short task sensor invalid')
                scan_ms.append(float(plant.last_scan_ms));encode_ms.append(float(plant.last_encode_ms))
                trace.append({'sim_time':float(plant.data.time),'distance_m':float(info['distance_m']),
                    'speed_m_s':float(info['speed_m_s']),'position':plant.data.qpos[:3].tolist(),
                    'velocity':plant.data.qvel[:3].tolist(),'action':action.tolist(),'reward':float(r),
                    'physics_ticks':round(float(plant.data.time)/.002)-tick,
                    'lidar_source_time_s':info['lidar_source_time_s'],'lidar_age_s':info['lidar_age_s'],
                    'lidar_frame_count':info['lidar_frame_count']})
                if t or tr:
                    record={'task_identity_sha256':task['identity_sha256'],'base_identity_sha256':task['base_identity_sha256'],
                        'layout':task['layout'],'start_xyz':task['start_xyz'],'target_xyz':task['target_xyz'],
                        'terminal_class':classification.terminal_class,'raw_termination_reason':classification.raw_termination_reason,
                        'steps':step,'episode_return':total,'trace':trace}
                    rows.append(record);atomic_json(OUT/f'episode_{i:02d}.json',record,exclusive=True)
                    break
            else:raise OpsError('short task missing terminal by original375 limit')
            if record['terminal_class']!='success':break
    finally:env.close()
    del env;gc.collect()
    if (BudgetStore(HOME).status()!=budget or {str(p.relative_to(HOME)):sha256(p) for p in paths}!=before
        or source_identity(ROOT,reg['source_paths'])!=reg['source_identity']):raise OpsError('short feasibility budget/source changed')
    result={'status':'PASS' if len(rows)==12 and all(r['terminal_class']=='success' for r in rows) else 'FAIL',
        'experiment_id':'AUTORL_R5_SHORT_FEASIBILITY_ONLY','registration_sha256':digest(reg),
        'source_identity_sha256':digest(reg['source_identity']),'protocol_sha256':digest(spec),
        'short_manifest_sha256':digest(manifest),'rows':rows,'optimizer_updates':0,'training_timesteps':0,
        'validation_episodes':0,'test_episodes':0,'budget_unchanged':True,'budget_files_sha256':before,
        'budget':{k:v for k,v in budget.items() if k!='attempts'},'wall_seconds':time.monotonic()-begun,
        'cpu_seconds':time.process_time()-cpu,'worker_ru_maxrss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'min_sampled_mem_available_mib':minimum,'physics_ticks':sum(v['physics_ticks'] for r in rows for v in r['trace']),
        'environment_step_ms':_stats(step_ms),'scan_ms_last_sample_per_step':_stats(scan_ms),
        'encoding_ms_last_sample_per_step':_stats(encode_ms),'claim':'STATE_ONLY_SCRIPTED_FEASIBILITY_NOT_LEARNED_AVOIDANCE'}
    if result['status']=='PASS':validate_feasibility(result,manifest)
    atomic_json(OUT/'result.json',result,exclusive=True)
    atomic_json(OUT/'complete.json',{'result_sha256':sha256(OUT/'result.json')},exclusive=True)
    return {'status':result['status'],'result_sha256':sha256(OUT/'result.json'),'optimizer_updates':0}


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    import json
    print(json.dumps(run(),allow_nan=False))
