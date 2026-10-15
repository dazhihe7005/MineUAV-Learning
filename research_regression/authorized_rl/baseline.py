"""Three frozen empty Train scenes, state-only heuristic, no learning."""
from pathlib import Path
import time
from research_ops.storage import FileLock,OpsError,atomic_json,sha256
from .runtime import ROOT,HOME,verify_current,check_resources,monitor_process,validate_launch


def run_baseline():
    from research_ops_v2.manager import OperationsV2
    m=OperationsV2(ROOT)
    with FileLock(m.lock_path):
        _,registration=verify_current()
        from research_ops.storage import process_identity,read_json,digest
        from research_ops.environment import runtime_environment,resource_status
        import os,subprocess
        work=HOME/'baseline_runtime';work.mkdir(exist_ok=False)
        permit={'experiment_id':'ARL_BASELINE_R1','parent_pid':os.getpid(),
                'parent_identity':process_identity(),'protocol_sha256':digest(registration)}
        atomic_json(work/'permit.json',permit,exclusive=True)
        envlock=read_json(ROOT/'research_ops/environment_lock.json')
        with (work/'console.txt').open('xb') as console:
            child=subprocess.Popen([envlock['interpreter'],'-m','research_regression.authorized_rl.baseline'],
                cwd=ROOT,env=runtime_environment(ROOT,envlock),stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
            def probe():
                from .budget import BudgetStore
                r=resource_status(ROOT);check_resources(r['mem_available_mib'],r['disk_free_bytes'])
                if BudgetStore(HOME).status()['stop_requested']:raise OpsError('baseline stop requested')
            monitor_process(child,120,probe)
        if child.returncode!=0:raise OpsError('baseline worker failed; no automatic retry')
        return verify_baseline(registration)


def verify_baseline(registration):
    from research_ops.storage import read_json,digest
    result=read_json(HOME/'baseline.json')
    complete=read_json(HOME/'baseline.complete.json')
    if complete['sha256']!=sha256(HOME/'baseline.json'):raise OpsError('baseline evidence changed')
    if (result.get('all_success') is not True or result.get('optimizer_updates')!=0
        or result.get('registration_sha256')!=digest(registration)
        or result.get('source_identity_sha256')!=digest(registration['source_identity'])):raise OpsError('baseline authority mismatch')
    prereg_path=ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json'
    if sha256(prereg_path)!='c468490154627ba8f71df9c73bc8c80769f557054d9692313139188971e332c8':
        raise OpsError('baseline frozen manifest file changed')
    prereg=read_json(prereg_path)
    expected={r['identity_sha256'] for r in prereg['scenario_manifest']['train'] if r['layout']=='empty'}
    if {r['scenario_identity_sha256'] for r in result['rows']}!=expected or len(result['rows'])!=3:
        raise OpsError('baseline scene identities differ')
    for r in result['rows']:
        if (r['terminal_class']!='success' or not 1<=r['steps']<=375 or len(r['trace'])!=r['steps']
            or len(r['trace'])<5 or any(v['distance_m']>=.1 or v['speed_m_s']>=.15 for v in r['trace'][-5:])):
            raise OpsError('baseline successful stop not evidenced')
    return {'all_success':True,'episodes':3,'sha256':complete['sha256'],'outcomes':['success']*3}


def produce_baseline():
        from research_ops.storage import read_json,digest
        from .budget import BudgetStore
        from .lifetime import start_guard,stop_guard
        permit=read_json(HOME/'baseline_runtime/permit.json');validate_launch(permit,'ARL_BASELINE_R1')
        watcher=start_guard(permit['parent_identity'],time.monotonic()+120,ROOT)
        from research_ops_v2.manager import OperationsV2
        registration=OperationsV2(ROOT)._load()[1]['registration']
        if permit['protocol_sha256']!=digest(registration):raise OpsError('baseline registered source changed')
        path=HOME/'baseline.json'
        attempt=HOME/'baseline.attempt.json'
        if path.exists() or attempt.exists():raise OpsError('baseline already attempted; never overwrite')
        atomic_json(attempt,{'kind':'STATE_ONLY_SCRIPTED_FEASIBILITY','optimizer_updates':0},exclusive=True)
        import numpy as np
        from lidar_integration.ppo_formal_scheduler import MultiScenarioTrainingEnv,load_l6_prereg
        from lidar_integration.terminal_accounting import classify_terminal
        from test_env_scripted_policy import scripted_action
        from .worker import runtime_sources
        prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        scheduler=MultiScenarioTrainingEnv(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0)
        rows=[];started=time.monotonic()
        try:
            for row in prereg['scenario_manifest']['train']:
                if row['layout']!='empty':continue
                obs,info=scheduler.reset(options={'scenario_identity_sha256':row['identity_sha256']})
                trace=[];total=0.
                for step in range(1,376):
                    validate_launch(permit,'ARL_BASELINE_R1')
                    if BudgetStore(HOME).status()['stop_requested']:raise OpsError('baseline stop requested')
                    if time.monotonic()-started>120:raise OpsError('bounded baseline wall deadline')
                    action=scripted_action(obs['state'])
                    obs,r,t,tr,info=scheduler.step(action);total+=r
                    trace.append({'sim_time':float(scheduler.active.unwrapped.data.time),
                        'distance_m':float(info['distance_m']),'speed_m_s':float(info['speed_m_s']),
                        'position':scheduler.active.unwrapped.data.qpos[:3].tolist()})
                    if t or tr:
                        outcome=classify_terminal(info,t,tr)
                        rows.append({'scenario_identity_sha256':row['identity_sha256'],
                            'terminal_class':outcome.terminal_class,'raw_termination_reason':outcome.raw_termination_reason,
                            'steps':step,'episode_return':float(total),'final_distance_m':float(info['distance_m']),
                            'min_distance_m':min(v['distance_m'] for v in trace),'trace':trace})
                        break
                else:raise OpsError('baseline did not terminate by frozen episode limit')
            result={'kind':'STATE_ONLY_SCRIPTED_NOT_LIDAR_POLICY','all_success':len(rows)==3 and all(r['terminal_class']=='success' for r in rows),
                'rows':rows,'optimizer_updates':0,'test_episode_count':0,'runtime_sources':runtime_sources(prereg),
                'wall_seconds':time.monotonic()-started,'registration_sha256':digest(registration),
                'source_identity_sha256':digest(registration['source_identity'])}
            atomic_json(path,result,exclusive=True)
            atomic_json(HOME/'baseline.complete.json',{'sha256':sha256(path)},exclusive=True)
            return {'all_success':result['all_success'],'episodes':len(rows),'sha256':sha256(path),
                    'outcomes':[r['terminal_class'] for r in rows]}
        finally:
            scheduler.close();stop_guard(watcher)


if __name__=='__main__':produce_baseline()
