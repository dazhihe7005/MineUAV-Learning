"""R27 fixed foreground two-episode altered-heading diagnostic; no train CLI."""
import argparse,json,os,time
from pathlib import Path
from research_ops.storage import OpsError,FileLock,atomic_json,read_json,digest,sha256,process_identity
from research_ops.environment import enforce_resources
from research_ops.identity import source_identity
from .runtime import ROOT,HOME,verify_current
from .budget import BudgetStore
from .goal_heading_diagnostic import make_protocol,evaluate

OUT=ROOT/'.research_local/autorl_r27_goal_heading_diagnostic/run_01'

def verify_checks(reg,root=ROOT):
    root=Path(root)
    try:
        p=root/'.research_local/autorl_r27_checks_01.json';a=read_json(p)
        if a['status']!='PASS' or a['registration_sha256']!=digest(reg) or a['source_identity_sha256']!=digest(reg['source_identity']):raise OpsError('R27 stale admission')
        values={}
        for key in ('operations','tensor','historical','runtime_binding'):
            ref=a['evidence'][key];path=root/ref['path']
            if ref['path']!=f'.research_local/autorl_r27_{key}_final_01.json' or sha256(path)!=ref['sha256']:raise OpsError('R27 admission artifact changed')
            values[key]=read_json(path)
        op,te,hi,bi=(values[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or (op['tests_run'],op['failures'],op['errors'],op['optimizer_updates'])!=(238,0,0,0)
                or len(op['results'])!=238 or any(v['outcome']!='PASS' for v in op['results'])
                or op['source_identity_sha256']!=a['source_identity_sha256'] or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
                or te['returncode'] or te['tests_run']!=76 or te['optimizer_updates'] or 'Ran 76 tests' not in te['stderr']
                or te['source_identity_sha256']!=a['source_identity_sha256'] or te['registration_sha256']!=digest(reg)
                or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
                or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results']) or bi['returncode']
                or bi['source_before_sha256']!=a['source_identity_sha256'] or bi['source_after_sha256']!=a['source_identity_sha256']
                or bi['registration_sha256']!=digest(reg) or bi['final']!=a['evidence']['historical']):raise OpsError('R27 incomplete current no-update regression')
        return sha256(p)
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid R27 no-update evidence') from error

def probe(reg,spec):
    import resource,torch
    from stable_baselines3 import PPO
    from lidar_integration.ppo_formal_scheduler import load_l6_prereg
    from .interrupted_yaw_diagnostic_run import load_readonly_model
    from .goal_heading_trace import collect
    def forbidden(*args,**kwargs):raise AssertionError('NO_OPTIMIZER_OR_TRAINING_R27')
    PPO.learn=forbidden;PPO.train=forbidden;torch.optim.Adam.step=forbidden;torch.set_num_threads(1)
    started=time.monotonic();cpu=time.process_time();minimum=float('inf')
    before={str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()};budget=BudgetStore(HOME).status()
    def guard():
        nonlocal minimum
        available=enforce_resources(ROOT);minimum=min(minimum,available['mem_available_mib'])
        if time.monotonic()-started>spec['wall_seconds']-5 or BudgetStore(HOME).status()['stop_requested']:raise OpsError('R27 duration/stop protection')
        if source_identity(ROOT,reg['source_paths'])!=reg['source_identity']:raise OpsError('R27 runtime source changed')
    prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json');guard()
    model,meta=load_readonly_model(spec['model'],prereg);cases=[]
    for i,task in enumerate(spec['cases']):
        guard();path=OUT/f'trace_{i}.jsonl'
        with path.open('x',encoding='utf-8') as stream:
            def emit(row):stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');stream.flush()
            result=collect(model,task,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',guard=guard,on_record=emit)
        if result['policy_sha256_before_after']!=meta['policy_sha256']:raise OpsError('R27 wrong inference policy')
        result['trace_file']={'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'bytes':path.stat().st_size}
        cases.append(result);atomic_json(OUT/f'case_{i}.json',result,exclusive=True)
        if result['outcome'] not in ('success','collision','time_limit'):raise OpsError('R27 unsafe diagnostic terminal')
    assessment=evaluate(cases,spec);guard()
    after={str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()}
    if before!=after or BudgetStore(HOME).status()!=budget or spec!=make_protocol(reg):raise OpsError('R27 budget/history/protocol changed')
    result={'status':'PASS','registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity']),'protocol_sha256':digest(spec),
        'model':meta,'cases':cases,'assessment':assessment,'new_diagnostic_validation_episodes':2,'reused_control_episodes':2,
        'actual_policy_steps':sum(v['policy_steps'] for v in cases),'physics_ticks':sum(v['physics_ticks'] for v in cases),
        'training_timesteps':0,'optimizer_updates':0,'test_episodes':0,'budget_snapshot_sha256':digest(budget),
        'authorization_files_unchanged':True,'authorization_file_count':len(before),'wall_seconds':time.monotonic()-started,
        'cpu_seconds':time.process_time()-cpu,'worker_hwm_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'guard_sampled_min_MemAvailable_mib':minimum,'claim_limit':spec['claim_limit']}
    atomic_json(OUT/'result.json',result,exclusive=True);atomic_json(OUT/'complete.json',{'result_sha256':sha256(OUT/'result.json')},exclusive=True)
    return {'status':'PASS','result_sha256':sha256(OUT/'result.json'),'new_episodes':2,'optimizer_updates':0}

def run():
    if os.getpgrp()!=os.getpid():raise OpsError('R27 requires owned isolated process group')
    from research_ops_v2.manager import OperationsV2
    m=OperationsV2(ROOT);parent=process_identity(os.getppid())
    with FileLock(m.lock_path):
        _,reg=verify_current()
        if reg['phase_id']!='AUTORL-R27-GOAL-HEADING-DIAGNOSTIC':raise OpsError('wrong fixed R27 registration')
        verify_checks(reg);enforce_resources(ROOT);spec=read_json(ROOT/'mujoco/reports/autonomous_rl_goal_heading_r27_prereg.json')
        if spec!=make_protocol(reg):raise OpsError('R27 frozen protocol differs')
        from .lifetime import start_guard,stop_guard
        watcher=start_guard(parent,time.monotonic()+spec['wall_seconds'],ROOT)
        try:
            OUT.mkdir(parents=True,exist_ok=False)
            atomic_json(OUT/'attempt.json',{'experiment_id':spec['experiment_id'],'kind':spec['kind'],'owner':process_identity(),'parent':parent,
                'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity']),'protocol_sha256':digest(spec)},exclusive=True)
            try:return probe(reg,spec)
            except BaseException as error:
                import traceback
                atomic_json(OUT/'failure.json',{'status':'FAIL','exception_type':type(error).__name__,'message':str(error)[:2000],
                    'traceback':traceback.format_exc()[-12000:],'partial_traces':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in OUT.glob('trace_*.jsonl')],
                    'context':'unique new partial alteredtask diagnostic,no retry/no oldresult modification'},exclusive=True)
                raise
        finally:stop_guard(watcher)

def main():
    argparse.ArgumentParser(description=__doc__).parse_args();print(json.dumps(run(),allow_nan=False))

if __name__=='__main__':main()
