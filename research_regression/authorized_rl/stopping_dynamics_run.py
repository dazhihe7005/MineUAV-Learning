"""Fixed bounded R18 physical diagnostic, no training or arbitrary task args."""
import argparse,gc,json,os,time
from research_ops.storage import FileLock,OpsError,atomic_json,read_json,digest,sha256,process_identity
from research_ops.environment import enforce_resources
from .runtime import ROOT,HOME,verify_current
from .budget import BudgetStore
from .stopping_dynamics import configuration,make_protocol,verify_checks

OUT=ROOT/'.research_local/autorl_r18_stopping_dynamics/run_01'


def run():
    if os.getpgrp()!=os.getpid():raise OpsError('R18 requires isolated owned process group')
    from research_ops_v2.manager import OperationsV2
    m=OperationsV2(ROOT);parent=process_identity(os.getppid())
    with FileLock(m.lock_path):
        _,reg=verify_current()
        if reg['phase_id']!='AUTORL-R18-STOPPING-DYNAMICS':raise OpsError('wrong reviewed R18 phase')
        verify_checks(reg);enforce_resources(ROOT)
        spec=read_json(ROOT/'mujoco/reports/autonomous_rl_stopping_dynamics_r18_prereg.json')
        if spec!=make_protocol(reg):raise OpsError('R18 frozen identity differs')
        from .lifetime import start_guard,stop_guard
        watcher=start_guard(parent,time.monotonic()+spec['wall_seconds'],ROOT)
        try:
            OUT.mkdir(parents=True,exist_ok=False)
            atomic_json(OUT/'attempt.json',{'experiment_id':spec['experiment_id'],'kind':spec['kind'],
                'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity']),
                'protocol_sha256':digest(spec),'owner':process_identity(),'parent':parent},exclusive=True)
            try:return probe(reg,spec)
            except BaseException as e:
                import traceback
                atomic_json(OUT/'failure.json',{'status':'FAIL','exception_type':type(e).__name__,'message':str(e)[:2000],
                    'traceback':traceback.format_exc()[-12000:],
                    'partial_traces':[{'path':str(p.relative_to(ROOT)),'sha256':sha256(p),'bytes':p.stat().st_size} for p in OUT.glob('trace_*.jsonl')],
                    'context':'new diagnostic partial records only; no complete result assumed; unique attempt never retried'},exclusive=True)
                raise
        finally:stop_guard(watcher)


def probe(reg,spec):
    import resource,torch
    from stable_baselines3 import PPO
    from lidar_integration.ppo_formal_scheduler import load_l6_prereg
    from lidar_integration.ppo_formal_checkpoint import CheckpointManager
    from research_ops.identity import source_identity
    from .state_centered_policy import verify_centered_policy
    from .stopping_trace import collect_trace
    def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_OR_TRAINING_R18')
    PPO.learn=forbidden;PPO.train=forbidden;torch.optim.Adam.step=forbidden
    torch.set_num_threads(1);begun=time.monotonic();cpu=time.process_time()
    prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
    row=next(v for v in prereg['scenario_manifest']['validation'] if v['identity_sha256']==spec['validation_identity'])
    budget=BudgetStore(HOME).status();before={str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()}
    results=[];minimum=float('inf')
    def guard():
        nonlocal minimum
        available=enforce_resources(ROOT);minimum=min(minimum,available['mem_available_mib'])
        if time.monotonic()-begun>spec['wall_seconds']-5 or BudgetStore(HOME).status()['stop_requested']:
            raise OpsError('R18 absolute time/stop protection')
    for i,ref in enumerate(spec['models']):
        guard()
        model=CheckpointManager(ROOT/ref['checkpoint_path'].rsplit('/',1)[0],prereg,
            execution_protocol_sha256=ref['protocol_sha256']).load_verified(None,run_id=ref['run_id'])
        architecture=verify_centered_policy(model)
        trace_path=OUT/f'trace_{i}.jsonl'
        with trace_path.open('x',encoding='utf-8') as f:
            def emit(record):
                f.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n');f.flush()
            result=collect_trace(model,row,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',
                max_steps=spec['max_policy_steps_per_model'],guard=guard,on_record=emit)
        if result['policy_sha256_before_after']!=ref['policy_sha256']:raise OpsError('R18 loaded policy identity differs')
        old=ref['historical_same_scene_summary'];tol=spec['historical_summary_reproduction_tolerance']
        if (result['outcome']!=old['terminal_class'] or result['policy_steps']!=old['policy_steps']
            or abs(result['final_target_error_m']-old['final_target_error_m'])>tol
            or abs(result['episode_return']-old['episode_return'])>tol):raise OpsError('R18 instrumentation does not reproduce historical same-scene summary')
        del result['trace']
        results.append({'reference':ref,'architecture':architecture,**result,'historical_summary_reproduction_verified':True,
            'trace_file':{'path':str(trace_path.relative_to(ROOT)),'sha256':sha256(trace_path),'bytes':trace_path.stat().st_size}})
        del model;gc.collect()
    after={str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()}
    if (before!=after or BudgetStore(HOME).status()!=budget or spec!=make_protocol(reg)
        or source_identity(ROOT,reg['source_paths'])!=reg['source_identity']):raise OpsError('R18 changed source/history/budget')
    history=__import__('research_ops_v2.manager',fromlist=['OperationsV2']).OperationsV2(ROOT).verify()['history']
    result={'status':'PASS','experiment_id':spec['experiment_id'],'registration_sha256':digest(reg),
        'source_identity_sha256':digest(reg['source_identity']),'protocol_sha256':digest(spec),'models':results,
        'training_timesteps':0,'optimizer_updates':0,'episodes':3,'validation_episodes':3,'test_episodes':0,
        'physics_steps':sum(v['physics_ticks'] for v in results),'actual_evaluation_policy_steps':sum(v['policy_steps'] for v in results),
        'authorization_files_sha256':before,'authorization_file_count':len(before),'all_authorization_files_unchanged':True,
        'budget_snapshot_sha256':digest(budget),'history':history,'wall_seconds':time.monotonic()-begun,
        'cpu_seconds':time.process_time()-cpu,'worker_hwm_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        'per_policy_guard_sampled_min_mem_available_mib':minimum,'claim_limit':spec['claim_limit']}
    atomic_json(OUT/'result.json',result,exclusive=True)
    atomic_json(OUT/'complete.json',{'result_sha256':sha256(OUT/'result.json')},exclusive=True)
    return {'status':'PASS','result_sha256':sha256(OUT/'result.json'),'optimizer_updates':0}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(),allow_nan=False))


if __name__=='__main__':main()
