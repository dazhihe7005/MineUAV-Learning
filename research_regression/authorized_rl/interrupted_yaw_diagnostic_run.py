"""R26 foreground-only fixed eight-episode inference diagnostic, never PPO learn."""
import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from research_ops.storage import FileLock,OpsError,atomic_json,read_json,digest,sha256,process_identity
from research_ops.environment import enforce_resources
from research_ops.identity import source_identity
from .runtime import ROOT,HOME,verify_current
from .budget import BudgetStore
from .interrupted_yaw_diagnostic import make_protocol,references,evaluate_retention

OUT=ROOT/'.research_local/autorl_r26_interrupted_yaw_diagnostic/run_01'


def load_readonly_model(reference,prereg,root=ROOT):
    # Validate trusted paths and full evidence before unpickling an artifact.
    if reference not in references(root)['models']:raise OpsError('R26 untrusted artifact reference')
    import torch
    from lidar_integration.ppo_formal_checkpoint import CheckpointManager
    from lidar_integration.ppo_training_runner import _weights_sha
    from .state_centered_policy import verify_centered_policy
    from .yaw_coverage import verify_yaw_temporal_configuration
    model=CheckpointManager(Path(root)/Path(reference['checkpoint_path']).parent,prereg,
        execution_protocol_sha256=reference['execution_protocol_sha256']).load_verified(None,run_id=reference['run_id'])
    architecture=verify_centered_policy(model);temporal=verify_yaw_temporal_configuration(model)
    fingerprint=_weights_sha(model.policy)
    if reference['expected_policy_sha256'] is not None and fingerprint!=reference['expected_policy_sha256']:
        raise OpsError('R26 loaded initial policy differs')
    def finite(value):
        if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
        return not isinstance(value,float) or math.isfinite(value)
    if (not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())
            or not finite(model.policy.optimizer.state_dict()) or model.num_timesteps!=reference['step']):
        raise OpsError('R26 loaded state nonfinite or wrong training step')
    return model,{'policy_sha256':fingerprint,'architecture':architecture,'temporal':temporal,
        'legacy_loader_mode':'WARM_START_NAME_DESERIALIZATION_ONLY_NO_TRAINING',
        'scope':'verified deserialization env=None;not experiment initialization,training or resume'}


def verify_checks(reg,root=ROOT):
    root=Path(root)
    try:
        a=read_json(root/'.research_local/autorl_r26_checks_01.json')
        if a['status']!='PASS' or a['registration_sha256']!=digest(reg) or a['source_identity_sha256']!=digest(reg['source_identity']):raise OpsError('R26 stale checks')
        values={}
        for key in ('operations','tensor','historical','runtime_binding'):
            ref=a['evidence'][key];path=root/ref['path']
            if ref['path']!=f'.research_local/autorl_r26_{key}_final_01.json' or sha256(path)!=ref['sha256']:raise OpsError('R26 test evidence changed')
            values[key]=read_json(path)
        op,te,hi,bi=(values[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or (op['tests_run'],op['failures'],op['errors'],op['optimizer_updates'])!=(231,0,0,0)
                or len(op['results'])!=231 or any(v['outcome']!='PASS' for v in op['results'])
                or op['source_identity_sha256']!=a['source_identity_sha256'] or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
                or te['returncode'] or te['tests_run']!=75 or te['optimizer_updates'] or 'Ran 75 tests' not in te['stderr']
                or te['source_identity_sha256']!=a['source_identity_sha256'] or te['registration_sha256']!=digest(reg)
                or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
                or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results'])
                or bi['returncode'] or bi['source_before_sha256']!=a['source_identity_sha256']
                or bi['source_after_sha256']!=a['source_identity_sha256'] or bi['registration_sha256']!=digest(reg)
                or bi['final']!=a['evidence']['historical']):raise OpsError('R26 incomplete current no-update regression')
        return sha256(root/'.research_local/autorl_r26_checks_01.json')
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid R26 no-update evidence') from error


def probe(reg,spec):
    import resource
    import torch
    from stable_baselines3 import PPO
    from lidar_integration.ppo_formal_scheduler import load_l6_prereg
    from .stopping_trace import collect_trace
    def forbidden(*args,**kwargs):raise AssertionError('NO_OPTIMIZER_OR_TRAINING_R26')
    PPO.learn=forbidden;PPO.train=forbidden;torch.optim.Adam.step=forbidden
    torch.set_num_threads(1);started=time.monotonic();cpu=time.process_time();minimum=float('inf')
    prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
    before={str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()}
    budget=BudgetStore(HOME).status();results=[];models=[]
    def guard():
        nonlocal minimum
        available=enforce_resources(ROOT);minimum=min(minimum,available['mem_available_mib'])
        if time.monotonic()-started>spec['wall_seconds']-5 or BudgetStore(HOME).status()['stop_requested']:
            raise OpsError('R26 absolute duration/stop protection')
        if source_identity(ROOT,reg['source_paths'])!=reg['source_identity']:raise OpsError('R26 runtime source changed')
    for i,reference in enumerate(spec['models']):
        guard();model,meta=load_readonly_model(reference,prereg);models.append({'reference':reference,**meta})
        for j,case in enumerate(spec['cases']):
            guard();path=OUT/f'trace_{i}_{j}.jsonl'
            with path.open('x',encoding='utf-8') as stream:
                def emit(record):stream.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n');stream.flush()
                result=collect_trace(model,case['scenario'],ROOT/'.research_local/dependencies/MuJoCo-LiDAR',
                    max_steps=spec['max_policy_steps_per_episode'],guard=guard,on_record=emit)
            if result['policy_sha256_before_after']!=meta['policy_sha256']:raise OpsError('R26 policy changed between paired cases')
            del result['trace']
            result.update(checkpoint_step=reference['step'],distance_stage=case['distance_stage'],
                trace_file={'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'bytes':path.stat().st_size})
            results.append(result);atomic_json(OUT/f'case_{i}_{j}.json',result,exclusive=True)
            if result['outcome'] not in ('success','collision','time_limit'):raise OpsError('R26 unsafe diagnostic terminal state')
        del model;gc.collect()
    assessment=evaluate_retention(results,spec)
    after={str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()}
    if before!=after or BudgetStore(HOME).status()!=budget or spec!=make_protocol(reg):raise OpsError('R26 changed budget/history/protocol')
    guard()
    result={'status':'PASS','experiment_id':spec['experiment_id'],'registration_sha256':digest(reg),
        'source_identity_sha256':digest(reg['source_identity']),'protocol_sha256':digest(spec),'models':models,'cases':results,'assessment':assessment,
        'training_timesteps':0,'optimizer_updates':0,'episodes':8,'validation_diagnostic_episodes':8,'test_episodes':0,
        'actual_policy_steps':sum(v['policy_steps'] for v in results),'physics_ticks':sum(v['physics_ticks'] for v in results),
        'authorization_files_unchanged':True,'authorization_files_sha256':before,'authorization_file_count':len(before),
        'budget_snapshot_sha256':digest(budget),'R24_original_scientific_result':'STILL_NOT_TESTED',
        'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,
        'worker_hwm_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'per_policy_guard_sampled_min_MemAvailable_mib':minimum,'claim_limit':spec['claim_limit']}
    atomic_json(OUT/'result.json',result,exclusive=True);atomic_json(OUT/'complete.json',{'result_sha256':sha256(OUT/'result.json')},exclusive=True)
    return {'status':'PASS','result_sha256':sha256(OUT/'result.json'),'episodes':8,'optimizer_updates':0}


def run():
    if os.getpgrp()!=os.getpid():raise OpsError('R26 needs isolated owned process group')
    from research_ops_v2.manager import OperationsV2
    m=OperationsV2(ROOT);parent=process_identity(os.getppid())
    with FileLock(m.lock_path):
        _,reg=verify_current()
        if reg['phase_id']!='AUTORL-R26-INTERRUPTED-YAW-DIAGNOSTIC':raise OpsError('wrong reviewed R26 phase')
        verify_checks(reg);enforce_resources(ROOT)
        spec=read_json(ROOT/'mujoco/reports/autonomous_rl_interrupted_yaw_r26_prereg.json')
        if spec!=make_protocol(reg):raise OpsError('R26 frozen protocol differs')
        from .lifetime import start_guard,stop_guard
        watcher=start_guard(parent,time.monotonic()+spec['wall_seconds'],ROOT)
        try:
            OUT.mkdir(parents=True,exist_ok=False)
            atomic_json(OUT/'attempt.json',{'experiment_id':spec['experiment_id'],'kind':spec['kind'],'owner':process_identity(),
                'parent':parent,'protocol_sha256':digest(spec),'source_identity_sha256':digest(reg['source_identity'])},exclusive=True)
            try:return probe(reg,spec)
            except BaseException as error:
                import traceback
                atomic_json(OUT/'failure.json',{'status':'FAIL','exception_type':type(error).__name__,'message':str(error)[:2000],
                    'traceback':traceback.format_exc()[-12000:],'partial_trace_files':[{'path':str(p.relative_to(ROOT)),
                    'sha256':sha256(p),'bytes':p.stat().st_size} for p in OUT.glob('trace_*.jsonl')],
                    'context':'new diagnostic partial evidence,not completed R24/no retry/no training'},exclusive=True)
                raise
        finally:stop_guard(watcher)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(),allow_nan=False))


if __name__=='__main__':main()
