"""One R12 fixed-checkpoint inference audit. No user-selected code or training."""
import argparse,gc,os,time
from research_ops.storage import FileLock,OpsError,atomic_json,read_json,digest,sha256,process_identity
from research_ops.environment import enforce_resources
from .runtime import ROOT,HOME,verify_current
from .budget import BudgetStore
from .feedback_followup import configuration,references,make_protocol,verify_checks,load_background,validate_background

OUT=ROOT/'.research_local/autorl_r12_feedback/run_01'


def run():
    if os.getpgrp()!=os.getpid():raise OpsError('R12 requires isolated owned process group')
    from research_ops_v2.manager import OperationsV2
    m=OperationsV2(ROOT);parent=process_identity(os.getppid())
    with FileLock(m.lock_path):
        _,reg=verify_current()
        if reg['phase_id']!='AUTORL-R12-LOW-KL-FEEDBACK':raise OpsError('wrong R12 registered phase')
        verify_checks(reg);enforce_resources(ROOT)
        spec=read_json(ROOT/'mujoco/reports/autonomous_rl_low_kl_feedback_r12_prereg.json')
        if spec!=make_protocol(reg):raise OpsError('R12 protocol/source/budget/history changed')
        from .lifetime import start_guard,stop_guard
        watcher=start_guard(parent,time.monotonic()+configuration()['wall_seconds'],ROOT)
        try:
            OUT.mkdir(parents=True,exist_ok=False)
            atomic_json(OUT/'attempt.json',{'experiment_id':spec['experiment_id'],'kind':spec['kind'],
                'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity']),
                'protocol_sha256':digest(spec),'owner':process_identity(),'parent':parent},exclusive=True)
            try:return probe(reg,spec)
            except BaseException as e:
                atomic_json(OUT/'failure.json',{'status':'FAIL','exception_type':type(e).__name__,'message':str(e)[:2000],
                    'context':'inference-only; no complete result presumed; no retry'},exclusive=True)
                raise
        finally:stop_guard(watcher)


def probe(reg,spec):
    import numpy as np
    import torch,resource
    from stable_baselines3 import PPO
    from lidar_integration.ppo_formal_scheduler import load_l6_prereg
    from lidar_integration.ppo_formal_checkpoint import CheckpointManager
    from research_ops.identity import source_identity
    from .feedback_probe import infer_panel,verify_snapshot
    from .feedback_followup import BACKGROUND_REL
    def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_OR_ROLLOUT_R12')
    PPO.learn=forbidden;PPO.train=forbidden;torch.optim.Adam.step=forbidden
    torch.set_num_threads(1);begun=time.monotonic();cpu=time.process_time()
    prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
    budget=BudgetStore(HOME).status()
    before={str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()}
    obs,seal,prior,scripted=load_background();results=[];minimum=float('inf')
    for ref in spec['models']:
        available=enforce_resources(ROOT);minimum=min(minimum,available['mem_available_mib'])
        if time.monotonic()-begun>100 or BudgetStore(HOME).status()['stop_requested']:raise OpsError('R12 bounded/stop protection')
        model=CheckpointManager(ROOT/ref['checkpoint_path'].rsplit('/',1)[0],prereg,
            execution_protocol_sha256=ref['protocol_sha256']).load_verified(None,run_id=ref['run_id'])
        if ref['experiment_id']=='ARL_R11_PRIOR_LOWER_LR_SEED0':
            from .lower_lr_training import verify_effective_lr
            verify_effective_lr(model)
        result=infer_panel(model,obs,spec['cases'])
        if result['policy_sha256']!=ref['policy_sha256']:raise OpsError('R12 actual loaded policy differs')
        results.append({'reference':ref,**result});del model;gc.collect()
    after={str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()}
    validate_background(obs);verify_snapshot(ROOT/BACKGROUND_REL,obs,seal)
    if (before!=after or BudgetStore(HOME).status()!=budget or spec!=make_protocol(reg)
        or source_identity(ROOT,reg['source_paths'])!=reg['source_identity']):raise OpsError('R12 changed source/history/budget/model evidence')
    from research_ops_v2.manager import OperationsV2
    history=OperationsV2(ROOT).verify()['history']
    result={'status':'PASS','experiment_id':spec['experiment_id'],'registration_sha256':digest(reg),
        'source_identity_sha256':digest(reg['source_identity']),'protocol_sha256':digest(spec),
        'case_count':62,'new_model_count':4,'new_inference_rows':248,'models':results,
        'snapshot_files':seal,'snapshot_usage':'HISTORICAL_R4_REAL_MUJOCO_RESET_REUSED_NO_NEW_SENSOR_SAMPLE',
        'historical_prior_baseline':prior,'scripted_reference':scripted,'training_timesteps':0,'optimizer_updates':0,
        'physics_steps':0,'new_scans':0,'episodes':0,'validation_episodes':0,'test_episodes':0,
        'all_authorization_files_unchanged':True,'authorization_file_count':len(before),'authorization_files_sha256':before,
        'budget_snapshot_sha256':digest(budget),'history':history,'wall_seconds':time.monotonic()-begun,
        'cpu_seconds':time.process_time()-cpu,'worker_hwm_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        'min_sampled_mem_available_mib':minimum,'rng_scope':'within each infer_panel only; checkpoint construction/load may consume local RNG',
        'claim_limit':configuration()['claim_limit']}
    atomic_json(OUT/'result.json',result,exclusive=True)
    atomic_json(OUT/'complete.json',{'result_sha256':sha256(OUT/'result.json')},exclusive=True)
    return {'status':'PASS','result_sha256':sha256(OUT/'result.json'),'optimizer_updates':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    import json
    print(json.dumps(run(),allow_nan=False))


if __name__=='__main__':main()
