"""One isolated, bounded, inference-only audit; fixed inputs/paths, no train mode."""
import argparse
import gc
import os
import time
from research_ops.storage import FileLock,OpsError,atomic_json,read_json,digest,sha256,process_identity
from research_ops.environment import enforce_resources
from .runtime import ROOT,HOME,verify_current
from .budget import BudgetStore
from .feedback_spec import checkpoint_references,probe_cases,verify_probe_checks

OUT=ROOT/'.research_local/autorl_r4_feedback/run_01'


def run():
    parent=process_identity(os.getppid())
    if os.getpgrp()!=os.getpid():raise OpsError('feedback audit needs its own isolated process group')
    from research_ops_v2.manager import OperationsV2
    manager=OperationsV2(ROOT)
    with FileLock(manager.lock_path):
        _,reg=verify_current()
        if reg['phase_id']!='AUTORL-R4-FEEDBACK':raise OpsError('wrong registered feedback phase')
        verify_probe_checks(reg);enforce_resources(ROOT)
        budget=BudgetStore(HOME).status()
        if budget['stop_requested'] or any(a['status']!='COMPLETE' for a in budget['attempts']):
            raise OpsError('stop or unresolved training attempt; audit first')
        refs=checkpoint_references();spec=read_json(ROOT/'mujoco/reports/autonomous_rl_feedback_r4_prereg.json')
        if (spec['registration_sha256']!=digest(reg) or spec['source_identity_sha256']!=digest(reg['source_identity'])
            or spec['cases']!=probe_cases() or spec['models']!=refs):raise OpsError('frozen feedback protocol differs')
        from .lifetime import start_guard,stop_guard
        watcher=start_guard(parent,time.monotonic()+120,ROOT)
        try:
            OUT.mkdir(parents=True,exist_ok=False)
            atomic_json(OUT/'attempt.json',{'kind':'INFERENCE_ONLY_NOT_TRAINING','source_identity_sha256':digest(reg['source_identity']),
                'registration_sha256':digest(reg),'protocol_sha256':digest(spec),'owner':process_identity(),'parent':parent},exclusive=True)
            try:return probe(reg,budget,refs,spec)
            except BaseException as e:
                atomic_json(OUT/'failure.json',{'status':'FAIL','exception_type':type(e).__name__,'message':str(e)[:2000],
                    'context':'inference-only audit; no complete result presumed'},exclusive=True)
                raise
        finally:stop_guard(watcher)


def probe(reg,budget,refs,spec):
    import numpy as np
    import torch
    import resource
    from stable_baselines3 import PPO
    from lidar_integration.ppo_formal_scheduler import load_l6_prereg,MultiScenarioTrainingEnv
    from lidar_integration.ppo_formal_checkpoint import CheckpointManager
    from lidar_integration.ppo_training_runner import _weights_sha
    from research_ops.identity import source_identity
    from .feedback_probe import infer_panel,seal_snapshot,verify_snapshot
    from test_env_scripted_policy import scripted_action
    def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_OR_ROLLOUT_FEEDBACK_AUDIT')
    PPO.learn=forbidden;PPO.train=forbidden;torch.optim.Adam.step=forbidden
    torch.set_num_threads(1);begun=time.monotonic()
    prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
    before_files={str(p.relative_to(HOME)):sha256(p) for p in [HOME/'grant.json',HOME/'head.json',*sorted((HOME/'events').iterdir())]}
    env=MultiScenarioTrainingEnv(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0)
    try:
        row=prereg['scenario_manifest']['train'][0]
        obs,info=env.reset(options={'scenario_identity_sha256':row['identity_sha256']})
        plant=env.active.unwrapped
        if (plant.data.time!=0 or plant.episode_steps!=0 or plant.lidar_frame_count!=1
            or info['scenario_split']!='train'):raise OpsError('snapshot advanced physics or used wrong split')
        snapshot={'kind':'REAL_MUJOCO_TRAIN_RESET_SNAPSHOT_NOT_EPISODE','scenario_identity_sha256':row['identity_sha256'],
            'source':'SIM_GROUND_TRUTH_STATE','simulation_time':float(plant.data.time),'frame_timestamp':plant.last_frame.frame_timestamp,
            'clock_domain':plant.last_frame.clock_domain,'ray_count':plant.last_frame.ray_count,
            'qpos':plant.data.qpos.tolist(),'qvel':plant.data.qvel.tolist(),'ctrl':plant.data.ctrl.tolist(),
            'state':obs['state'].tolist(),'heading':obs['heading'].tolist(),'lidar_age':obs['lidar_age'].tolist()}
        plant.last_frame.save_npz(OUT/'sensor_frame.npz')
        np.savez_compressed(OUT/'base_observation.npz',**obs)
        obs={k:v.copy() for k,v in obs.items()}
        atomic_json(OUT/'snapshot.json',snapshot,exclusive=True)
        sealed=seal_snapshot(OUT,obs)
    finally:env.close()
    if env.active_environment_count:raise OpsError('snapshot environment did not close')
    del plant,env;gc.collect()
    cases=probe_cases();results=[];minimum=float('inf')
    for ref in refs:
        resources=enforce_resources(ROOT);minimum=min(minimum,resources['mem_available_mib'])
        if time.monotonic()-begun>110 or BudgetStore(HOME).status()['stop_requested']:raise OpsError('bounded feedback audit stopped')
        model=CheckpointManager(ROOT/ref['checkpoint_path'].rsplit('/',1)[0],prereg,
            execution_protocol_sha256=ref['protocol_sha256']).load_verified(None,run_id=ref['run_id'])
        if _weights_sha(model.policy)!=ref['policy_sha256']:raise OpsError('feedback policy weight identity differs')
        result=infer_panel(model,obs,cases);results.append({'reference':ref,**result})
        del model;gc.collect()
    reference=[{'case_id':c['case_id'],'normalized_action':scripted_action(np.asarray(c['state'])).tolist(),
        'role':'STATE_ONLY_SCRIPTED_REFERENCE_NOT_AVOIDANCE_EXPERT'} for c in cases]
    after_files={str(p.relative_to(HOME)):sha256(p) for p in [HOME/'grant.json',HOME/'head.json',*sorted((HOME/'events').iterdir())]}
    if (BudgetStore(HOME).status()!=budget or before_files!=after_files
        or source_identity(ROOT,reg['source_paths'])!=reg['source_identity']
        or checkpoint_references()!=refs):raise OpsError('feedback audit changed budget/source/model evidence')
    from research_ops_v2.manager import OperationsV2
    history=OperationsV2(ROOT).verify()['history']
    verify_snapshot(OUT,obs,sealed)
    result={'status':'PASS','protocol_sha256':digest(spec),'registration_sha256':digest(reg),
        'source_identity_sha256':digest(reg['source_identity']),'snapshot':snapshot,'snapshot_files':sealed,
        'case_count':len(cases),'models':results,'scripted_reference':reference,
        'training_timesteps':0,'optimizer_updates':0,'physics_steps':0,'episodes':0,'validation_episodes':0,'test_episodes':0,
        'budget_unchanged':True,'budget_files_sha256':before_files,'budget':{k:v for k,v in budget.items() if k!='attempts'},
        'history':history,'wall_seconds':time.monotonic()-begun,
        'worker_ru_maxrss_mib_including_model_loads':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'min_sampled_mem_available_mib':minimum,'claim_limit':'fixed counterfactual inputs, not physical trajectory/navigation success'}
    atomic_json(OUT/'result.json',result,exclusive=True)
    atomic_json(OUT/'complete.json',{'result_sha256':sha256(OUT/'result.json')},exclusive=True)
    return {'status':'PASS','result_sha256':sha256(OUT/'result.json'),'optimizer_updates':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    import json
    print(json.dumps(run(),allow_nan=False))


if __name__=='__main__':main()
