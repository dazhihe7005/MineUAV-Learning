"""Serial, atomic evaluation and exact repeats. Never calls learn/backward/step optimizer."""
import json
import gc
from pathlib import Path
import numpy as np
import torch
from uav_ppo_bc_nominal import (PARTS,CONTROLLERS,MODELS,build_manifest,frozen_controller,initial_snapshot)
from uav_bc_policy import parameter_hash
from uav_bc_robustness import rollout,validate_record,validate_cohort
from uav_bc_safety import atomic_json,atomic_npz
from latent_dynamics_data import file_hash,json_hash

def motion_metrics(trace):
    o=trace['observations'];valid=trace['physical_state_valid'];p=trace['positions']
    pair=valid[:-1]&valid[1:]
    length=float(np.linalg.norm(np.diff(p,axis=0),axis=1)[pair].sum())
    d=np.linalg.norm(o[:,:3],axis=1);v=np.linalg.norm(o[:,3:6],axis=1)
    joint=valid&(d<.1)&(v<.15)
    # Quantities are policy-boundary samples; no fabricated continuous tracking.
    return dict(trajectory_length_m=length,trajectory_length_valid_segments=int(pair.sum()),
        final_velocity_xyz_m_s=o[-1,3:6].astype(float).tolist() if valid[-1] else None,
        ever_joint_threshold=bool(joint.any()),joint_threshold_samples=int(joint.sum()),
        maximum_success_streak=int(np.max(trace.get('success_streak',[0]))),
        sampled_minimum_distance_m=float(d[valid].min()),
        failed_after_entering_target=bool((d[valid]<.1).any()),
        peak_speed_after_first_target_entry_m_s=float(v[np.flatnonzero(valid&(d<.1))[0]:][valid[np.flatnonzero(valid&(d<.1))[0]:]].max()) if (valid&(d<.1)).any() else None)

def validate_execution(record,identity):
    payload={k:v for k,v in record.items() if k!='sha256'}
    if record.get('identity')!=identity or record.get('sha256')!=json_hash(payload) or not record.get('exact_arrays_equal'):
        raise ValueError('invalid or stale deterministic repeat')

def evaluate(env,snapshot,predict,state,name,path,identity):
    if path.exists():
        row=json.loads(path.read_text());validate_record(row,identity,state,name)
        with np.load(row['trace_path'],allow_pickle=False) as trace:
            if any(row.get(k)!=v for k,v in motion_metrics(trace).items()):raise ValueError('trace/metric inconsistency')
        return row,False
    row,trace=rollout(env,snapshot,predict);row.update(motion_metrics(trace))
    if row['initial_snapshot_sha256']!=state['initial_state']['snapshot_sha256']:raise ValueError('initial state mismatch')
    raw=path.with_suffix('.npz');atomic_npz(raw,**trace)
    row.update(identity=identity,key=state['key'],task_key=state['key']+'/'+name,controller=name,
        condition=state['condition'],split=state['split'],target_id=state['target_id'],env_seed=state['env_seed'],target=state['target'],
        trace_path=str(raw),trace_sha256=file_hash(raw))
    row['record_sha256']=json_hash(row);atomic_json(path,row)
    validate_record(row,identity,state,name)
    return row,True

def repeat_check(env,snap,predict,row,parts,name,identity):
    path=parts/f'repeat_{name}.json'
    if path.exists():
        value=json.loads(path.read_text());validate_execution(value,identity);return value,False
    again,trace=rollout(env,snap,predict);again.update(motion_metrics(trace))
    with np.load(row['trace_path'],allow_pickle=False) as original:
        if set(trace)!=set(original.files):raise ValueError('repeat trace schema changed')
        for k in trace:np.testing.assert_array_equal(trace[k],original[k],err_msg=f'non-deterministic {name}/{k}')
    if any(row.get(k)!=v for k,v in again.items()):raise ValueError('repeat termination/metrics changed')
    value=dict(identity=identity,controller=name,task_key=row['task_key'],kind='technical repeat not statistical sample',
        compared_trace_sha256=row['trace_sha256'],exact_arrays_equal=True,episode_executions=1)
    value['sha256']=json_hash(value);atomic_json(path,value)
    return value,True

def run(root):
    from ppo_pi_env import MineUAVPIEnv
    root=Path(root);parts=root/'mujoco/reports'/PARTS;torch.set_num_threads(1)
    manifest=build_manifest(root);env=MineUAVPIEnv(reward_version='v2');rows=[];frozen={};repeats={};new_executions=0
    identity=dict(manifest_sha256=manifest['sha256'],model_identity=manifest['identity']['models'])
    try:
        for name in CONTROLLERS:
            predict,module,meta=frozen_controller(root,name)
            if meta!=manifest['identity']['models'][name]:raise ValueError('frozen policy metadata changed')
            before=parameter_hash(module) if module is not None else None
            selected=[s for s in manifest['states'] if s['split']=='fresh' or name=='ppo7d']
            cohort=[]
            for i,state in enumerate(selected):
                snap,initial=initial_snapshot(env,state)
                if initial!=state['initial_state']:raise ValueError('snapshot not deterministic')
                path=parts/f"{state['split']}_{state['index']:03d}_{name}.json"
                row,is_new=evaluate(env,snap,predict,state,name,path,identity)
                new_executions+=int(is_new);rows.append(row);cohort.append(row)
                if i==0:
                    repeat,new_repeat=repeat_check(env,snap,predict,row,parts,name,identity)
                    repeats[name]=repeat;new_executions+=int(new_repeat)
                if (i+1)%25==0:print('nominal',name,i+1,'/',len(selected),'success',sum(x['success'] for x in cohort),flush=True)
            after=parameter_hash(module) if module is not None else None
            if before!=after:raise ValueError('policy parameters mutated')
            frozen[name]=dict(parameter_sha256_before=before,parameter_sha256_after=after,
                checkpoint_sha256_before=meta.get('checkpoint_sha256'),checkpoint_sha256_after=file_hash(root/meta['path']) if name!='scripted' else None,
                eval_mode=not module.training if module is not None else None,
                requires_grad_false=all(not p.requires_grad for p in module.parameters()) if module is not None else None)
            if name!='scripted' and frozen[name]['checkpoint_sha256_after']!=MODELS[name][1]:raise ValueError('checkpoint changed')
            del predict,module;gc.collect()
    finally:env.close()
    expected=[s['key']+'/'+n for s in manifest['states'] for n in (CONTROLLERS if s['split']=='fresh' else ['ppo7d'])]
    validate_cohort(rows,expected)
    for n,sha in manifest['identity']['source_sha256'].items():
        if file_hash(root/'mujoco'/n)!=sha:raise ValueError('immutable physics/controller/source changed')
    value=dict(manifest_sha256=manifest['sha256'],formal_episodes=len(rows),records=rows,frozen=frozen,
        deterministic_repeats=repeats,workers=1,new_episode_executions_this_invocation=new_executions,
        unique_experiment_executions=len(rows)+len(repeats),raw_traces_local_only=True)
    atomic_json(parts/'evaluation.json',value)
    print('complete formal',len(rows),'unique executions',value['unique_experiment_executions'],'new',new_executions,flush=True)
    return value

if __name__=='__main__':run(Path(__file__).resolve().parents[2])
