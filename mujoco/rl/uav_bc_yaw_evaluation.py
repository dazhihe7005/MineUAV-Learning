"""Frozen four-controller exact-state evaluation; Test labels are posthoc only."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from latent_dynamics_data import file_hash,json_hash
from uav_bc_safety import atomic_json
from uav_bc_policy import load,parameter_hash
from uav_bc_yaw_data import MANIFEST,PARTS,MODEL_NAMES,MODEL_SHA
from uav_bc_robustness import (conditions,sample_perturbation,prepare_snapshot,evaluate_task,validate_cohort,
    validate_record,controller_summary,paired_summary,sustained_time)
from uav_bc_evaluation import action_metrics
from test_env_scripted_policy import scripted_action

CONTROLLERS=('scripted','original','nominal_expanded','yaw_augmented')

def extra_metrics(z):
    o=z['observations'][z['physical_state_valid']];d=np.linalg.norm(o[:,:3],axis=1);v=np.linalg.norm(o[:,3:6],axis=1)
    peak=int(np.argmax(v));settled=sustained_time(v[peak:],.15,.04)
    return dict(distance_rebound_overshoot_m=float(np.max(d-np.minimum.accumulate(d))),
        post_peak_velocity_recovery_s=peak*.04+settled if settled is not None else None)

def audit_task(env,snapshot,controller,state,name,path,identity):
    r=evaluate_task(env,snapshot,controller,state,name,path,identity)
    with np.load(r['trace_path'],allow_pickle=False) as z:extra=extra_metrics(z)
    if any(k in r and r[k]!=v for k,v in extra.items()):raise ValueError('derived trajectory metrics changed')
    if any(k not in r for k in extra):
        r.update(extra);r['record_sha256']=json_hash({k:v for k,v in r.items() if k!='record_sha256'});atomic_json(path,r)
    return r

def final_states(env,targets):
    out=[]
    for condition in conditions():
        for i,t in enumerate(targets):
            p=sample_perturbation(condition,'benchmark',i);_,meta=prepare_snapshot(env,t,p)
            out.append(dict(t,key=f'{condition}/final/{i:03d}',condition=condition,split='final',index=i,
                perturbation=p,initial_state=meta))
    return out

def offline_arrays(o,y,groups,steps,raw,clipped,std):
    def measure(mask,pred):return action_metrics(pred[mask],y[mask],std) if mask.any() else None
    d=np.linalg.norm(o[:,:3],axis=1)
    return dict(raw=action_metrics(raw,y,std),clipped=action_metrics(clipped,y,std),
        by_initial_yaw={str(g):dict(raw=measure(groups==g,raw),clipped=measure(groups==g,clipped)) for g in np.unique(groups)},
        early_large_yaw=measure((groups==30)&(steps<10),clipped),
        near_braking=measure((d<.2)&(steps>=10),clipped),
        raw_out_of_bounds_element_fraction=float((abs(raw)>1).mean()))

def summaries(rows):
    from latent_mpc_evaluation import distribution
    result={}
    for condition in conditions():
        by={n:[r for r in rows if r['condition']==condition and r['controller']==n] for n in CONTROLLERS}
        result[condition]={n:controller_summary(a) for n,a in by.items()}
        for n,a in by.items():
            for field in ('distance_rebound_overshoot_m','post_peak_velocity_recovery_s'):
                v=[r[field] for r in a if r[field] is not None]
                result[condition][n][field]=dict(measured_episodes=len(v),statistics=distribution(v) if v else None)
        result[condition]['paired']={f'{a}_minus_{b}':paired_summary(by[b],by[a])
            for a,b in [('yaw_augmented','nominal_expanded'),('yaw_augmented','original'),('nominal_expanded','original'),
                        ('original','scripted'),('nominal_expanded','scripted'),('yaw_augmented','scripted')]}
    return result

def run(root,smoke=False):
    from ppo_pi_env import MineUAVPIEnv
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS;models=root/'mujoco/rl/models';torch.set_num_threads(1)
    m=json.loads((reports/MANIFEST).read_text());training=json.loads((parts/'training.json').read_text())
    paths={'original':models/'uav_bc_mlp_seed0.pt',**{n:models/f for n,f in MODEL_NAMES.items()}}
    policies={n:load(p)[0] for n,p in paths.items()};hashes={n:file_hash(p) for n,p in paths.items()}
    parameters={n:parameter_hash(p.actor) for n,p in policies.items()}
    if hashes['original']!=MODEL_SHA:raise ValueError('Original BC changed')
    for n in MODEL_NAMES:
        if hashes[n]!=training[n]['checkpoint_sha256'] or json_hash(policies[n].stats)!=m['preflight']['normalization_sha256']:
            raise ValueError('wrong checkpoint selection/normalization')
    env=MineUAVPIEnv(reward_version='v2');rows=[]
    try:
        states=final_states(env,m['final_targets'])
        identity=dict(data_manifest_sha256=m['sha256'],model_sha256=hashes,parameters=parameters,
            source={n:file_hash(root/'mujoco/rl'/n) for n in ['uav_bc_yaw_evaluation.py','uav_bc_robustness.py','uav_bc_policy.py']})
        evaluation_manifest=dict(identity=identity,states=states);evaluation_manifest['sha256']=json_hash(evaluation_manifest)
        path=parts/'evaluation_manifest.json'
        if path.exists() and json.loads(path.read_text())!=evaluation_manifest:raise ValueError('frozen final cohort/source changed')
        atomic_json(path,evaluation_manifest)
        selected=[s for s in states if not smoke or s['index']<2]
        for i,s in enumerate(selected):
            snap,meta=prepare_snapshot(env,s,s['perturbation'])
            if meta!=s['initial_state']:raise ValueError('paired snapshot reconstruction mismatch')
            for n in CONTROLLERS:
                controller=scripted_action if n=='scripted' else policies[n].predict
                path=parts/f"closed_{s['condition']}_{s['index']:03d}_{n}.json"
                rows.append(audit_task(env,snap,controller,s,n,path,identity))
            if (i+1)%25==0 or i+1==len(selected):print('Yaw final states',i+1,'/',len(selected),'episodes',len(rows),flush=True)
        validate_cohort(rows,[s['key']+'/'+n for s in selected for n in CONTROLLERS])
    finally:env.close()
    after={n:parameter_hash(p.actor) for n,p in policies.items()}
    if after!=parameters or {n:file_hash(p) for n,p in paths.items()}!=hashes:raise ValueError('frozen policy mutated')
    result=dict(rows=rows,identity=identity,evaluation_manifest_sha256=evaluation_manifest['sha256'],
        frozen_hashes_before=hashes,frozen_hashes_after=hashes,parameter_hashes_before=parameters,parameter_hashes_after=after,
        all_frozen=all(not p.actor.training and all(not q.requires_grad for q in p.actor.parameters()) for p in policies.values()))
    if not smoke:
        result['metrics']=summaries(rows);xs=[];ys=[];gs=[];ts=[]
        for r in rows:
            if r['controller']!='scripted' or r['condition'] not in ('nominal','yaw_small','yaw_large'):continue
            with np.load(r['trace_path'],allow_pickle=False) as z:
                valid=z['physical_state_valid'][:-1];o=z['observations'][:-1][valid];y=z['actions'][valid]
                xs.append(o);ys.append(y);gs.append(np.full(len(o),{'nominal':0,'yaw_small':10,'yaw_large':30}[r['condition']]));ts.append(z['step_index'][valid])
        x,y,g,t=[np.concatenate(a) for a in (xs,ys,gs,ts)]
        result['offline_test']={n:offline_arrays(x,y,g,t,p.raw_actions(x),np.clip(p.raw_actions(x),-1,1),p.stats['action']['std']) for n,p in policies.items()}
        result['offline_test']['source']='Final unseen100-target Scripted nominal/yaw10/yaw30 real traces, only valid pre-action states; never used in training/selection'
    atomic_json(parts/('smoke.json' if smoke else 'evaluation.json'),result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run(Path(__file__).resolve().parents[2],a.smoke)
