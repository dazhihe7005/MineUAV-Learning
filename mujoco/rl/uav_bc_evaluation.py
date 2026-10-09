"""Independent frozen BC execution; teacher is data/baseline/posthoc only."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from latent_dynamics_data import file_hash,json_hash
from latent_mpc_evaluation import crossings,distribution
from uav_bc_data import PARTS,MANIFEST,split_targets,identity,collect_episode,load_split,validate_record
from uav_bc_policy import load,parameter_hash
from uav_bc_training import MODEL
from uav_bc_safety import atomic_json,atomic_npz

EVALUATION='uav_bc_evaluation_manifest_seed0.json'
CANONICAL='42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9'

def correlation(x,y):
    if len(x)<2 or np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])

def action_metrics(predicted,expert,std):
    p,y=np.asarray(predicted,float),np.asarray(expert,float);s=np.asarray(std,float)
    if p.shape!=y.shape or p.ndim!=2 or p.shape[1]!=4 or not len(p) or not np.isfinite([p,y]).all() or (s<=0).any():
        raise ValueError('finite aligned nonempty 4D actions required')
    e=p-y
    return dict(samples=len(y),mse=float(np.mean(e**2)),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(np.abs(e))),
        normalized_mse=float(np.mean((e/s)**2)),normalized_rmse=float(np.sqrt(np.mean((e/s)**2))),
        per_dimension=dict(names=['vx_cmd','vy_cmd','vz_cmd','yaw_rate_cmd'],rmse=np.sqrt(np.mean(e**2,axis=0)).tolist(),
            mae=np.mean(np.abs(e),axis=0).tolist(),mse=np.mean(e**2,axis=0).tolist(),correlation=[correlation(p[:,i],y[:,i]) for i in range(4)]))

def region_metrics(p,y,obs,std,thresholds):
    d=np.linalg.norm(obs[:,:3],axis=1);v=np.linalg.norm(obs[:,3:6],axis=1);a=np.linalg.norm(y,axis=1)
    groups={'distance':dict(zip(['<0.1','0.1-0.2','0.2-0.5','>=0.5'],[d<.1,(d>=.1)&(d<.2),(d>=.2)&(d<.5),d>=.5]))}
    for key,values in [('speed',v),('action_magnitude',a)]:
        lo,hi=thresholds[key];groups[key]={'small':values<lo,'medium':(values>=lo)&(values<hi),'large':values>=hi}
    return {key:{name:action_metrics(p[mask],y[mask],std) if mask.any() else None for name,mask in bins.items()} for key,bins in groups.items()}

def trace_metrics(trace):
    o,a=trace['observations'],trace['actions'];d=np.linalg.norm(o[:,:3],axis=1);v=np.linalg.norm(o[:,3:6],axis=1)
    near=d[:-1]<.1
    return dict(near_steps=int(near.sum()),near_actual_sum=float(v[:-1][near].sum()),
        near_command_sum=float(np.linalg.norm(a[near,:3]*[1.5,1.5,1.],axis=1).sum()),
        action_saturation_elements=int((np.abs(a)>=.95).sum()),crossings=crossings(o),
        initial_distance_m=float(d[0]),mean_distance_m=float(d.mean()),
        distance_reduction_m=float(d[0]-d[-1]),minimum_distance_m=float(d.min()),
        mean_speed_m_s=float(v.mean()),action_mean=a.mean(0).tolist(),action_std=a.std(0).tolist())

def summarize(rows):
    if not rows:raise ValueError('all episodes must be counted')
    c=Counter(r['termination_reason'] for r in rows);n=len(rows);steps=sum(r['steps'] for r in rows);near=sum(r['near_steps'] for r in rows)
    successes=c['success'];times=[r['simulated_seconds'] for r in rows if r['success']]
    return dict(episodes=n,successes=successes,success_rate=successes/n,physical_failures=n-successes-c['time_limit'],
        timeouts=c['time_limit'],termination_reasons=dict(c),final_distance_m=distribution([r['final_distance_m'] for r in rows]),
        completion_time_s=distribution(times) if times else None,all_episode_duration_s=distribution([r['simulated_seconds'] for r in rows]),
        total_decisions=steps,near_target_steps=near,near_target_actual_speed_m_s=sum(r['near_actual_sum'] for r in rows)/near if near else None,
        near_target_command_speed_m_s=sum(r['near_command_sum'] for r in rows)/near if near else None,
        action_saturation_fraction=sum(r['action_saturation_elements'] for r in rows)/(4*steps),
        crossing_episode_fraction=float(np.mean([r['crossings']>0 for r in rows])),
        initial_distance_m=distribution([r['initial_distance_m'] for r in rows]),
        mean_distance_reduction_m=float(np.mean([r['distance_reduction_m'] for r in rows])),
        by_initial_distance={name:dict(episodes=sum(mask),successes=sum(r['success'] for r,m in zip(rows,mask) if m))
            for name,mask in [(name,[lo<=r['initial_distance_m']<hi for r in rows]) for name,lo,hi in
                [('0-0.5',0,.5),('0.5-1',.5,1),('1-2',1,2),('>=2',2,float('inf'))]]})

def closed_loop_identity(root,model_sha):
    root=Path(root)
    return dict(collection=identity(root),model_sha256=model_sha,
        source_sha256={n:file_hash(root/'mujoco/rl'/n) for n in ['uav_bc_policy.py','uav_bc_evaluation.py']},
        environment='unchanged PI RewardV2',controller_protocol='BC(obs) alone; scripted only separate baseline')

def evaluate_one(env,target,controller,path,ident,condition,task):
    path=Path(path)
    if path.exists():
        r=json.loads(path.read_text());validate_record(r,ident,target['target_id'])
        if any(r[k]!=v for k,v in dict(condition=condition,task=task,env_seed=target['env_seed'],target=target['target']).items()):
            raise ValueError('closed-loop cache mislabeled')
        return r
    row,arrays=collect_episode(env,target,controller);row.update(trace_metrics(arrays))
    raw=path.with_suffix('.npz');atomic_npz(raw,**arrays)
    row.update(identity=ident,condition=condition,task=task,path=str(raw),sha256=file_hash(raw));atomic_json(path,row)
    return row

def shift_summary(obs,stats,train_obs):
    rms=np.sqrt(np.mean(((obs-np.asarray(stats['obs']['mean']))/stats['obs']['std'])**2,axis=1))
    tr=np.sqrt(np.mean(((train_obs-np.asarray(stats['obs']['mean']))/stats['obs']['std'])**2,axis=1))
    lower,upper=np.quantile(train_obs,[.025,.975],axis=0)
    return dict(train_standardized_rms=distribution(tr),standardized_rms=distribution(rms),
        outside_train_central95_any_dimension_fraction=float(((obs<lower)|(obs>upper)).any(1).mean()),
        definition='marginal box departure is a proxy, not full state OOD; yaw statistics have near-zero nominal variance')

def run(root):
    from ppo_pi_env import MineUAVPIEnv
    from test_env_scripted_policy import scripted_action
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS;torch.set_num_threads(1)
    canonical=root/'mujoco/rl/models/joint_latent_world_model_v3_autonomous_consistency.pt'
    if file_hash(canonical)!=CANONICAL:raise ValueError('canonical World Model changed')
    model_path=root/'mujoco/rl/models'/MODEL;model_sha=file_hash(model_path);policy,meta=load(model_path);before=parameter_hash(policy.actor)
    manifest=json.loads((reports/MANIFEST).read_text());train=load_split(manifest,'train');test=load_split(manifest,'test')
    std=policy.stats['action']['std'];pred=policy.raw_actions(test['observations'])
    thresholds=dict(speed=np.quantile(np.linalg.norm(train['observations'][:,3:6],axis=1),[1/3,2/3]).tolist(),
        action_magnitude=np.quantile(np.linalg.norm(train['actions'],axis=1),[1/3,2/3]).tolist())
    offline=dict(raw=action_metrics(pred,test['actions'],std),deployed_clipped=action_metrics(np.clip(pred,-1,1),test['actions'],std),
        raw_out_of_bounds_element_fraction=float((np.abs(pred)>1).mean()),regions=region_metrics(pred,test['actions'],test['observations'],std,thresholds),
        thresholds=thresholds,threshold_provenance='Train only',test_target_ids=np.unique(test['target_id']).tolist())
    _,targets,target_meta=split_targets(root);ident=closed_loop_identity(root,model_sha);records={};summaries={};shifts={}
    env=MineUAVPIEnv(reward_version='v2')
    try:
        for condition,controller in [('scripted',scripted_action),('bc',policy.predict)]:
            records[condition]={};summaries[condition]={};shifts[condition]={}
            for task,entries in targets.items():
                rows=[];observations=[];raw_clipped=0;decisions=0;posthoc_squared=0.
                for i,(seed,target) in enumerate(entries):
                    t=dict(target_id=i,env_seed=int(seed),target=list(target))
                    row=evaluate_one(env,t,controller,parts/f'closed_{condition}_{task}_{i:03d}.json',ident,condition,task)
                    rows.append(row)
                    with np.load(row['path'],allow_pickle=False) as trace:
                        x=trace['observations'][:-1].copy();observations.append(x)
                        if condition=='bc':
                            raw=policy.raw_actions(x);raw_clipped+=int((np.abs(raw)>1).sum());decisions+=len(x)
                            # POSTHOC labels only; never used in collect_episode or executed output.
                            expert=np.asarray([scripted_action(o) for o in x]);posthoc_squared+=float(np.sum((np.clip(raw,-1,1)-expert)**2))
                    if (i+1)%25==0:print('BC closed',condition,task,i+1,'success',sum(r['success'] for r in rows),flush=True)
                if len(rows)!=100 or len({r['target_id'] for r in rows})!=100:raise ValueError('duplicate/omitted closed-loop episodes')
                records[condition][task]=rows;summaries[condition][task]=summarize(rows)
                shifts[condition][task]=shift_summary(np.concatenate(observations),policy.stats,train['observations'])
                if condition=='bc':
                    summaries[condition][task].update(raw_action_clipping_element_fraction=raw_clipped/(4*decisions),
                        posthoc_visited_state_expert_action_rmse=float(np.sqrt(posthoc_squared/(4*decisions))))
    finally:env.close()
    after=parameter_hash(policy.actor)
    if before!=after or file_hash(model_path)!=model_sha or file_hash(canonical)!=CANONICAL:raise ValueError('frozen checkpoint mutated')
    result=dict(offline=offline,closed_loop=summaries,state_distribution=shifts,records=records,evaluation_targets=target_meta,
        frozen_bc=dict(checkpoint_sha256=model_sha,parameter_hash_before=before,parameter_hash_after=after,all_parameters_frozen=True),
        canonical_world_model=dict(before=CANONICAL,after=file_hash(canonical),used=False),checkpoint_metadata=meta)
    atomic_json(reports/EVALUATION,dict(records=records,identity=ident,targets=target_meta))
    atomic_json(parts/'evaluation.json',result);return result

if __name__=='__main__':run(Path(__file__).resolve().parents[2])
