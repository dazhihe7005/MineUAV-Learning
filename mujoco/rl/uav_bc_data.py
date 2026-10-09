"""Actual scripted commands, immutable target grouping, atomic episode resume."""
import json,platform
from pathlib import Path
import numpy as np
import torch,mujoco
from latent_dynamics_data import file_hash,json_hash
from latent_mpc_evaluation import load_targets
from test_env_scripted_policy import scripted_action
from uav_bc_safety import atomic_json,atomic_npz,check_running,available_memory

PARTS='uav_behavior_cloning_seed0_parts'
MANIFEST='uav_bc_dataset_manifest_seed0.json'
SPLITS='uav_bc_target_splits_seed0.json'

def check_splits(groups,evaluation):
    ids=[];coordinates=[]
    for split,rows in groups.items():
        for r in rows:
            if r['target_id'] in ids:raise ValueError('target ID reused across split')
            value=np.asarray(r['target'],float)
            if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('invalid target')
            if any(np.linalg.norm(value-x)<1e-8 for x in coordinates):raise ValueError('target coordinate overlap')
            ids.append(r['target_id']);coordinates.append(value)
    for rows in evaluation.values():
        for _,target in rows:
            if any(np.linalg.norm(np.asarray(target)-x)<1e-8 for x in coordinates):raise ValueError('fixed evaluation overlaps BC dataset')

def split_targets(root):
    root=Path(root);source=root/'mujoco/rl/datasets/pi_hidden_state_seed0/manifest.json'
    old=json.loads(source.read_text());groups={s:[dict(target_id=int(i),env_seed=2026100900+int(i),target=old['targets'][i])
        for i in old['target_splits'][s]] for s in ('train','val','test')}
    evaluation,meta=load_targets(root);check_splits(groups,evaluation)
    return groups,evaluation,dict(source=str(source),source_sha256=file_hash(source),original_split=old['target_splits'],
        overlap_count=0,evaluation=meta,target_sha256=json_hash(groups))

def identity(root):
    root=Path(root);groups,_,meta=split_targets(root)
    names=['rl/uav_bc_data.py','rl/test_env_scripted_policy.py','rl/mine_uav_env.py','rl/ppo_pi_env.py','rl/audit_velocity_pi.py',
        'control/velocity_command_controller.py','control/hover_controller.py','control/position_controller.py',
        'control/control_allocator.py','models/mine_uav_dynamics_v2.xml','models/rotor_actuators.xml',
        'models/rotor_config.json','reports/dynamics_v2_report.json']
    return dict(target_sha256=meta['target_sha256'],runtime=dict(python=platform.python_version(),mujoco=mujoco.__version__,
        numpy=np.__version__,torch=str(torch.__version__)),source_sha256={n:file_hash(root/'mujoco'/n) for n in names},
        expert='existing scripted_action(7D obs), no privileged state',environment='unchanged MineUAVPIEnv RewardV2 full',workers=1)

def collect_episode(env,target,controller):
    obs,info=env.reset(seed=target['env_seed'],options={'target_position':target['target']})
    np.testing.assert_array_equal(info['target_position_m'],target['target'])
    observations=[obs.copy()];actions=[];previous=[];prior=np.zeros(4,np.float32);reasons=[]
    for step in range(env.max_episode_steps):
        if step%25==0:check_running(available_memory(),0,0)
        action=np.asarray(controller(obs),np.float32)
        if action.shape!=(4,) or not np.isfinite(action).all() or (np.abs(action)>1).any():raise ValueError('invalid policy command')
        previous.append(prior.copy());actions.append(action.copy())
        obs,_,terminated,truncated,info=env.step(action)
        np.testing.assert_array_equal(info['velocity_command_m_s'],action[:3].astype(float)*[1.5,1.5,1.])
        observations.append(obs.copy());prior=action.copy()
        if terminated or truncated:break
    else:raise RuntimeError('task did not terminate at unchanged timeout')
    reason=info['termination_reason'];n=len(actions)
    row=dict(target_id=target['target_id'],env_seed=target['env_seed'],episode_id=target['env_seed'],target=target['target'],
        steps=n,success=reason=='success',failure=reason!='success',termination_reason=reason,
        final_distance_m=float(info['distance_m']),final_speed_m_s=float(info['speed_m_s']) if np.isfinite(info['speed_m_s']) else None,
        simulated_seconds=float(env.data.time))
    arrays=dict(observations=np.asarray(observations,np.float32),actions=np.asarray(actions,np.float32),
        previous_actions=np.asarray(previous,np.float32),step_index=np.arange(n,dtype=np.int64),
        target_id=np.full(n,target['target_id'],np.int64),episode_id=np.full(n,target['env_seed'],np.int64),
        success=np.full(n,row['success'],bool),failure=np.full(n,row['failure'],bool))
    return row,arrays

def validate_record(record,expected,target_id):
    if record['identity']!=expected or record['target_id']!=target_id or file_hash(record['path'])!=record['sha256']:
        raise ValueError('stale/mislabeled/corrupted episode cache; never silently reuse')

def collect(root):
    from ppo_pi_env import MineUAVPIEnv
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS;groups,_,meta=split_targets(root);ident=identity(root)
    atomic_json(reports/SPLITS,dict(groups=groups,provenance=meta,sha256=json_hash(groups)))
    records={s:[] for s in groups};env=MineUAVPIEnv(reward_version='v2')
    try:
        for split,targets in groups.items():
            for t in targets:
                path=parts/f'expert_{split}_{t["target_id"]:03d}.json'
                if path.exists():
                    r=json.loads(path.read_text());validate_record(r,ident,t['target_id'])
                else:
                    row,arrays=collect_episode(env,t,scripted_action);raw=path.with_suffix('.npz');atomic_npz(raw,**arrays)
                    r=dict(row,identity=ident,path=str(raw),sha256=file_hash(raw),split=split);atomic_json(path,r)
                if r['split']!=split or r['env_seed']!=t['env_seed'] or r['target']!=t['target']:raise ValueError('cache split/seed/target mismatch')
                records[split].append(r)
            print('BC expert',split,len(records[split]),'episodes',sum(x['steps'] for x in records[split]),'steps',flush=True)
    finally:env.close()
    if {r['target_id'] for rows in records.values() for r in rows}!={r['target_id'] for rows in groups.values() for r in rows}:raise ValueError('omitted/duplicate target')
    result=dict(identity=ident,records=records,split_manifest=dict(path=str(reports/SPLITS),sha256=file_hash(reports/SPLITS)),
        counts={s:dict(episodes=len(rows),steps=sum(r['steps'] for r in rows),successes=sum(r['success'] for r in rows)) for s,rows in records.items()})
    atomic_json(reports/MANIFEST,result);return result

def load_split(manifest,split):
    records=manifest['records'][split];values={k:[] for k in ('observations','actions','target_id','step_index')}
    for r in records:
        validate_record(r,manifest['identity'],r['target_id'])
        with np.load(r['path'],allow_pickle=False) as a:
            n=r['steps']
            if len(a['observations'])!=n+1 or a['actions'].shape!=(n,4):raise ValueError('observation/action time mismatch')
            np.testing.assert_array_equal(a['previous_actions'][0],np.zeros(4))
            np.testing.assert_array_equal(a['previous_actions'][1:],a['actions'][:-1])
            np.testing.assert_array_equal(a['step_index'],np.arange(n))
            for k in values:values[k].append(a[k][:-1].copy() if k=='observations' else a[k].copy())
    return {k:np.concatenate(v) for k,v in values.items()}

if __name__=='__main__':collect(Path(__file__).resolve().parents[2])
