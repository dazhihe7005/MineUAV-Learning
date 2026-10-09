"""Controlled real expert coverage; immutable split, physical pairs and unique samples."""
import json,math
from pathlib import Path
from collections import Counter
import numpy as np
from latent_dynamics_data import file_hash,json_hash
from uav_bc_safety import atomic_json,atomic_npz
from uav_bc_robustness import prepare_snapshot,rollout,manifest_identity,conditions,MODEL_SHA
from uav_bc_policy import load
from test_env_scripted_policy import scripted_action

BASE='b05cf751cb776f2a17d21b19f4c4e629de9a8775'
SEED=2026101101
PARTS='uav_bc_yaw_coverage_seed0_parts'
MANIFEST='uav_bc_yaw_coverage_manifest_seed0.json'
MODEL_NAMES={'nominal_expanded':'uav_bc_nominal_expanded_seed0.pt','yaw_augmented':'uav_bc_yaw_augmented_seed0.pt'}
PHASES={'early':.10,'near_braking':.30,'high_speed':.25,'approach_other':.35}

def expert_tasks(targets,split,source):
    rows=[];sid=int(split=='val')
    for i,t in enumerate(targets):
        for rep in range(10):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,sid,i,rep]))
            p=rng.normal(size=3);v=rng.normal(size=3)
            p=p/np.linalg.norm(p)*rng.uniform(0,.10);v=v/np.linalg.norm(v)*rng.uniform(0,.15)
            group=0 if rep<4 else 10 if rep<7 else 30
            sign=0 if group==0 else (1 if (i+rep)%2==0 else -1)
            yaw=0. if source=='nominal_expanded' else math.radians(group*sign)
            rows.append(dict(key=f'{split}/{source}/{i:03d}/{rep:02d}',split=split,source=source,
                target_id=t['target_id'],target=t['target'],env_seed=950110000+sid*10000+i*10+rep,
                replicate=rep,yaw_group=group,yaw_sign=sign,initial_yaw_degrees=math.degrees(yaw),
                perturbation=dict(position_m=p.tolist(),velocity_m_s=v.tolist(),yaw_rad=yaw)))
    return rows

def assert_isolated(fresh,old):
    ids=set();seeds=set();points=[np.asarray(t['target'],float) for t in old]
    oldids={t['target_id'] for t in old};oldseeds={t['env_seed'] for t in old}
    for t in fresh:
        p=np.asarray(t['target'],float)
        if t['target_id'] in ids|oldids or t['env_seed'] in seeds|oldseeds or any(np.linalg.norm(p-q)<1e-8 for q in points):
            raise ValueError('target/reset identity overlap')
        if p.shape!=(3,) or not np.isfinite(p).all() or (abs(p[:2])>2).any() or not .7<=p[2]<=1.5:raise ValueError('illegal task target')
        ids.add(t['target_id']);seeds.add(t['env_seed']);points.append(p)

def final_targets(old):
    rng=np.random.default_rng(SEED)
    rows=[dict(target_id=f'yaw-final-{SEED}-{i:03d}',env_seed=950130000+i,
        target=np.r_[rng.uniform(-2,2,2),rng.uniform(.7,1.5)].tolist()) for i in range(100)]
    assert_isolated(rows,old);return rows

def exclusion_targets(bc,adapt):
    old=[t for rows in bc['groups'].values() for t in rows]+[t for rows in adapt['splits'].values() for t in rows]
    for split,v in bc['provenance']['evaluation'].items():
        old.extend(dict(t,target_id=f'old-bc-{split}-{i:03d}') for i,t in enumerate(v['targets']))
    return old

def phase_labels(obs,indices):
    o=np.asarray(obs);d=np.linalg.norm(o[:,:3],axis=1);v=np.linalg.norm(o[:,3:6],axis=1)
    return np.where(np.asarray(indices)<10,'early',np.where(d<.2,'near_braking',np.where(v>=.5,'high_speed','approach_other')))

def choose_samples(pool,total,seed):
    if len({r['key'] for r in pool})!=len(pool):raise ValueError('duplicate sample identity')
    rng=np.random.default_rng(seed);chosen=[]
    for phase,weight in PHASES.items():
        n=int(round(total*weight))
        for group,w in [(0,.4),(10,.3),(30,.3)]:
            count=int(round(n*w));signs=[0] if group==0 else [-1,1]
            quotas=[count] if group==0 else [count//2,count-count//2]
            for sign,q in zip(signs,quotas):
                candidates=[r for r in pool if (r['phase'],r['yaw_group'],r['yaw_sign'])==(phase,group,sign)]
                if len(candidates)<q:raise ValueError(f'fixed dataset quota shortage {phase}/{group}/{sign}: {len(candidates)} < {q}')
                chosen.extend(candidates[int(j)] for j in rng.choice(len(candidates),q,replace=False))
    if len(chosen)!=total:raise ValueError('nonintegral fixed quota')
    return [chosen[int(i)] for i in rng.permutation(total)]

def validate_expert_trace(a):
    obs=a['observations'];act=a['actions'];n=len(act);valid=a['physical_state_valid'][:-1]
    if obs.shape!=(n+1,7) or act.shape!=(n,4) or not np.isfinite(act).all():raise ValueError('expert time/action shape')
    if not np.array_equal(a['step_index'],np.arange(n)) or not np.array_equal(a['previous_actions'][1:],act[:-1]) or np.any(a['previous_actions'][0]):
        raise ValueError('previous action/step indexing')
    labels=np.asarray([scripted_action(o) for o in obs[:-1][valid]])
    if not np.array_equal(act[valid],labels):raise ValueError('expert label mismatch / teacher contamination')
    if (abs(obs[:-1][valid,6])>np.pi+1e-6).any():raise ValueError('unwrapped yaw label')

def check_record(r,identity,key):
    payload={k:v for k,v in r.items() if k!='record_sha256'}
    if r['key']!=key or r['identity']!=identity or json_hash(payload)!=r['record_sha256'] or file_hash(r['trace_path'])!=r['trace_sha256']:
        raise ValueError('stale/corrupt/mislabeled expert cache')

def preflight(root):
    from uav_bc_data import load_split
    root=Path(root);m=json.loads((root/'mujoco/reports/uav_bc_dataset_manifest_seed0.json').read_text())
    a=load_split(m,'train');yaw=a['observations'][:,6];policy,_=load(root/'mujoco/rl/models/uav_bc_mlp_seed0.pt')
    if not np.array_equal(a['actions'],np.array([scripted_action(o) for o in a['observations']])):raise ValueError('original expert label error')
    return dict(original_train_samples=len(yaw),yaw_mean_rad=float(yaw.mean()),yaw_std_rad=float(yaw.std()),
        yaw_min_rad=float(yaw.min()),yaw_max_rad=float(yaw.max()),yaw_abs_quantiles_rad=np.quantile(abs(yaw),[.5,.95,.99,1]).tolist(),
        yaw_abs_max_degrees=float(np.rad2deg(abs(yaw).max())),ten_thirty_in_train_std=(np.deg2rad([10,30])/policy.stats['obs']['std'][6]).tolist(),
        observation_names=['world_target_error_x','world_target_error_y','world_target_error_z','world_vx','world_vy','world_vz','wrapped_target_minus_actual_yaw'],
        action_frame='normalized world velocity xyz ×[1.5,1.5,1.0]m/s and world yaw-rate ×1rad/s; NOT body velocity or motor commands',
        yaw_wrap='atan2(sin(target_yaw - yaw_from_wxyz), cos(...))',
        expert='translation clip(error/3) independent of yaw; yaw_rate clip(yaw_error), same observation/no privileged input',
        coupling='world velocity PI creates desired world acceleration/thrust; yaw-dependent desired attitude and rotor allocation change real transient dynamics',
        information_gap='no missing inputs relative to expert; 7D is not full physical/controller Markov state',
        stats=policy.stats,normalization_sha256=json_hash(policy.stats),interface_error_found=False)

def make_manifest(root):
    from ppo_pi_env import MineUAVPIEnv
    root=Path(root);reports=root/'mujoco/reports';bc=json.loads((reports/'uav_bc_target_splits_seed0.json').read_text())
    adapt=json.loads((reports/'onpolicy_adaptation_target_splits_seed0.json').read_text())
    old=exclusion_targets(bc,adapt)
    final=final_targets(old);states=[]
    for source in MODEL_NAMES:states+=expert_tasks(bc['groups']['train'],'train',source)
    states+=expert_tasks(bc['groups']['val'],'val','shared_validation')
    env=MineUAVPIEnv(reward_version='v2')
    try:
        for t in states:
            _,t['initial_state']=prepare_snapshot(env,t,t['perturbation'])
    finally:env.close()
    identity=dict(environment=manifest_identity(root),sources={n:file_hash(root/'mujoco/rl'/n) for n in
        ['uav_bc_yaw_data.py','uav_bc_robustness.py','uav_bc_evaluation.py','latent_mpc_evaluation.py','uav_bc_safety.py']})
    value=dict(base_commit=BASE,identity=identity,generation_seed=SEED,preflight=preflight(root),
        target_sources={n:file_hash(reports/n) for n in ['uav_bc_target_splits_seed0.json','onpolicy_adaptation_target_splits_seed0.json']},
        train_target_groups=bc['groups']['train'],val_target_groups=bc['groups']['val'],final_targets=final,
        excluded_target_ids=[t['target_id'] for t in old],overlap_count=0,expert_states=states,
        config=dict(train_per_model=20000,validation_common=4000,repeats=10,phase_weights=PHASES,
            yaw_mixture=[.4,.3,.3],base_position_radius=[0,.10],base_velocity_radius=[0,.15],
            phase_rule='early step<10; else distance<.2 near_braking; else speed>=.5 high_speed; else approach_other',
            sample_selection='unique real valid inputs, no replacement, per-phase yaw/sign quotas; B shadow groups only for paired selection',
            training='seed0 fresh actor,original Train stats,Adam .001,batch512,100epochs,4000updates,ValMSE only',
            final_conditions=conditions(),acceptance=dict(yaw30_success=90,nominal_success=98)))
    value['sha256']=json_hash(value);path=reports/MANIFEST
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('immutable design/source/state manifest changed')
    else:atomic_json(path,value)
    return value

def collect(root):
    from ppo_pi_env import MineUAVPIEnv
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS;m=make_manifest(root);records=[]
    env=MineUAVPIEnv(reward_version='v2')
    try:
        for i,t in enumerate(m['expert_states']):
            path=parts/('expert_'+t['key'].replace('/','_')+'.json')
            if path.exists():r=json.loads(path.read_text());check_record(r,m['sha256'],t['key'])
            else:
                s,meta=prepare_snapshot(env,t,t['perturbation'])
                if meta!=t['initial_state']:raise ValueError('paired initial reconstruction changed')
                row,a=rollout(env,s,scripted_action);validate_expert_trace(a);raw=path.with_suffix('.npz');atomic_npz(raw,**a)
                r=dict(row,key=t['key'],identity=m['sha256'],source=t['source'],split=t['split'],target_id=t['target_id'],
                    env_seed=t['env_seed'],yaw_group=t['yaw_group'],yaw_sign=t['yaw_sign'],initial_yaw_degrees=t['initial_yaw_degrees'],
                    perturbation=t['perturbation'],trace_path=str(raw),trace_sha256=file_hash(raw))
                r['record_sha256']=json_hash(r);atomic_json(path,r)
            records.append(r)
            if (i+1)%100==0:print('Yaw expert episodes',i+1,'/',len(m['expert_states']),flush=True)
    finally:env.close()
    selected={}
    for source,total in [('nominal_expanded',20000),('yaw_augmented',20000),('shared_validation',4000)]:
        pool=[];lookup={};raws={}
        for r in records:
            if r['source']!=source:continue
            with np.load(r['trace_path'],allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
            validate_expert_trace(a);raws[r['key']]=a;lookup[r['key']]=r
            phases=phase_labels(a['observations'][:-1],a['step_index'])
            for j in np.flatnonzero(a['physical_state_valid'][:-1]):
                pool.append(dict(key=r['key']+f'/{j:03d}',episode=r['key'],step=int(j),phase=str(phases[j]),yaw_group=r['yaw_group'],yaw_sign=r['yaw_sign']))
        chosen=choose_samples(pool,total,SEED+11)
        ids={lookup[c['episode']]['target_id'] for c in chosen}
        expected={t['target_id'] for t in (m['val_target_groups'] if source=='shared_validation' else m['train_target_groups'])}
        if ids!=expected:raise ValueError('selected dataset omitted a fixed target group')
        x=np.array([raws[c['episode']]['observations'][c['step']] for c in chosen],np.float32)
        y=np.array([raws[c['episode']]['actions'][c['step']] for c in chosen],np.float32)
        if len(np.unique(np.c_[x,y],axis=0))!=total:raise ValueError('duplicate physical imitation samples; never pad')
        path=parts/(source+'_selected.npz');atomic_npz(path,observations=x,actions=y,
            phase=np.array([c['phase'] for c in chosen]),yaw_group=np.array([c['yaw_group'] for c in chosen]),
            initial_yaw_degrees=np.array([lookup[c['episode']]['initial_yaw_degrees'] for c in chosen]),
            sample_key=np.array([c['key'] for c in chosen]),target_id=np.array([lookup[c['episode']]['target_id'] for c in chosen]),
            step_index=np.array([c['step'] for c in chosen]))
        selected[source]=dict(path=str(path),sha256=file_hash(path),samples=total,sample_key_sha256=json_hash([c['key'] for c in chosen]),
            phases=dict(Counter(c['phase'] for c in chosen)),yaw_groups=dict(Counter(str(c['yaw_group']) for c in chosen)),
            initial_yaw_counts=dict(Counter(str(lookup[c['episode']]['initial_yaw_degrees']) for c in chosen)),
            pool_samples=len(pool),target_groups=len(ids),source_episode_outcomes=dict(Counter(lookup[k]['termination_reason'] for k in lookup)))
    value=dict(manifest_sha256=m['sha256'],records=records,selected=selected)
    atomic_json(parts/'dataset.json',value);return value

def load_selected(root,name):
    root=Path(root);reports=root/'mujoco/reports';m=json.loads((reports/MANIFEST).read_text());d=json.loads((reports/PARTS/'dataset.json').read_text())
    if d['manifest_sha256']!=m['sha256'] or json_hash({k:v for k,v in m.items() if k!='sha256'})!=m['sha256']:raise ValueError('dataset manifest altered')
    r=d['selected'][name]
    if file_hash(r['path'])!=r['sha256']:raise ValueError('selected dataset corrupted')
    with np.load(r['path'],allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

if __name__=='__main__':collect(Path(__file__).resolve().parents[2])
