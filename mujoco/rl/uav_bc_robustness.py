"""Evaluation-only, real physical initial perturbations and exact paired resets."""
import json
import math
from pathlib import Path
import mujoco
import numpy as np
from decision_fidelity_snapshot import capture, restore, fingerprint
from latent_dynamics_data import file_hash, json_hash
from uav_bc_safety import atomic_json,atomic_npz,check_running,available_memory

BASE='11d3e6bed47d87e86956349db207e89727d08bef'
MODEL_SHA='7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c'
MODEL='uav_bc_mlp_seed0.pt'
PARTS='uav_bc_robustness_seed0_parts'
MANIFEST='uav_bc_robustness_manifest_seed0.json'
REPORT='uav_bc_initial_state_robustness_seed0.json'
PERTURBATION_SEED=2026101001


def conditions():
    result={'nominal':dict(position=0.,velocity=0.,yaw_degrees=0.)}
    for kind in ['position','velocity','yaw','combined']:
        for level,p,v,y in [('small',.1,.15,10.),('large',.3,.4,30.)]:
            result[f'{kind}_{level}']=dict(position=p if kind in ('position','combined') else 0.,
                velocity=v if kind in ('velocity','combined') else 0.,yaw_degrees=y if kind in ('yaw','combined') else 0.)
    return result


def sample_perturbation(condition,split,index):
    if split not in ('benchmark','holdout') or not 0<=index<100:
        raise ValueError('fixed split and episode index required')
    c=conditions()[condition];sid=int(split=='holdout')
    rng=np.random.default_rng(np.random.SeedSequence([PERTURBATION_SEED,sid,index]))
    p=rng.normal(size=3);p=p/np.linalg.norm(p)
    v=rng.normal(size=3);v=v/np.linalg.norm(v)
    signs=np.random.default_rng(np.random.SeedSequence([PERTURBATION_SEED,sid,1000])).permutation(np.r_[-np.ones(50),np.ones(50)])
    return dict(position_m=(p*c['position']).tolist(),velocity_m_s=(v*c['velocity']).tolist(),
        yaw_rad=float(signs[index]*math.radians(c['yaw_degrees'])))


def prepare_snapshot(env,target,perturbation):
    p=np.asarray(perturbation['position_m'],float);v=np.asarray(perturbation['velocity_m_s'],float);y=float(perturbation['yaw_rad'])
    if p.shape!=(3,) or v.shape!=(3,) or not np.isfinite(np.r_[p,v,y]).all():
        raise ValueError('finite position, world velocity and yaw required')
    env.reset(seed=target['env_seed'],options={'target_position':target['target']})
    env.data.qpos[:3]+=p;env.data.qvel[:3]+=v
    # Nominal orientation is identity; yaw only about world +Z, wxyz unit quaternion.
    env.data.qpos[3:7]=[math.cos(y/2),0.,0.,math.sin(y/2)]
    mujoco.mj_forward(env.model,env.data)
    env.previous_distance=float(np.linalg.norm(env.target_position-env.data.qpos[:3]))
    geom=env.model.geom('coarse_collision').id
    rotation=env.data.geom_xmat[geom].reshape(3,3)
    clearance=float(env.data.geom_xpos[geom,2]-np.abs(rotation[2])@env.model.geom_size[geom])
    penetration=any(env.data.contact[i].dist < -1e-9 for i in range(env.data.ncon))
    if env._failure_reason() is not None or clearance<=0 or penetration or not env.observation_space.contains(env._get_obs()):
        raise ValueError('illegal initial physical state, never silently repair/filter')
    if not np.isclose(np.linalg.norm(env.data.qpos[3:7]),1.,atol=1e-12):raise ValueError('non-unit quaternion')
    snap=capture(env,np.zeros(4,np.float32))
    meta=dict(snapshot_sha256=fingerprint(snap),qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist(),
        observation=env._get_obs().tolist(),ground_clearance_m=clearance,initial_failure_reason=None,
        previous_distance_m=env.previous_distance,previous_action=[0.]*4,pi_integral=env.velocity_controller.integral_error.tolist(),
        controller_yaw_target_rad=env.velocity_controller.yaw_target,episode_steps=env.episode_steps,success_streak=env.success_streak)
    return snap,meta


def manifest_identity(root):
    import platform,torch
    root=Path(root);names=['rl/uav_bc_robustness.py','rl/uav_bc_policy.py','rl/decision_fidelity_snapshot.py',
        'rl/test_env_scripted_policy.py','rl/mine_uav_env.py','rl/ppo_pi_env.py','rl/audit_velocity_pi.py',
        'control/velocity_command_controller.py','control/hover_controller.py','control/position_controller.py',
        'control/control_allocator.py','models/mine_uav_dynamics_v2.xml','models/rotor_actuators.xml',
        'models/rotor_config.json','reports/dynamics_v2_report.json']
    checkpoint=root/'mujoco/rl/models'/MODEL
    if file_hash(checkpoint)!=MODEL_SHA:raise ValueError('canonical frozen BC checkpoint hash mismatch')
    return dict(checkpoint_sha256=MODEL_SHA,source_sha256={n:file_hash(root/'mujoco'/n) for n in names},
        runtime=dict(python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__,torch=str(torch.__version__)),
        conditions=conditions(),perturbation_seed=PERTURBATION_SEED,workers=1)


def build_manifest(root):
    from ppo_pi_env import MineUAVPIEnv
    root=Path(root);reports=root/'mujoco/reports';source=reports/'uav_bc_target_splits_seed0.json'
    old=json.loads(source.read_text());targets=old['provenance']['evaluation']
    identity=manifest_identity(root);rows=[];env=MineUAVPIEnv(reward_version='v2')
    try:
        for condition in conditions():
            for split in ('benchmark','holdout'):
                entries=targets[split]['targets']
                if len(entries)!=100:raise ValueError('fixed target manifest incomplete')
                for i,t in enumerate(entries):
                    target=dict(target_id=i,env_seed=int(t['env_seed']),target=t['target'])
                    perturbation=sample_perturbation(condition,split,i)
                    _,meta=prepare_snapshot(env,target,perturbation)
                    rows.append(dict(key=f'{condition}/{split}/{i:03d}',condition=condition,split=split,index=i,
                        **target,perturbation=perturbation,initial_state=meta))
    finally:env.close()
    config=dict(identity=identity,base_commit=BASE,target_manifest=dict(path=str(source),sha256=file_hash(source)),
        sampling='Gaussian unit directions, fixed norm; direction shared across severity and component/combined conditions; balanced permuted yaw signs per split',
        snapshot='complete MjData copy + all nonresource env fields/RNG + PI/yaw/allocator + previous action',
        engineering_feasibility='All initial collision boxes clear floor, no penetration/flight-area/tilt failure. Bounds are test values, not certified recovery thresholds.',
        episodes_per_controller=1800,total_controller_episodes=3600,states=rows)
    digest=json_hash(config);manifest=dict(config,sha256=digest)
    path=reports/MANIFEST
    if path.exists():
        existing=json.loads(path.read_text())
        if existing!=manifest:raise ValueError('frozen preflight manifest changed; refuse stale resume')
    else:atomic_json(path,manifest)
    return manifest


def wilson(successes,n):
    if n<=0 or not 0<=successes<=n:raise ValueError('valid nonempty success cohort required')
    z=1.959963984540054;p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-radius),min(1.,center+radius)]


def sustained_time(values,threshold,dt,steps=5):
    for i in range(len(values)-steps+1):
        if np.all(np.asarray(values[i:i+steps])<threshold):return float(i*dt)
    return None


def rollout(env,snapshot,controller):
    from uav_bc_evaluation import trace_metrics
    previous=restore(env,snapshot);initial=fingerprint(capture(env,previous))
    obs=env._get_obs();observations=[obs.copy()];positions=[env.data.qpos[:3].copy()]
    actions=[];priors=[];streaks=[env.success_streak];valid=[env._state_is_finite()]
    for step in range(env.max_episode_steps):
        if step%25==0:check_running(available_memory(),0,0)
        action=np.asarray(controller(obs),np.float32)
        if action.shape!=(4,) or not np.isfinite(action).all() or (np.abs(action)>1).any():raise ValueError('invalid executed command')
        priors.append(previous.copy());actions.append(action.copy())
        obs,_,terminated,truncated,info=env.step(action)
        np.testing.assert_array_equal(info['velocity_command_m_s'],action[:3].astype(float)*[1.5,1.5,1.])
        observations.append(obs.copy());positions.append(env.data.qpos[:3].copy());streaks.append(env.success_streak);valid.append(env._state_is_finite())
        previous=action.copy()
        if terminated or truncated:break
    else:raise RuntimeError('unchanged task failed to terminate')
    reason=info['termination_reason'];success=reason=='success';n=len(actions)
    arrays=dict(observations=np.asarray(observations,np.float32),positions=np.asarray(positions,np.float64),
        actions=np.asarray(actions,np.float32),previous_actions=np.asarray(priors,np.float32),step_index=np.arange(n),
        success_streak=np.asarray(streaks,np.int64),physical_state_valid=np.asarray(valid,bool))
    valid=np.asarray(valid,bool)
    speed=np.linalg.norm(arrays['observations'][:,3:6],axis=1);yaw=np.abs(arrays['observations'][:,6])
    distance=np.linalg.norm(arrays['observations'][:,:3],axis=1)
    finite_speed=bool(valid[-1]) and math.isfinite(float(info['speed_m_s']))
    row=dict(initial_snapshot_sha256=initial,steps=n,success=success,failure=not success,
        physical_failure=not success and reason!='time_limit',timeout=reason=='time_limit',termination_reason=reason,
        simulated_seconds=float(env.data.time),completion_time_s=float(env.data.time) if success else None,
        final_distance_m=float(info['distance_m']) if valid[-1] else None,final_speed_m_s=float(info['speed_m_s']) if finite_speed else None,
        physical_valid_observation_count=int(valid.sum()),physical_unavailable_observation_count=int((~valid).sum()),
        reached_target_region=bool(np.any(distance[valid]<.1)),max_position_excursion_m=float(np.linalg.norm(arrays['positions']-arrays['positions'][0],axis=1)[valid].max()),
        maximum_speed_m_s=float(speed[valid].max()),velocity_settled_below_015_s=sustained_time(speed[valid],.15,env.policy_dt),
        initial_abs_yaw_error_rad=float(yaw[0]),final_abs_yaw_error_rad=float(yaw[-1]) if valid[-1] else None,maximum_abs_yaw_error_rad=float(yaw[valid].max()),
        yaw_settled_below_2deg_s=sustained_time(yaw[valid],math.radians(2),env.policy_dt))
    row.update(trace_metrics(arrays))
    if not valid.all():
        # The environment's zero observation on numerical failure is only an
        # API placeholder, never a measured arrival/heading/speed. It can only
        # occur at the terminal state, so pre-action near metrics remain valid.
        from latent_mpc_evaluation import crossings
        row.update(distance_reduction_m=None,mean_distance_m=float(distance[valid].mean()),
            minimum_distance_m=float(distance[valid].min()),mean_speed_m_s=float(speed[valid].mean()),
            crossings=crossings(arrays['observations'][valid]))
    return row,arrays


def validate_record(record,identity,state,controller):
    digest=record.get('record_sha256');payload={k:v for k,v in record.items() if k!='record_sha256'}
    expected=dict(identity=identity,key=state['key'],task_key=state['key']+'/'+controller,controller=controller,
        condition=state['condition'],split=state['split'],target_id=state['target_id'],env_seed=state['env_seed'],
        target=state['target'],initial_snapshot_sha256=state['initial_state']['snapshot_sha256'])
    if digest!=json_hash(payload) or any(record.get(k)!=v for k,v in expected.items()):
        raise ValueError('cache identity/labels/record hash corrupted')
    if file_hash(record['trace_path'])!=record['trace_sha256']:raise ValueError('cache trace hash corrupted')
    with np.load(record['trace_path'],allow_pickle=False) as a:
        n=record['steps']
        if a['observations'].shape!=(n+1,7) or a['actions'].shape!=(n,4) or a['positions'].shape!=(n+1,3):
            raise ValueError('trace shape/episode boundary corrupted')
        np.testing.assert_array_equal(a['observations'][0],state['initial_state']['observation'])
        np.testing.assert_array_equal(a['previous_actions'][0],state['initial_state']['previous_action'])
        np.testing.assert_array_equal(a['previous_actions'][1:],a['actions'][:-1])
        np.testing.assert_array_equal(a['step_index'],np.arange(n))
    if record['success']!=(record['termination_reason']=='success') or record['timeout']!=(record['termination_reason']=='time_limit'):
        raise ValueError('outcome labels inconsistent')
    if record['completion_time_s']!=(record['simulated_seconds'] if record['success'] else None):raise ValueError('failure falsely counted as completion')


def evaluate_task(env,snapshot,controller,state,controller_name,path,identity):
    path=Path(path)
    if path.exists():
        record=json.loads(path.read_text());validate_record(record,identity,state,controller_name);return record
    row,arrays=rollout(env,snapshot,controller)
    if row['initial_snapshot_sha256']!=state['initial_state']['snapshot_sha256']:raise ValueError('paired physical state mismatch')
    raw=path.with_suffix('.npz');atomic_npz(raw,**arrays)
    row.update(identity=identity,key=state['key'],task_key=state['key']+'/'+controller_name,controller=controller_name,
        condition=state['condition'],split=state['split'],target_id=state['target_id'],env_seed=state['env_seed'],target=state['target'],
        trace_path=str(raw),trace_sha256=file_hash(raw))
    row['record_sha256']=json_hash(row);atomic_json(path,row);return row


def validate_cohort(rows,keys):
    actual=[r['task_key'] for r in rows]
    if len(actual)!=len(keys) or len(set(actual))!=len(actual) or set(actual)!=set(keys):raise ValueError('duplicate/omitted/unexpected episodes')


def paired_summary(scripted,bc):
    from latent_mpc_evaluation import distribution
    a={r['key']:r for r in scripted};b={r['key']:r for r in bc}
    if not a or a.keys()!=b.keys() or len(a)!=len(scripted) or len(b)!=len(bc):raise ValueError('paired episode cohort mismatch')
    keys=sorted(a);delta=np.asarray([int(b[k]['success'])-int(a[k]['success']) for k in keys])
    both=[k for k in keys if a[k]['success'] and b[k]['success']]
    boot=np.random.default_rng(0).choice(delta,size=(4096,len(keys)),replace=True).mean(1)
    measured=[k for k in keys if a[k]['final_distance_m'] is not None and b[k]['final_distance_m'] is not None]
    return dict(episodes=len(keys),success_delta=float(delta.mean()),success_delta_percentage_points=float(100*delta.mean()),
        exploratory_paired_bootstrap_95_ci=np.quantile(boot,[.025,.975]).tolist(),
        uncertainty_note='descriptive fixed-target paired bootstrap; degenerate intervals when all pairs agree do not prove zero population risk',
        discordance=dict(both_success=len(both),bc_only_success=int((delta==1).sum()),scripted_only_success=int((delta==-1).sum()),
            both_failed=sum(not a[k]['success'] and not b[k]['success'] for k in keys)),
        finite_final_distance_pairs=len(measured),unavailable_final_distance_pairs=len(keys)-len(measured),
        final_distance_delta_m=distribution([b[k]['final_distance_m']-a[k]['final_distance_m'] for k in measured]) if measured else None,
        both_success_pairs=len(both),completion_time_delta_s=distribution([b[k]['completion_time_s']-a[k]['completion_time_s'] for k in both]) if both else None,
        failure_reason_pairs={f'{x} -> {y}':sum(a[k]['termination_reason']==x and b[k]['termination_reason']==y for k in keys)
            for x,y in sorted({(a[k]['termination_reason'],b[k]['termination_reason']) for k in keys})})


def controller_summary(rows):
    from collections import Counter
    from latent_mpc_evaluation import distribution
    def known(field):
        values=[r[field] for r in rows if r[field] is not None]
        return distribution(values) if values else None
    counts=Counter(r['termination_reason'] for r in rows);n=len(rows);steps=sum(r['steps'] for r in rows);near=sum(r['near_steps'] for r in rows)
    times=[r['completion_time_s'] for r in rows if r['success']]
    reductions=[r['distance_reduction_m'] for r in rows if r['distance_reduction_m'] is not None]
    out=dict(episodes=n,successes=counts['success'],success_rate=counts['success']/n,
        physical_failures=n-counts['success']-counts['time_limit'],timeouts=counts['time_limit'],termination_reasons=dict(counts),
        final_distance_m=known('final_distance_m'),finite_final_distance_episodes=sum(r['final_distance_m'] is not None for r in rows),
        unavailable_final_distance_episodes=sum(r['final_distance_m'] is None for r in rows),
        completion_time_s=distribution(times) if times else None,all_episode_duration_s=known('simulated_seconds'),
        total_decisions=steps,near_target_steps=near,near_target_actual_speed_m_s=sum(r['near_actual_sum'] for r in rows)/near if near else None,
        near_target_command_speed_m_s=sum(r['near_command_sum'] for r in rows)/near if near else None,
        action_saturation_fraction=sum(r['action_saturation_elements'] for r in rows)/(4*steps),
        crossing_episode_fraction=float(np.mean([r['crossings']>0 for r in rows])),initial_distance_m=known('initial_distance_m'),
        mean_distance_reduction_m=float(np.mean(reductions)) if reductions else None,
        by_initial_distance={name:dict(episodes=sum(mask),successes=sum(r['success'] for r,m in zip(rows,mask) if m))
            for name,mask in [(name,[lo<=r['initial_distance_m']<hi for r in rows]) for name,lo,hi in
                [('0-0.5',0,.5),('0.5-1',.5,1),('1-2',1,2),('>=2',2,float('inf'))]]})
    out['success_wilson_95_ci']=wilson(out['successes'],len(rows))
    for field in ['max_position_excursion_m','maximum_speed_m_s','final_abs_yaw_error_rad','maximum_abs_yaw_error_rad','final_speed_m_s']:
        values=[r[field] for r in rows if r[field] is not None]
        out[field]=distribution(values) if values else None
    for field in ['velocity_settled_below_015_s','yaw_settled_below_2deg_s']:
        values=[r[field] for r in rows if r[field] is not None]
        out[field]=dict(episodes_with_sustained_recovery=len(values),time_s=distribution(values) if values else None,
            definition='first 5 successive sampled states under threshold; velocity can later increase to travel to target')
    out['reached_target_region_count']=sum(r['reached_target_region'] for r in rows)
    return out


def run_evaluation(root,smoke=False):
    import torch
    from ppo_pi_env import MineUAVPIEnv
    from test_env_scripted_policy import scripted_action
    from uav_bc_policy import load,parameter_hash
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS;torch.set_num_threads(1)
    manifest=build_manifest(root);policy,_=load(root/'mujoco/rl/models'/MODEL);before=parameter_hash(policy.actor)
    identity=dict(manifest_sha256=manifest['sha256'],checkpoint_sha256=MODEL_SHA,policy_parameter_hash=before)
    rows=[];states=[s for s in manifest['states'] if not smoke or s['index']<2]
    env=MineUAVPIEnv(reward_version='v2')
    try:
        for index,state in enumerate(states):
            snap,meta=prepare_snapshot(env,state,state['perturbation'])
            if meta!=state['initial_state']:raise ValueError('manifest snapshot reproduction mismatch')
            for name,controller in [('scripted',scripted_action),('bc',policy.predict)]:
                path=parts/f"{state['condition']}_{state['split']}_{state['index']:03d}_{name}.json"
                rows.append(evaluate_task(env,snap,controller,state,name,path,identity))
            if (index+1)%25==0 or index+1==len(states):print('BC robustness states',index+1,'/',len(states),'episodes',len(rows),flush=True)
    finally:env.close()
    validate_cohort(rows,[s['key']+'/'+c for s in states for c in ('scripted','bc')])
    if file_hash(root/'mujoco/rl/models'/MODEL)!=MODEL_SHA or parameter_hash(policy.actor)!=before:raise ValueError('frozen checkpoint/parameters mutated')
    result=dict(manifest_sha256=manifest['sha256'],records=rows,episodes=len(rows),smoke=smoke,
        frozen_bc=dict(checkpoint_sha256_before=MODEL_SHA,checkpoint_sha256_after=file_hash(root/'mujoco/rl/models'/MODEL),
            parameter_hash_before=before,parameter_hash_after=parameter_hash(policy.actor),eval_mode=not policy.actor.training,
            all_requires_grad_false=all(not p.requires_grad for p in policy.actor.parameters())),actual_workers=1)
    atomic_json(parts/('smoke.json' if smoke else 'evaluation.json'),result);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run_evaluation(Path(__file__).resolve().parents[2],args.smoke)
