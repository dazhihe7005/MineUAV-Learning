"""Single-worker physical evaluation, censored metrics and fail-closed cache."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from latent_dynamics_data import file_hash,json_hash
from uav_bc_safety import atomic_json,atomic_npz
from uav_bc_policy import load,parameter_hash
from uav_bc_external_disturbance import (DisturbanceEnv,initial_snapshot,build_manifest,
    conditions,MODELS,MANIFEST,PARTS)
from uav_bc_robustness import rollout,validate_record,validate_cohort,controller_summary,paired_summary
from latent_mpc_evaluation import distribution

CONTROLLERS=('scripted','original','yaw_augmented')
REVIEWED_SOURCE_REVISIONS={'uav_bc_external_evaluation.py':{
    '42115dc66b7217a5c58b18ac4335a8160c4621b2878eb438c59c7f02d41911d7':
    'Post-acquisition review: exclude partial/failed terminal samples and require complete gust exposure and consecutive policy boundaries. No simulation reacquisition; all cached cohort metrics must recompute unchanged.'}}

def evaluation_provenance(root,identity,allowed_revisions=None):
    """Keep acquisition identity intact; allow only explicitly reviewed processors."""
    allowed=REVIEWED_SOURCE_REVISIONS if allowed_revisions is None else allowed_revisions
    current={};revisions={}
    for name,acquired in identity['source_sha256'].items():
        current[name]=file_hash(Path(root)/'mujoco/rl'/name)
        if current[name]!=acquired:
            reason=allowed.get(name,{}).get(acquired)
            if not reason:raise ValueError('unreviewed evaluation source revision')
            revisions[name]=dict(acquisition_sha256=acquired,processor_sha256=current[name],reason=reason)
    return dict(acquisition_identity=identity,metric_verifier_source_sha256=current,source_revisions=revisions)

def attach_physics(env,z):
    return dict(z,policy_times=np.asarray(env.policy_times,float),
        physics_force_times=np.asarray(env.force_times,float),physics_forces=np.asarray(env.force_vectors,float).reshape(-1,3),
        allocator_saturation_counts=np.asarray(env.allocator_saturations,int),control_update_counts=np.asarray(env.control_updates,int))

def metrics(row,z,condition):
    o=z['observations'];valid=z['physical_state_valid'];t=z['policy_times']
    d=np.linalg.norm(o[:,:3],axis=1);v=np.linalg.norm(o[:,3:6],axis=1)
    known=d[valid];gust=condition.startswith('gust_');constant=condition.startswith('constant_')
    exposed=np.linalg.norm(z['physics_forces'],axis=1)>1e-12
    recovery=None;confirmation=None
    full_gust=np.count_nonzero(exposed&(z['physics_force_times']>=2.-1e-9)&(z['physics_force_times']<4.-1e-9))==1000
    if gust and full_gust:
        boundary=np.isclose(t/.04,np.rint(t/.04),atol=1e-7,rtol=0)
        eligible=(t>=4.-1e-9)&valid&boundary&(d<.1)&(v<.15)
        if row.get('physical_failure',False):eligible[-1]=False
        for i in range(len(t)-4):
            if eligible[i:i+5].all() and np.allclose(np.diff(t[i:i+5]),.04,atol=1e-8,rtol=0):
                recovery=float(max(0.,t[i]-4.));confirmation=float(max(0.,t[i+4]-4.));break
    window=(t>=t[-1]-1.-1e-9)&valid
    steady=constant and t[-1]>=5.-1e-9 and valid[-1] and window.sum()>=25
    return dict(distance_rebound_overshoot_m=float(np.max(known-np.minimum.accumulate(known))),
        maximum_target_distance_m=float(known.max()),
        force_exposed_physics_ticks=int(exposed.sum()),force_exposed_seconds=float(exposed.sum()*.002),
        terminated_before_gust_start=bool(gust and not exposed.any()),
        survived_gust_end=bool(gust and t[-1]>=4.-1e-9),
        gust_recovery_onset_lag_s=recovery,gust_recovery_confirmation_lag_s=confirmation,
        steady_window_eligible=bool(steady),steady_window_samples=int(window.sum()) if steady else 0,
        steady_position_error_m=float(d[window].mean()) if steady else None,
        steady_position_error_std_m=float(d[window].std()) if steady else None,
        steady_velocity_m_s=float(v[window].mean()) if steady else None,
        steady_velocity_std_m_s=float(v[window].std()) if steady else None,
        steady_distance_slope_m_s=float(np.polyfit(t[window]-t[window][0],d[window],1)[0]) if steady else None,
        steady_target_hold_fraction=float(((d[window]<.1)&(v[window]<.15)).mean()) if steady else None)

def verify_arrays(row,z,condition,direction):
    from run_uav_bc_yaw_coverage import verify_rollout_metrics
    verify_rollout_metrics(row,z)
    n=row['steps'];t=z['policy_times'];ft=z['physics_force_times'];forces=z['physics_forces']
    if t.shape!=(n+1,) or forces.shape!=(len(ft),3) or not np.isfinite(t).all() or not np.isfinite(forces).all():
        raise ValueError('invalid physics trace')
    np.testing.assert_allclose(t[0],0.,atol=1e-12);np.testing.assert_allclose(t[-1],row['simulated_seconds'],atol=1e-12)
    np.testing.assert_allclose(t[:-1],np.arange(n)*.04,atol=1e-8,rtol=0)
    ticks=round(row['simulated_seconds']/.002)
    if len(ft)!=ticks:raise ValueError('missing/duplicated physical force application')
    np.testing.assert_allclose(ft,np.arange(ticks)*.002,atol=1e-8,rtol=0)
    c=conditions()[condition];k=np.arange(ticks)
    active=(k>=c['start_tick'])&(True if c['end_tick'] is None else k<c['end_tick'])
    expected=active[:,None]*np.asarray(direction)*c['fraction_mg']*7.*9.81
    np.testing.assert_allclose(forces,expected,atol=1e-12,rtol=0)
    if z['physical_state_valid'][-1] and not np.isfinite(z['positions']).all():raise ValueError('unmarked nonfinite physical state')
    if row['success'] and (z['success_streak'][-1]!=5 or not np.all(np.linalg.norm(z['observations'][-5:,:3],axis=1)<.1) or not np.all(np.linalg.norm(z['observations'][-5:,3:6],axis=1)<.15)):
        raise ValueError('success hold changed')

def validate_task(row,identity,state,name):
    validate_record(row,identity,state,name)
    if row.get('direction_world')!=state['direction_world']:raise ValueError('force direction relabelled')
    with np.load(row['trace_path'],allow_pickle=False) as z:
        verify_arrays(row,z,state['condition'],state['direction_world'])
        expected=metrics(row,z,state['condition'])
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('physical metric cache corrupted')

def evaluate_task(env,snapshot,controller,state,name,path,identity):
    path=Path(path)
    if path.exists():
        row=json.loads(path.read_text());validate_task(row,identity,state,name);return row
    row,z=rollout(env,snapshot,controller);z=attach_physics(env,z)
    if row['initial_snapshot_sha256']!=state['initial_state']['snapshot_sha256']:raise ValueError('unpaired initial state')
    verify_arrays(row,z,state['condition'],state['direction_world']);row.update(metrics(row,z,state['condition']))
    raw=path.with_suffix('.npz');atomic_npz(raw,**z)
    row.update(identity=identity,key=state['key'],task_key=state['key']+'/'+name,controller=name,condition=state['condition'],
        split=state['split'],target_id=state['target_id'],env_seed=state['env_seed'],target=state['target'],
        direction_world=state['direction_world'],trace_path=str(raw),trace_sha256=file_hash(raw),
        allocator_saturation_fraction=float(z['allocator_saturation_counts'].sum()/max(1,z['control_update_counts'].sum())))
    row['record_sha256']=json_hash(row);atomic_json(path,row);return row

def paired(first,second):
    p=paired_summary(first,second);a={r['key']:r for r in first};b={r['key']:r for r in second}
    p['timeout_discordance']=dict(first_only_timeout=sum(a[k]['timeout'] and not b[k]['timeout'] for k in a),
        second_only_timeout=sum(b[k]['timeout'] and not a[k]['timeout'] for k in a),both_timeout=sum(a[k]['timeout'] and b[k]['timeout'] for k in a))
    p['discordance']={k.replace('bc','second').replace('scripted','first'):v for k,v in p['discordance'].items()}
    p['peak_speed_delta_m_s']=distribution([b[k]['maximum_speed_m_s']-a[k]['maximum_speed_m_s'] for k in a])
    near=[k for k in a if a[k]['near_steps'] and b[k]['near_steps']]
    p['near_speed_delta_m_s']=distribution([b[k]['near_actual_sum']/b[k]['near_steps']-a[k]['near_actual_sum']/a[k]['near_steps'] for k in near]) if near else None
    p['near_speed_paired_episodes']=len(near);return p

def summaries(rows):
    result={}
    for c in conditions():
        by={n:[r for r in rows if r['condition']==c and r['controller']==n] for n in CONTROLLERS}
        result[c]={}
        for n,a in by.items():
            s=controller_summary(a)
            for field in ['distance_rebound_overshoot_m','maximum_target_distance_m','allocator_saturation_fraction',
                'steady_position_error_m','steady_position_error_std_m','steady_velocity_m_s','steady_velocity_std_m_s',
                'steady_distance_slope_m_s','steady_target_hold_fraction','gust_recovery_onset_lag_s','gust_recovery_confirmation_lag_s']:
                known=[r[field] for r in a if r[field] is not None]
                s[field]=dict(measured_episodes=len(known),statistics=distribution(known) if known else None)
            s['exposure']=dict(force_exposed_episodes=sum(r['force_exposed_physics_ticks']>0 for r in a),
                terminated_before_gust_start=sum(r['terminated_before_gust_start'] for r in a),
                survived_gust_end=sum(r['survived_gust_end'] for r in a),
                recovered_after_gust=sum(r['gust_recovery_onset_lag_s'] is not None for r in a),
                steady_window_eligible_episodes=sum(r['steady_window_eligible'] for r in a))
            result[c][n]=s
        result[c]['paired']={f'{a}_minus_{b}':paired(by[b],by[a]) for a,b in
            [('yaw_augmented','original'),('yaw_augmented','scripted'),('original','scripted')]}
    return result

def model_identity(root):
    policies={};hashes={};parameters={}
    for n,(f,expected) in MODELS.items():
        path=Path(root)/'mujoco/rl/models'/f;policies[n]=load(path)[0];hashes[n]=file_hash(path)
        parameters[n]=parameter_hash(policies[n].actor)
        if hashes[n]!=expected or policies[n].actor.training or any(p.requires_grad for p in policies[n].actor.parameters()):
            raise ValueError('canonical model not frozen/changed')
    return policies,hashes,parameters

def run(root,smoke=False):
    from test_env_scripted_policy import scripted_action
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS;torch.set_num_threads(1)
    m=build_manifest(root);policies,hashes,parameters=model_identity(root)
    identity=dict(manifest_sha256=m['sha256'],checkpoint_sha256=hashes,parameters=parameters,
        source_sha256={n:file_hash(root/'mujoco/rl'/n) for n in ['uav_bc_external_evaluation.py','uav_bc_robustness.py',
            'run_uav_bc_yaw_coverage.py','uav_bc_yaw_evaluation.py','uav_bc_evaluation.py','latent_mpc_evaluation.py','uav_bc_policy.py','uav_bc_safety.py']})
    env=DisturbanceEnv();rows=[];selected=[s for s in m['states'] if not smoke or s['index']<2]
    try:
        for i,s in enumerate(selected):
            env.configure(s['condition'],s['direction_world']);snap,meta=initial_snapshot(env,s)
            if meta!=s['initial_state']:raise ValueError('manifest physical snapshot mismatch')
            for n in CONTROLLERS:
                fn=scripted_action if n=='scripted' else policies[n].predict
                rows.append(evaluate_task(env,snap,fn,s,n,parts/f"{s['condition']}_{s['index']:03d}_{n}.json",identity))
            if (i+1)%20==0 or i+1==len(selected):print('External-force states',i+1,'/',len(selected),'episodes',len(rows),flush=True)
        validate_cohort(rows,[s['key']+'/'+n for s in selected for n in CONTROLLERS])
    finally:env.close()
    after={n:parameter_hash(p.actor) for n,p in policies.items()}
    if after!=parameters or {n:file_hash(root/'mujoco/rl/models'/f) for n,(f,h) in MODELS.items()}!=hashes:raise ValueError('frozen parameters mutated')
    value=dict(rows=rows,identity=identity,checkpoint_hashes_before=hashes,checkpoint_hashes_after=hashes,
        parameter_hashes_before=parameters,parameter_hashes_after=after,all_frozen=True)
    if not smoke:value['metrics']=summaries(rows)
    atomic_json(parts/('smoke.json' if smoke else 'evaluation.json'),value);return value

def verify(root):
    root=Path(root);reports=root/'mujoco/reports';m=build_manifest(root)
    ev=json.loads((reports/PARTS/'evaluation.json').read_text());policies,hashes,parameters=model_identity(root)
    if ev['identity']['manifest_sha256']!=m['sha256'] or hashes!=ev['checkpoint_hashes_before'] or hashes!=ev['checkpoint_hashes_after'] or parameters!=ev['parameter_hashes_before'] or parameters!=ev['parameter_hashes_after']:
        raise ValueError('frozen model/manifest hash changed')
    provenance=evaluation_provenance(root,ev['identity'])
    for n,revision in provenance['source_revisions'].items():
        archived=reports/PARTS/('acquisition_'+n)
        if not archived.exists() or file_hash(archived)!=revision['acquisition_sha256']:
            raise ValueError('original acquisition evaluator missing/changed')
    for n,h in m['target_sources'].items():
        if file_hash(reports/n)!=h:raise ValueError('target source changed')
    states={s['key']:s for s in m['states']};validate_cohort(ev['rows'],[s+'/'+n for s in states for n in CONTROLLERS])
    from test_env_scripted_policy import scripted_action
    pairs={}
    for r in ev['rows']:
        s=states[r['key']];validate_task(r,ev['identity'],s,r['controller']);pairs.setdefault(r['key'],[]).append(r['initial_snapshot_sha256'])
        with np.load(r['trace_path'],allow_pickle=False) as z:
            valid=z['physical_state_valid'][:-1];o=z['observations'][:-1][valid];actions=z['actions'][valid]
            if r['controller']=='scripted':expected=np.asarray([scripted_action(x) for x in o])
            else:expected=np.clip(policies[r['controller']].raw_actions(o),-1,1)
            np.testing.assert_allclose(actions,expected,atol=3e-6,rtol=0)
            valid=z['physical_state_valid'];np.testing.assert_allclose(np.asarray(r['target'])-z['positions'][valid],z['observations'][valid,:3],atol=1e-6,rtol=0)
            measured=float(z['allocator_saturation_counts'].sum()/max(1,z['control_update_counts'].sum()))
            if r['allocator_saturation_fraction']!=measured:raise ValueError('allocator saturation misreported')
    if len(pairs)!=600 or any(len(x)!=3 or len(set(x))!=1 for x in pairs.values()):raise ValueError('unpaired/omitted initial states')
    if summaries(ev['rows'])!=ev['metrics']:raise ValueError('aggregate metrics corrupted')
    result=dict(passed=True,controller_episodes=1800,paired_states=600,all_source_checkpoint_parameter_trace_hashes_verified=True,
        every_physics_tick_force_verified=True,controller_independent_actions_verified=True,
        no_duplicate_or_missing_episodes=True,targets_isolated=True,
        cached_metrics_recomputed_unchanged=True,postprocessing_source_revisions=provenance['source_revisions'])
    atomic_json(reports/PARTS/'verification.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--verify',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    verify(root) if a.verify else run(root,a.smoke)
