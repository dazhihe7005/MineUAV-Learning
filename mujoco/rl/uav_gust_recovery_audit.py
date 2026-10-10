"""Read-only audit of saved physical traces. Never imports/executes a simulator."""
import argparse,hashlib,json,math
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from uav_bc_safety import atomic_json

BASE='e273c24eda37a3f06c21d845a4bc5e94be065b98'
OLD_PARTS='uav_bc_external_disturbance_seed0_parts'
PARTS='uav_gust_recovery_audit_parts'
REPORT='uav_gust_recovery_audit.json'
CONDITIONS=('constant_medium','gust_medium','gust_high')
CONTROLLERS=('scripted','original','yaw_augmented')
PHASES={'pre_0_2':(0.,2.),'force_2_4':(2.,4.),'post_4_6':(4.,6.),'late_6_end':(6.,math.inf)}
REPRESENTATIVES=(0,33,66,99)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def stats(values):
    a=np.asarray(values,float)
    if not len(a):return None
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),
        max=float(a.max()),p95=float(np.quantile(a,.95)))

@contextmanager
def verified_trace(row):
    p=Path(row['trace_path'])
    if sha(p)!=row['trace_sha256']:raise ValueError('raw trace hash mismatch')
    with np.load(p,allow_pickle=False) as z:yield z
    if sha(p)!=row['trace_sha256']:raise ValueError('raw trace mutated during analysis')

def signal_availability(z):
    return {name:('recorded' if field in z else 'not recorded') for name,field in
        [('position','positions'),('velocity','observations'),('target_error','observations'),
         ('commanded_velocity','actions'),('yaw_error','observations'),('external_force','physics_forces'),
         ('success_streak','success_streak'),('pi_integral_state','integral_error_m'),
         ('pi_output','desired_acceleration_m_s2'),('anti_windup_freeze_count','anti_windup_freeze_count'),
         ('actual_attitude_quaternion','quaternions'),('rotor_thrust','rotor_thrust') ]}

def validate_arrays(row,z):
    n=row['steps'];t=z['policy_times'];o=z['observations'];a=z['actions']
    if o.shape!=(n+1,7) or a.shape!=(n,4) or z['positions'].shape!=(n+1,3):raise ValueError('trace shape mismatch')
    if not all(np.isfinite(z[k]).all() for k in ['observations','positions','actions','policy_times']):raise ValueError('nonfinite trace')
    np.testing.assert_allclose(t[:-1],np.arange(n)*.04,atol=1e-8,rtol=0)
    np.testing.assert_allclose(t[-1],row['simulated_seconds'],atol=1e-10,rtol=0)
    if not 0<t[-1]-t[-2]<=.04+1e-8:raise ValueError('invalid terminal interval')
    np.testing.assert_array_equal(z['previous_actions'][0],np.zeros(4))
    np.testing.assert_array_equal(z['previous_actions'][1:],a[:-1])
    np.testing.assert_array_equal(z['step_index'],np.arange(n))
    np.testing.assert_allclose(np.asarray(row['target'])-z['positions'],o[:,:3],atol=1e-6,rtol=0)
    if np.abs(a).max()>1 or np.abs(o[:,6]).max()>math.pi+1e-6:raise ValueError('action bounds or yaw wrap')
    ticks=round(t[-1]/.002);ft=z['physics_force_times'];forces=z['physics_forces']
    if ft.shape!=(ticks,) or forces.shape!=(ticks,3):raise ValueError('force sampling shape')
    np.testing.assert_allclose(ft,np.arange(ticks)*.002,atol=1e-8,rtol=0)
    c=row['condition'];fraction=.2 if c=='gust_high' else .1
    active=np.ones(ticks,bool) if c=='constant_medium' else (np.arange(ticks)>=1000)&(np.arange(ticks)<2000)
    np.testing.assert_allclose(forces,active[:,None]*np.asarray(row['direction_world'])*fraction*7.*9.81,atol=1e-12,rtol=0)

def longest_interval(mask,dt):
    run=best=0.
    for ok,width in zip(mask,dt):
        run=run+float(width) if ok else 0.;best=max(best,run)
    return best

def boundary_runs(mask,t):
    """Count adjacent complete observations; duration is observed span, not future extrapolation."""
    best=run=0;start=0;span=0.;confirmation=onset=None
    for i,ok in enumerate(mask):
        contiguous=i>0 and abs(t[i]-t[i-1]-.04)<1e-8
        if ok:
            if not run or not contiguous:run=1;start=i
            else:run+=1
            best=max(best,run);span=max(span,float(t[i]-t[start]))
            if run==5 and confirmation is None:onset=float(t[start]);confirmation=float(t[i])
        else:run=0
    return dict(longest_samples=best,longest_span_s=span,onset_s=onset,confirmation_s=confirmation)

def verify_success(row,z):
    streak=0;counts=[0];t=z['policy_times'];last=len(t)-1
    distances=np.linalg.norm(z['observations'][:,:3],axis=1)
    speeds=np.linalg.norm(z['observations'][:,3:6],axis=1)
    for i in range(1,len(t)):
        complete=abs(t[i]-t[i-1]-.04)<1e-8
        valid=bool(z['physical_state_valid'][i]) and not (i==last and row['physical_failure'])
        # MineUAVEnv.step resets its streak on failure, even if final geometry is near target.
        streak=streak+1 if complete and valid and distances[i]<.1 and speeds[i]<.15 else 0
        counts.append(streak)
    if not np.array_equal(counts,z['success_streak']) or row['success']!=(streak>=5):
        raise ValueError('original success mismatch')
    return streak

def analyze_episode(row,z):
    t=z['policy_times'];o=np.asarray(z['observations'],float);v=o[:,3:6];e=o[:,:3]
    d=np.linalg.norm(e,axis=1);speed=np.linalg.norm(v,axis=1)
    cmd=z['actions'][:,:3].astype(float)*[1.5,1.5,1.];dt=np.diff(t)
    gated=(np.linalg.norm(cmd[:,:2],axis=1)>=.02)&(np.linalg.norm(v[:-1,:2],axis=1)>=.15)
    opposite=np.einsum('ij,ij->i',cmd[:,:2],v[:-1,:2])<0
    outward=np.einsum('ij,ij->i',e[:-1],v[:-1])<0
    phases={}
    for name,(lo,hi) in PHASES.items():
        m=(t[:-1]>=lo-1e-9)&(t[:-1]<hi-1e-9)&z['physical_state_valid'][:-1]
        if not m.any():phases[name]=None;continue
        width=dt[m];total=width.sum();both=m&gated;indices=np.flatnonzero(m)
        avg=lambda value:float(np.average(value[m],weights=width))
        phases[name]=dict(samples=int(m.sum()),duration_s=float(total),distance_mean_m=avg(d[:-1]),
            speed_mean_m_s=avg(speed[:-1]),speed_peak_m_s=float(speed[:-1][m].max()),
            command_speed_mean_m_s=avg(np.linalg.norm(cmd,axis=1)),
            command_actual_rmse_m_s=float(np.sqrt(avg(np.sum((cmd-v[:-1])**2,axis=1)))),
            outward_duration_fraction=float(dt[m&outward].sum()/total),
            opposing_gated_duration_s=float(dt[both].sum()),
            opposing_fraction_given_gated=float(dt[both&opposite].sum()/dt[both].sum()) if both.any() else None,
            longest_opposing_run_s=longest_interval((gated&opposite)[indices],width),
            near_target_samples=int(np.count_nonzero(m&(d[:-1]<.1))),
            near_target_moving_samples=int(np.count_nonzero(m&(d[:-1]<.1)&(speed[:-1]>=.15))),
            mean_yaw_error_rad=avg(np.abs(o[:-1,6])))
    post=None
    if row['condition'].startswith('gust_'):
        complete=np.isclose(t/.04,np.rint(t/.04),atol=1e-7,rtol=0)&z['physical_state_valid']
        if row['physical_failure']:complete[-1]=False
        m=(t>=4.-1e-9)&complete
        observed=np.count_nonzero(np.linalg.norm(z['physics_forces'],axis=1)>1e-12)==1000
        joint=m&(d<.1)&(speed<.15)&observed
        jr=boundary_runs(joint,t);dr=boundary_runs(m&(d<.1),t);vr=boundary_runs(m&(speed<.15),t)
        at=lambda when,values:float(values[np.flatnonzero(np.isclose(t,when,atol=1e-8,rtol=0))[0]]) if np.isclose(t,when,atol=1e-8,rtol=0).any() else None
        d4=at(4.,d);v4=at(4.,speed);d6=at(6.,d);post_valid=d[m]
        post=dict(full_gust_exposed=bool(observed),post_end_window_s=float(max(0.,t[-1]-4.)),
            distance_at4_m=d4,speed_at4_m_s=v4,distance_at6_m=d6,
            max_distance_excess_over_at4_m=float(post_valid.max()-d4) if len(post_valid) and d4 is not None else None,
            distance_grows_after4=bool(len(post_valid) and d4 is not None and post_valid.max()>d4+1e-6),
            post_distance_peak_time_s=float(t[m][np.argmax(post_valid)]) if len(post_valid) else None,
            post_speed_peak_m_s=float(speed[m].max()) if len(post_valid) else None,
            post_speed_peak_time_s=float(t[m][np.argmax(speed[m])]) if len(post_valid) else None,
            final_minus_at4_distance_m=float(d[-1]-d4) if d4 is not None else None,
            distance_only_ever=bool(dr['longest_samples']),speed_only_ever=bool(vr['longest_samples']),
            joint_ever=bool(jr['longest_samples']),joint_longest_samples=jr['longest_samples'],
            joint_longest_run_s=jr['longest_span_s'],distance_only_longest_run_s=dr['longest_span_s'],
            joint_recovery_onset_s=jr['onset_s'],joint_recovery_confirmation_s=jr['confirmation_s'],
            final_joint_qualifying=bool(d[-1]<.1 and speed[-1]<.15 and complete[-1]),
            final_distance_qualifying=bool(d[-1]<.1),final_speed_qualifying=bool(speed[-1]<.15),
            last1s_speed_mean_m_s=float(speed[(t>=t[-1]-1.-1e-9)&complete].mean()))
    return dict(**{k:row[k] for k in ['condition','controller','success']},
        final_distance_m=float(d[-1]),final_speed_m_s=float(speed[-1]),peak_speed_m_s=float(speed.max()),
        maximum_distance_m=float(d.max()),final_yaw_error_rad=float(abs(o[-1,6])),
        duration_s=float(t[-1]),phases=phases,post_gust=post)

def paired_metrics(rows,first,c1,second,c2):
    a={r['index']:r for r in rows if r['controller']==first and r['condition']==c1}
    b={r['index']:r for r in rows if r['controller']==second and r['condition']==c2}
    if not a or a.keys()!=b.keys() or any(a[i]['target_id']!=b[i]['target_id'] for i in a):raise ValueError('paired IDs missing/mismatched')
    return dict(n=len(a),first_only_success=sum(a[i]['success'] and not b[i]['success'] for i in a),
        second_only_success=sum(b[i]['success'] and not a[i]['success'] for i in a),
        **{field+'_delta_m' if field=='final_distance' else field+'_delta_m_s':stats([b[i][field+'_m' if field=='final_distance' else field+'_m_s']-a[i][field+'_m' if field=='final_distance' else field+'_m_s'] for i in a]) for field in ['final_distance','peak_speed']})

def summarize(rows):
    result={}
    for c in CONDITIONS:
        result[c]={}
        for controller in CONTROLLERS:
            group=[r for r in rows if r['condition']==c and r['controller']==controller]
            s=dict(episodes=len(group),successes=sum(r['success'] for r in group))
            for name in ['final_distance_m','final_speed_m_s','peak_speed_m_s','duration_s']:
                s[name]=stats([r[name] for r in group])
            s['phases']={}
            for phase in PHASES:
                valid=[r['phases'][phase] for r in group if r['phases'][phase] is not None]
                s['phases'][phase]=dict(eligible_episodes=len(valid),statistics={
                    k:stats([r[k] for r in valid if r[k] is not None]) for k in (valid[0] if valid else [])})
            if c.startswith('gust_'):
                p=[r['post_gust'] for r in group];s['post_gust']={}
                for k in p[0]:
                    values=[r[k] for r in p if r[k] is not None]
                    s['post_gust'][k]=sum(values) if values and isinstance(values[0],bool) else stats(values)
            result[c][controller]=s
    return result

def validate_initial_pairs(states):
    groups={}
    for s in states:
        if s['condition'] not in CONDITIONS:continue
        index=s['index'];group=groups.setdefault(index,{})
        if s['condition'] in group:raise ValueError('duplicate initial condition')
        group[s['condition']]=s
    for group in groups.values():
        if set(group)!=set(CONDITIONS):raise ValueError('incomplete initial pairing')
        # Snapshot fingerprint includes condition config, which intentionally differs.
        def fields(s):return dict(target_id=s['target_id'],env_seed=s['env_seed'],target=s['target'],
            direction_world=s['direction_world'],initial_state={k:v for k,v in s['initial_state'].items() if k!='snapshot_sha256'})
        if any(fields(s)!=fields(group[CONDITIONS[0]]) for s in group.values()):raise ValueError('initial physical/controller pairing differs')
    return len(groups)

def analyze(root):
    root=Path(root);reports=root/'mujoco/reports';parts=reports/OLD_PARTS
    public_path=reports/'uav_bc_external_disturbance_seed0.json';manifest_path=reports/'uav_bc_external_disturbance_manifest_seed0.json';cache_path=parts/'evaluation.json'
    public=json.loads(public_path.read_text());manifest=json.loads(manifest_path.read_text());ev=json.loads(cache_path.read_text())
    if sha(manifest_path)!=public['manifest']['file_sha256'] or digest({k:v for k,v in manifest.items() if k!='sha256'})!=manifest['sha256']:raise ValueError('manifest identity corrupt')
    if sha(cache_path)!=public['evaluation_provenance']['acquisition_cache_sha256'] or ev['identity']!=public['evaluation_provenance']['acquisition_identity']:raise ValueError('acquisition cache changed')
    inputs={str(p.relative_to(root)):sha(p) for p in [public_path,manifest_path,cache_path]}
    for n,h in manifest['identity']['source_sha256'].items():inputs['mujoco/'+n]=h
    inputs['mujoco/rl/uav_bc_external_disturbance.py']=manifest['identity']['external_source_sha256']
    for key in ['metric_verifier_source_sha256','publication_source_sha256']:
        inputs.update({'mujoco/rl/'+n:h for n,h in public['evaluation_provenance'][key].items()})
    for v in manifest['identity']['models'].values():inputs[v['path']]=v['sha256']
    if any(sha(root/name)!=h for name,h in inputs.items()):raise ValueError('historical source/checkpoint identity changed')
    if ev['metrics']!=public['metrics'] or len(ev['rows'])!=1800:raise ValueError('incomplete/different source results')
    initial_pairs=validate_initial_pairs(manifest['states'])
    if initial_pairs!=100:raise ValueError('incomplete100-target initial pairing')
    states={s['key']:s for s in manifest['states']};seen=set();inventory={};rows=[];schemas=[]
    for original in ev['rows']:
        key=original['task_key']
        if key in seen or digest({k:v for k,v in original.items() if k!='record_sha256'})!=original['record_sha256']:raise ValueError('duplicate/corrupt source record')
        seen.add(key);path=Path(original['trace_path']).resolve()
        if path.parent!=parts.resolve() or sha(path)!=original['trace_sha256']:raise ValueError('incorrect raw identity/path')
        inventory[str(path.relative_to(root))]=original['trace_sha256']
        if original['condition'] not in CONDITIONS:continue
        state=states[original['key']]
        if original['initial_snapshot_sha256']!=state['initial_state']['snapshot_sha256'] or original['target_id']!=state['target_id'] or original['identity']!=ev['identity']:raise ValueError('unpaired/relabeled source')
        with verified_trace(original) as z:
            validate_arrays(original,z);a=analyze_episode(original,z);schemas.append(signal_availability(z))
            for old,new in [('final_distance_m','final_distance_m'),('final_speed_m_s','final_speed_m_s'),('maximum_speed_m_s','peak_speed_m_s'),('maximum_target_distance_m','maximum_distance_m')]:
                if not math.isclose(original[old],a[new],abs_tol=2e-6,rel_tol=0):raise ValueError('original scalar recomputation mismatch')
            verify_success(original,z)
            if a['post_gust'] is not None:
                onset=a['post_gust']['joint_recovery_onset_s'];lag=None if onset is None else onset-4.
                if original['gust_recovery_onset_lag_s']!=lag:raise ValueError('original recovery metric mismatch')
            a.update(index=state['index'],target_id=state['target_id'],task_key=key,trace_sha256=original['trace_sha256'],
                initial_snapshot_sha256=original['initial_snapshot_sha256'],timeout=original['timeout'],
                physical_failure=original['physical_failure'],command_saturation_elements=original['action_saturation_elements'],
                allocator_saturation_fraction=original['allocator_saturation_fraction'])
            rows.append(a)
    if seen!={s+'/'+c for s in states for c in CONTROLLERS} or len(rows)!=900 or any(schema!=schemas[0] for schema in schemas):raise ValueError('cohort/schema omission')
    group={(c,n):[r for r in rows if r['condition']==c and r['controller']==n] for c in CONDITIONS for n in CONTROLLERS}
    if any(len(v)!=100 or {r['index'] for r in v}!=set(range(100)) for v in group.values()):raise ValueError('bad denominators')
    paired={}
    for n in CONTROLLERS:
        paired[n+'_constant_to_gust_medium']=paired_metrics(rows,n,'constant_medium',n,'gust_medium')
        paired[n+'_gust_medium_to_high']=paired_metrics(rows,n,'gust_medium',n,'gust_high')
    for c in ['gust_medium','gust_high']:
        for n in ['original','yaw_augmented']:paired[c+'_scripted_to_'+n]=paired_metrics(rows,'scripted',c,n,c)
    for name,h in inputs.items():
        if sha(root/name)!=h:raise ValueError('historical input changed during audit')
    for name,h in inventory.items():
        if sha(root/name)!=h:raise ValueError('raw source changed during audit')
    return dict(base_commit=BASE,branch='feat/uav-gust-recovery-audit',experiment='READ-ONLY UAV GUST RECOVERY AUDIT',
        provenance=dict(input_sha256=inputs,raw_trace_sha256=inventory,all1800_source_trace_hashes_unchanged=True,
            source_rows_verified=1800,analyzed_episodes=900,no_new_simulation_episode=True,no_training=True,
            original_metrics_recomputed_consistent=True,condition_initial_physical_controller_pairs_verified=initial_pairs),signal_availability=schemas[0],
        timeline=dict(physics_dt_s=.002,control_dt_s=.01,policy_dt_s=.04,max_episode_s=15.,gust_start_s=2.,gust_end_s=4.,
            post_force_budget_s=11.,success='distance<.10m AND speed<.15m/s for5consecutive policy steps',
            recovery='same thresholds and5complete consecutive observations, but only post4s with full gust exposure',
            observation_dwell_for5_samples_s=.16,original_success_control_intervals_s=.20,force_after4_verified_zero=True),
        definitions=dict(phases={k:[v[0],None if math.isinf(v[1]) else v[1]] for k,v in PHASES.items()},
            phase_statistics='pre-action boundaries with actual interval duration weighting; no posttermination extrapolation',
            aggregate_statistics='equal episode weight, eligibility counts; not pooled unequal exposure',
            opposing_direction='XY dot(command,velocity)<0, gated command>=.02m/s and actual>=.15m/s; can be intentional braking',
            joint_run_duration='span between observed complete boundaries; no extrapolation beyond final frame',
            representative_selection='indices0/33/66/99 fixed before metrics, same targets across all9groups'),
        representatives=list(REPRESENTATIVES),metrics=summarize(rows),paired=paired,episode_metrics=rows,
        pi_source_audit=dict(kp_velocity=[1.5,1.5,2.],ki=[.5,.5,.8],integral_acceleration_limits_xy_norm_z_abs=[1.5,1.5],
            desired_acceleration_limits_xy_norm_z_abs=[3.,3.],update_hz=100,reset='episode reset only; no gust-on/off reset',
            update='integral_error += (world_velocity_command-world_velocity)*.01 before Ki contribution',
            clamping='XY vector-norm clamp of Ki integral; Z component clamp',
            anti_windup='Freeze candidate increment if P+candidateI exceeds acceleration limit and deltaI pushes outward',
            output='Limited world acceleration -> m(a-g) -> desired quaternion/thrust -> attitude PD wrench -> allocator',
            integral_trace='not recorded',pi_output_trace='not recorded',anti_windup_trace='not recorded',
            cannot_identify=['actual integrator saturation/clamping duration','integral retention or reversed compensation',
                'anti-windup events','actual desired acceleration limiting','attitude/rotor transient mechanism']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args()
    result=analyze(a.root);atomic_json(a.root/'mujoco/reports'/PARTS/'analysis.json',result)
    print('Read-only audit:',{k:v for k,v in result['provenance'].items()
        if k not in ('input_sha256','raw_trace_sha256')},flush=True)
