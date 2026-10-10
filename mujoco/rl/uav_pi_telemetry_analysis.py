"""Measured two-episode analysis/publication. No simulator integration or tuning."""
import argparse
import json
from pathlib import Path

import numpy as np

from uav_gust_recovery_audit import sha,digest,stats,verify_success
from uav_bc_safety import atomic_json
from run_uav_pi_internal_telemetry import PARTS,BASE,CONDITIONS,identities,validate_completed

REPORT='uav_pi_internal_telemetry.json'
PHASES={'0_2s':(0,2),'2_4s':(2,4),'4_6s':(4,6),'6_10s':(6,10),'10_15s':(10,15)}
EVENTS=('integral_clamp_xy','integral_clamp_z','anti_windup_freeze_xy','anti_windup_freeze_z',
        'output_limited_xy','output_limited_z','allocator_saturated')


def window_mask(t,start,end):
    t=np.asarray(t);return (t>=start-1e-9)&(t<end-1e-9)


def threshold_gates(error,velocity):
    d=np.linalg.norm(error,axis=1)<.1;v=np.linalg.norm(velocity,axis=1)<.15
    return dict(joint=d&v,distance_only=d&~v,speed_only=v&~d)


def braking_masks(velocity,p,i,output):
    moving=np.linalg.norm(velocity[:,:2],axis=1)>=.15
    dot=lambda a,b:np.einsum('ij,ij->i',a[:,:2],b[:,:2])
    return dict(p_braking_i_accelerating=moving&(dot(p,velocity)<0)&(dot(i,velocity)>0),
        output_braking=moving&(dot(output,velocity)<0),
        p_i_opposition=(np.linalg.norm(p[:,:2],axis=1)>1e-8)&(np.linalg.norm(i[:,:2],axis=1)>1e-8)&(dot(p,i)<0),
        moving=moving)


def zero_crossings(times,values):
    times=np.asarray(times);values=np.asarray(values);index=np.flatnonzero(np.abs(values)>1e-9)
    return [[float(times[a]),float(times[b])] for a,b in zip(index[:-1],index[1:]) if values[a]*values[b]<0]


def attitude_error_degrees(actual,desired):
    actual=np.asarray(actual,float);desired=np.asarray(desired,float)
    actual=actual/np.linalg.norm(actual,axis=1,keepdims=True)
    desired=desired/np.linalg.norm(desired,axis=1,keepdims=True)
    dot=np.abs(np.einsum('ij,ij->i',actual,desired))
    return np.degrees(2*np.arccos(np.clip(dot,0,1)))


def body_z_world(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=1,keepdims=True)
    w,x,y,z=q.T
    return np.c_[2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)]


def phase_metrics(z,start,end):
    m=window_mask(z['control_time_s'],start,end)
    if not m.any():return None
    e=z['control_target_m']-z['control_position_m'];v=z['control_measured_velocity_m_s']
    cmd=z['control_commanded_velocity_m_s'];i=z['control_i_contribution_m_s2'];p=z['control_p_contribution_m_s2_derived']
    out=z['control_output_after_limiting_m_s2'];bm=braking_masks(v,p,i,out)
    gates=threshold_gates(e,v);speed=np.linalg.norm(v,axis=1);dist=np.linalg.norm(e,axis=1)
    def mean(x):return float(np.mean(np.asarray(x)[m]))
    return dict(samples=int(m.sum()),observed_control_hold_duration_s=float(m.sum()*.01),
        distance_mean_m=mean(dist),distance_min_m=float(dist[m].min()),distance_max_m=float(dist[m].max()),
        speed_mean_m_s=mean(speed),speed_max_m_s=float(speed[m].max()),command_speed_mean_m_s=mean(np.linalg.norm(cmd,axis=1)),
        integral_xy_contribution_mean_m_s2=mean(np.linalg.norm(i[:,:2],axis=1)),
        integral_xy_contribution_max_m_s2=float(np.linalg.norm(i[:,:2],axis=1)[m].max()),
        integral_world_mean_m_s2=np.mean(i[m],axis=0).tolist(),
        p_i_opposition_fraction=mean(bm['p_i_opposition']),
        p_braking_i_accelerating_samples=int((bm['p_braking_i_accelerating']&m).sum()),
        output_braking_samples=int((bm['output_braking']&m).sum()),moving_samples=int((bm['moving']&m).sum()),
        distance_only_samples=int((gates['distance_only']&m).sum()),speed_only_samples=int((gates['speed_only']&m).sum()),
        joint_threshold_samples=int((gates['joint']&m).sum()),
        attitude_tracking_error_degrees_mean=mean(attitude_error_degrees(z['control_actual_quaternion_wxyz'],z['control_desired_quaternion_wxyz'])),
        events={f:int(z['control_'+f][m].sum()) for f in EVENTS})


def response_metrics(row,z,direction):
    t=z['control_time_s'];e=z['control_target_m']-z['control_position_m'];v=z['control_measured_velocity_m_s']
    i=z['control_i_contribution_m_s2'];p=z['control_p_contribution_m_s2_derived'];out=z['control_output_after_limiting_m_s2']
    d=np.linalg.norm(e,axis=1);speed=np.linalg.norm(v,axis=1);post=t>=4-1e-9;indices=np.flatnonzero(post)
    at4=int(np.argmin(np.abs(t-4)));peak=int(indices[np.argmax(speed[post])]);dp=int(indices[np.argmax(d[post])])
    axis=np.asarray(direction);ratt=body_z_world(z['control_actual_quaternion_wxyz']);datt=body_z_world(z['control_desired_quaternion_wxyz'])
    projection={'position_error':e@axis,'velocity':v@axis,'commanded_velocity':z['control_commanded_velocity_m_s']@axis,
        'p_contribution':p@axis,'i_contribution':i@axis,'pi_output':out@axis,
        'actual_body_z':ratt@axis,'desired_body_z':datt@axis,'cached_qacc':z['control_cached_qacc_world_m_s2']@axis}
    crossing={name:zero_crossings(t[post],value[post]) for name,value in projection.items()}
    gates=threshold_gates(e,v);close=post&(d<.1);slow=post&(speed<.15)
    boundary_gates=threshold_gates(z['observations'][:, :3],z['observations'][:,3:6])
    checkpoints={}
    for query in (0,2,4,6,10,15):
        if query>t[-1]+.010000001:continue
        k=int(np.argmin(abs(t-query)))
        checkpoints[str(query)]=dict(requested_time_s=query,actual_control_timestamp_s=float(t[k]),distance_m=float(d[k]),speed_m_s=float(speed[k]),
            integral_state_m=z['control_integral_after_m'][k].tolist(),i_contribution_m_s2=i[k].tolist(),p_contribution_m_s2=p[k].tolist(),
            pi_output_m_s2=out[k].tolist(),commanded_velocity_m_s=z['control_commanded_velocity_m_s'][k].tolist(),
            force_world_n=z['control_external_force_world_n'][k].tolist(),
            force_axis_projections={name:float(value[k]) for name,value in projection.items()})
    bm=braking_masks(v,p,i,out)
    return dict(termination={k:row[k] for k in ('steps','simulated_seconds','success','timeout','physical_failure','termination_reason','final_distance_m','final_speed_m_s')},
        control_samples=len(t),physics_force_samples=len(z['physics_force_times']),policy_boundary_samples=len(z['policy_times']),
        events={f:int(z['control_'+f].sum()) for f in EVENTS},
        phases={name:phase_metrics(z,*limits) for name,limits in PHASES.items()},checkpoints=checkpoints,
        integral_xy_contribution_max_m_s2=float(np.linalg.norm(i[:,:2],axis=1).max()),
        post4_peak_speed=dict(time_s=float(t[peak]),speed_m_s=float(speed[peak]),distance_m=float(d[peak]),
            p_projection_m_s2=float(projection['p_contribution'][peak]),i_projection_m_s2=float(projection['i_contribution'][peak]),
            requested_acceleration_projection_m_s2=float(projection['pi_output'][peak]),actual_body_z_projection=float(projection['actual_body_z'][peak])),
        post4_distance_growth=dict(distance_at4_m=float(d[at4]),peak_distance_m=float(d[dp]),peak_time_s=float(t[dp]),excess_over_at4_m=float(d[dp]-d[at4])),
        post4_integral_braking_opposition_samples=int((post&bm['p_braking_i_accelerating']).sum()),
        force_axis_zero_crossing_brackets_s=crossing,
        thresholds=dict(control100hz={key:int((value&post).sum()) for key,value in gates.items()},
            policy25hz={key:int((value&(z['policy_times']>=4-1e-9)).sum()) for key,value in boundary_gates.items()},
            minimum_speed_when_distance_lt010_after4_m_s=float(speed[close].min()) if close.any() else None,
            minimum_distance_when_speed_lt015_after4_m=float(d[slow].min()) if slow.any() else None,
            success_streak_max=int(z['success_streak'].max()),rule='original25Hz: distance<0.10m AND speed<0.15m/s for5consecutive completed policy steps'),
        current_vs_desired_attitude=dict(error_degrees=stats(attitude_error_degrees(z['control_actual_quaternion_wxyz'],z['control_desired_quaternion_wxyz'])),
            note='Measured wxyz quaternions; angular discrepancy and body-Z projections are deterministic derived diagnostics, not causal transfer-function estimates.'))


def resource_report(parts):
    phases={p.stem.removeprefix('resource_'):json.loads(p.read_text()) for p in parts.glob('resource_*.json')}
    if any(r['oom_count_delta'] or r['failure'] or r['workers']!=1 for r in phases.values()):raise ValueError('unsafe/unfinished resource phase')
    expected_failed={'acquisition','validator_red'}
    if any(r['exit_code'] and name not in expected_failed for name,r in phases.items()):raise ValueError('unresolved resource phase failure')
    return dict(workers=1,oom_occurred=False,swap_modified=False,
        peak_sampled_process_tree_rss_bytes=max(r['process_tree_peak_sampled_rss_bytes'] for r in phases.values()),
        maximum_sum_process_hwm_bytes=max(r['process_tree_peak_hwm_bytes'] for r in phases.values()),
        phases=phases,technical_error='Initial acquisition validator failure and its intentional RED test retained; corrected GREEN+bitwise verified acquisition. No simulation failure/OOM.')


def conclusions():
    return {
        'A_integral_retention':dict(grade='Confirmed',finding='Nonzero bounded integral persists after force removal, can oppose P braking in part of the response. This is integral memory, not proof of pathological windup.'),
        'A_abnormal_integral_or_windup':dict(grade='Unresolved',finding='No clamp, anti-windup freeze, output/allocator saturation occurs in either episode; saturation-driven windup explanation not supported. Retention causal contribution is not isolated.'),
        'B_integral_clamping':dict(grade='Confirmed',finding='No actual XY or Z integral-clamp event in either selected episode.'),
        'C_braking_request':dict(grade='Confirmed',finding='P and then committed PI output request reversal/braking before actual body thrust direction reverses. No evidence of a stale software command or omitted100Hz updates.'),
        'C_dynamic_lag':dict(grade='Supported hypothesis',finding='Measured desired/actual attitude separation plus thrust-direction reversal timing supports finite attitude/plant response lag contributing to speed overshoot; not a unique-cause proof.'),
        'D_oscillatory_response':dict(grade='Confirmed',finding='Repeated signed position-error/velocity reversals after force removal and decreasing but nonsettled motion within15s. Underdamped cascade interpretation is supported, not an identified linear-system damping ratio.'),
        'E_success_criterion':dict(grade='Confirmed',finding='Gust fails even instantaneous joint position+speed gates at100Hz as well as25Hz; not merely a5-step hold artifact. Finite timeout censors eventual settling; no evidence to change task criterion.'),
        'F_specific_failure_link':dict(grade='Supported hypothesis',finding='Shared velocity-PI→attitude→physical translation cascade exhibits integral/P phase opposition and delayed actual thrust reversal; waypoint crossings occur at high speed while low-speed turning points stay outside position threshold. No single module uniquely proven causal.'),
        'F_generalization':dict(grade='Unresolved',finding='Only one target/direction, Scripted, Medium telemetry. No new High/BC internal data; do not claim this uniquely establishes the mechanism of all600 historical failures.'),
        'controller_change_required':dict(grade='Unresolved',finding='Evidence warrants a narrowly controlled lower-cascade validation before tuning, not automatic PI gain/reset/anti-windup modification.'),
        'ppo_sac_readiness':dict(grade='Unresolved',finding='Nominal policy baselines exist, but shared disturbance-recovery limitation is not fixed or uniquely isolated. Do not claim PPO/SAC can resolve it or formally start training in this task.'),
        'next_single_recommendation':'One separate, same-state offline/local attitude-cascade response validation to distinguish requested braking from actual thrust-direction lag, retaining current gains and task criteria; not executed.'}


def publish(root,finalize=False):
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS
    ev=json.loads((parts/'acquisition.json').read_text());identity=ev['identity']
    before=identities(root)
    if before!=ev['original_hashes_before'] or before!=ev['original_hashes_after']:raise ValueError('original source/checkpoint changed')
    analyses={};arrays={};records=[]
    for c in CONDITIONS:
        r=validate_completed(json.loads((parts/f'{c}.json').read_text()),identity,c)
        with np.load(r['raw_path'],allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
        verify_success(r,z)
        analyses[c]=response_metrics(r,z,identity['selected_states'][0]['direction_world']);arrays[c]=z
        records.append({k:v for k,v in r.items() if k not in ('identity','raw_path','reference_json')})
    if {r['task_id'] for r in records}!=set(CONDITIONS):raise ValueError('invalid cohort')
    if finalize:
        previous=json.loads((reports/REPORT).read_text())
        if previous['measured_results']!=analyses:raise ValueError('analysis not reproducible')
        figures=previous['figures']
        if any(sha(reports/n)!=h for n,h in figures.items()):raise ValueError('figure modified')
    else:
        from uav_pi_telemetry_plots import plots
        figures=plots(root,arrays,identity['selected_states'][0]['direction_world'])
    tests=json.loads((parts/'test_evidence.json').read_text())
    if tests['failures'] or tests['errors']:raise ValueError('algorithmic tests failed')
    value=dict(experiment='MINIMAL PI GUST RECOVERY TELEMETRY',date='2026-10-10',branch='feat/uav-pi-internal-telemetry',base_commit=BASE,
        design=dict(unique_episodes=2,total_flight_executions_including_technical_repeat=ev['total_flight_executions_including_technical_repeat'],
            no_training=True,no_controller_changes=True,no_physics_changes=True,workers=1,target=identity['selected_states'][0],
            force_n=6.867,force_schedules=dict(constant='all physics ticks',gust='ticks1000..1999, t2inclusive..4exclusive'),
            physics_dt_s=.002,control_dt_s=.01,policy_dt_s=.04,max_episode_s=15.),
        instrumentation=dict(method='externalCPython sys.settrace, exact function/frame identity and source-AST line probes; copies only; production sources unchanged',
            time='100Hz entry/PI-update pre-integration timestamp; post-updateI/command share that time; termination only original25Hz boundaries',
            force_alignment='actual applied500Hz force indexed to control tick after real force hook; no stale previous-tick buffer',
            p_term='Derived using recorded kv and actual velocity error; no separately stored P variable. Not an independent measured variable.',
            output_before_limit='Actual helper acceleration local after anti-windup committedI, before XY/Z clipping',
            anti_windup_trial='Actual raw local before conditionalIfreeze; distinct from committed pre-limit output',
            commanded_thrust='Actual allocator achieved_wrench[0] plus actual data.ctrl; command, not measured aerodynamic/rotor thrust',
            cached_qacc='Actual pre-update MjData.qacc cache from prior physics integration, NOT instantaneous derivative of the new command; boundary t4 may reflect final forced tick',
            not_applicable=['positionPI integrator: Scripted uses proportional position→velocity only','EMA/target encoder/model learning','separate anti-windup back-calculation state: implementation uses counters/conditional freeze'],
            not_observable=['independent measured rotor aerodynamic thrust; command and physical attitude are observed','hypothetical dynamics with I disabled or different gains; no intervention performed']),
        controller_code_audit=dict(scripted_position_velocity_gain_per_s=1/3,velocity_kp_per_s=[1.5,1.5,2.],velocity_ki_per_s2=[.5,.5,.8],
            integral_state='VelocityCommandController._integral_error, metres, updated at100Hz',
            integral_caps=dict(xy_I_acceleration_norm_m_s2=1.5,z_I_acceleration_abs_m_s2=1.5),
            anti_windup='Freeze candidateI update when trial acceleration exceeds3m/s² AND proposedI increment worsens saturated demand; then clip committed output.',
            reset='reset zerosI,last acceleration,counters and wraps yaw target; full snapshot restoresMjData/controller/allocator/environment/RNG.',
            attitude='Quaternion PD kp=[6,6,2],kd=[4,4,2.8]; inertia torque+gyro compensation; bounded rotor allocator; no motor lag added.'),
        initial_physical_controller_sha256=ev['initial_physical_controller_sha256'],original_hashes_before=before,original_hashes_after=identities(root),
        acquisition_identity=identity,records=records,measured_results=analyses,interpretation=conclusions(),
        reproducibility=dict(two_unique_tasks=True,no_omissions_or_duplicates=True,full_baseline_arrays_bitwise_equal=ev['bitwise_reproduced_all_original_traces'],
            verification_new_episodes=json.loads((parts/'verification.json').read_text())['new_episodes_this_invocation'],
            technical_repeat=json.loads((parts/'verifier_recovery.json').read_text()) if (parts/'verifier_recovery.json').exists() else None),
        resources=resource_report(parts),tests=tests,figures=figures,
        analysis_source_sha256={n:sha(root/'mujoco/rl'/n) for n in ('uav_pi_telemetry_analysis.py','uav_pi_telemetry_plots.py','test_uav_pi_internal_telemetry.py')},
        limitations=['one preselected target/direction','Scripted Medium only','100Hz internals,500Hz force,25Hz original success','finite15s timeout, no asymptotic stability proof',
            'no causalI/attitude intervention','cannot establish unique cause for all600 earlier gust failures'],finalized=finalize)
    atomic_json(reports/REPORT,value)
    print('Measured report published; unique episodes2, flight executions',value['design']['total_flight_executions_including_technical_repeat'],flush=True)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');a=p.parse_args()
    publish(Path(__file__).resolve().parents[2],a.finalize)
