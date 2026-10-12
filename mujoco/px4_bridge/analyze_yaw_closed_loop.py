"""Offline-only analysis of the single Phase2F restricted yaw SITL recording."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'mujoco/reports/px4_yaw_closed_loop_parts/yaw_01/smoke_summary.json'
OUT=ROOT/'mujoco/reports/px4_mujoco_yaw_closed_loop_validation.json'
FIG=ROOT/'mujoco/reports/figures/px4_yaw_closed_loop'
PROOF=ROOT/'mujoco/reports/px4_disarmed_verification_parts/disarmed_01/smoke_summary.json'
PROOF_REPORT=ROOT/'mujoco/reports/px4_mujoco_disarmed_verification.json'
PX4_ROOT=Path('/home/keyousuke/PX4-Autopilot-mineuav-yaw')
IDENTITY_MANIFEST=ROOT/'mujoco/px4_bridge/compat/px4_yaw_build_identity.json'
EXPECTED_RAW_SHA256='edf5a709c307bc9972f28d27d94b23edc1e1aa08be540d7ede3be49be92278da'
EXPECTED_CONSOLE_SHA256='24efc147409a1c3855d4b4354c7cbb9c5c53ec579dd018a3e37a4dc3fb24e471'
EXPECTED_ULOG_SHA256='a3bf79d84d4548eed3714f7f7ac45f1f73992aafad6bdafb6cd8dfed408fb754'


def onset_after_step(times, values, step, threshold, direction=1):
    """First of two consecutive above-threshold samples at/after the step."""
    t=np.asarray(times,dtype=float);v=np.asarray(values,dtype=float)
    if (t.ndim!=1 or v.shape!=t.shape or not np.isfinite(t).all()
        or not np.isfinite(v).all() or not np.isfinite([step,threshold]).all()
        or threshold<=0 or direction not in (-1,1) or np.any(np.diff(t)<=0)):
        raise ValueError('finite, aligned, strictly increasing sampled signal required')
    hit=(t>=step)&(direction*v>threshold)
    for i in range(len(t)-1):
        if hit[i] and hit[i+1]:return float(t[i])
    return None


def motor_model_response(normalized_controls,max_thrust_n,max_reaction_nm,rotation_signs):
    """Model-derived per-motor thrust and reaction torque; not a force sensor."""
    c=np.asarray(normalized_controls,dtype=float)
    signs=np.asarray(rotation_signs,dtype=float)
    if (c.ndim!=2 or c.shape[1]!=4 or signs.shape!=(4,)
        or not np.isfinite(c).all() or not np.isfinite(signs).all()
        or np.any((c<0)|(c>1)) or not np.isin(signs,[-1,1]).all()
        or not np.isfinite([max_thrust_n,max_reaction_nm]).all()
        or min(max_thrust_n,max_reaction_nm)<=0):
        raise ValueError('four bounded nonreversible motor commands and physical coefficients required')
    reaction=-c*signs*max_reaction_nm # FLU-z to FRD-z changes sign.
    return {'thrust_n':c*max_thrust_n,'reaction_torque_frd_nm':reaction,
        'total_reaction_yaw_frd_nm':reaction.sum(axis=1)}


def control_chain_status(evidence):
    """Never infer physical yaw success from estimator or allocator reports alone."""
    step=evidence['step']
    if step is None:
        return {'step_issued':'NOT TESTED','attitude_rate_request':'NOT TESTED',
            'allocator_torque':'NOT TESTED','motor_differential':'NOT TESTED',
            'physical_yaw_direction':'NOT TESTED','yaw_settling':'NOT TESTED',
            'safety_monitor':'PASS' if evidence['query_healthy'] else 'FAIL',
            'control_chain_prerequisite':'BLOCKED','phase3_entry':'NOT TESTED'}
    complete=bool(evidence['complete'])
    status={'step_issued':'PASS' if step>=32 and evidence['px4_setpoint_observed'] else 'FAIL',
        'attitude_rate_request':'PASS' if evidence['rate_request_observed'] and evidence['torque_request_observed'] else 'FAIL',
        'allocator_torque':'PASS' if evidence['allocator_sample_count']>0
            and evidence['allocator_achieved_count']==evidence['allocator_sample_count']
            and evidence['unallocated_yaw_max_abs'] is not None
            and evidence['unallocated_yaw_max_abs']<1e-3 else 'FAIL',
        'motor_differential':'PASS' if evidence['motor_yaw_torque_max_nm'] is not None
            and evidence['motor_yaw_torque_max_nm']>.01 and evidence['bounds_valid']
            and evidence['motor_saturation_count']==0 else 'FAIL',
        'physical_yaw_direction':'PASS' if evidence['physical_change_deg'] is not None
            and evidence['physical_change_deg']>1 else 'FAIL',
        'yaw_settling':'BLOCKED' if not complete else 'PASS' if evidence['settling_time_s'] is not None else 'FAIL',
        'safety_monitor':'PASS' if complete and evidence['query_healthy'] and evidence['ekf_healthy'] else 'FAIL',
        'phase3_entry':'NOT TESTED'}
    status['control_chain_prerequisite']='PASS' if all(value=='PASS' for key,value in status.items()
        if key not in ('phase3_entry','control_chain_prerequisite')) else 'BLOCKED'
    return status


def _physical_record_time_coverage(records,duration,final):
    """Check that saved 50 Hz truth spans the complete flown state, not a prefix."""
    try:
        if not records or not isinstance(duration,(int,float)) or not np.isfinite(duration):return False
        t=np.asarray([r['time_s'] for r in records],dtype=float)
        clock=np.asarray([r['clock_usec'] for r in records],dtype=float)
        if (not np.isfinite(t).all() or not np.isfinite(clock).all()
            or t[0]<0 or t[0]>.0241 or abs(t[-1]-duration)>.0041
            or np.any(np.diff(t)<=0) or np.any(np.diff(t)>.0241)
            or np.any(np.diff(clock)<=0)
            or np.max(np.abs(clock-(1000000+np.rint(t*1e6))))>1):return False
        qpos=np.asarray(final['qpos'],dtype=float)
        qvel=np.asarray(final['qvel'],dtype=float)
        last=records[-1]
        last_qpos=np.r_[last['position_enu_m'],last['quaternion_enu_flu']]
        last_qvel=np.r_[last['velocity_origin_enu_m_s'],last['angular_velocity_flu_rad_s']]
        return bool(qpos.shape==(7,) and qvel.shape==(6,)
            and np.isfinite(np.r_[qpos,qvel]).all()
            and np.allclose(qpos,last_qpos,atol=1e-8,rtol=0)
            and np.allclose(qvel,last_qvel,atol=1e-8,rtol=0))
    except (KeyError,TypeError,ValueError,IndexError):return False


def flight_safety_checks(raw):
    """Fail-closed read-only checks on the actual bounded flight record."""
    checks={}
    def check(name,passed):checks[name]='PASS' if passed else 'FAIL'
    phase=raw.get('phase2') or {};fresh=phase.get('telemetry_fix') or {}
    metrics=raw.get('runtime_measurements') or {};samples=metrics.get('samples') or {}
    queries=metrics.get('queries') or []
    query_events=[e for e in fresh.get('events',[]) if e.get('kind')=='query']
    duration=raw.get('simulation_seconds');final=(raw.get('final_physical_state') or {}).get('time_s')
    finite=lambda value:isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value)
    check('complete_45s',raw.get('complete') is True and finite(duration)
        and 45-1e-8<=duration<=45.008 and finite(final) and abs(duration-final)<1e-8
        and raw.get('exception') is None)
    check('armed_only_after_request',raw.get('kind')=='bounded_yaw_validation'
        and raw.get('flight_command_sent') is True and raw.get('arm_request_at_s',0)>=11
        and {(176,0),(400,0)}<=set((ack.get('command'),ack.get('result')) for ack in raw.get('command_acks',[])))
    check('shutdown',raw.get('px4_exit_code')==0 and raw.get('zero_control_on_stop')==[0,0,0,0])
    check('health',fresh.get('fault') is None and (fresh.get('last_decision') or {}).get('safe_stop') is False
        and (fresh.get('last_decision') or {}).get('query_pending') is False)
    def within_envelope(record):
        try:
            p=np.asarray(record['position_enu_m'],dtype=float)
            v=np.asarray(record['velocity_origin_enu_m_s'],dtype=float)
            w=np.asarray(record['angular_velocity_flu_rad_s'],dtype=float)
            q=np.asarray(record['quaternion_enu_flu'],dtype=float)
            if (p.shape!=(3,) or v.shape!=(3,) or w.shape!=(3,) or q.shape!=(4,)
                or not np.isfinite(np.r_[p,v,w,q]).all() or abs(np.linalg.norm(q)-1)>1e-6):return False
            tilt=np.arccos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))
            return bool(-.01<=p[2]<=2.6 and np.linalg.norm(p[:2])<=.5
                and np.linalg.norm(v)<=1.5 and np.linalg.norm(w)<=2.5
                and tilt<=np.deg2rad(20))
        except (KeyError,TypeError,ValueError):return False
    records=raw.get('records') or []
    check('sampled_flight_envelope',bool(records) and all(within_envelope(record) for record in records))
    check('physical_record_time_coverage',_physical_record_time_coverage(records,duration,
        raw.get('final_physical_state') or {}))
    query_ok=(len(queries)>=70 and len(queries)==len(query_events)
        and len(queries)==(metrics.get('scheduler') or {}).get('starts')
        and len(queries)==metrics.get('row_valid_at_completions'))
    last_end_wall=None;last_start_wall=None;last_start_clock=None
    for query,event in zip(queries,query_events):
        try:
            start=query['start_wall_s'];end=query['end_wall_s']
            start_clock=query['start_clock_usec'];end_clock=query['end_clock_usec']
            query_ok &= (all(finite(x) for x in (start,end,start_clock,end_clock))
                and 0<=start<=end and end-start<=.15+1e-9
                and 0<=end_clock-start_clock<=150000
                and start_clock==event['clock_usec'] and abs(start-event['wall_s'])<1e-6)
            if last_start_wall is not None:
                query_ok &= (last_end_wall<=start and 0<start-last_start_wall<=.65+1e-9
                    and 0<start_clock-last_start_clock<=650000)
            last_end_wall=end;last_start_wall=start;last_start_clock=start_clock
        except (KeyError,TypeError,ValueError):query_ok=False
    decision=fresh.get('last_decision') or {}
    if queries and finite(final):
        final_clock=1000000+round(final*1e6)
        query_ok &= (7000000<=queries[0]['start_clock_usec']<=7650000
            and queries[-1]['start_clock_usec']>=final_clock-650000
            and finite(decision.get('clock_usec')) and 0<=final_clock-decision['clock_usec']<=8000
            and finite(decision.get('wall_s')) and finite(raw.get('wall_seconds'))
            and 0<=raw['wall_seconds']-decision['wall_s']<=.65)
    check('query_coverage_and_deadlines',bool(queries) and bool(query_ok)
        and (samples.get('query_response_seconds') or {}).get('n')==len(queries)
        and finite((samples.get('query_response_seconds') or {}).get('max'))
        and samples['query_response_seconds']['max']<=.15+1e-9)
    actuator=samples.get('actuator_interval_wall_seconds') or {}
    received=(raw.get('messages_received') or {}).get('HIL_ACTUATOR_CONTROLS',0)
    check('actuator_watchdog',received>=11000 and actuator.get('n')==received-1-raw.get('stale_bootstrap_messages_discarded',0)
        and finite(actuator.get('max')) and actuator['max']<=.25+1e-9)
    check('bounded_parser',finite(metrics.get('parser_peak_buffer_chars'))
        and metrics['parser_peak_buffer_chars']<=32768
        and finite(metrics.get('recent_peak_chars')) and metrics['recent_peak_chars']<=65536
        and metrics.get('dropped')=={})
    check('identity_preserved',raw.get('history_unchanged') is True
        and bool(raw.get('history_hashes_before'))
        and raw.get('history_hashes_after')==raw.get('history_hashes_before'))
    return {'checks':checks,'overall':'PASS' if all(value=='PASS' for value in checks.values()) else 'FAIL',
        'first_failed_checks':[name for name,value in checks.items() if value=='FAIL']}


def _finite_onset(times,values,step,threshold):
    t=np.asarray(times,dtype=float);v=np.asarray(values,dtype=float)
    valid=np.isfinite(t)&np.isfinite(v)
    return onset_after_step(t[valid],v[valid],step,threshold) if valid.sum()>=2 else None


def _last_valid(times,values):
    t=np.asarray(times,dtype=float);v=np.asarray(values,dtype=float)
    good=np.flatnonzero(np.isfinite(t)&np.isfinite(v))
    return None if not len(good) else {'time_s':float(t[good[-1]]),'value':float(v[good[-1]])}


def build_report(raw_path=RAW,write=False,external_peak_rss_kib=None):
    """Recompute all decisions from frozen raw evidence; never launches SITL."""
    import mujoco
    from .actuator_adapter import ActuatorAdapter
    from .allocation_math import physical_audit
    from .allocation_runner import AllocationYawProtocol,allocation_status
    from .analyze_disarmed_verification import assess_run
    from .analyze_yaw import analyze_run,load_topics,simulation_time,yaw_signal
    from .analyze_yaw_allocation import motor_response
    from .analyze_yaw_telemetry import distribution,load_auxiliary,response_metrics
    from .diagnostics import atomic_json,sha256
    from .magnetic_model import field_from_px4
    from .report_disarmed_verification import PROTECTED_MODELS,console_evidence
    from .simulation_runner import MODEL,CONFIG,source_identity,streaming_runtime_identity,validate_streaming_proof
    from .yaw_validation import angle_error

    raw_path=Path(raw_path).resolve()
    if sha256(raw_path)!=EXPECTED_RAW_SHA256:raise ValueError('raw summary SHA256 differs from the sole frozen flight')
    raw=json.loads(raw_path.read_text());proof=json.loads(PROOF.read_text())
    proof_report=json.loads(PROOF_REPORT.read_text())
    if (sha256(PROOF)!=proof_report['raw_evidence']['summary_sha256']
        or assess_run(proof)['overall']!='PASS'):
        raise ValueError('Phase2E proof is missing, changed, or no longer accepted')
    identity=source_identity(PX4_ROOT,IDENTITY_MANIFEST)
    runtime_hashes=streaming_runtime_identity()
    validate_streaming_proof(proof,runtime_hashes)
    if raw.get('streaming_runtime_hashes')!=runtime_hashes or raw.get('px4')!=identity:
        raise ValueError('flown runtime or PX4 identity differs from current verified source')
    model=mujoco.MjModel.from_xml_path(str(MODEL));adapter=ActuatorAdapter(model,CONFIG)
    history={'model_xml':sha256(MODEL),'rotor_config':sha256(CONFIG),
        'rotor_actuators':sha256(MODEL.parent/'rotor_actuators.xml')}
    protocol=AllocationYawProtocol(field_from_px4(PX4_ROOT),np.asarray(physical_audit(adapter)['raw_px4_matrix'])[2],
        active=True,identity=identity,history=history)
    protocol.validate_proof(proof)
    matched_protocol={name:proof.get(name)==raw.get(name) for name in
        ('px4_environment','initial_qpos','synthetic_sensor_noise')}
    matched_protocol['magnetic_field']=proof['phase2']['field_sha256']==raw['phase2']['field_sha256']
    matched_protocol['model_history']=raw.get('history_hashes_before')==history==raw.get('history_hashes_after')
    if not all(matched_protocol.values()):raise ValueError('flown model/startup/sensor protocol differs from Phase2E proof')
    if mujoco.__version__!='3.14.0' or abs(float(model.body_mass.sum())-7.)>1e-9:
        raise ValueError('MuJoCo version or vehicle mass changed')
    models={name:sha256(ROOT/name) for name in PROTECTED_MODELS}
    if models!=PROTECTED_MODELS:raise ValueError('protected historical model changed')
    logs=sorted(raw_path.parent.rglob('*.ulg'))
    if len(logs)!=1:raise ValueError('exactly one PX4 ULog required for the one flown episode')
    console_path=raw_path.parent/'px4_console.bin'
    if not console_path.exists():raise ValueError('raw PX4 console archive missing')
    if sha256(console_path)!=EXPECTED_CONSOLE_SHA256 or sha256(logs[0])!=EXPECTED_ULOG_SHA256:
        raise ValueError('raw console/ULog SHA256 differs from the sole frozen flight')
    flight,trace=analyze_run(raw_path)
    aux,params=load_auxiliary(logs[0],flight['simulation_end_s'])
    topics,_=load_topics(logs[0],flight['simulation_end_s'])
    expected_params=adapter.px4_geometry()
    geometry_matches={name:name in params and bool(np.isclose(float(params[name]),float(value),atol=1e-7,rtol=1e-6))
        for name,value in expected_params.items()}
    if not all(geometry_matches.values()):raise ValueError('actual ULog allocator/motor parameters do not match model')
    audit=physical_audit(adapter)
    console=console_evidence(raw)
    expected_yaw=np.asarray(audit['raw_px4_matrix'])[2]
    if not (console['allocator_row_all_match'] and console['allocator_row_records']>=len(raw['runtime_measurements']['queries'])
        and allocation_status(raw['px4_console'],expected_yaw)):
        raise ValueError('flown allocator row not consistently equal to verified physical mapping')

    step=flight['actual_step_time_s'];response=response_metrics(trace['t'],trace['true'],trace['rate'],step,flight['complete'])
    post=trace['t']>=step if step is not None else np.zeros(len(trace['t']),dtype=bool)
    motor=motor_response(trace['t'],trace['motors'],np.asarray(audit['physical_com_frd_matrix_per_normalized_motor'])[2],step)
    physical_motor=motor_model_response(trace['motors'],audit['maximum_motor_thrust_n'],
        audit['maximum_motor_reaction_torque_nm'],adapter.signs)
    physical_yaw_model=trace['motors']@np.asarray(audit['physical_com_frd_matrix_per_normalized_motor'])[2]
    if not np.allclose(physical_yaw_model,physical_motor['total_reaction_yaw_frd_nm'],atol=1e-10,rtol=1e-10):
        raise ValueError('motor force/reaction torque mapping inconsistency')
    current_times=trace['t'];desired=trace['desired']
    attitude_sp=topics.get('vehicle_attitude_setpoint',{})
    attitude_sp_times=simulation_time(attitude_sp.get('timestamp',[]))
    attitude_sp_yaw=yaw_signal(attitude_sp,'q_d') if attitude_sp else np.array([])
    setpoint_onset=_finite_onset(attitude_sp_times,
        np.degrees(angle_error(attitude_sp_yaw,np.deg2rad(90))),step,2.) if step is not None else None
    rate_data=aux.get('vehicle_rates_setpoint',{});torque_data=aux.get('vehicle_torque_setpoint',{})
    rate_onset=_finite_onset(simulation_time(rate_data.get('timestamp',[])),rate_data.get('yaw',[]),step,.02) if step is not None else None
    torque_onset=_finite_onset(simulation_time(torque_data.get('timestamp',[])),torque_data.get('xyz[2]',[]),step,.005) if step is not None else None
    motor_onset=_finite_onset(current_times,physical_yaw_model,step,.01) if step is not None else None
    physical_onset=_finite_onset(current_times,trace['rate'],step,.02) if step is not None else None
    first_post=int(np.flatnonzero(post)[0]) if post.any() else None
    actual_change=float(np.degrees(angle_error(trace['true'][-1],trace['true'][first_post]))) if first_post is not None else None
    alloc=aux.get('control_allocator_status',{})
    amask=simulation_time(alloc['timestamp'])>=step if alloc and step is not None else np.zeros(len(alloc.get('timestamp',[])),dtype=bool)
    allocated=int(np.sum(alloc['torque_setpoint_achieved'][amask])) if alloc else 0
    alloc_n=int(amask.sum());unallocated=distribution(alloc['unallocated_torque[2]'][amask]) if alloc else None
    alloc_sat=[int(np.count_nonzero(alloc[f'actuator_saturation[{i}]'][amask])) for i in range(4)] if alloc else None
    position=trace['position'];velocity=trace['velocity'];quats=np.asarray([row['quaternion_enu_flu'] for row in raw['records']])
    tilts=np.degrees(np.arccos(np.clip(1-2*(quats[:,1]**2+quats[:,2]**2),-1,1)))
    local=topics.get('vehicle_local_position',{});flags=topics.get('estimator_status_flags',{});vehicle=topics.get('vehicle_status',{})
    def all_after(topic,fields):
        if not topic or step is None:return False
        mask=simulation_time(topic['timestamp'])>=step
        return bool(mask.any() and all(np.all(topic[name][mask]==value) for name,value in fields.items()))
    ekf_healthy=(all_after(local,{'heading_good_for_control':1})
        and all_after(flags,{'cs_mag_aligned_in_flight':1,'cs_yaw_align':1,'cs_in_air':1,'cs_gnss_pos':1})
        and all_after(vehicle,{'arming_state':2,'nav_state':14,'failsafe':0,'pre_flight_checks_pass':1}))
    safety=flight_safety_checks(raw)
    motor_sat=int(motor['motor_saturation_samples'])+sum(alloc_sat or [])
    evidence={'step':step,'complete':flight['complete'],'settling_time_s':response['settling_time_s'],
        'physical_change_deg':actual_change,'motor_yaw_torque_max_nm':motor['maximum_physical_yaw_torque_nm'],
        'allocator_achieved_count':allocated,'allocator_sample_count':alloc_n,
        'unallocated_yaw_max_abs':None if unallocated is None else unallocated['max_abs'],
        'motor_saturation_count':motor_sat,'bounds_valid':motor['bounds_valid'],
        'ekf_healthy':ekf_healthy,'query_healthy':safety['overall']=='PASS',
        'px4_setpoint_observed':setpoint_onset is not None,
        'rate_request_observed':rate_onset is not None,'torque_request_observed':torque_onset is not None}
    status=control_chain_status(evidence)
    status['flight_safety']='PASS' if safety['overall']=='PASS' else 'FAIL'
    status['final_ekf_heading_alignment']='PASS' if ekf_healthy else 'FAIL'
    status['yaw_allocator_row']='PASS'
    status['no_motor_saturation']='PASS' if motor_sat==0 else 'FAIL'
    status['bounded_height_xy_coupling']=('PASS' if post.any()
        and np.max(np.abs(position[post,2]-2.051))<.15
        and np.max(np.linalg.norm(position[post,:2],axis=1))<.2 else 'FAIL')

    queries=raw['runtime_measurements']['queries'];starts=np.asarray([q['start_wall_s'] for q in queries])
    durations=np.asarray([q['end_wall_s']-q['start_wall_s'] for q in queries]);query_stats=distribution(durations)
    report={'phase':'PX4-MuJoCo Phase2F Complete Yaw Closed-Loop Validation',
        'date':'2026-10-12','branch':'feat/px4-mujoco-yaw-closed-loop-validation',
        'base_commit':'10c70d0cfa53234d241d476b79002b675d9422b8',
        'experiment':{'sitl_runs':1,'armed_runs':1,'workers':raw['workers'],'mode':'allocation_runner --mode yaw',
            'simulation_seconds':raw['simulation_seconds'],'bridge_wall_seconds':raw['wall_seconds'],
            'physics_hz':raw['physics_hz'],'sensor_hz':raw['sensor_hz'],
            'initial_target_yaw_ned_deg':90.,'final_target_yaw_ned_deg':100.,
            'requested_step_not_before_s':32.,'actual_step_s':step,'target_height_ned_m':-2.,
            'settling_rule':'truth |wrap(100deg-yaw)|<2deg AND |FRD body-z rate|<0.05rad/s continuously >=2s; sample gaps <=0.15s'},
        'status':status,'preflight':{'phase2e_proof':'PASS','protocol_match':matched_protocol,
            'actual_ulog_allocator_parameters_match':geometry_matches,'mujoco_version':mujoco.__version__,
            'vehicle_mass_kg':float(model.body_mass.sum()),'protected_models_unchanged':True},
        'raw_evidence':{'summary_path':str(raw_path.relative_to(ROOT)),'summary_sha256':sha256(raw_path),
            'console_path':str(console_path.relative_to(ROOT)),'console_sha256':sha256(console_path),
            'ulog_path':str(logs[0].relative_to(ROOT)),'ulog_sha256':sha256(logs[0]),
            'phase2e_proof_sha256':sha256(PROOF),'runtime_source_hashes':runtime_hashes,'px4':identity,
            'model_hashes':history,'raw_local_only':True},
        'yaw_response':{**response,'true_yaw_at_step_deg':float(np.degrees(trace['true'][first_post])) if first_post is not None else None,
            'true_yaw_final_deg':float(np.degrees(trace['true'][-1])),
            'actual_yaw_change_deg':actual_change,
            'last_supported_estimated_yaw':_last_valid(current_times,np.degrees(trace['estimate'])),
            'last_supported_px4_desired_yaw':_last_valid(current_times,np.degrees(desired)),
            'estimate_minus_truth_poststep_deg':flight['poststep_estimate_minus_truth_deg'],
            'response_onsets_s':{'px4_attitude_setpoint':setpoint_onset,'px4_yaw_rate_setpoint':rate_onset,
                'px4_yaw_torque_request':torque_onset,'model_derived_motor_yaw_torque':motor_onset,
                'physical_yaw_rate':physical_onset},
            'yaw_control_start_delay_s':None if physical_onset is None or step is None else physical_onset-step,
            'onset_rules':'Two consecutive samples above: desired yaw change >2deg, requested/actual yaw rate >0.02rad/s, requested PX4 yaw torque >0.005 normalized, model-derived yaw torque >0.01Nm. Discrete first-crossing bounds include the source sample interval; they are not pure component latencies.'},
        'control_allocation':{'yaw_row_px4':expected_yaw.tolist(),'queried_row_records':console['allocator_row_records'],
            'queried_rows_all_match':console['allocator_row_all_match'],
            'requested_yaw_torque_px4_units':distribution(torque_data['xyz[2]'][simulation_time(torque_data['timestamp'])>=step]) if torque_data and step is not None else None,
            'requested_yaw_rate_rad_s':distribution(rate_data['yaw'][simulation_time(rate_data['timestamp'])>=step]) if rate_data and step is not None else None,
            'unallocated_yaw_torque_px4_units':unallocated,'torque_setpoint_achieved_count':allocated,
            'allocator_poststep_samples':alloc_n,'allocator_saturation_count_by_motor':alloc_sat,
            'px4_units_note':'PX4 normalized torque command/shortfall, not Nm.'},
        'motors':{'normalized_output_poststep':[distribution(trace['motors'][post,i]) for i in range(4)],
            'model_derived_thrust_n_poststep':[distribution(physical_motor['thrust_n'][post,i]) for i in range(4)],
            'model_derived_reaction_torque_frd_nm_poststep':[distribution(physical_motor['reaction_torque_frd_nm'][post,i]) for i in range(4)],
            'net_yaw_torque_frd_nm_poststep':motor['physical_yaw_torque_model_derived_not_measured'],
            'first_positive_model_yaw_torque_s':motor_onset,'positive_differential_max':motor['positive_differential_max'],
            'command_saturation_samples':motor['motor_saturation_samples'],
            'rotation_signs_flu':adapter.signs.tolist(),'max_thrust_per_motor_n':audit['maximum_motor_thrust_n'],
            'max_reaction_torque_per_motor_nm':audit['maximum_motor_reaction_torque_nm'],
            'assumption':'No ESC lag. Forces and reaction torques are reconstructed from verified model kf/km and executed omega^2, not measured hardware forces.'},
        'flight_state':{'height_poststep_error_m':distribution(position[post,2]-2.051),
            'max_abs_height_poststep_error_m':float(np.max(np.abs(position[post,2]-2.051))) if post.any() else None,
            'max_xy_poststep_m':float(np.max(np.linalg.norm(position[post,:2],axis=1))) if post.any() else None,
            'max_xy_whole_episode_m':flight['maximum_xy_excursion_m'],
            'max_speed_whole_episode_m_s':flight['maximum_speed_m_s'],
            'max_tilt_poststep_deg':float(np.max(tilts[post])) if post.any() else None,
            'final_position_enu_m':position[-1].tolist(),'final_velocity_enu_m_s':velocity[-1].tolist(),
            'first_final_heading_good_s':flight['alignment']['first_final_heading_good_s'],
            'first_inflight_mag_alignment_s':flight['alignment']['first_inflight_mag_alignment_s'],
            'poststep_ekf_healthy':ekf_healthy,'px4_estimator_source':'ULog timestamped uORB, not physical truth'},
        'safety':{'checks':safety['checks'],'overall':safety['overall'],
            'last_gate_decision':raw['phase2']['telemetry_fix']['last_decision'],
            'monitor_fault':raw['phase2']['telemetry_fix']['fault'],
            'query_count':len(queries),'query_interval_wall_s':distribution(np.diff(starts)),
            'query_response_recomputed_wall_s':query_stats,
            'query_service_timeout_count_observed':0 if safety['checks']['query_coverage_and_deadlines']=='PASS' else None,
            'actuator_wall_interval_s':raw['runtime_measurements']['samples']['actuator_interval_wall_seconds'],
            'parser_peak_buffer_chars':raw['runtime_measurements']['parser_peak_buffer_chars'],
            'query_pending_max':raw['runtime_measurements']['pending_query_max'],
            'telemetry_events_count':len(raw['phase2']['telemetry_fix']['events']),
            'message_counts_sent':raw['messages_sent'],'message_counts_received':raw['messages_received'],
            'px4_exit_code':raw['px4_exit_code'],'zero_motor_ctrl_on_stop':raw['zero_control_on_stop'],
            'console_warnings':console['px4_console_warnings']},
        'resources':{'bridge_peak_rss_bytes':raw['bridge_peak_rss_bytes'],
            'external_peak_rss_bytes':None if external_peak_rss_kib is None else external_peak_rss_kib*1024,
            'px4_sampled_peak_rss_bytes':raw['px4_sampled_peak_rss_bytes'],
            'system_min_available_bytes':raw['system_min_available_bytes'],
            'bridge_cpu_seconds':raw['bridge_cpu_seconds'],
            'px4_sampled_cpu_seconds':raw['runtime_measurements']['px4_sampled_cpu_seconds'],
            'oom_observed_in_owned_run':False,'swap_changed':False},
        'limitations':['One nominal SITL positive yaw step, no negative-yaw or disturbed-flight replication.',
            'Physical thrust/yaw torque are model-derived from ideal motor coefficients and executed controls, not hardware measurements.',
            'PX4 ULog torque request units are normalized and cannot be numerically equated to physical Nm.',
            'Finite console/source polling may miss a transient between queries; no hard real-time guarantee.',
            'This verifies the control chain only. External Odometry, ROS and Phase3 integration were not tested.'],
        'next_recommendation':'Separately authorize Phase3 external odometry interface preflight only after reviewing this bounded control-chain evidence; not executed.'}
    if write:
        from .yaw_closed_loop_figures import write_figures
        report['figures']=write_figures(FIG,trace,raw,aux,report)
        atomic_json(OUT,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=RAW)
    parser.add_argument('--dependency-search-path',type=Path)
    parser.add_argument('--external-peak-rss-kib',type=int)
    args=parser.parse_args()
    if args.dependency_search_path:sys.path.append(str(args.dependency_search_path))
    report=build_report(args.raw,write=True,external_peak_rss_kib=args.external_peak_rss_kib)
    print(json.dumps({'status':report['status'],'response':report['yaw_response'],
        'report':str(OUT)},allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
