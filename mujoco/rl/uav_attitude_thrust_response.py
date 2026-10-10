"""Read-only saved-state attitude/force audit. No time integration or controller calls.

Force and current acceleration outputs are conditional model-derived quantities,
not historical sensor samples. The recorded qacc cache is deliberately not used.
"""
import numpy as np
import mujoco


def rotation(quat):
    q=np.asarray(quat,float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-10:
        raise ValueError('unit finite MuJoCo wxyz quaternion required')
    result=np.empty(9);mujoco.mju_quat2Mat(result,q)
    return result.reshape(3,3)


def model_facts(model):
    for name in dir(mujoco):
        if name.startswith('get_mjcb_') and getattr(mujoco,name)() is not None:
            raise ValueError('callback prevents independent offline reconstruction')
    if (model.nq,model.nv,model.nu,model.na,model.nbody,model.nmocap,model.nplugin)!=(7,6,4,0,2,0,0):
        raise ValueError('single rigid body/stateless four-motor model required')
    if (np.any(model.actuator_dyntype!=int(mujoco.mjtDyn.mjDYN_NONE)) or np.any(model.actuator_delay)
            or np.any(model.actuator_history) or np.any(model.actuator_gaintype!=int(mujoco.mjtGain.mjGAIN_FIXED))
            or np.any(model.actuator_biastype!=int(mujoco.mjtBias.mjBIAS_NONE))
            or np.any(model.actuator_gainprm[:,0]!=1) or np.any(model.actuator_plugin!=-1)):
        raise ValueError('stateless immediate unity-gain motor model required')
    if (np.any(model.actuator_trntype!=int(mujoco.mjtTrn.mjTRN_SITE)) or np.any(model.actuator_trnid[:,1]!=-1)
            or np.any(model.site_bodyid[model.actuator_trnid[:,0]]!=1)):
        raise ValueError('body-local rotor site transmission required')
    if (np.any(model.dof_damping) or np.any(model.dof_armature) or model.opt.density or model.opt.viscosity):
        raise ValueError('unsupported passive dynamics')
    if np.any(model.actuator_ctrllimited) or np.any(model.actuator_forcelimited):
        raise ValueError('unsupported actuator limiting')
    r=rotation(model.body_iquat[1]);inertia=r@np.diag(model.body_inertia[1])@r.T
    return dict(mujoco_version=mujoco.__version__,nq=7,nv=6,nu=4,na=0,body_name=model.body(1).name,
        mass_kg=float(model.body_mass[1]),com_body_m=model.body_ipos[1].tolist(),inertia_body_kg_m2=inertia.tolist(),
        gravity_world_m_s2=model.opt.gravity.tolist(),physics_dt_s=float(model.opt.timestep),
        actuator_delay_s=model.actuator_delay.tolist(),actuator_history=model.actuator_history.tolist(),
        rotor_site_gear=model.actuator_gear.tolist(),motor_dynamic_state='not applicable: na0/dyntypeNONE/no history or delay',
        force_semantics='actuator_force is scalar omega² output; site gear converts it to Newton/Newton-metre wrench',
        physical_mass_inertia_provenance='engineering estimates in unchanged MJCF, not measured flight hardware')


def reconstruct(model,position,quat,velocity,omega,ctrl,force,time):
    """Evaluate current post-command/pre-integration state in a NEW scratch MjData."""
    model_facts(model)
    vectors=[np.asarray(x,float) for x in (position,velocity,omega,ctrl,force)]
    if any(x.shape!=shape or not np.isfinite(x).all() for x,shape in zip(vectors,[(3,),(3,),(3,),(4,),(3,)])) or not np.isfinite(time):
        raise ValueError('finite saved state/control/force/time required')
    r=rotation(quat);p,v,w,u,f=vectors
    data=mujoco.MjData(model)
    data.time=float(time);data.qpos[:3]=p;data.qpos[3:]=quat;data.qvel[:3]=v;data.qvel[3:]=w
    data.ctrl[:]=u;data.xfrc_applied[1,:3]=f
    mujoco.mj_forward(model,data)  # NO mj_step, no integration, no controller.
    if data.time!=time or not np.array_equal(data.qpos,np.r_[p,quat]) or not np.array_equal(data.qvel,np.r_[v,w]):
        raise ValueError('scratch forward unexpectedly changed state/time')
    if data.ncon or data.nefc or np.any(data.qfrc_constraint):raise ValueError('contact/constraint state unsupported')
    if np.max(np.abs(data.qfrc_passive))>1e-12:raise ValueError('unsupported passive force')
    if not np.array_equal(data.actuator_force,u):raise ValueError('motor scalar force differs from immediate ctrl')
    force_world=np.zeros(3);torque_world=np.zeros(3)
    for i,site in enumerate(model.actuator_trnid[:,0]):
        sr=data.site_xmat[site].reshape(3,3)
        sf=sr@model.actuator_gear[i,:3]*data.actuator_force[i]
        st=sr@model.actuator_gear[i,3:]*data.actuator_force[i]
        force_world+=sf;torque_world+=np.cross(data.site_xpos[site]-data.xipos[1],sf)+st
    if np.max(np.abs(force_world-data.qfrc_actuator[:3]))>1e-10:
        raise ValueError('site/generalized world-force mismatch')
    mass=model.body_mass[1];net=force_world+f+mass*model.opt.gravity
    c=model.body_ipos[1];alpha=data.qacc[3:].copy()
    offset_acceleration=r@(np.cross(alpha,c)+np.cross(w,np.cross(w,c)))
    origin=data.qacc[:3].copy();com=origin+offset_acceleration
    balance=float(np.max(abs(com-net/mass)))
    if balance>1e-10:raise ValueError('COM Newton force balance inconsistent')
    generalized_external=np.zeros(6)
    mujoco.mj_applyFT(model,data,f,np.zeros(3),data.xipos[1],1,generalized_external)
    return dict(time_s=float(data.time),actuator_scalar_output=data.actuator_force.copy(),
        actuator_generalized_force=data.qfrc_actuator.copy(),external_generalized_force=generalized_external,
        rotor_force_world_n=force_world,rotor_torque_com_body_nm=r.T@torque_world,
        external_force_world_n=f.copy(),net_force_com_world_n=net,
        origin_acceleration_world_m_s2=origin,com_acceleration_world_m_s2=com,
        origin_to_com_acceleration_world_m_s2=offset_acceleration,angular_acceleration_body_rad_s2=alpha,
        actual_body_z_world=r[:,2].copy(),force_balance_max_abs_m_s2=balance,contact_count=int(data.ncon))


def reconstruct_series(model,z):
    keys=['control_position_m','control_actual_quaternion_wxyz','control_measured_velocity_m_s',
          'control_actual_angular_velocity_body','control_actual_ctrl_u','control_external_force_world_n','control_time_s']
    n=len(z[keys[-1]])
    if not n or any(len(z[k])!=n for k in keys):raise ValueError('unaligned saved signals')
    t=z[keys[-1]]
    if n>1 and not np.allclose(np.diff(t),.01,atol=1e-10,rtol=0):raise ValueError('control clock not100Hz')
    rows=[reconstruct(model,*(z[k][j] for k in keys)) for j in range(n)]
    return {k:np.asarray([row[k] for row in rows]) for k in rows[0]}


def crossing(times,signal,start,end,epsilon=1e-9,hold=3,sign=1):
    """First sustained opposite-to-requested-sign reversal, with a neutral interval.

    Event bracket: last sample beyond negative deadband to FIRST sample of
    confirmed positive run, not its confirmation time. No interpolation,
    smoothing or extrapolation; sign=-1 detects a positive-to-negative reversal.
    """
    t=np.asarray(times,float);y=np.asarray(signal,float)*sign
    if (t.ndim!=1 or y.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(y).all()
            or np.any(np.diff(t)<=0) or epsilon<0 or hold<1 or int(hold)!=hold or sign not in (-1,1) or end<=start):
        raise ValueError('invalid event signal/clock/config')
    index=np.flatnonzero((t>=start-1e-9)&(t<=end+1e-9));negative=None
    for local,j in enumerate(index):
        if y[j]<-epsilon:negative=j
        if negative is not None and y[j]>epsilon:
            run=index[local:local+hold]
            if len(run)==hold and np.all(y[run]>epsilon):
                return dict(bracket_s=[float(t[negative]),float(t[j])],confirmation_time_s=float(t[run[-1]]),
                    neutral_samples=int(np.sum(abs(y[negative+1:j])<=epsilon)),epsilon=float(epsilon),hold_samples=int(hold))
    return None


def interval_difference(later,earlier):
    """Bounds on crossing separation; explicitly not an estimated pure delay."""
    return [float(later[0]-earlier[1]),float(later[1]-earlier[0])]


def force_error_components(z,derived,mass):
    """Exact force/geometry identity, NOT causal module attribution."""
    desired_z=z['control_desired_rotation_body_to_world'][:,:,2]
    thrust=np.linalg.norm(derived['rotor_force_world_n'],axis=1)
    orientation=(thrust/mass)[:,None]*(derived['actual_body_z_world']-desired_z)
    magnitude=((thrust-z['control_desired_thrust_n'])/mass)[:,None]*desired_z
    external=z['control_external_force_world_n']/mass
    offset=-derived['origin_to_com_acceleration_world_m_s2']
    observed=derived['origin_acceleration_world_m_s2']-z['control_output_after_limiting_m_s2']
    return dict(orientation_m_s2=orientation,magnitude_m_s2=magnitude,external_m_s2=external,
        origin_offset_m_s2=offset,decomposition_max_abs_m_s2=float(np.max(abs(observed-orientation-magnitude-external-offset))))


def input_identity(root):
    """Check original acquisition provenance and all12 unchanged old trace arrays."""
    import hashlib,json,subprocess
    from pathlib import Path
    from uav_gust_recovery_audit import sha
    from run_uav_pi_internal_telemetry import identities,validate_completed
    root=Path(root);reports=root/'mujoco/reports';parts=reports/'uav_pi_internal_telemetry_parts'
    acquisition=json.loads((parts/'acquisition.json').read_text())
    current=identities(root)
    if current!=acquisition['original_hashes_before'] or current!=acquisition['original_hashes_after']:
        raise ValueError('original source/checkpoint changed')
    tracked=['mujoco/reports/uav_pi_internal_telemetry.json','mujoco/rl/UAV_PI_INTERNAL_TELEMETRY.md',
             'mujoco/reports/dynamics_v2_report.json','mujoco/reports/uav_bc_external_disturbance_manifest_seed0.json']
    hashes={name:sha(root/name) for name in tracked}
    for name,actual in hashes.items():
        original=subprocess.check_output(['git','show','c84389c3457bfa88d6def6be517055ec3c8e8161:'+name],cwd=root)
        if hashlib.sha256(original).hexdigest()!=actual:raise ValueError('historical report/model metadata modified')
    raw={};records={}
    for case in ('constant_medium','gust_medium'):
        path=parts/(case+'.json');row=validate_completed(json.loads(path.read_text()),acquisition['identity'],case)
        if not row['passivity']['bitwise_equal']:raise ValueError('original baseline not bitwise reproduced')
        raw[case]=row['raw_sha256'];records[case]=sha(path)
    selected=acquisition['identity']['selected_states']
    if len(selected)!=2 or len({s['target_id'] for s in selected})!=1:raise ValueError('preselected paired cohort changed')
    return dict(target_id=selected[0]['target_id'],seed=selected[0]['env_seed'],direction_world=selected[0]['direction_world'],
        initial_physical_controller_sha256=acquisition['initial_physical_controller_sha256'],protected_hashes=current,
        historical_reports_sha256=hashes,acquisition_sha256=sha(parts/'acquisition.json'),raw=raw,records=records,
        baseline_arrays_bitwise_equal=True,new_flight_executions=0)


def preserve_identity(path,identity):
    import json
    from pathlib import Path
    from uav_bc_safety import atomic_json
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=identity:raise ValueError('input identity changed; refuse stale resume')
    else:atomic_json(path,identity)


def distribution(values):
    v=np.asarray(values,float)
    if not v.size:return None
    if not np.isfinite(v).all():raise ValueError('nonfinite diagnostic')
    return dict(mean=float(v.mean()),median=float(np.median(v)),min=float(v.min()),max=float(v.max()),rms=float(np.sqrt(np.mean(v*v))))


def case_analysis(z,d,direction,mass=7.,inertia=None):
    """Deterministic measured/derived audit with fixed3..8s focus and4..8s events."""
    from uav_pi_telemetry_analysis import attitude_error_degrees,zero_crossings,threshold_gates
    t=z['control_time_s'];axis=np.asarray(direction,float)
    if axis.shape!=(3,) or abs(axis[2])>1e-12 or abs(np.linalg.norm(axis)-1)>1e-12:raise ValueError('unit worldXY audit axis required')
    if not np.array_equal(t,d['time_s']):raise ValueError('derived current physics clock changed')
    e=z['control_target_m']-z['control_position_m'];v=z['control_measured_velocity_m_s'];cmd=z['control_commanded_velocity_m_s']
    p=z['control_p_contribution_m_s2_derived'];i=z['control_i_contribution_m_s2'];out=z['control_output_after_limiting_m_s2']
    actual_z=d['actual_body_z_world'];desired_z=z['control_desired_rotation_body_to_world'][:,:,2]
    actual_tilt=np.arctan2(actual_z@axis,actual_z[:,2]);desired_tilt=np.arctan2(desired_z@axis,desired_z[:,2])
    signals=dict(velocity=v@axis,relative_velocity_request=(cmd-v)@axis,p_contribution=p@axis,i_contribution=i@axis,
        pi_output=out@axis,desired_tilt=desired_tilt,actual_tilt=actual_tilt,
        rotor_force=d['rotor_force_world_n']@axis,com_acceleration=d['com_acceleration_world_m_s2']@axis,
        origin_acceleration=d['origin_acceleration_world_m_s2']@axis,
        speed_power=np.einsum('ij,ij->i',v,d['origin_acceleration_world_m_s2']),position_error=e@axis)
    deadbands=dict(velocity=.005,relative_velocity_request=.005,p_contribution=.01,i_contribution=.01,pi_output=.01,
        desired_tilt=.001,actual_tilt=.001,rotor_force=.07,com_acceleration=.01,origin_acceleration=.01,speed_power=.001,position_error=.002)
    signs={key:(-1 if key in ('velocity','speed_power') else 1) for key in signals}
    events_zero={key:crossing(t,y,4,8,epsilon=1e-9,hold=3,sign=signs[key]) for key,y in signals.items()}
    events_band={key:crossing(t,y,4,8,epsilon=deadbands[key],hold=3,sign=signs[key]) for key,y in signals.items()}
    separations={}
    for later,earlier in [('pi_output','p_contribution'),('actual_tilt','desired_tilt'),('rotor_force','actual_tilt'),
                          ('com_acceleration','rotor_force'),('origin_acceleration','com_acceleration'),('speed_power','pi_output')]:
        a,b=events_zero[later],events_zero[earlier]
        separations[later+'_minus_'+earlier]=interval_difference(a['bracket_s'],b['bracket_s']) if a and b else None
    focus=(t>=3-1e-9)&(t<8-1e-9);post=t>=4-1e-9
    components=force_error_components(z,d,mass)
    if components['decomposition_max_abs_m_s2']>1e-10:raise ValueError('force geometry/decomposition invalid')
    thrust=np.linalg.norm(d['rotor_force_world_n'],axis=1)
    error_angle=attitude_error_degrees(z['control_actual_quaternion_wxyz'],z['control_desired_quaternion_wxyz'])
    norm_error=np.linalg.norm(e,axis=1);speed=np.linalg.norm(v,axis=1)
    gates=threshold_gates(e,v)
    opposing=(p@axis>.05)&(i@axis<-.01)&(v@axis<-.15)&post&(t<8)
    power_out=np.einsum('ij,ij->i',v,out);power_actual=signals['speed_power']
    requested_braking_actual_accelerating=focus&(power_out<-.001)&(power_actual>.001)
    current_gradient=np.gradient(v,t,axis=0,edge_order=2) if len(t)>2 else np.zeros_like(v)
    checkpoints={}
    for query in (3,4,4.56,4.82,5.51,6.11,8):
        if not len(t) or query>t[-1]+1e-9:continue
        k=int(np.argmin(abs(t-query)))
        checkpoints[str(query)]=dict(time_s=float(t[k]),speed_m_s=float(speed[k]),distance_m=float(norm_error[k]),
            attitude_error_deg=float(error_angle[k]),force_axis={name:float(y[k]) for name,y in signals.items()},
            rotor_force_world_n=d['rotor_force_world_n'][k].tolist(),origin_acceleration_world_m_s2=d['origin_acceleration_world_m_s2'][k].tolist(),
            com_acceleration_world_m_s2=d['com_acceleration_world_m_s2'][k].tolist(),p_world_m_s2=p[k].tolist(),i_world_m_s2=i[k].tolist())
    peak_index=np.flatnonzero(post)[np.argmax(speed[post])] if post.any() else None
    angular_error=None
    if inertia is not None:
        w=z['control_actual_angular_velocity_body'];inertia=np.asarray(inertia)
        rhs=z['control_desired_wrench_body'][:,1:]-np.cross(w,w@inertia.T)
        expected=np.linalg.solve(inertia,rhs.T).T
        angular_error=float(np.max(abs(expected-d['angular_acceleration_body_rad_s2'])))
        if angular_error>1e-10:raise ValueError('requested torque/angular acceleration balance inconsistent')
    return dict(control_samples=len(t),focus_samples=int(focus.sum()),events_zero=events_zero,events_deadband=events_band,
        event_separation_bounds_s=separations,event_rules=dict(window_s=[4,8],hold_samples=3,raw_zero_epsilon=1e-9,deadbands=deadbands,
            meaning='last opposite-sign sample→first sample of3-consecutive new-sign samples; confirmation occurs later; intervals are crossing separations, NOT pure delays'),
        attitude_tracking_error_degrees_3_8=distribution(error_angle[focus]),angular_speed_body_rad_s_3_8=distribution(np.linalg.norm(z['control_actual_angular_velocity_body'][focus],axis=1)),
        thrust_scalar_error_n=distribution(abs(thrust-z['control_desired_thrust_n'])),
        commanded_vs_model_derived_com_torque_max_abs_nm=float(np.max(abs(d['rotor_torque_com_body_nm']-z['control_desired_wrench_body'][:,1:]))),
        commanded_torque_euler_angular_acceleration_max_abs_rad_s2=angular_error,
        actuator_scalar_ctrl_max_abs_difference=float(np.max(abs(d['actuator_scalar_output']-z['control_actual_ctrl_u']))),
        force_balance_max_abs_m_s2=float(np.max(d['force_balance_max_abs_m_s2'])),decomposition_max_abs_m_s2=components['decomposition_max_abs_m_s2'],
        acceleration_error_components_rms_3_8={key:distribution(np.linalg.norm(value[focus],axis=1)) for key,value in components.items() if isinstance(value,np.ndarray)},
        origin_com_acceleration_difference_3_8=distribution(np.linalg.norm(d['origin_to_com_acceleration_world_m_s2'][focus],axis=1)),
        finite_difference_velocity_vs_current_model_acceleration_3_8=distribution(np.linalg.norm((current_gradient-d['origin_acceleration_world_m_s2'])[focus],axis=1)),
        finite_difference_note='100Hz centered velocity difference spans ±10ms; current forward acceleration is instantaneous. Not bitwise alignment or independent sensor truth.',
        p_i_opposition_braking_samples=int(opposing.sum()),pi_over_p_projected_ratio_when_braking_opposed=distribution((out@axis)[opposing]/(p@axis)[opposing]),
        requested_braking_actual_accelerating_samples_3_8=int(requested_braking_actual_accelerating.sum()),
        requested_braking_actual_accelerating_duration_s_3_8=float(requested_braking_actual_accelerating.sum()*.01),
        post4_speed_peak=None if peak_index is None else dict(time_s=float(t[peak_index]),speed_m_s=float(speed[peak_index])),
        post4_signed_position_crossings_s=zero_crossings(t[post],signals['position_error'][post]),
        post4_signed_velocity_crossings_s=zero_crossings(t[post],signals['velocity'][post]),
        post4_joint_threshold_samples=int((post&gates['joint']).sum()),checkpoints=checkpoints,
        contact_count_max=int(d['contact_count'].max()),cached_qacc_used_as_current_acceleration=False)


def resource_evidence(parts):
    import json
    phases={p.stem.removeprefix('resource_'):json.loads(p.read_text()) for p in parts.glob('resource_*.json')}
    if any(r['failure'] or r['oom_count_delta'] or r['workers']!=1 for r in phases.values()):
        raise ValueError('resource guard failure/unfinished phase')
    if any(r['exit_code'] and name!='core_tests' for name,r in phases.items()):
        raise ValueError('unresolved test or analysis phase failure')
    return dict(workers=1,oom_occurred=False,swap_changed=False,phases=phases,
        peak_sampled_process_tree_rss_bytes=max((r['process_tree_peak_sampled_rss_bytes'] for r in phases.values()),default=0),
        max_sum_process_hwm_bytes=max((r['process_tree_peak_hwm_bytes'] for r in phases.values()),default=0),
        pre_final_failure_note='core_tests initially failed only on an overstrict decimal-clock fixture; corrected to preserve saved float timestamps bitwise plus1e-12 lattice check, final suite green')


def publish(root,finalize=False):
    """Only reads historical data and writes NEW audit artifacts, never integrates."""
    import json
    from pathlib import Path
    from uav_gust_recovery_audit import sha
    from uav_bc_safety import atomic_json,atomic_npz
    root=Path(root);before=input_identity(root)
    reports=root/'mujoco/reports';parts=reports/'uav_attitude_thrust_response_parts'
    preserve_identity(parts/'input_identity.json',before)
    model=mujoco.MjModel.from_xml_path(str(root/'mujoco/models/mine_uav_dynamics_v2.xml'))
    facts=model_facts(model);mass=facts['mass_kg'];inertia=np.asarray(facts['inertia_body_kg_m2'])
    output=reports/'uav_attitude_thrust_response_audit.json'
    previous=json.loads(output.read_text()) if finalize else None
    cases={};arrays={};raw_meta={}
    for case in ('constant_medium','gust_medium'):
        with np.load(reports/'uav_pi_internal_telemetry_parts'/(case+'.npz'),allow_pickle=False) as saved:
            z={k:saved[k] for k in saved.files}
        derived=reconstruct_series(model,z)
        cases[case]=case_analysis(z,derived,before['direction_world'],mass,inertia)
        row=json.loads((reports/'uav_pi_internal_telemetry_parts'/(case+'.json')).read_text())
        cases[case]['termination']={k:row[k] for k in ('success','timeout','physical_failure','steps','simulated_seconds','termination_reason','final_distance_m','final_speed_m_s')}
        raw=parts/(case+'_derived.npz')
        if finalize:
            if sha(raw)!=previous['derived_local_only'][case]['sha256']:raise ValueError('derived cache hash changed')
            with np.load(raw,allow_pickle=False) as old:
                if set(old.files)!=set(derived):raise ValueError('derived signal schema changed')
                for key in old.files:np.testing.assert_array_equal(old[key],derived[key])
        else:atomic_npz(raw,**derived)
        raw_meta[case]=dict(path=str(raw.relative_to(root)),sha256=sha(raw),samples=len(derived['time_s']))
        arrays[case]=(z,derived)
        print(case,':',len(derived['time_s']),'saved states;0integrated steps;balance',cases[case]['force_balance_max_abs_m_s2'],flush=True)
    if finalize:
        if previous['cases']!=cases or previous['inputs_before']!=before:raise ValueError('audit analysis/input not reproducible')
        figures=previous['figures']
        if any(sha(reports/path)!=digest for path,digest in figures.items()):raise ValueError('audit figure changed')
    else:
        from uav_attitude_thrust_response_plots import plots
        figures=plots(reports,arrays,before['direction_world'])
    after=input_identity(root)
    if before!=after:raise ValueError('original data/controller/checkpoint mutated')
    tests=json.loads((parts/'tests_evidence.json').read_text())
    if tests['failures'] or tests['errors'] or tests['skipped']:raise ValueError('final algorithmic tests not green')
    value=dict(experiment='GUST RECOVERY ATTITUDE-THRUST RESPONSE AUDIT',date='2026-10-10',
        branch='feat/uav-attitude-thrust-response-audit',base_commit='c84389c3457bfa88d6def6be517055ec3c8e8161',
        execution=dict(new_unique_episodes=0,new_flight_executions=0,integrated_physics_steps=0,reused_unique_trajectories=2,
            scratch_state_forward_evaluations_per_full_analysis=sum(r['control_samples'] for r in cases.values()),
            historical_previous_stage_executions=3,workers=1,training=False,controller_changed=False,motor_changed=False),
        model_facts=facts,inputs_before=before,inputs_after=after,cases=cases,tests=tests,resources=resource_evidence(parts),
        frames_and_signals=dict(world='right-handed,+Zup,gravity[0,0,-9.81]',body='MJCF rigid-body axes;+Zrotor force',
            quaternion='MuJoCo wxyz body→world, qpos[3:7]',linear_velocity='qvel[:3], world-frame body-origin velocity, NOT COM velocity',
            angular_velocity='qvel[3:6], body-frame rad/s',
            recorded='100Hz actual qpos/quaternion/qvel/rotor ctrl and desired attitude/thrust/P/I; actual500Hz xfrc_applied after original hook',
            model_derived='scratch mj_forward: actuator scalar output, generalized actuator wrench, site resultant, net COM force, qacc/current origin and COM acceleration; not independently logged physical sensors',
            current_time='recorded100Hz pre-integration state, newly applied ctrl, matching500Hz post-hook force',
            motor_state='not applicable, no activation/delay/history; ctrl is omega², actuator_force scalar omega² is NOT Newton rotor force',
            generalized_force='qfrc_actuator is generalized6D; first3 world force; rotational3 body generalized moments at freejoint origin, distinct from COM torque',
            com='origin acceleration + R(alpha×c + omega×(omega×c)); c=compiled body_ipos',
            cached_acceleration='previous-tick qacc cache not used as current acceleration',
            not_observable='independent aerodynamic thrust sensor, real motor lag/wind/aerodynamics; none introduced'),
        evidence=dict(confirmed=['all2saved traces verified bitwise against previous original arrays; no inputs modified',
            'compiled direct-drive motor has no delay/activation/history; model scalar output matches ctrl exactly',
            'model site forces and generalized force agree; current COM Newton balance and angular Euler balance pass',
            'allocated scalar thrust and COM torque match requested wrench to floating precision; persistent desired/actual attitude differences remain',
            'post-gust position/velocity crossing phases do not satisfy joint gates within original15s'],
            supported_hypotheses=['largest non-external acceleration discrepancy in3..8s is requested-versus-actual thrust orientation, with bounded integral opposition upstream',
                'finite attitude/cascade response together with outer-loop/PI phase contributes to transient braking and oscillatory settling'],
            unresolved=['unique causality PI vs attitude dynamics vs outer position cascade','optimal gain or necessary controller structure change',
                'counterfactual effect of any tuning','eventual settling beyond15s','generalization to High/BC/alltargets'],
            controller_structure_change='not established; do not automatically redesign',specific_pi_tuning='not supported by this two-trace observational audit',
            stop='bounded audit complete; no additional experiments or training'),
        official_api_references=['https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-forward',
            'https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-motor',
            'https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model'],
        derived_local_only=raw_meta,figures=figures,
        analysis_source_sha256={name:sha(root/'mujoco/rl'/name) for name in ('uav_attitude_thrust_response.py','uav_attitude_thrust_response_plots.py','test_uav_attitude_thrust_response.py')},
        finalized=finalize)
    atomic_json(output,value)
    print('Audit published; original inputs unchanged; new flight executions0',flush=True)
    return value


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--finalize',action='store_true');args=parser.parse_args()
    publish(Path(__file__).resolve().parents[2],args.finalize)
