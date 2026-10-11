"""Offline Phase2 evidence; no simulator, network, arming or training entrypoint."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from .diagnostics import atomic_json,sha256,own_peak_rss_bytes
from .yaw_validation import angle_error,yaw_ned,latest_topics,settling_time

BASE="3b36e9d77f2f6b89699b1ebd065e1e38310999be"


def simulation_time(clock_usec):
    return (np.asarray(clock_usec,dtype=float)-1000000)/1000000


def interpolate(times,values,query,circular=False):
    t,v,q=map(lambda x:np.asarray(x,dtype=float),(times,values,query))
    if t.ndim!=1 or v.shape!=t.shape or not np.isfinite(np.r_[t,v]).all() or np.any(np.diff(t)<=0):
        raise ValueError("finite strictly increasing aligned source required")
    if not len(t):return np.full(q.shape,np.nan)
    out=np.interp(q,t,np.unwrap(v) if circular else v,left=np.nan,right=np.nan)
    return angle_error(out,0) if circular else out


def hold_interpolate(times,values,query):
    t,v,q=map(lambda x:np.asarray(x,dtype=float),(times,values,query))
    if t.ndim!=1 or v.shape!=t.shape or not np.isfinite(np.r_[t,v]).all() or np.any(np.diff(t)<=0):
        raise ValueError("finite strictly increasing aligned source required")
    out=np.full(q.shape,np.nan)
    if not len(t):return out
    valid=(q>=t[0])&(q<=t[-1])
    out[valid]=v[np.searchsorted(t,q[valid],side="right")-1]
    return out


def px4_yaw(quaternions):
    q=np.asarray(quaternions,dtype=float)
    if q.ndim!=2 or q.shape[1]!=4 or not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-5):
        raise ValueError("unit finite NED-FRD wxyz quaternions required")
    w,x,y,z=q.T
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))


def alignment_times(local_times,heading_good,flag_times,flags):
    t,h,ft=np.asarray(local_times),np.asarray(heading_good,dtype=bool),np.asarray(flag_times)
    mag=np.asarray(flags.get("cs_mag_aligned_in_flight",[]),dtype=bool)
    air=np.asarray(flags.get("cs_in_air",[]),dtype=bool)
    if not len(ft) or len(mag)!=len(ft) or len(air)!=len(ft):
        return {"first_inflight_mag_alignment_s":None,"first_final_heading_good_s":None,"final_heading_good":False}
    aligned=mag&air
    index=np.searchsorted(ft,t,side="right")-1
    valid=index>=0
    selected=np.zeros(t.shape,dtype=bool);selected[valid]=aligned[index[valid]]
    final=h&selected
    return {"first_inflight_mag_alignment_s":float(ft[np.flatnonzero(aligned)[0]]) if aligned.any() else None,
            "first_final_heading_good_s":float(t[np.flatnonzero(final)[0]]) if final.any() else None,
            "final_heading_good":bool(len(final) and final[-1])}


def response_status(complete,step_time,settled):
    if step_time is None:return "NOT TESTED"
    if not complete:return "BLOCKED"
    return "PASS" if settled is not None else "FAIL"


def bias_status(summary):
    if summary is None:return "NOT TESTED"
    return "PASS" if np.isfinite(summary["mean"]) and abs(summary["mean"])<.05 else "FAIL"


def verify_preserved(hashes,runs):
    if not hashes or any(not item["unchanged"] for item in hashes.values()) or any(not r["history_unchanged"] for r in runs):
        raise ValueError("protected historical identities changed; do not publish PASS")


def stats(values):
    v=np.asarray(values,dtype=float);v=v[np.isfinite(v)]
    if not len(v):return None
    return {"n":int(len(v)),"mean":float(np.mean(v)),"min":float(np.min(v)),
            "max":float(np.max(v)),"max_abs":float(np.max(np.abs(v))),"rms":float(np.sqrt(np.mean(v*v)))}


def load_topics(path,end):
    from pyulog import ULog
    names={"vehicle_attitude","vehicle_attitude_setpoint","vehicle_angular_velocity",
           "vehicle_local_position","estimator_status_flags","vehicle_status"}
    u=ULog(str(path));out={}
    for d in u.data_list:
        if d.name in names and d.multi_id==0:
            mask=simulation_time(d.data["timestamp"])<=end+1e-8
            out[d.name]={k:v[mask] for k,v in d.data.items()}
    return out,dict(u.initial_parameters)


def yaw_signal(topic,prefix):
    q=np.column_stack([topic[f"{prefix}[{i}]"] for i in range(4)])
    return px4_yaw(q)


def analyze_run(path):
    path=Path(path);s=json.loads(path.read_text());end=s["final_physical_state"]["time_s"]
    logs=list(path.parent.rglob("*.ulg"))
    if len(logs)!=1:raise ValueError(f"one ULog required for {path}")
    topics,params=load_topics(logs[0],end)
    r=s["records"];t=np.array([p["time_s"] for p in r]);clock=simulation_time([p["clock_usec"] for p in r])
    if not np.allclose(clock,t,rtol=0,atol=1e-8) or np.any(np.diff(t)<=0):raise ValueError("physical/ULog clock mismatch")
    true=np.array([yaw_ned(p["quaternion_enu_flu"]) for p in r])
    body_rate=np.array([-p["angular_velocity_flu_rad_s"][2] for p in r])
    att=topics.get("vehicle_attitude",{});sp=topics.get("vehicle_attitude_setpoint",{})
    estimated=interpolate(simulation_time(att.get("timestamp_sample",[])),yaw_signal(att,"q") if att else [],t,True)
    desired=hold_interpolate(simulation_time(sp.get("timestamp",[])),yaw_signal(sp,"q_d") if sp else [],t)
    local=topics.get("vehicle_local_position",{});flags=topics.get("estimator_status_flags",{})
    health=alignment_times(simulation_time(local.get("timestamp",[])),local.get("heading_good_for_control",[]),
                           simulation_time(flags.get("timestamp",[])),flags)
    phase=s.get("phase2",{});step=phase.get("actual_step_time_s")
    hold_end=step if step is not None else end
    hold=(t>=hold_end-5)&(t<hold_end)&np.isfinite(estimated)
    errors=np.degrees(angle_error(estimated,true))
    position=np.array([p["position_enu_m"] for p in r]);velocity=np.array([p["velocity_origin_enu_m_s"] for p in r])
    motors=np.array([p["ctrl_omega_squared"] for p in r])/((8500*2*np.pi/60)**2)
    post=(t>=step)&np.isfinite(desired)&np.isfinite(estimated) if step is not None else np.zeros(t.shape,dtype=bool)
    settled=None
    # An interrupted trajectory is censored, not evidence of successful settling.
    if s["complete"] and step is not None and post.any():
        settled=settling_time(t[post],angle_error(np.deg2rad(phase["yaw_target_deg"]),true[post]),body_rate[post],step)
    console_topics=latest_topics(s["px4_console"][-65536:])
    final_clock=1000000+round(end*1000000)
    ages={k:(final_clock-v["timestamp"])/1e6 for k,v in console_topics.items() if "timestamp" in v}
    result={"summary_path":str(path),"summary_sha256":sha256(path),"ulog_path":str(logs[0]),"ulog_sha256":sha256(logs[0]),
        "complete":s["complete"],"simulation_end_s":end,"wall_s":s["wall_seconds"],"physical_samples":len(r),
        "exception":s.get("exception"),"px4_exit_code":s["px4_exit_code"],"zero_control_on_stop":s["zero_control_on_stop"],
        "history_unchanged":s["history_unchanged"],"history_hashes":s["history_hashes_after"],
        "messages_received":s.get("messages_received"),"messages_sent":s.get("messages_sent"),
        "bridge_peak_rss_bytes":s["bridge_peak_rss_bytes"],"px4_sampled_peak_rss_bytes":s["px4_sampled_peak_rss_bytes"],
        "system_min_available_bytes":s["system_min_available_bytes"],"px4":s["px4"],"mujoco_version":s["mujoco_version"],
        "declination_parameter_deg":params.get("EKF2_MAG_DECL"),"declination_type":params.get("EKF2_DECL_TYPE"),
        "mag_type":params.get("EKF2_MAG_TYPE"),"alignment":health,
        "prestep_or_final_5s_estimate_minus_truth_deg":stats(errors[hold]),
        "poststep_estimate_minus_truth_deg":stats(errors[post]),
        "poststep_desired_minus_estimate_deg":stats(np.degrees(angle_error(desired[post],estimated[post]))),
        "poststep_desired_minus_truth_deg":stats(np.degrees(angle_error(desired[post],true[post]))),
        "poststep_requested_target_minus_truth_deg":stats(np.degrees(angle_error(np.deg2rad(phase.get("yaw_target_deg",100)),true[post]))),
        "poststep_observed_duration_s":end-step if step is not None else None,
        "actual_step_time_s":step,"gate_evidence":phase.get("gate_evidence_at_step"),
        "settling_time_s":settled,"yaw_response_status":response_status(s["complete"],step,settled),
        "console_final_topic_ages_s":ages,
        "maximum_xy_excursion_m":float(np.max(np.linalg.norm(position[:,:2],axis=1))),
        "maximum_height_m":float(np.max(position[:,2])),"maximum_speed_m_s":float(np.max(np.linalg.norm(velocity,axis=1))),
        "motor_output_fraction":stats(motors),"motor_saturation_samples":sum(p["saturated"] for p in r),
        "poststep_true_yaw_change_deg":float(np.degrees(angle_error(true[-1],true[np.flatnonzero(post)[0]]))) if post.any() else None}
    traces={"t":t,"true":true,"estimate":estimated,"desired":desired,"rate":body_rate,
            "position":position,"velocity":velocity,"motors":motors,"local":local,"flags":flags}
    return result,traces


def plot_results(out,results,traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def finish(fig,name):
        fig.tight_layout();fig.savefig(out/name,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,6))
    for key in ("phase1_hover","disarmed_01","yaw_03"):
        p=traces[key];axes[0].plot(p["t"],np.degrees(angle_error(p["estimate"],p["true"])),label=key)
    axes[0].set(ylabel="Estimate minus truth (deg)",xlabel="Simulation time (s)",xlim=(3,33),ylim=(-.08,3.6))
    axes[0].set_title("Post-initialization heading bias (startup unaligned samples excluded from view)");axes[0].legend()
    p=traces["yaw_03"];axes[1].plot(p["t"],p["position"][:,2],label="Physical body height")
    local=p["local"];flags=p["flags"]
    axes[1].plot(simulation_time(local["timestamp"]),local["heading_good_for_control"],label="heading_good_for_control")
    axes[1].plot(simulation_time(flags["timestamp"]),flags["cs_mag_aligned_in_flight"],label="mag_aligned_in_flight")
    axes[1].axhline(1.5,color="grey",ls=":",label="Source HAGL reset threshold (not body height)")
    axes[1].set(xlabel="Simulation time (s)",ylabel="Height (m) / boolean");axes[1].legend(fontsize=8)
    finish(fig,"px4_yaw_heading_alignment.png")
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True)
    for signal,label in (("desired","PX4 attitude setpoint"),("estimate","PX4 estimate"),("true","MuJoCo truth")):
        axes[0].plot(p["t"],np.degrees(p[signal]),label=label)
    axes[0].axhline(100,color="grey",ls=":",label="Requested yaw after32.004s");axes[0].legend(fontsize=8);axes[0].set(ylabel="NED yaw (deg)")
    axes[1].plot(p["t"],p["rate"]);axes[1].set(ylabel="FRD body-z rate (rad/s)")
    axes[2].plot(p["t"],p["motors"]);axes[2].set(ylabel="Motor normalized thrust",xlabel="Simulation time (s)")
    for ax in axes:ax.axvline(results["yaw_03"]["actual_step_time_s"],ls="--",color="black");ax.set_xlim(30.5,32.4);ax.grid(alpha=.2)
    axes[0].set_title("Censored response: safety-gate abort at32.196s; no settling claim")
    finish(fig,"px4_yaw_censored_step_response.png")
    fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True)
    axes[0].plot(p["t"],p["position"]);axes[0].set(ylabel="ENU body-origin position (m)")
    axes[1].plot(p["t"],p["velocity"]);axes[1].set(ylabel="ENU origin velocity (m/s)",xlabel="Simulation time (s)")
    finish(fig,"px4_yaw_physical_envelope.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-root",type=Path,required=True)
    parser.add_argument("--dependency-search-path",required=True)
    args=parser.parse_args();sys.path.append(args.dependency_search_path)
    root=Path(__file__).resolve().parents[2];out=root/"mujoco/reports"
    from .magnetic_model import field_from_px4
    from .simulation_runner import source_identity
    identity=source_identity(args.px4_root)
    results={};traces={}
    paths={"phase1_hover":out/"px4_sitl_phase1_parts/hover_01/smoke_summary.json"}
    paths.update({k:out/f"px4_yaw_phase2_parts/{k}/smoke_summary.json" for k in ("disarmed_01","yaw_01","yaw_02","yaw_03")})
    for name,path in paths.items():results[name],traces[name]=analyze_run(path)
    protected=["mujoco/models/mine_uav_dynamics_v2.xml","mujoco/models/rotor_actuators.xml","mujoco/models/rotor_config.json",
               "mujoco/rl/models/uav_bc_mlp_seed0.pt","mujoco/rl/models/uav_bc_yaw_augmented_seed0.pt",
               "mujoco/rl/models/joint_latent_world_model_v3_autonomous_consistency.pt","mujoco/reports/px4_mujoco_phase1.json"]
    hashes={}
    import hashlib
    for name in protected:
        baseline=hashlib.sha256(subprocess.check_output(["git","show",f"{BASE}:{name}"],cwd=root)).hexdigest()
        hashes[name]={"base_sha256":baseline,"current_sha256":sha256(root/name),"unchanged":baseline==sha256(root/name)}
    final=results["yaw_03"]
    verify_preserved(hashes,results.values())
    report={"phase":"PX4-MuJoCo Phase2 heading/yaw validation","date":"2026-10-11","base_commit":BASE,
        "branch":"feat/px4-mujoco-yaw-validation","px4":identity,"runs":results,
        "magnetic_reference":field_from_px4(args.px4_root),"protected_identities":hashes,
        "status":{
            "declination_mismatch_source":"PASS","frame_unit_tests":"PASS","field_bias_correction":bias_status(final["prestep_or_final_5s_estimate_minus_truth_deg"]),
            "inflight_final_heading_alignment":"PASS" if final["alignment"]["final_heading_good"] else "FAIL",
            "yaw_step_authorized_with_real_health":"PASS" if final["gate_evidence"] else "BLOCKED",
            "complete_yaw_response":final["yaw_response_status"],"settling_time":"NOT TESTED",
            "actuator_mapping_offline":"PASS","yaw_actuator_dynamics_validation":"NOT TESTED",
            "external_odometry_phase_entry":"BLOCKED","external_odometry_integration":"NOT TESTED"},
        "timing_method":"physical clock=t*1e6+1e6; EKF attitude uses timestamp_sample and circular-unwrapped interpolation; setpoint uses publication timestamp and causalzero-orderhold; no endpoint extrapolation",
        "gate_abort_explanation":{"confirmed":"At32.196s current clock33196000 minus most recent queried estimator_status_flags32192000 is1.004s, exceeding fixed1s gate. PX4 source publishes flags~1Hz, console queried2Hz. ULog final aligned/heading-good remaintrue; this is diagnostics freshness mismatch, not evidence of EKF or physical failure.",
            "source":"PX4 src/modules/ekf2/EKF2.cpp::PublishStatusFlags (>=last+1_s or flag change)",
            "no_gate_relaxation_or_rerun":True},
        "execution_integrity":{"phase2_sitl_invocations":4,"complete_disarmed_runs":1,"complete_flight_runs":0,
            "workflow_error":"yaw02 launched in tool batch before failed regression result read; immediately interrupted at23.556s, no yawstep. Fixedprimaryinstance selection and51 tests verified in separate call beforefinalyaw03. All raw attempts retained.",
            "yaw01":"stopped32.004s beforestep: listener title3instances notrecognized; passiveparser corrected",
            "yaw02":"SIGINT exit130, incomplete summary; exceptionNone doesnot implysuccess",
            "yaw03":"step32.004s, automaticzero/shutdown32.196s, incomplete response",
            "budget_revision":"original singleflight expanded only for passiveinstrumentation faults to two incomplete attempts plusonefinal; no furtherflight afterfinalblock"},
        "verification":{"independent_readonly_review":{"bridge_tests_passed":59,"skipped":0,"critical":0,"important_gate_findings":2,
                "checked":"all5rawsummary/ULGhashpairs,7protectedidentities,3figures,bias/alignment/extrema,exactstepgateprefix/source"},
            "single_offline_fix_pass":{"bridge_tests_passed":69,"skipped":0,"new_simulator_runs":0,
                "fixes":["failclosedpartialtopicheaders/bodies; no reuseofoldhealth","explicitGNSSposition/yaw/heightfaultandvehiclefailsafeFalse;missingflagsreject"],
                "actual_recorded_gate_prefix_still_passes":True},
            "known_operational_blocker":"fixed1sfreshnessgate cannot support fullresponse with1Hzflags/query2Hz; kept failclosed andBLOCKED",
            "deferred_minor":"boolean/heldsetpoint plottedaslines mayvisuallyslope discretechanges; numericalZOH iscausal",
            "no_second_review_or_flight":True,
            "source_sha256":{p.name:sha256(p) for p in sorted((root/"mujoco/px4_bridge").glob("*.py"))}},
        "resources":{"workers":1,"simultaneous_px4_max":1,"oom_observed":False,"swap_modified":False,
            "runtime_reserve_bytes":int(1.5*1024**3),"startup_reserve_bytes":int(2.5*1024**3),
            "max_bridge_run_hwm_bytes":max(x["bridge_peak_rss_bytes"] for k,x in results.items() if k!="phase1_hover"),
            "max_px4_sampled_rss_bytes":max(x["px4_sampled_peak_rss_bytes"] for k,x in results.items() if k!="phase1_hover"),
            "minimum_available_bytes":min(x["system_min_available_bytes"] for k,x in results.items() if k!="phase1_hover")},
        "limitations":["same nominal physical startup, one +10deg request, observation censored after0.192s",
            "strict gate retained; cannot conclude yawtracking normal/abnormal or fullsettling",
            "WMM2020 pinnedtable consistent withPX4, not contemporary measuredmagnetometer or2026truefield",
            "ideal normalized-thrust adapter, no validatedESC/motorlag/no realhardware",
            "historical fullRLtest suite notrun; completebridge suite only; dynamics/modelhashes verified"],
        "next_recommendation":"Before Phase3, make live safety telemetry timestamp semantics source-consistent (periodic flags vs asynchronous query), retain genuine fault/heading checks, then repeat only this bounded yaw protocol after offline tests; notexecuted."}
    plot_results(out,results,traces)
    report["resources"]["offline_analysis_peak_rss_bytes"]=own_peak_rss_bytes()
    report["figures"]={name:sha256(out/name) for name in ("px4_yaw_heading_alignment.png","px4_yaw_censored_step_response.png","px4_yaw_physical_envelope.png")}
    atomic_json(out/"px4_mujoco_yaw_validation.json",report)
    print(json.dumps({"status":report["status"],"corrected_yaw03_bias":final["prestep_or_final_5s_estimate_minus_truth_deg"],"alignment":final["alignment"],"resources":report["resources"]},indent=2))


if __name__=="__main__":main()
