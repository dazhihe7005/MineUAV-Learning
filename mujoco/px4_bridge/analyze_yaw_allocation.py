"""Phase2C offline evidence, preserving all raw runs and historical reports."""
import argparse
import json
from pathlib import Path
import sys
import time
import hashlib
import subprocess
import numpy as np
import mujoco
from .diagnostics import atomic_json,sha256,own_peak_rss_bytes
from .analyze_yaw import analyze_run,simulation_time
from .analyze_yaw_telemetry import distribution,response_metrics,freshness_summary,load_auxiliary
from .allocation_math import physical_audit
from .actuator_adapter import ActuatorAdapter
from .simulation_runner import MODEL,CONFIG,source_identity
from .allocation_runner import allocation_status
from .yaw_validation import angle_error

BASE="fbd273be0f7169f738f6c8024e7ec7be37e14941"


def verify_recording(raw,identity,history):
    if not (raw.get("px4")==identity and raw.get("history_hashes_before")==history
        and raw.get("history_hashes_after")==history and raw.get("history_unchanged") is True):
        raise ValueError("recording source/binary/config/physics identity mismatch")


def entry_status(required):return "PASS" if required and all(v=="PASS" for v in required.values()) else "BLOCKED"


def maneuver_status(step,complete,aligned,has_allocator,post_samples,motor_sat,
                    unallocated_max,bounds,allocator_sat,yaw_change=0):
    observed="NOT TESTED" if not has_allocator else "PASS" if bounds and motor_sat==0 and allocator_sat==0 else "FAIL"
    post_observed=step is not None and post_samples>0
    return {"flight_completion":"PASS" if complete else "BLOCKED",
        "heading_alignment":"PASS" if aligned else "BLOCKED" if not complete else "FAIL",
        "yaw_direction":"NOT TESTED" if step is None else "PASS" if yaw_change>0 else "FAIL",
        "yaw_torque_allocated":"NOT TESTED" if not post_observed or unallocated_max is None else "PASS" if unallocated_max<1e-3 else "FAIL",
        "no_motor_saturation":observed if post_observed else "NOT TESTED",
        "observed_no_motor_saturation":observed}


def shutdown_deadline(raw):
    """Reconstruct the minimum deadline, not a guessed actuator outage."""
    fix=raw.get("phase2",{}).get("telemetry_fix",{});d=fix.get("last_decision") or {}
    candidates={}
    if d.get("last_query"):candidates["query_service"]=d["last_query"][1]+.65
    if d.get("last_actuator"):candidates["actuator_communication"]=d["last_actuator"][1]+.25
    for name,topic in d.get("topics",{}).items():
        candidates["receipt:"+name]=topic["received_wall_s"]+.65
    if d.get("query_pending") and d.get("last_query"):
        candidates["pending_response"]=d["last_query"][1]+.15
    limiting=min(candidates,key=candidates.get) if candidates else None
    return {"monitor_fault":fix.get("fault"),"outer_exception":raw.get("exception"),
        "limiting_deadline":limiting,"deadline_wall_s":candidates.get(limiting),
        "candidate_deadlines_wall_s":candidates,
        "last_check_actuator_age_s":d["wall_s"]-d["last_actuator"][1] if d.get("last_actuator") else None,
        "scope":"last recorded monitor state; outer receive error names all minimum safety deadlines, not proof of actuator loss"}


def parser_diagnostic(raw,expected):
    """Post-run microbenchmark only; no change to the flown runtime."""
    console=raw["px4_console"];begin=time.perf_counter()
    ok=all(allocation_status(console,expected) for _ in range(100));elapsed=time.perf_counter()-begin
    queries=[e["wall_s"] for e in raw["phase2"]["telemetry_fix"]["events"] if e["kind"]=="query"]
    return {"console_characters":len(console),"offline_100_checks_wall_s":elapsed,
        "mean_parser_ms":elapsed*10,"all_checks_true":ok,
        "last_5_query_wall_intervals_s":np.diff(queries)[-5:].tolist(),
        "scope":"full post-run console microbenchmark, not measured in-flight CPU attribution or unique cause"}


def topic_rows(rows,name):
    return [r for r in rows if name in (r.get("telemetry_gate") or {}).get("topics",{})]


def preservation_evidence(root,parts):
    protected=json.loads((parts/"protected_identities.json").read_text());result={}
    extra=("mujoco/reports/px4_mujoco_phase1.json","mujoco/reports/px4_mujoco_yaw_validation.json",
           "mujoco/reports/px4_mujoco_yaw_telemetry_fix.json")
    for name in [*protected,*extra]:
        base=hashlib.sha256(subprocess.check_output(["git","show",f"{BASE}:{name}"],cwd=root)).hexdigest()
        current=sha256(root/name)
        if current!=base:raise ValueError("protected historical file changed: "+name)
        result[name]={"base_sha256":base,"current_sha256":current,"unchanged":True}
    frozen=json.loads((parts/"flight_runtime_hashes.json").read_text())
    matches={name:sha256(root/name)==digest for name,digest in frozen["runtime_hashes"].items()}
    if not all(matches.values()):raise ValueError("final runtime differs from authorized flight; validation attribution invalid")
    return {"protected_files":result,"flight_runtime_freeze":frozen,"runtime_unchanged_after_flight":matches,
            "original_px4":source_identity(Path("/home/keyousuke/PX4-Autopilot"))}


def motor_response(times,motors,physical_yaw_row,step):
    t,c,y=map(lambda v:np.asarray(v,dtype=float),(times,motors,physical_yaw_row))
    if t.ndim!=1 or c.shape!=(len(t),4) or y.shape!=(4,) or np.any(np.diff(t)<=0) or not np.isfinite(np.r_[t,c.ravel(),y]).all():
        raise ValueError("aligned finite physical motor samples required")
    post=t>=step if step is not None else np.zeros(t.shape,dtype=bool)
    torque=c@y;differential=c@np.array([-1,1,-1,1])/4
    positive=post&(torque>1e-4)
    return {"physical_yaw_torque_model_derived_not_measured":distribution(torque[post]),
        "maximum_physical_yaw_torque_nm":float(np.max(torque[post])) if post.any() else None,
        "first_positive_physical_torque_s":float(t[np.flatnonzero(positive)[0]]) if positive.any() else None,
        "positive_differential_max":float(np.max(differential[post])) if post.any() else None,
        "bounds_valid":bool(np.all((c>=0)&(c<=1))),
        "motor_saturation_samples":int(np.sum(np.any(c>=1-1e-6,axis=1))),
        "poststep_motor_ranges":[distribution(c[post,i]) for i in range(4)]}


def plots(out,trace,raw,aux,response,audit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files=[];t=trace["t"];step=raw["phase2"]["actual_step_time_s"]
    view=(30,45) if step is not None else (0,float(t[-1]))
    def finish(fig,name):
        fig.tight_layout();fig.savefig(out/name,dpi=150);plt.close(fig);files.append(name)
    def style(axes):
        for ax in axes:
            if step is not None:ax.axvline(step,ls="--",color="grey")
            if not raw["complete"]:ax.axvline(t[-1],ls=":",color="red",label="Safety stop; no yaw step")
            ax.grid(alpha=.2)
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    axes[0].step(t,np.degrees(trace["desired"]),where="post",label="PX4 published target")
    axes[0].plot(t,np.degrees(trace["estimate"]),label="PX4 estimate")
    axes[0].plot(t,np.degrees(trace["true"]),label="MuJoCo true NED yaw")
    axes[0].set(ylabel="NED yaw (deg)",title="Yaw maneuver response: "+response["status"]+(" (stopped before step)" if step is None else ""));axes[0].legend()
    target=np.where(t>=step,100.,90.) if step is not None else np.full(t.shape,90.)
    axes[1].plot(t,np.degrees(angle_error(np.deg2rad(target),trace["true"])),label="Bridge hold/step command minus truth")
    axes[1].plot(t,np.degrees(angle_error(trace["estimate"],trace["true"])),label="Estimate minus truth")
    axes[1].set(ylabel="Yaw error (deg)");axes[1].legend()
    rates=aux.get("vehicle_rates_setpoint",{})
    if rates:axes[2].step(simulation_time(rates["timestamp"]),rates["yaw"],where="post",label="Desired FRD-z rate")
    axes[2].plot(t,trace["rate"],label="Actual FRD-z rate");axes[2].set(xlabel="Simulation time (s)",ylabel="rad/s",xlim=view);axes[2].legend()
    style(axes);finish(fig,"px4_allocation_yaw_response.png")
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    for i in range(4):axes[0].step(t,trace["motors"][:,i],where="post",label=f"Motor{i+1}")
    axes[0].set(ylabel="Normalized thrust");axes[0].legend()
    torque=aux.get("vehicle_torque_setpoint",{});alloc=aux.get("control_allocator_status",{})
    if torque:axes[1].step(simulation_time(torque["timestamp"]),torque["xyz[2]"],where="post",label="PX4 yaw request")
    if alloc:axes[1].step(simulation_time(alloc["timestamp"]),alloc["unallocated_torque[2]"],where="post",label="Unallocated yaw")
    axes[1].set(ylabel="PX4 normalized units");axes[1].legend()
    physical=np.array(audit["physical_com_frd_matrix_per_normalized_motor"])[2]
    axes[2].plot(t,trace["motors"]@physical,label="Model-derived actuator yaw torque")
    axes[2].set(xlabel="Simulation time (s)",ylabel="FRD Nm",xlim=view);axes[2].legend()
    style(axes);finish(fig,"px4_allocation_motor_and_torque.png")
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True)
    axes[0].plot(t,trace["position"][:,2]);axes[0].axhline(2.051,ls=":",color="grey");axes[0].set(ylabel="ENU height (m)")
    axes[1].plot(t,trace["position"][:,:2]);axes[1].set(ylabel="ENU XY (m)")
    local=trace["local"];flags=trace["flags"]
    axes[2].step(simulation_time(local["timestamp"]),local["heading_good_for_control"],where="post",label="Heading good")
    axes[2].step(simulation_time(flags["timestamp"]),flags["cs_mag_aligned_in_flight"],where="post",label="Final mag aligned")
    rows=[r for r in raw["records"] if r.get("telemetry_gate")]
    axes[2].step([r["time_s"] for r in rows],[r["telemetry_gate"]["safe_stop"] for r in rows],where="post",label="Recorded gate stop (last sample precedes latch)")
    axes[2].set(xlabel="Simulation time (s)",ylabel="State boolean");axes[2].legend()
    style(axes);finish(fig,"px4_allocation_position_and_health.png")
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for name in ("vehicle_local_position","estimator_status_flags","vehicle_status"):
        rows=topic_rows(raw["records"],name)
        axes[0].plot([r["time_s"] for r in rows],[r["telemetry_gate"]["topics"][name]["source_age_s"] for r in rows],label=name)
    axes[0].set(ylabel="Original source age (s)");axes[0].legend(fontsize=8)
    axes[1].plot(t,trace["velocity"]);axes[1].set(xlabel="Simulation time (s)",ylabel="ENU velocity (m/s)")
    style(axes);finish(fig,"px4_allocation_freshness_and_velocity.png")
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--px4-root",type=Path,required=True)
    p.add_argument("--compatibility-manifest",type=Path,required=True);p.add_argument("--dependency-search-path",required=True)
    args=p.parse_args();sys.path.append(args.dependency_search_path)
    root=Path(__file__).resolve().parents[2];out=root/"mujoco/reports";parts=out/"px4_yaw_allocation_parts"
    m=mujoco.MjModel.from_xml_path(str(MODEL));adapter=ActuatorAdapter(m,CONFIG);audit=physical_audit(adapter)
    identity=source_identity(args.px4_root,args.compatibility_manifest)
    preservation=preservation_evidence(root,parts)
    history={"model_xml":sha256(MODEL),"rotor_config":sha256(CONFIG),"rotor_actuators":sha256(MODEL.parent/"rotor_actuators.xml")}
    runs={};raws={};traces={}
    for name in ("disarmed_01","yaw_01"):
        path=parts/name/"smoke_summary.json";raws[name]=json.loads(path.read_text())
        verify_recording(raws[name],identity,history)
        runs[name],traces[name]=analyze_run(path)
        runs[name]["actual_yaw_axis_retained"]=allocation_status(raws[name]["px4_console"],np.array(audit["raw_px4_matrix"])[2])
        runs[name]["freshness"]=freshness_summary(raws[name]["records"],raws[name]["phase2"]["telemetry_fix"]["events"])
    s=raws["yaw_01"];flight=runs["yaw_01"];trace=traces["yaw_01"];step=flight["actual_step_time_s"]
    response=response_metrics(trace["t"],trace["true"],trace["rate"],step,flight["complete"])
    aux,params=load_auxiliary(flight["ulog_path"],flight["simulation_end_s"])
    # Actual flight parameters must still match immutable adapter geometry.
    expected=adapter.px4_geometry()
    matches={k:bool(k in params and np.isclose(float(params[k]),float(v),atol=1e-7,rtol=1e-6)) for k,v in expected.items()}
    if not all(matches.values()):raise ValueError("flight geometry or adapter configuration changed")
    motors=motor_response(trace["t"],trace["motors"],np.array(audit["physical_com_frd_matrix_per_normalized_motor"])[2],step)
    post=trace["t"]>=step if step is not None else np.zeros(trace["t"].shape,dtype=bool)
    flight["poststep_height_error_m"]=distribution(trace["position"][post,2]-2.051)
    flight["poststep_xy_excursion_m"]=distribution(np.linalg.norm(trace["position"][post,:2],axis=1))
    quats=np.array([r["quaternion_enu_flu"] for r in s["records"]]);upright=1-2*(quats[:,1]**2+quats[:,2]**2)
    flight["maximum_tilt_deg"]=float(np.max(np.degrees(np.arccos(np.clip(upright,-1,1)))))
    alloc=aux.get("control_allocator_status",{})
    if alloc:
        mask=simulation_time(alloc["timestamp"])>=step if step is not None else np.zeros(len(alloc["timestamp"]),dtype=bool)
        flight["allocator_poststep_samples"]=int(mask.sum())
        flight["allocator_poststep_torque_achieved_count"]=int(np.sum(alloc["torque_setpoint_achieved"][mask]))
        flight["allocator_unallocated_yaw"]=distribution(alloc["unallocated_torque[2]"][mask])
        flight["allocator_active_motor_saturation_count"]=int(sum(np.count_nonzero(alloc[f"actuator_saturation[{i}]"][mask]) for i in range(4)))
        arm=simulation_time(alloc["timestamp"])>=s.get("arm_request_at_s",float("inf"))
        flight["allocator_observed_motor_saturation_count"]=int(sum(np.count_nonzero(alloc[f"actuator_saturation[{i}]"][arm]) for i in range(4)))
        flight["allocator_prearm_lower_limit_count"]=int(sum(np.count_nonzero(alloc[f"actuator_saturation[{i}]"][~arm]==-2) for i in range(4)))
        flight["allocator_observed_scope"]="time after recorded non-force arm request; no poststep samples if step absent"
    for topic,column in (("vehicle_rates_setpoint","yaw"),("vehicle_torque_setpoint","xyz[2]")):
        data=aux.get(topic,{})
        flight[topic+"_poststep"]=distribution(data[column][simulation_time(data["timestamp"])>=step]) if data and step is not None else None
    status={"physical_mapping":"PASS","isolated_verified_binary":"PASS",
        "disarmed_communication":"PASS" if runs["disarmed_01"]["complete"] else "BLOCKED",
        "yaw_axis_retained":"PASS" if flight["actual_yaw_axis_retained"] else "FAIL",
        "physical_yaw_response":response["status"],"settling":response["settling_status"],
        "flight_safety":"PASS" if s["complete"] and not s["phase2"]["telemetry_fix"]["fault"] else "BLOCKED",
        "safe_stop_cleanup":"PASS" if s["px4_exit_code"]==0 and s["zero_control_on_stop"]==[0.,0.,0.,0.] else "FAIL"}
    status.update(maneuver_status(step,flight["complete"],flight["alignment"]["final_heading_good"],bool(alloc),
        flight.get("allocator_poststep_samples",0),motors["motor_saturation_samples"],
        (flight.get("allocator_unallocated_yaw") or {}).get("max_abs"),motors["bounds_valid"],
        flight.get("allocator_observed_motor_saturation_count"),flight.get("poststep_true_yaw_change_deg") or 0))
    status["phase3_entry"]=entry_status(status)
    from pyulog import ULog
    previous=next((out/"px4_yaw_telemetry_fix_parts/yaw_01").rglob("*.ulg"))
    prior=ULog(str(previous),message_name_filter_list=[])
    keys=sorted(k for k in prior.initial_parameters if k.startswith(("MC_","MPC_","EKF2_","CA_")))
    differences={k:{"previous":prior.initial_parameters[k],"current":params.get(k)}
                 for k in keys if params.get(k)!=prior.initial_parameters[k]}
    previous_raw=out/"px4_yaw_telemetry_fix_parts/yaw_01/smoke_summary.json"
    prior_raw=json.loads(previous_raw.read_text())
    protocol_same={key:prior_raw[key]==s[key] for key in ("px4_environment","initial_qpos","synthetic_sensor_noise")}
    protocol_same["magnetic_field"]=prior_raw["phase2"]["field_sha256"]==s["phase2"]["field_sha256"]
    if not all(protocol_same.values()) or set(differences)-{"EKF2_MAG_DECL"}:
        raise ValueError("configured control/EKF/environment protocol differs unexpectedly")
    report={"phase":"PX4-MuJoCo Phase2C yaw allocation compatibility","date":"2026-10-11",
        "branch":"feat/px4-mujoco-yaw-allocation-fix","base_commit":BASE,"status":status,
        "px4":identity,"physical_audit":audit,
        "preservation":preservation,
        "verification":json.loads((parts/"verification.json").read_text()),
        "previous_protocol_comparison":{"explicit_protocol_identical":protocol_same,
            "previous_summary_sha256":sha256(previous_raw),"previous_ulog_sha256":sha256(previous),
            "controller_ekf_allocator_parameters_compared":len(keys),"ulog_parameter_differences":differences,
            "declination_difference_scope":"volatile EKF2_MAG_DECL is saved without notification by EKF2::UpdateMagCalibration; both builds have configured default0 and DECL_TYPE3, identical magnetic/GNSS inputs. Old log starts1.344s, new .304s; exact unlogged change time not observable. No gain/EKF tuning."},
        "flight_parameters_match":matches,"runs":runs,"yaw_response":response,"motor_response":motors,
        "shutdown_diagnostics":shutdown_deadline(s),
        "post_run_parser_diagnostic":parser_diagnostic(s,np.array(audit["raw_px4_matrix"])[2]),
        "conclusions":{"confirmed":["Pinned weak-row filter removes original yaw despite rank-four physical map",
                "Custom live allocator retains unchanged yaw row with original motor bounds",
                "Limited flight stopped before yaw request on host query-service deadline; cleanup zeros motors and exits PX4",
                "No actual +10degree maneuver or settling result exists in this stage"],
            "supported_hypothesis":["Full accumulated console parsing in the high-frequency added yaw-row gate contributes to increasing wall-clock query intervals; offline microbenchmark supports plausibility, not unique attribution"],
            "unresolved":["Runtime scheduling/diagnostic overhead contribution versus other host load", "Complete yaw tracking, coupled hover, and final EKF alignment in this custom run"],
            "next_recommendation":"One separately authorized offline bounded/incremental allocator diagnostic and wall-versus-simulation query-scheduling audit, preserving fail-closed deadlines; no automatic new flight or Phase3."},
        "execution":{"recorded_sitl_invocations":len(list(parts.rglob("smoke_summary.json"))),
            "disarmed_recordings":sum(r["kind"]=="disarmed_communication_smoke" for r in raws.values()),
            "flight_recordings":sum(r.get("flight_command_sent",False) for r in raws.values()),
            "workers_recorded":[r["workers"] for r in raws.values()],
            "scope":"counts are verified recorded runs, not an OS-wide proof of historical launches"},
        "resources":{"build":json.loads((parts/"build/build_resources.json").read_text()),
            "bridge_peak_rss_bytes":max(r["bridge_peak_rss_bytes"] for r in runs.values()),
            "px4_sampled_peak_rss_bytes":max(r["px4_sampled_peak_rss_bytes"] for r in runs.values()),
            "min_available_bytes":min(r["system_min_available_bytes"] for r in runs.values()),
            "offline_peak_rss_bytes":own_peak_rss_bytes(),"oom_telemetry_available":False,
            "oom_observed_during_owned_runs":any(r.get("exception",{}).get("type")=="MemoryError" for r in raws.values() if r.get("exception")),
            "swap_change_audit":"no swap command used in this stage; no system-wide monitoring"},
        "limitations":["Single bounded positive yaw step; no negative or aggressive flight test",
            "SITL-only local source guard, not upstream PX4 or real flight approval",
            "Ideal actuator without ESC/motor delay; km estimated, rotor handedness requires hardware verification",
            "Physical force/torque model-derived, not independent measured hardware values",
            "Unchanged normalized PX4 torque request units are not Nm; saturation may prioritize other axes",
            "Console health polling is finite fail-closed, not hard-real-time or an observer of every transient"]}
    files=plots(out,trace,s,aux,response,audit);report["figures"]={n:sha256(out/n) for n in files}
    report["resources"]["offline_peak_rss_bytes"]=own_peak_rss_bytes()
    atomic_json(out/"px4_mujoco_yaw_allocation_fix.json",report)
    print(json.dumps({"status":status,"yaw":response,"motor":motors,"hold":flight["poststep_height_error_m"]},indent=2))


if __name__=="__main__":main()
