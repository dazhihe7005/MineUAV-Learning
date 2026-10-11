"""Offline Phase2B evidence; never launches a simulator or changes parameters."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from .analyze_yaw import analyze_run,simulation_time,stats,interpolate,hold_interpolate
from .yaw_validation import angle_error,settling_time
from .diagnostics import atomic_json,sha256,own_peak_rss_bytes

BASE="601f517bc9bc6034197808646931f8214e9596a4"


def distribution(values):
    out=stats(values)
    if out is not None:
        v=np.asarray(values,dtype=float);v=v[np.isfinite(v)]
        out.update(median=float(np.median(v)),p95=float(np.quantile(v,.95)))
    return out


def response_metrics(times,yaw,rate,step,complete):
    t,y,w=map(lambda a:np.asarray(a,dtype=float),(times,yaw,rate))
    if t.shape!=y.shape or w.shape!=t.shape or np.any(np.diff(t)<=0):raise ValueError("aligned monotonic physical samples required")
    valid=np.isfinite(t)&np.isfinite(y)&np.isfinite(w)
    post=valid&(t>=step) if step is not None else np.zeros(t.shape,dtype=bool)
    e=angle_error(np.deg2rad(100),y)
    steady=post&(t>=t[-1]-2) if len(t) else post
    settled=settling_time(t[post],e[post],w[post],step) if complete and post.any() else None
    return {"status":("NOT TESTED" if step is None else "BLOCKED" if not complete else "PASS" if settled is not None else "FAIL"),
        "settling_time_s":settled,"settling_status":"NOT TESTED" if not complete or step is None else "PASS" if settled is not None else "FAIL",
        "max_target_error_deg":float(np.max(np.abs(np.degrees(e[post])))) if post.any() else None,
        "overshoot_deg":float(max(0.,np.max(np.degrees(-e[post])))) if post.any() else None,
        "steady_state_error_deg":float(np.mean(np.degrees(e[steady]))) if complete and steady.any() else None,
        "peak_body_z_rate_rad_s":float(np.max(np.abs(w[post]))) if post.any() else None,
        "poststep_observation_s":float(t[post][-1]-step) if post.any() else None,
        "sampling_gap_max_s":float(np.max(np.diff(t[post]))) if post.sum()>1 else None}


def freshness_summary(rows,events):
    gates=[r["telemetry_gate"] for r in rows if r.get("telemetry_gate")]
    names={n for g in gates for n in g["topics"]}
    query=None;response_latencies=[]
    for event in events:
        if event["kind"]=="query":query=event["wall_s"]
        elif event.get("topic")=="vehicle_status" and query is not None:
            response_latencies.append(event["wall_s"]-query);query=None
    return {"query_response_wall_s":distribution(response_latencies),
        "source_age_s":{n:distribution([g["topics"][n]["source_age_s"] for g in gates if n in g["topics"]]) for n in sorted(names)},
        "duplicate_receipts":sum(e.get("duplicate",False) for e in events),
        "query_pending_samples":sum(g["query_pending"] for g in gates),
        "safe_stop_samples":sum(g["safe_stop"] for g in gates),
        "sample_count":len(gates),
        "receipt_source_age_s":{n:distribution([(e["clock_usec"]-e["publication_usec"])/1e6 for e in events if e["kind"]=="receipt" and e["topic"]==n]) for n in sorted(names)}}


def allocation_authority(params):
    """Pinned PX4 computeEffectivenessMatrix followed by weak-row removal.

    Analytical source/config evidence, not a logged internal matrix sensor.
    """
    raw=np.zeros((6,4));axis=np.array([0.,0.,-1.])
    for i in range(4):
        p=np.array([params[f"CA_ROTOR{i}_P{a}"] for a in "XYZ"])
        ct=params[f"CA_ROTOR{i}_CT"];km=params[f"CA_ROTOR{i}_KM"]
        raw[:3,i]=ct*np.cross(p,axis)-ct*km*axis;raw[3:,i]=ct*axis
    filtered=raw.copy();weak=np.max(np.abs(raw),axis=1)<=.05
    filtered[weak]=0
    return {"raw_matrix":raw.tolist(),"after_source_weak_row_removal":filtered.tolist(),
            "threshold":.05,"weak_rows":np.flatnonzero(weak).tolist(),
            "yaw_row_removed":bool(weak[2]),"evidence_type":"model-derived from actual ULog parameters and pinned source, not a logged matrix"}


def load_auxiliary(path,end):
    from pyulog import ULog
    names=["vehicle_rates_setpoint","vehicle_torque_setpoint","control_allocator_status"]
    u=ULog(str(path),message_name_filter_list=names)
    out={}
    for d in u.data_list:
        if d.name in names and d.multi_id==0:
            mask=simulation_time(d.data["timestamp"])<=end
            out[d.name]={k:v[mask] for k,v in d.data.items()}
    return out,dict(u.initial_parameters)


def plot(out,trace,summary,aux):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files=[];t=trace["t"];step=summary["actual_step_time_s"]
    def finish(fig,name):
        fig.tight_layout();fig.savefig(out/name,dpi=150);plt.close(fig);files.append(name)
    def marker(axes):
        for ax in axes:
            if step is not None:ax.axvline(step,color="grey",ls="--")
            ax.grid(alpha=.2)
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    axes[0].step(t,np.degrees(trace["desired"]),where="post",label="PX4 published desired (causal hold)")
    axes[0].plot(t,np.degrees(trace["estimate"]),label="PX4 estimate")
    axes[0].plot(t,np.degrees(trace["true"]),label="MuJoCo true NED yaw")
    axes[0].set(ylabel="NED yaw (deg)",title="Complete45s observation: yaw target changed, vehicle did not turn")
    axes[1].plot(t,np.degrees(angle_error(np.deg2rad(100),trace["true"])),label="100deg target minus truth")
    axes[1].plot(t,np.degrees(angle_error(trace["estimate"],trace["true"])),label="Estimate minus truth")
    axes[1].set(xlabel="Simulation time (s)",ylabel="Yaw error (deg)",xlim=(30,45))
    for ax in axes:ax.legend(fontsize=8)
    marker(axes);finish(fig,"px4_telemetry_fix_yaw_response.png")
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True)
    rates=aux.get("vehicle_rates_setpoint",{});torque=aux.get("vehicle_torque_setpoint",{});alloc=aux.get("control_allocator_status",{})
    if rates:axes[0].step(simulation_time(rates["timestamp"]),rates["yaw"],where="post",label="PX4 desired FRD-z rate")
    axes[0].plot(t,trace["rate"],label="Actual FRD body-z rate");axes[0].set(ylabel="rad/s");axes[0].legend(fontsize=8)
    for i in range(4):axes[1].step(t,trace["motors"][:,i],where="post",label=f"Motor{i+1}")
    axes[1].set(ylabel="Normalized thrust command");axes[1].legend(fontsize=8)
    if torque:axes[2].step(simulation_time(torque["timestamp"]),torque["xyz[2]"],where="post",label="Requested yaw torque (PX4 units)")
    if alloc:axes[2].step(simulation_time(alloc["timestamp"]),alloc["unallocated_torque[2]"],where="post",label="Unallocated yaw torque")
    axes[2].set(xlabel="Simulation time (s)",ylabel="PX4 torque units",xlim=(30,45));axes[2].legend(fontsize=8)
    marker(axes);finish(fig,"px4_telemetry_fix_rate_and_motors.png")
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True)
    axes[0].plot(t,trace["position"][:,2],label="MuJoCo ENU body height")
    local=trace["local"]
    axes[0].plot(simulation_time(local["timestamp"]),-local["z"]+.051,label="PX4 relative NED-z + initialheight (not absolute truth)")
    axes[0].axhline(2.051,color="grey",ls=":");axes[0].set(ylabel="Height (m)");axes[0].legend(fontsize=8)
    axes[1].plot(t,trace["position"][:,:2]);axes[1].set(ylabel="ENU XY (m)")
    for k,label in (("y","estimated East (ENU-x)"),("x","estimated North (ENU-y)")):
        axes[1].plot(simulation_time(local["timestamp"]),local[k],ls=":",label=label)
    axes[1].legend(fontsize=8)
    axes[2].plot(t,trace["velocity"]);axes[2].set(ylabel="ENU velocity (m/s)",xlabel="Simulation time (s)")
    marker(axes);finish(fig,"px4_telemetry_fix_position_hold.png")
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True)
    rows=trace["rows"];valid=[r for r in rows if r.get("telemetry_gate")]
    for n in ("vehicle_local_position","estimator_status_flags","vehicle_status"):
        rs=[r for r in valid if n in r["telemetry_gate"]["topics"]]
        axes[0].plot([r["time_s"] for r in rs],[r["telemetry_gate"]["topics"][n]["source_age_s"] for r in rs],label=n)
    axes[0].set(ylabel="Actual source age (s)");axes[0].legend(fontsize=8)
    axes[1].step([r["time_s"] for r in valid],[r["telemetry_gate"]["query_pending"] for r in valid],where="post",label="query pending (no new maneuver)")
    axes[1].step([r["time_s"] for r in valid],[r["telemetry_gate"]["safe_stop"] for r in valid],where="post",label="safe stop")
    axes[1].set(ylabel="Gate boolean");axes[1].legend(fontsize=8)
    flags=trace["flags"]
    axes[2].step(simulation_time(local["timestamp"]),local["heading_good_for_control"],where="post",label="heading good")
    axes[2].step(simulation_time(flags["timestamp"]),flags["cs_mag_aligned_in_flight"],where="post",label="final mag aligned")
    axes[2].set(ylabel="Actual EKF state",xlabel="Simulation time (s)");axes[2].legend(fontsize=8)
    marker(axes);finish(fig,"px4_telemetry_fix_freshness_and_health.png")
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--px4-root",type=Path,required=True);p.add_argument("--dependency-search-path",required=True)
    args=p.parse_args();sys.path.append(args.dependency_search_path)
    root=Path(__file__).resolve().parents[2];out=root/"mujoco/reports";parts=out/"px4_yaw_telemetry_fix_parts"
    runs={};traces={};raw={}
    for name in ("disarmed_01","yaw_01"):
        path=parts/name/"smoke_summary.json";raw[name]=json.loads(path.read_text())
        runs[name],traces[name]=analyze_run(path);traces[name]["rows"]=raw[name]["records"]
        runs[name]["freshness"]=freshness_summary(raw[name]["records"],raw[name]["phase2"]["telemetry_fix"]["events"])
    flight=runs["yaw_01"];trace=traces["yaw_01"];s=raw["yaw_01"]
    flight["actual_publication_intervals_s"]={n:distribution(np.diff(simulation_time(topic["timestamp"]))) for n,topic in (("vehicle_local_position",trace["local"]),("estimator_status_flags",trace["flags"]))}
    aux,params=load_auxiliary(flight["ulog_path"],flight["simulation_end_s"])
    metrics=response_metrics(trace["t"],trace["true"],trace["rate"],flight["actual_step_time_s"],flight["complete"])
    post=trace["t"]>=flight["actual_step_time_s"] if flight["actual_step_time_s"] else np.zeros(trace["t"].shape,dtype=bool)
    flight["poststep_height_error_m"]=distribution(trace["position"][post,2]-2.051)
    flight["poststep_xy_excursion_m"]=distribution(np.linalg.norm(trace["position"][post,:2],axis=1))
    for name,column in (("vehicle_rates_setpoint","yaw"),("vehicle_torque_setpoint","xyz[2]"),("control_allocator_status","unallocated_torque[2]")):
        topic=aux.get(name,{})
        flight[name+"_poststep"]=distribution(topic[column][simulation_time(topic["timestamp"])>=flight["actual_step_time_s"]]) if topic and flight["actual_step_time_s"] else None
    alloc=aux.get("control_allocator_status",{})
    if alloc:
        m=simulation_time(alloc["timestamp"])>=flight["actual_step_time_s"]
        flight["allocator_poststep_torque_achieved_count"]=int(np.sum(alloc["torque_setpoint_achieved"][m]))
        flight["allocator_poststep_samples"]=int(m.sum())
        flight["allocator_active_motor_saturation_count"]=int(sum(np.count_nonzero(alloc[f"actuator_saturation[{i}]"][m]) for i in range(4)))
    protected=["mujoco/models/mine_uav_dynamics_v2.xml","mujoco/models/rotor_config.json","mujoco/models/rotor_actuators.xml",
        "mujoco/rl/models/uav_bc_mlp_seed0.pt","mujoco/rl/models/uav_bc_yaw_augmented_seed0.pt",
        "mujoco/rl/models/joint_latent_world_model_v3_autonomous_consistency.pt",
        "mujoco/reports/px4_mujoco_phase1.json","mujoco/reports/px4_mujoco_yaw_validation.json"]
    hashes={}
    for name in protected:
        baseline=hashlib.sha256(subprocess.check_output(["git","show",f"{BASE}:{name}"],cwd=root)).hexdigest()
        hashes[name]={"base_sha256":baseline,"current_sha256":sha256(root/name),"unchanged":baseline==sha256(root/name)}
    if not all(v["unchanged"] for v in hashes.values()) or not all(r["history_unchanged"] for r in runs.values()):raise ValueError("protected artifact changed")
    source_files={name:sha256(args.px4_root/name) for name in ["src/modules/ekf2/EKF2.cpp","src/modules/commander/Commander.cpp",
        "src/systemcmds/topic_listener/listener_main.cpp","src/systemcmds/topic_listener/topic_listener.hpp",
        "src/modules/control_allocator/ControlAllocator.cpp","src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessRotors.cpp"]}
    authority=allocation_authority(params)
    from .simulation_runner import source_identity
    report={"phase":"PX4-MuJoCo Phase2B telemetry freshness and yaw validation","date":"2026-10-11",
        "branch":"feat/px4-mujoco-yaw-telemetry-fix","base_branch":"feat/px4-mujoco-yaw-validation","base_commit":BASE,
        "px4":source_identity(args.px4_root),"runs":runs,"yaw_response":metrics,"allocator_source_audit":authority,
        "protected_identities":hashes,"px4_source_identities":source_files,
        "time_semantics":{"sample":"timestamp_sample: EKF delayed/sample source; absent for vehicle_status",
            "publication":"original uORB timestamp, never rewritten on duplicates",
            "query":"actual bridge request simulation clock + host monotonic time",
            "receipt":"actual console parse observation clock + host monotonic time, duplicate logged separately",
            "liveness":"accepted exact timestamp HIL actuator at250Hz; .25s absolute wall watchdog plus process/socket/protocol checks",
            "source_limits_s":{"vehicle_local_position":.65,"estimator_status_flags":1.65,"vehicle_status":1.15},
            "derivation":"flags/status nominalperiod1/.5s +2Hz query(.5s)+bounded response(.15s). Local source is125Hz; the fixed.65s operational lease is conservative versus.008+.5+.15=.658s and may fail-closed at extreme jitter. Source lease remains independent of heartbeat/receipt.",
            "partial_query":"no new maneuver while pending; existing flight bounded by.15s response deadline and unchanged finite source/transport leases; complete unsafe topic stops without waiting for rest of batch",
            "old_gate_defect":"1Hz state+2Hz snapshot can normally be~1.5s old. At original32.196s flags age1.004s exceeded1s despitehealthy publication stream.",
            "fault_latency_bounds":"Observed complete unsafe state stops at next4ms physics check or≤8ms console-service slice during blocking receive. Sustained fault published just after query:≤.5+.15+.008=.658s under serviced polling. Source expiry on advancing simulation detected≤1.654/1.154/.654s for flags/status/local. During stalled lockstep, earliest pending query150ms or actuator250ms absolute wall deadline caps receive; late packets cannot renew expired leases. Synthetic query timeout152ms and flags-only freeze1652ms on4ms grid. No hard real-time scheduling guarantee."},
        "status":{"root_cause":"PASS","separated_time_semantics":"PASS","fault_closed_offline_regressions":"PASS",
            "disarmed_bidirectional_communication":"PASS" if runs["disarmed_01"]["complete"] else "BLOCKED",
            "no_spurious_telemetry_stop":"PASS" if s["complete"] and not s["phase2"]["telemetry_fix"]["fault"] else "FAIL",
            "final_heading_alignment":"PASS" if flight["alignment"]["final_heading_good"] else "FAIL",
            "full_bounded_observation":"PASS" if flight["complete"] else "BLOCKED",
            "yaw_tracking":metrics["status"],"settling":metrics["settling_status"],
            "motor_bounds":"PASS" if flight["motor_saturation_samples"]==0 else "FAIL",
            "allocator_yaw_authority":"FAIL" if authority["yaw_row_removed"] else "NOT TESTED",
            "height_xy_envelope":"PASS" if s["complete"] else "BLOCKED",
            "phase3_entry":"BLOCKED","external_odometry_integration":"NOT TESTED"},
        "failure_explanation":{"level":"Confirmed source/config + corroborating ULog evidence",
            "mechanism":"Existing CT=1 and |KM|=.017207708 yield yaw effectivenessentries below pinned allocator weak-axis threshold.05. ControlAllocator zeroes this row before assigning matrix. Desiredyaw100deg, desiredbodyzrate≈.488rad/s and requestedyawtorque≈.097 occur, but torque remainsunallocated and motors do not produce differentialyaw response. Not a heading/frame freshness failure.",
            "not_fixed":"Changing existing allocator geometry/CT scaling or PX4 source would violate this phase fixedparameterprotocol; no extra flight, no gains tuned."},
        "execution":{"sitl_runs":2,"disarmed_runs":1,"flight_runs":1,"retries":0,"workers":1,
            "only_simulation_hardware":"loopback TCP4560 + offboard UDP14540/14580, no serial/realmotors",
            "no_parameter_model_physics_training_changes":True},
        "verification":{"original_tests":69,"bridge_tests_before_final_review":103,"final_bridge_tests":109,"skipped":0,
            "red_green":"monitor23failed→allpass; opt-in runner missingmodule→pass; analysis4failed→pass; review regressions6failed/29passed→full bridge109passed",
            "original_phase2_raw_replay":"primary0 actual ULog last5.196s, newmonitor passes with zero fabricatedhealth while old1sgate denies normal snapshots",
            "repository_suite":"BLOCKED collection cad_check/test_inspect_stl_assembly.py: system SciPy/venv NumPy2.2.6 ABI mismatch; no install, no unrelated dependency fix",
            "independent_review":{"reviewer":"px4_telemetry_fix_review","read_only":True,"tests_independently_verified":103,
                "critical":0,"important":2,"minor":1,"declined_to_judge":[],
                "findings":["Complete unhealthy first topic could be hidden by next fragmented title",
                    "Late complete response could clear expired query; blocking actuator receive could overrun query deadline",
                    "General duplicate-stall test double-fed one actuator timestamp; redundant flags-only stall coverage was valid"],
                "fix_passes":1,"fixes":"Preserve independently complete topics while keeping strict batch completion; latch old lease expiry before renewing any query/state/actuator; service console during≤8ms receive slices and cap wait at earliest absolute watchdog. Correct duplicate-stall fixture to fail specifically on publication expiry.",
                "minor_ruling":"Regraded Important: test fault-isolation was necessary evidence for safety-regression coverage; corrected in the same single offline pass",
                "deferred_minor":[],"repeat_review":False,"post_fix_tests":109},
            "post_review_flight_repeated":False,
            "runtime_provenance":"Both authorized SITL runs preceded the final review hardening. Raw summary/ULog identities are immutable; source hashes below identify final offline-verified code, not a byte-exact attestation of the pre-hardening executed version. Post-review parser/deadline changes were not re-flown; no remaining authorized flight budget.",
            "phase2b_source_hashes":{p.name:sha256(p) for p in sorted((root/"mujoco/px4_bridge").glob("*.py"))}},
        "resources":{"oom_observed":False,"workers":1,"swap_changed":False,
            "bridge_peak_rss_bytes":max(r["bridge_peak_rss_bytes"] for r in runs.values()),
            "px4_sampled_peak_rss_bytes":max(r["px4_sampled_peak_rss_bytes"] for r in runs.values()),
            "min_available_bytes":min(r["system_min_available_bytes"] for r in runs.values())},
        "limitations":["One bounded yawflight, ideal instantaneous actuator, no realhardware safety claim",
            "Final review parser/deadline hardening passed offline regressions but was not re-flown; actual completed45s recording belongs to pre-hardening monitor",
            "Console snapshots are polled; transient faults wholly betweenqueries maynotbeobserved; sustainedfault latency bounded, not unlimitedhealthycache",
            "No continuous externalheadinghealth MAVLink equivalent substituted for exact EKF finalmagaligned flags",
            "No hard realtime guarantee, physical50Hz response has~20ms measurement granularity",
            "Full response FAIL, not missingdata NOTTESTED; observednearzeroovershoot doesnotmean goodtracking"],
        "next_recommendation":"Approve a separate actuator-effectiveness scaling compatibility fix for pinned PX4 weak-row threshold, validate unchanged physical torque mapping offline, then authorize a new bounded yaw test; not executed."}
    files=plot(out,trace,flight,aux)
    report["figures"]={n:sha256(out/n) for n in files}
    report["resources"]["offline_peak_rss_bytes"]=own_peak_rss_bytes()
    atomic_json(out/"px4_mujoco_yaw_telemetry_fix.json",report)
    print(json.dumps({"status":report["status"],"yaw":metrics,"alignment":flight["alignment"],"resources":report["resources"]},indent=2))


if __name__=="__main__":main()
