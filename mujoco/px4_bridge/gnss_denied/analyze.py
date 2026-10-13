"""Read-only Phase3D evidence assessment; no PX4 execution entry point."""
import argparse
import json
from pathlib import Path
import numpy as np

from ..diagnostics import atomic_json,sha256
from ..compatibility_identity import MANIFEST
from ..logger_configuration import startup_tree_hash
from ..analyze_odometry_reverification import (read_log,stats,log_is_complete,
                                              artifact_hashes_match,passive_status)
from ..analyze_odometry_fusion import (assess_aid_source,_end_flags,_local_valid,
                                     assess_disarmed,cut_at_end,local_comparison)
from ..odometry_reverification_evidence import assess_direct,compare_payloads,END_USEC
from .runner import KIND,PARAMETERS,SOURCE_PARAMETERS,protocol_content,runtime_hashes,ATTEMPT_PATH
from .evidence import aggregate,assess_boundary,assess_gnss_absence,assess_non_gnss_sources


def assess_safety(raw):
    from ..analyze_disarmed_verification import assess_run
    from ..telemetry_freshness import TelemetryFreshness
    historical=assess_run(raw)
    names=("no_safety_fault","new_streaming_runtime","query_count","query_deadlines",
           "allocator_yaw_row","actuator_watchdog","incremental_parser","instrumentation_complete","simulation_boundaries")
    checks={name:historical["checks"][name] for name in names}
    decision=raw.get("phase2",{}).get("telemetry_fix",{}).get("last_decision") or {}
    clock=decision.get("clock_usec",np.nan);wall=decision.get("wall_s",np.nan)
    leases=True
    for name,limit in TelemetryFreshness.SOURCE_LIMITS.items():
        topic=decision.get("topics",{}).get(name,{})
        publication=topic.get("publication_usec",np.nan);received=topic.get("received_wall_s",np.nan)
        leases &= bool(np.isfinite([clock,wall,publication,received]).all()
                       and 0<=clock-publication<=limit and 0<=wall-received<=.65)
    checks["final_source_leases"]="PASS" if leases else "FAIL"
    checks["no_flight_authority"]="PASS" if decision.get("allow_new_maneuver") is False and decision.get("flight_authorized") is False else "FAIL"
    return {"status":"PASS" if all(v=="PASS" for v in checks.values()) else "FAIL","checks":checks,
            "query_count":historical["query_count"],"monitor_query_count":historical["monitor_query_count"]}


def fresh_run_identity(raw,protocol_hash):
    return (raw.get("kind")==KIND and raw.get("not_flight_proof") is True
            and raw.get("cold_start_no_gnss") is True and raw.get("protocol_sha256")==protocol_hash)


def global_origin_evidence(topics):
    local=topics.get("vehicle_local_position",{})
    result={"fields":{},"global_position_records":len(topics.get("vehicle_global_position",{}).get("timestamp",()))}
    valid=True
    for key in ("ref_timestamp","xy_global","z_global"):
        values=np.asarray(local.get(key,()),dtype=float)
        ok=len(values)>0 and np.isfinite(values).all() and np.all(values==0)
        valid &= ok
        result["fields"][key]={"samples":len(values),"nonzero_samples":int(np.sum(values!=0))}
    return {**result,"status":"PASS" if valid and result["global_position_records"]==0 else "BLOCKED",
            "meaning":"No observed global origin; any nonzero origin requires source explanation, not assumed GNSS."}


def fusion_evidence(topics,params,sent):
    visual=topics.get("vehicle_visual_odometry")
    direct=assess_direct(visual,end_usec=END_USEC)
    comparison,_=compare_payloads(sent,visual) if direct["status"]=="PASS" else ({"status":"BLOCKED"},None)
    aids={key:assess_aid_source(topics.get("estimator_aid_src_"+key),end_usec=END_USEC) for key in ("ev_pos","ev_vel")}
    local=_local_valid(topics.get("vehicle_local_position"),end_usec=END_USEC)
    result={"direct":direct,"source_vs_received":comparison,"aid_sources":aids,
            "local_status":"PASS" if local else "FAIL",
            "direct_status":comparison["status"] if direct["status"]=="PASS" else direct["status"]}
    for key,aid in aids.items():
        result[key+"_status"]=("BLOCKED" if aid["status"]=="BLOCKED" else "PASS" if aid["pass"] and local
            and params.get("EKF2_EV_CTRL")==5
            and _end_flags(topics.get("estimator_status_flags"),("cs_"+key,),end_usec=END_USEC) else "FAIL")
    return result


def analyze(runtime,report_path,figure_dir):
    runtime=Path(runtime);raw_path=runtime/"smoke_summary.json";source_path=runtime/"odometry_source.json"
    raw=json.loads(raw_path.read_text());source=json.loads(source_path.read_text());protocol=source["protocol"]
    if source["summary_sha256"]!=sha256(raw_path):raise ValueError("immutable summary identity mismatch")
    logs=sorted(runtime.rglob("*.ulg"));full={};params={};logmeta={};log_error=None
    if len(logs)==1:
        try:full,params,logmeta=read_log(logs[0])
        except (ValueError,OSError,IndexError) as error:log_error=str(error)
    else:log_error="exactly one independent ULog required"
    topics,excluded=cut_at_end(full,END_USEC)
    sent=source["sent_samples"];samples=source["source_samples"]
    stamps=np.asarray([s["sample_time_usec"] for s in sent],dtype=np.int64)
    origin_stamps=np.asarray([s["sample_time_usec"] for s in samples],dtype=np.int64)
    wall=np.asarray([s["send_monotonic_s"] for s in sent],dtype=float)
    producer_ok=bool(len(stamps)==len(origin_stamps)==1000 and np.array_equal(stamps,origin_stamps)
        and stamps[0]==1_000_000 and stamps[-1]==20_980_000 and np.all(np.diff(stamps)==20_000)
        and np.isfinite(wall).all() and np.all(np.diff(wall)>0))
    fusion=fusion_evidence(topics,params,sent)
    counts=raw.get("messages_sent",{})
    gnss=assess_gnss_absence(topics,params,counts,end_usec=END_USEC)
    other=assess_non_gnss_sources(topics,params,end_usec=END_USEC)
    origin=global_origin_evidence(topics)
    parameter_ok=all(params.get(k)==v for k,v in {**PARAMETERS,**SOURCE_PARAMETERS}.items())
    current=None;identity_error=None
    try:current=protocol_content(Path(protocol["px4"]["binary"]).parents[3],MANIFEST)
    except (OSError,ValueError) as error:identity_error=str(error)
    identity_ok=bool(current==protocol and source["runtime_source_sha256"]==runtime_hashes()==source["runtime_hashes_after"]
        and raw.get("px4")==protocol["px4"]==source.get("px4_identity_after")
        and all(protocol[key]==source.get(key+"_after") for key in ("model_hashes","checkpoint_hashes","history_hashes"))
        and source.get("post_run_identity_error") is None and fresh_run_identity(raw,source["protocol_sha256"]))
    attempt_path=ATTEMPT_PATH
    attempt=json.loads(attempt_path.read_text()) if attempt_path.is_file() else {}
    protocol_path=Path(attempt.get("protocol_path","/nonexistent/phase3d_protocol"))
    attempt_ok=bool(protocol_path.is_file() and sha256(protocol_path)==source["protocol_sha256"]
        and attempt.get("attempt")==attempt.get("authorized_maximum")==1
        and attempt.get("protocol_sha256")==source["protocol_sha256"] and attempt.get("runtime")==str(runtime.resolve()))
    measurements=raw.get("runtime_measurements",{});queries=measurements.get("queries",[])
    durations=np.asarray([q["end_wall_s"]-q["start_wall_s"] for q in queries if q.get("end_wall_s") is not None])
    intervals=np.diff([q["start_wall_s"] for q in queries])
    fault=raw.get("phase2",{}).get("telemetry_fix",{}).get("fault")
    safety=assess_safety(raw)
    safety_ok=safety["status"]=="PASS" and raw.get("complete") is True and measurements.get("pending_query_max",999)<=1
    rows=raw.get("records",[])
    physical_zero=bool(rows and all(row.get("ctrl_omega_squared")==[0.,0.,0.,0.] for row in rows)
        and raw.get("final_physical_state",{}).get("ctrl")==[0.,0.,0.,0.]
        and raw.get("zero_control_on_stop")==[0.,0.,0.,0.])
    commands=np.asarray(topics.get("vehicle_command",{}).get("command",()))
    arm_commands=int(np.sum(commands==400));origin_commands=int(np.sum(commands==100000))
    motors=topics.get("actuator_outputs_sim",{})
    motor_values=[np.asarray(motors.get(f"output[{i}]",()),dtype=float) for i in range(4)]
    outputs_zero=bool(all(len(v)>0 and np.isfinite(v).all() and np.all(v==0) for v in motor_values))
    actual_logger=runtime/"etc/logging/logger_topics.txt"
    logger_ok=bool(actual_logger.is_file() and sha256(actual_logger)==protocol["logger"]["text_sha256"]
        and raw.get("logger_configuration",{}).get("logger_file_sha256")==protocol["logger"]["text_sha256"]
        and raw.get("logger_configuration",{}).get("private_etc") is True
        and startup_tree_hash(runtime/"etc")==protocol["px4"]["rootfs_sha256"])
    required=("vehicle_visual_odometry","estimator_status_flags","estimator_aid_src_ev_pos","estimator_aid_src_ev_vel",
              "estimator_aid_src_baro_hgt","estimator_aid_src_mag","vehicle_local_position","vehicle_status","vehicle_control_mode","actuator_outputs_sim")
    raw_ok=artifact_hashes_match(runtime,source.get("closed_raw_artifact_hashes"))
    log_ok=bool(all(len(topics.get(name,{}).get("timestamp",()))>0 for name in required)
        and log_is_complete(logmeta) and raw_ok and logger_ok and parameter_ok
        and "Too many subscriptions" not in raw.get("px4_console","")
        and logmeta.get("topic_instances_count",999)<=255
        and all(e["instance"]==0 for e in logmeta.get("topic_instances",[])))
    checks={
        "A":"PASS" if attempt_ok and fresh_run_identity(raw,source["protocol_sha256"]) and origin["status"]=="PASS" and origin_commands==0 and gnss["status"]=="PASS" else "BLOCKED",
        "B":"PASS" if counts.get("HIL_GPS")==0 else "FAIL",
        "C":"PASS" if params.get("EKF2_GPS_CTRL")==0 else "BLOCKED",
        "D":fusion["direct_status"],"E":fusion["ev_pos_status"],"F":fusion["ev_vel_status"],
        "G":gnss["status"],"H":gnss["status"],"I":fusion["local_status"],"J":other["status"],
        "K":"PASS" if safety_ok and passive_status(topics) else "FAIL",
        "L":"PASS" if raw.get("flight_command_sent") is False and arm_commands==0 and physical_zero and outputs_zero
                     and passive_status(topics) and assess_disarmed(topics.get("vehicle_status"),end_usec=END_USEC) else "FAIL",
        "M":assess_boundary(raw)["status"],
        "N":"PASS" if identity_ok and attempt_ok and log_ok and producer_ok and raw.get("px4_exit_code")==0 and raw.get("history_unchanged") is True else "BLOCKED"}
    navigation,_=local_comparison(samples,topics.get("vehicle_local_position"))
    flags=topics.get("estimator_status_flags",{});local=topics.get("vehicle_local_position",{})
    final_flags={key:bool(flags[key][-1]) for key in ("cs_tilt_align","cs_yaw_align","cs_ev_pos","cs_ev_vel","cs_ev_yaw",
        "cs_baro_hgt","cs_mag_hdg","cs_mag_3d","cs_in_air","cs_mag_aligned_in_flight") if key in flags and len(flags[key])}
    first_valid={}
    for key in ("xy_valid","z_valid","v_xy_valid","v_z_valid"):
        indices=np.flatnonzero(local.get(key,()))
        first_valid[key]=float((local["timestamp"][indices[0]]-1e6)/1e6) if len(indices) else None
    report={"phase":"Phase3D GNSS-Denied Disarmed EKF2 Fusion Verification","date":"2026-10-13",
        "branch":"feat/px4-mujoco-gnss-denied-disarmed","base_commit":protocol["base_commit"],"overall":aggregate(checks),"status":checks,
        "status_meaning":dict(zip("ABCDEFGHIJKLMN",["cold start without GNSS or global origin","zero HIL_GPS","actual GPS_CTRL0",
          "direct visual odometry receipt","EV position fusion","EV velocity fusion","GNSS position/velocity absent",
          "GNSS height/yaw absent","local position/velocity valid","non-GNSS height/heading sources","safe communication",
          "disarmed zero motor outputs","strict20s boundary","log/runtime identity/exit/history"])),
        "identity":{"protocol_sha256":source["protocol_sha256"],"px4":raw.get("px4"),"mujoco_version":raw.get("mujoco_version"),
            "runtime_hashes":source["runtime_source_sha256"],"final_runtime_hashes":runtime_hashes(),
            "final_matches_executed":identity_ok,"identity_error":identity_error,
            "model_hashes":protocol["model_hashes"],"checkpoint_hashes":protocol["checkpoint_hashes"],
            "history_hashes":protocol["history_hashes"],"summary_sha256":sha256(raw_path),"source_sha256":sha256(source_path),
            "raw_hashes":source["closed_raw_artifact_hashes"],"analysis_sha256":sha256(Path(__file__)),
            "source_definition_files_sha256":protocol["source_definition_files_sha256"]},
        "parameters":{"actual":{k:params.get(k) for k in {**PARAMETERS,**SOURCE_PARAMETERS}},"matched":parameter_ok},
        "logger":{"configuration":protocol["logger"],"metadata":logmeta,"error":log_error,"valid":log_ok},
        "evidence":{"fusion":fusion,"gnss_absence":gnss,"non_gnss_sources":other,"global_origin":origin,
            "local_estimate_vs_com":navigation,"producer_valid":producer_ok,"source_samples":len(samples),"sent_samples":len(sent),
            "final_estimator_flags":final_flags,"first_recorded_local_valid_simulation_s":first_valid,
            "final_heading_good_for_control":bool(local["heading_good_for_control"][-1]) if len(local.get("heading_good_for_control",())) else None,
            "arm_commands":arm_commands,"global_origin_commands":origin_commands,"commands":sorted(set(int(v) for v in commands)),
            "physical_controls_zero":physical_zero,"px4_outputs_zero":outputs_zero,"single_attempt_manifest":attempt,"safety":safety,
            "shutdown_publications_excluded":excluded,"source_interval_usec":stats(np.diff(stamps)),
            "send_wall_interval_seconds":stats(np.diff(wall)),
            "time_mapping":"SimulatorMavlink replaces sample time with PX4 receipt time. Exact logical HIL_SENSOR epoch pairing tested; host/PX4 transport latency NOT TESTED."},
        "boundary":assess_boundary(raw),
        "runtime":{"attempts":1,"workers":1,"armed_runs":0,"simulation_seconds":raw.get("simulation_seconds"),"wall_seconds":raw.get("wall_seconds"),
            "messages_sent":counts,"messages_received":raw.get("messages_received"),"exception":raw.get("exception"),"fault":fault,
            "query_count":len(queries),"query_response_seconds":stats(durations),"query_intervals_seconds":stats(intervals),
            "bridge_peak_rss_bytes":raw.get("bridge_peak_rss_bytes"),"px4_peak_rss_bytes":raw.get("px4_sampled_peak_rss_bytes"),
            "minimum_available_memory_bytes":raw.get("system_min_available_bytes"),"bridge_cpu_seconds":raw.get("bridge_cpu_seconds"),
            "px4_cpu_seconds_sampled":measurements.get("px4_sampled_cpu_seconds"),"px4_exit_code":raw.get("px4_exit_code"),
            "zero_control_on_stop":raw.get("zero_control_on_stop"),"measurements":{k:v for k,v in measurements.items() if k not in ("chunks","queries")}},
        "limitations":["SIMULATED GROUND TRUTH ODOMETRY, synthetic covariance; no FAST-LIO2/SLAM.",
            "Single static disarmed run; no GNSS-denied flight or policy control authorization.",
            "Inflight mag alignment, heading_good_for_control and flight prechecks NOT TESTED.",
            "Actual source-to-PX4 transport latency and moving-state estimator quality NOT TESTED."],
        "next_recommendation":"Review a separately authorized GNSS-denied flight safety and estimator-readiness protocol before any flight; no further execution in this phase."}
    report["figures"]=plots(raw,topics,Path(figure_dir))
    atomic_json(report_path,report);return report


def plots(raw,topics,directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True);made=[]
    def save(name,draw):
        fig,ax=plt.subplots(figsize=(8.5,3.7));draw(ax);ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(directory/name,dpi=150);plt.close(fig);made.append(name)
    def flags(ax,topic,names,title):
        data=topics.get(topic,{})
        for key in names:
            if key in data:ax.step((data["timestamp"]-1e6)/1e6,data[key],where="post",label=key)
        ax.set(xlabel="PX4 logical simulation time (s)",ylabel="Recorded flag",title=title);ax.legend(fontsize=7,ncol=2)
    def ev(ax):
        for key in ("ev_pos","ev_vel"):
            d=topics.get("estimator_aid_src_"+key,{})
            if d:ax.step((d["timestamp"]-1e6)/1e6,(d["time_last_fuse"].astype(float)-1e6)/1e6,where="post",label=key)
        ax.set(xlabel="PX4 publication time (s)",ylabel="Last fusion time (s)");ax.legend()
    save("ev_position_velocity_fusion.png",ev)
    save("gnss_fusion_state.png",lambda ax:flags(ax,"estimator_status_flags",("cs_gnss_pos","cs_gnss_vel","cs_gps_hgt","cs_gnss_yaw"),"GNSS fusion"))
    save("local_position_validity.png",lambda ax:flags(ax,"vehicle_local_position",("xy_valid","z_valid"),"Local position validity"))
    save("local_velocity_validity.png",lambda ax:flags(ax,"vehicle_local_position",("v_xy_valid","v_z_valid"),"Local velocity validity"))
    save("height_heading_sources.png",lambda ax:flags(ax,"estimator_status_flags",("cs_baro_hgt","cs_mag_hdg","cs_mag_3d","cs_yaw_align","cs_mag_aligned_in_flight"),"Ground sources; no flight readiness claim"))
    def queries(ax):
        q=raw.get("runtime_measurements",{}).get("queries",[])
        if q:
            ax.plot([r["start_wall_s"]-q[0]["start_wall_s"] for r in q if r.get("end_wall_s") is not None],
                    [(r["end_wall_s"]-r["start_wall_s"])*1000 for r in q if r.get("end_wall_s") is not None],"o-",ms=3)
        ax.axhline(150,color="r",label="150ms response deadline");ax.set(xlabel="Host wall time from first query (s)",ylabel="Query response (ms)");ax.legend()
    save("safety_query_timing.png",queries)
    def boundary(ax):
        physical=raw.get("final_physical_state",{}).get("time_s",np.nan)
        ax.bar(["sensor clock","physical clock"],[raw.get("simulation_seconds",np.nan),physical]);ax.axhline(20,color="r",label="20s maximum")
        ax.set(ylabel="Untrimmed final simulation time (s)");ax.legend()
    save("strict_stop_boundary.png",boundary)
    return made


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True);parser.add_argument("--figures",type=Path,required=True)
    a=parser.parse_args();result=analyze(a.runtime,a.report,a.figures)
    print(json.dumps({"overall":result["overall"],"status":result["status"],"runtime":{k:v for k,v in result["runtime"].items() if k!="measurements"}},indent=2))


if __name__=="__main__":main()
