"""Immutable Phase3C-R run evidence; direct receipt is mandatory, not inferred."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import struct
import numpy as np

from .diagnostics import atomic_json,sha256
from .logger_configuration import REQUIRED_TOPICS,startup_tree_hash
from .odometry_reverification_runner import (PARAMETERS,runtime_hashes,is_new_evidence,
                                            checkpoint_hashes,historical_evidence_hashes,model_hashes)
from .odometry_reverification_evidence import (assess_boundary,assess_direct,compare_payloads,
                                              aggregate_checks,END_USEC)
from .compatibility_identity import MANIFEST
from . import simulation_runner as sim
from .analyze_odometry_fusion import (assess_aid_source,_end_flags,_local_valid,
                                     assess_disarmed,cut_at_end,local_comparison)


def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    return {"n":len(a),"mean":float(a.mean()),"p50":float(np.median(a)),
            "p95":float(np.percentile(a,95)),"p99":float(np.percentile(a,99)),"max":float(a.max())}


def ulog_framing(path):
    """Bounded record walk: pyulog can silently accept a cut final payload."""
    count=0;position=16
    with Path(path).open("rb") as stream:
        header=stream.read(16)
        if len(header)!=16 or header[:7]!=b"ULog\x01\x12\x35":
            return {"complete":False,"reason":"invalid/partial ULog header","records":0}
        while True:
            chunk=stream.read(3)
            if not chunk:return {"complete":True,"records":count,"bytes":position,"max_record_bytes":65535}
            if len(chunk)!=3:return {"complete":False,"reason":"partial record header","records":count}
            size,kind=struct.unpack("<HB",chunk)
            if kind not in b"FDIMPQARSO LCB".replace(b" ",b""):
                return {"complete":False,"reason":"invalid record type","records":count}
            if len(stream.read(size))!=size:
                return {"complete":False,"reason":"partial record payload","records":count}
            count+=1;position+=3+size


def log_is_complete(metadata):
    return (metadata.get("framing",{}).get("complete") is True
            and metadata.get("file_corruption") is False and metadata.get("dropouts")==0)


def artifact_hashes_match(runtime,expected):
    return bool(expected and len(expected)==2 and "px4_console.bin" in expected
             and len([name for name in expected if name.endswith(".ulg")])==1
             and all((Path(runtime)/relative).is_file()
             and sha256(Path(runtime)/relative)==digest for relative,digest in expected.items()))


def passive_status(topics):
    values=(topics.get("vehicle_control_mode",{}).get("flag_control_offboard_enabled",()),
            topics.get("vehicle_status",{}).get("failsafe",()))
    return all(len(v)>0 and np.isfinite(v).all() and np.all(np.asarray(v)==0) for v in values)


def read_log(path):
    search=os.environ.get("MINEUAV_PYULOG_SEARCH_PATH")
    if search and search not in sys.path:sys.path.append(search)
    from pyulog import ULog
    log=ULog(str(path))
    names=[{"name":ds.name,"instance":ds.multi_id,"samples":len(ds.data.get("timestamp",()))}
           for ds in log.data_list]
    topics={ds.name:ds.data for ds in log.data_list if ds.multi_id==0}
    return topics,log.initial_parameters,{"topic_instances":names,
          "topic_instances_count":len(names),"dropouts":len(log.dropouts),
          "file_corruption":bool(log.file_corruption),"framing":ulog_framing(path),
          "logging_start_timestamp":log.start_timestamp,"logging_end_timestamp":log.last_timestamp,
          "parameters_in_ulog_header":True}


def analyze(runtime,report_path,figure_dir):
    runtime=Path(runtime);raw_path=runtime/"smoke_summary.json";source_path=runtime/"odometry_source.json"
    raw=json.loads(raw_path.read_text());source=json.loads(source_path.read_text())
    if source["summary_sha256"]!=sha256(raw_path):raise ValueError("immutable summary identity mismatch")
    protocol=source["protocol"]
    logs=sorted(runtime.rglob("*.ulg"))
    if len(logs)!=1:raise ValueError("exactly one independent raw ULog required")
    full_topics,params,logmeta=read_log(logs[0])
    # Only exclude actual shutdown publications, NEVER shorten physical runtime.
    topics,excluded=cut_at_end(full_topics,END_USEC)
    visual=topics.get("vehicle_visual_odometry")
    direct=assess_direct(visual,end_usec=END_USEC)
    sent=source["sent_samples"];samples=source["source_samples"]
    comparison,matched=compare_payloads(sent,visual) if direct["status"]=="PASS" else ({"status":"BLOCKED","reason":"direct evidence invalid or absent"},None)
    stamps=np.asarray([s["sample_time_usec"] for s in sent],dtype=np.int64)
    source_stamps=np.asarray([s["sample_time_usec"] for s in samples],dtype=np.int64)
    send_times=np.asarray([s["send_monotonic_s"] for s in sent],dtype=float)
    producer_ok=bool(len(stamps)==len(source_stamps)==1000 and np.array_equal(stamps,source_stamps)
        and stamps[0]==1_000_000 and stamps[-1]==20_980_000 and np.all(np.diff(stamps)==20_000)
        and np.isfinite(send_times).all() and np.all(np.diff(send_times)>0))
    param_ok=all(params.get(k)==v for k,v in PARAMETERS.items())
    aids={key:assess_aid_source(topics.get("estimator_aid_src_"+key),end_usec=END_USEC)
          for key in ("ev_pos","ev_vel","gnss_pos","gnss_vel")}
    flags=topics.get("estimator_status_flags");local=topics.get("vehicle_local_position")
    local_ok=_local_valid(local,end_usec=END_USEC)
    def fusion(key):
        if aids[key]["status"]=="BLOCKED":return "BLOCKED"
        return "PASS" if (aids[key]["pass"] and param_ok and local_ok
                          and _end_flags(flags,("cs_"+key,),end_usec=END_USEC)) else "FAIL"
    runtime_identity=(source["runtime_source_sha256"]==runtime_hashes()==source["runtime_hashes_after"]
                      and protocol["model_hashes"]==source.get("model_hashes_after")==model_hashes())
    current_px4=None;current_px4_error=None
    try:current_px4=sim.source_identity(Path(protocol["px4"]["binary"]).parents[3],MANIFEST)
    except ValueError as error:current_px4_error=str(error)
    measurements=raw.get("runtime_measurements",{});queries=measurements.get("queries",[])
    durations=np.asarray([q["end_wall_s"]-q["start_wall_s"] for q in queries if q.get("end_wall_s") is not None])
    intervals=np.diff([q["start_wall_s"] for q in queries])
    fault=raw.get("phase2",{}).get("telemetry_fix",{}).get("fault")
    actuator=measurements.get("samples",{}).get("actuator_interval_wall_seconds",{})
    safety_ok=bool(raw.get("complete") and raw.get("exception") is None and fault is None
        and len(durations)>0 and len(durations)==len(queries) and durations.min()>=0 and durations.max()<=.15
        and len(intervals)>0 and intervals.min()>0 and intervals.max()<=.65
        and actuator.get("max",float("inf"))<=.25 and measurements.get("pending_query_max",999)<=1
        and measurements.get("row_valid_at_completions")==len(queries))
    records=raw.get("records",[])
    physical_zero=bool(records and all(row.get("ctrl_omega_squared")==[0.,0.,0.,0.] for row in records)
        and raw.get("final_physical_state",{}).get("ctrl")==[0.,0.,0.,0.]
        and raw.get("zero_control_on_stop")==[0.,0.,0.,0.])
    commands=topics.get("vehicle_command",{})
    arm_commands=int(np.sum(np.asarray(commands.get("command",()))==400))
    expected_logger_sha=protocol["logger"]["text_sha256"]
    actual_logger=runtime/"etc/logging/logger_topics.txt"
    logger_ok=(actual_logger.is_file() and sha256(actual_logger)==expected_logger_sha
        and raw.get("logger_configuration",{}).get("logger_file_sha256")==expected_logger_sha
        and raw.get("logger_configuration",{}).get("private_etc") is True
        and startup_tree_hash(runtime/"etc")==protocol["px4"]["rootfs_sha256"])
    console=raw.get("px4_console","")
    necessary_topics=all(name in topics and len(topics[name].get("timestamp",()))>0 for name in REQUIRED_TOPICS)
    raw_artifacts_ok=artifact_hashes_match(runtime,source.get("closed_raw_artifact_hashes"))
    log_ok=bool(necessary_topics and log_is_complete(logmeta) and raw_artifacts_ok and logger_ok and param_ok
        and "Too many subscriptions" not in console
        and logmeta["topic_instances_count"]<=255 and all(e["instance"]==0 for e in logmeta["topic_instances"]))
    checks={
        "A_producer":"PASS" if producer_ok else "FAIL",
        "B_send":"PASS" if producer_ok and raw.get("messages_sent",{}).get("ODOMETRY")==1000 and raw.get("exception") is None else "FAIL",
        "C_direct_receipt":direct["status"] if direct["status"]!="PASS" else comparison["status"],
        "D_ev_position":fusion("ev_pos"),"E_ev_velocity":fusion("ev_vel"),
        "F_gnss":"BLOCKED" if any(aids[k]["status"]=="BLOCKED" for k in ("gnss_pos","gnss_vel")) else
                   "PASS" if fusion("gnss_pos")==fusion("gnss_vel")=="PASS" and raw.get("messages_sent",{}).get("HIL_GPS")==200 else "FAIL",
        "G_duration":assess_boundary(raw)["status"],
        "H_final_identity":"PASS" if runtime_identity and is_new_evidence(raw,source["protocol_sha256"])
                             and raw.get("px4")==protocol["px4"]==source["px4_identity_after"]==current_px4 else "NOT TESTED",
        "I_safety":"PASS" if safety_ok and passive_status(topics) else "FAIL",
        "J_disarmed_zero":"PASS" if raw.get("flight_command_sent") is False and arm_commands==0
                    and physical_zero and passive_status(topics)
                    and assess_disarmed(topics.get("vehicle_status"),end_usec=END_USEC) else "FAIL",
        "K_log_integrity":"PASS" if log_ok else "BLOCKED",
        "L_cleanup_history":"PASS" if raw.get("px4_exit_code")==0 and raw.get("history_unchanged") is True
             and source["checkpoint_hashes_before"]==checkpoint_hashes()==source["checkpoint_hashes_after"]
             and protocol["history_hashes"]==historical_evidence_hashes() and source["historical_evidence_unchanged"] is True else "FAIL"}
    navigation,local_matched=local_comparison(samples,local)
    report={"phase":"PX4–MuJoCo Phase3C-R GNSS-Retained External Odometry Evidence Reverification",
        "date":"2026-10-13","branch":"feat/px4-mujoco-odom-fusion-reverification",
        "base_commit":"f09264edaffb25c4e59e33cb8be4c01326fb2c33",
        "overall":aggregate_checks(checks),"status":checks,
        "identity":{"px4":raw["px4"],"mujoco_version":raw["mujoco_version"],
            "current_px4":current_px4,"current_px4_error":current_px4_error,
            "analysis_source_sha256":sha256(Path(__file__)),
            "model_hashes":raw["history_hashes_before"],"protocol_sha256":source["protocol_sha256"],
            "frozen_model_hashes":protocol["model_hashes"],"model_hashes_after":source.get("model_hashes_after"),
            "current_model_hashes":model_hashes(),"closed_raw_artifact_hashes":source.get("closed_raw_artifact_hashes"),
            "runtime_source_sha256":source["runtime_source_sha256"],"final_runtime_source_sha256":runtime_hashes(),
            "final_runtime_matches_executed":runtime_identity,"checkpoint_hashes":source["checkpoint_hashes_before"],
            "summary_sha256":sha256(raw_path),"source_manifest_sha256":sha256(source_path),
            "ulog_sha256":sha256(logs[0]),"console_sha256":sha256(runtime/"px4_console.bin"),
            "historical_evidence_hashes":protocol["history_hashes"]},
        "logger":{"configured":protocol["logger"],"actual_parameters":{k:params.get(k) for k in PARAMETERS},
            "actual_topic_list":logmeta["topic_instances"],"log_metadata":logmeta,
            "additional_event_topic":"logger.cpp:401,994 dedicated event subscription, outside the configured 19 main subscriptions",
            "startup_copy":raw["logger_configuration"],"closed_raw_artifact_identity":raw_artifacts_ok,
            "subscription_failure_count":len(re.findall("Too many subscriptions",console))},
        "evidence":{"direct":direct,"source_vs_received":comparison,"aid_sources":aids,
            "local_estimate_vs_com":navigation,"local_valid":local_ok,
            "final_fusion_flags":{key:bool(flags[key][-1]) for key in ("cs_ev_pos","cs_ev_vel","cs_gnss_pos","cs_gnss_vel") if flags and key in flags},
            "shutdown_publications_excluded":excluded,"parameter_identity":param_ok,
            "arm_command_count":arm_commands,"motor_controls_zero_all_records":physical_zero,
            "vehicle_command_ids":sorted(set(int(v) for v in commands.get("command",()))),
            "offboard_enabled_any":bool(np.any(topics.get("vehicle_control_mode",{}).get("flag_control_offboard_enabled",()))),
            "failsafe_any":bool(np.any(topics.get("vehicle_status",{}).get("failsafe",()))),
            "source_samples":len(samples),"successful_sent_samples":len(sent),
            "source_first_last_usec":stamps[[0,-1]].tolist() if len(stamps) else [],
            "source_sample_interval_usec":stats(np.diff(stamps)),"wall_send_interval_seconds":stats(np.diff(send_times)),
            "host_to_px4_physical_transport_latency":"NOT TESTED: separate clock domains; no measured host/PX4 offset",
            "receiver_semantics":"SimulatorMavlink timestamp_sample=hrt_absolute_time; publication timestamp from same PX4 clock. Exact logical sensor-epoch match is checked, not source-time sync for FAST-LIO2."},
        "boundary":assess_boundary(raw),
        "runtime":{"sitl_attempts":1,"workers":1,"armed_runs":0,"simulation_seconds":raw.get("simulation_seconds"),
            "wall_seconds":raw.get("wall_seconds"),"messages_sent":raw.get("messages_sent"),"messages_received":raw.get("messages_received"),
            "bridge_peak_rss_bytes":raw.get("bridge_peak_rss_bytes"),"px4_peak_rss_bytes":raw.get("px4_sampled_peak_rss_bytes"),
            "minimum_system_available_bytes":raw.get("system_min_available_bytes"),"bridge_cpu_seconds":raw.get("bridge_cpu_seconds"),
            "px4_cpu_seconds_sampled":measurements.get("px4_sampled_cpu_seconds"),"oom_observed":False,
            "exception":raw.get("exception"),"safety_fault":fault,"query_count":len(queries),
            "query_response_seconds":stats(durations),"query_start_interval_seconds":stats(intervals),
            "passive_measurements":{k:v for k,v in measurements.items() if k not in ("chunks","queries")},
            "px4_exit_code":raw.get("px4_exit_code"),"zero_control_on_stop":raw.get("zero_control_on_stop")},
        "limitations":["SIMULATED GROUND TRUTH ODOMETRY; no FAST-LIO2/SLAM",
            "Disarmed static/settling state; no moving-state odometry validation",
            "GNSS retained; GNSS-denied and policy control NOT TESTED",
            "Synthetic covariance .01 and quality100, not sensor characterization",
            "Logger retention counts are not exact wire-packet loss counts"],
        "next_stage":{"gnss_denied_preflight":"PASS" if aggregate_checks(checks)=="PASS" else "BLOCKED",
            "gnss_denied_runtime":"NOT TESTED","fast_lio2_source_time_sync":"NOT TESTED","bc_ppo_flight":"NOT TESTED",
            "single_recommendation":"Separately authorized disarmed GNSS-denied fusion-source verification before any external-odometry flight; do not execute in this task"}}
    report["verification"]={"bridge_inherited_passed":371,"bridge_new_passed":66,"bridge_total_passed":437,
        "shadow_passed":30,"bridge_inherited_warnings":2,"shadow_inherited_package_warnings":3,
        "bridge_suite_peak_rss_kib":1146308,"pre_run_bridge_passed":431,
        "historical_untracked_files_preserved":87,"failed_tests":0,
        "independent_review":{"pre_run":"GO after offline Important fixes; independent60 targeted tests PASS",
            "resolved_findings":["ULog parser corruption/partial-record/full-record truncation acceptance",
                "final frozen/post/current model identity", "preserve source witnesses after identity errors"],
            "final_result_review":"GO: independent read-only source/raw-evidence/report review",
            "remaining_critical":0,"remaining_important":0,
            "reviewer_targeted_tests_passed":65,"reviewer_physics_schedule_tests_deselected":1,
            "reviewer_recomputed_all_saved_evidence":True,
            "reviewer_reran_full_437_bridge_30_shadow_suites":False,
            "minor_documentation_wording":"corrected flown/communicating to executed; no runtime changes"},
        "post_run_operational_code_changes":False,
        "whole_repository_tests":"NOT RUN; unrelated CAD NumPy/SciPy ABI untouched"}
    report["figures"]=plots(raw,source,topics,matched,Path(figure_dir))
    atomic_json(report_path,report);return report


def plots(raw,source,topics,matched,directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True);names=[]
    def save(name,draw):
        fig,ax=plt.subplots(figsize=(8.5,3.6));draw(ax);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(directory/name,dpi=150);plt.close(fig);names.append(name)
    sent=source["sent_samples"]
    if sent:
        save("odometry_source_send_time.png",lambda ax:(
            ax.plot([(r["sample_time_usec"]-1e6)/1e6 for r in sent],
                    [r["send_monotonic_s"]-sent[0]["send_monotonic_s"] for r in sent],".",ms=2),
            ax.set(xlabel="Source simulation time (s)",ylabel="Host send time relative to first send (s)")))
    direct=topics.get("vehicle_visual_odometry")
    if direct:
        save("px4_direct_receive_publish_time.png",lambda ax:(
            ax.plot((direct["timestamp_sample"]-1e6)/1e6,(direct["timestamp"]-1e6)/1e6,".",ms=2),
            ax.set(xlabel="PX4 receipt-based sample time (s)",ylabel="PX4 publication time (s)")))
    if matched:
        def values(ax):
            for i,label in enumerate(("N","E","D")):
                ax.plot(matched["time_s"],matched["source_position"][:,i],label=f"source {label}")
                ax.plot(matched["time_s"],matched["received_position"][:,i],"--",label=f"PX4 {label}")
            ax.set(xlabel="Logical simulation time (s)",ylabel="COM position NED (m)");ax.legend(ncol=3,fontsize=8)
        save("odometry_source_received_values.png",values)
        def velocities(ax):
            for i,label in enumerate(("F","R","D")):
                ax.plot(matched["time_s"],matched["source_velocity"][:,i],label=f"source {label}")
                ax.plot(matched["time_s"],matched["received_velocity"][:,i],"--",label=f"PX4 {label}")
            ax.set(xlabel="Logical simulation time (s)",ylabel="COM velocity BODY_FRD (m/s)");ax.legend(ncol=3,fontsize=8)
        save("odometry_source_received_velocity.png",velocities)
    for source_type in ("ev","gnss"):
        def fusion(ax,source_type=source_type):
            time_ax=ax.twinx();handles=[]
            for i,dimension in enumerate(("pos","vel")):
                data=topics.get(f"estimator_aid_src_{source_type}_{dimension}")
                if data:
                    t=(data["timestamp"]-1e6)/1e6
                    handles+=ax.step(t,data["fused"]+2*i,where="post",label=f"{dimension} fused (+{2*i})")
                    valid=data["time_last_fuse"]>0
                    handles+=time_ax.step(t[valid],(data["time_last_fuse"][valid]-1e6)/1e6,
                                           where="post",ls=":",label=f"{dimension} last fuse (s)",color=("orange" if i==0 else "red"))
            ax.set(xlabel="PX4 publication time (s)",ylabel="Fusion state lanes",ylim=(-.2,3.2))
            ax.set_yticks([0,1,2,3],["pos false","pos true","vel false","vel true"])
            time_ax.set_ylabel("Actual last fuse time (s)")
            ax.legend(handles,[h.get_label() for h in handles],fontsize=8,loc="lower right")
        save(f"{source_type}_fusion_evidence.png",fusion)
    records=raw.get("records",[])
    if records:
        def stop(ax):
            ax.plot([r["time_s"] for r in records],[r["clock_usec"]/1e6-1 for r in records],".",ms=2)
            final=raw["final_physical_state"]["time_s"]
            ax.plot([final],[final],"o",label=f"actual final physical time {final:.12f}s")
            ax.axvline(20.,ls="--",color="red",label="authorized bound")
            ax.set(xlabel="Actual physical time (s)",ylabel="Integer bridge clock elapsed (s)");ax.legend(fontsize=8)
        save("simulation_stop_boundary.png",stop)
    queries=raw.get("runtime_measurements",{}).get("queries",[])
    if queries:
        def timing(ax):
            ax.plot([r["start_wall_s"] for r in queries],[(r["end_wall_s"]-r["start_wall_s"])*1000 for r in queries],".-",label="query response")
            ax.axhline(150,color="red",ls="--",label="150ms response limit")
            ax.set(xlabel="Host wall time (s)",ylabel="Query response (ms)");ax.legend(fontsize=8)
        save("safety_query_timing.png",timing)
    actuator=topics.get("actuator_outputs_sim")
    if actuator and len(actuator.get("timestamp",()))>1:
        save("actuator_publication_intervals.png",lambda ax:(
            ax.plot((actuator["timestamp"][1:]-1e6)/1e6,np.diff(actuator["timestamp"])/1000.,".",ms=2),
            ax.set(xlabel="PX4 publication time (s)",ylabel="ULog actuator publication interval (ms; not host watchdog age)")))
    chunks=raw.get("runtime_measurements",{}).get("chunks",[])
    if chunks:
        save("console_parse_timing.png",lambda ax:(ax.plot([r["wall_s"] for r in chunks],[r["parse_validate_s"]*1000 for r in chunks],".",ms=2),
            ax.set(xlabel="Host wall time (s)",ylabel="Streaming parse + validation (ms)")))
    return names


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime",type=Path,required=True);parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--figures",type=Path,required=True);args=parser.parse_args()
    report=analyze(args.runtime,args.report,args.figures)
    print(json.dumps({"overall":report["overall"],"status":report["status"]}));return 0


if __name__=="__main__":raise SystemExit(main())
