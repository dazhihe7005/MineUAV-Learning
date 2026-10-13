"""One protocol-bound, <=20s disarmed run with private finite Logger config."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading

import mujoco

from . import simulation_runner as sim
from .actuator_adapter import ActuatorAdapter
from .allocation_math import effectiveness
from .allocation_runner import AllocationYawProtocol
from .diagnostics import atomic_json, sha256
from .logger_configuration import (audit_topics,compiled_topic_names,render_topics,
                                   prepare_private_rootfs)
from .magnetic_model import field_from_px4
from .odometry_fusion_runner import (isolated_odometry_hooks,checkpoint_hashes,SMOKE_DURATION_S)
from .odometry_producer import GroundTruthOdometryProducer

KIND="disarmed_odometry_fusion_reverification"
PARAMETERS={"EKF2_EV_CTRL":5,"EKF2_GPS_CTRL":7,"EKF2_HGT_REF":1,
            "SDLOG_PROFILE":0,"SDLOG_MISSION":0}
EXPECTED_BINARY="2f6bbc24252577513ffb7f078bcb1048e258744f6c482918d3ea29c411e17004"
HISTORICAL_RUNTIME=sim.ROOT/"reports/px4_odometry_fusion_parts/disarmed_01"


def runtime_hashes():
    # Include all operational modules/guards, but not evolving report analysis/tests.
    root=Path(__file__).parent
    return {p.name:sha256(p) for p in sorted(root.glob("*.py"))
            if not p.name.startswith(("test_","analyze_"))}


def historical_evidence_hashes():
    files=[HISTORICAL_RUNTIME/"smoke_summary.json",HISTORICAL_RUNTIME/"odometry_source.json",
           HISTORICAL_RUNTIME/"px4_console.bin"]+sorted(HISTORICAL_RUNTIME.rglob("*.ulg"))
    if len(files)!=4 or not all(p.is_file() for p in files):
        raise ValueError("historical evidence missing; cannot verify preservation")
    return {str(p.relative_to(sim.ROOT)):sha256(p) for p in files}


def model_hashes():
    return {"model":sha256(sim.MODEL),"rotors":sha256(sim.CONFIG),
            "actuators":sha256(sim.ROOT/"models/rotor_actuators.xml")}


def closed_raw_hashes(runtime):
    runtime=Path(runtime);logs=sorted(runtime.rglob("*.ulg"))
    files=logs+[runtime/"px4_console.bin"]
    return {str(p.relative_to(runtime)):sha256(p) for p in files if p.is_file()}


def protocol_content(px4_root,manifest):
    identity=sim.source_identity(px4_root,manifest)
    if identity["binary_sha256"]!=EXPECTED_BINARY:raise ValueError("PX4 binary identity mismatch")
    if mujoco.__version__!="3.14.0":raise ValueError("MuJoCo version mismatch")
    return {"schema":1,"kind":KIND,"base_commit":"f09264edaffb25c4e59e33cb8be4c01326fb2c33",
            "px4":identity,"mujoco_version":mujoco.__version__,
            "runtime_hashes":runtime_hashes(),"checkpoint_hashes":checkpoint_hashes(),
            "history_hashes":historical_evidence_hashes(),
            "model_hashes":model_hashes(),
            "logger":audit_topics(render_topics(),compiled_topic_names(identity["binary"])),
            "parameters":PARAMETERS,"max_simulation_seconds":20.,"comparison_guard_seconds":1e-6,
            "expected_messages":{"HIL_SENSOR":5000,"HIL_GPS":200,"ODOMETRY":1000},
            "safety_deadlines_seconds":{"query_response":.15,"actuator":.25,"query_service":.65},
            "direct_evidence":{"min_samples":900,"max_gap_usec":250000,
                               "max_final_age_usec":250000,"field_absolute_tolerance":2e-6,
                               "require_exact_logical_clock_pairing":True},
            "aid_evidence":{"final_samples":3,"fused":True,"innovation_rejected":False,
                            "strict_fuse_time_advance":True,"max_final_age_usec":1000000},
            "only_one_disarmed_attempt":True,"label":"SIMULATED GROUND TRUTH ODOMETRY"}


def verify_protocol(protocol,px4_root,manifest):
    if protocol!=protocol_content(px4_root,manifest):raise ValueError("frozen protocol identity/configuration changed")


def require_new_runtime(runtime):
    p=Path(runtime)
    if p.is_symlink() or (p.exists() and (not p.is_dir() or any(p.iterdir()))):
        raise ValueError("new empty runtime required; never reuse an attempt")


def is_new_evidence(summary,protocol_hash):
    return (summary.get("kind")==KIND and summary.get("not_flight_proof") is True
            and summary.get("protocol_sha256")==protocol_hash)


@contextmanager
def reverification_hooks(rootfs,logger_metadata,sent,protocol_hash):
    if threading.active_count()!=1:raise RuntimeError("single-threaded run required")
    prepare,environment,transport,saver=(sim.prepare_runtime,sim.px4_environment,sim.LoopbackTransport,sim.atomic_json)

    def private_prepare(path):
        runtime=prepare(path)
        logger_metadata.update(prepare_private_rootfs(runtime,rootfs))
        return runtime

    def finite_environment(adapter):
        env=environment(adapter)
        env.update(PX4_PARAM_SDLOG_PROFILE="0",PX4_PARAM_SDLOG_MISSION="0")
        return env

    class PayloadTransport(transport):
        def send(self,msg,timeout=None):
            super().send(msg,timeout=timeout)
            if msg.get_type()=="ODOMETRY":
                # Base hook appended a witness only after successful transport.send.
                sent[-1]["message_fields"]=msg.to_dict()

    def new_summary(path,value):
        if Path(path).name=="smoke_summary.json":
            if value.get("kind")!="disarmed_communication_smoke" or value.get("flight_command_sent") is not False:
                raise ValueError("reverification cannot save flight evidence")
            value.update(kind=KIND,not_flight_proof=True,protocol_sha256=protocol_hash,
                         logger_configuration=dict(logger_metadata))
        # Bypass the historical summary-kind hook, never post-edit saved evidence.
        atomic_json(path,value)

    try:
        sim.prepare_runtime=private_prepare;sim.px4_environment=finite_environment
        sim.LoopbackTransport=PayloadTransport;sim.atomic_json=new_summary
        yield
    finally:
        sim.prepare_runtime=prepare;sim.px4_environment=environment
        sim.LoopbackTransport=transport;sim.atomic_json=saver


def run(px4_root,runtime,manifest,protocol_path,pymavlink_search_path=None):
    protocol_path=Path(protocol_path);protocol=json.loads(protocol_path.read_text())
    verify_protocol(protocol,px4_root,manifest);require_new_runtime(runtime)
    sim.require_startup_memory()
    protocol_hash=sha256(protocol_path)
    # Durable exclusive reservation applies across runtime paths. Never removed.
    attempt_path=protocol_path.with_suffix(".attempt.json")
    fd=os.open(attempt_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"w") as stream:
        json.dump({"protocol_sha256":protocol_hash,"runtime":str(Path(runtime).resolve()),
                   "attempt":1,"authorized_maximum":1},stream);stream.flush();os.fsync(stream.fileno())
    model=mujoco.MjModel.from_xml_path(str(sim.MODEL));adapter=ActuatorAdapter(model,sim.CONFIG)
    yaw_row=effectiveness(adapter)[2];del adapter,model
    phase=AllocationYawProtocol(field_from_px4(px4_root),yaw_row,active=False)
    producer=GroundTruthOdometryProducer();sent=[];logger_metadata={}
    with isolated_odometry_hooks(producer,sent):
        with reverification_hooks(protocol["px4"]["rootfs"],logger_metadata,sent,protocol_hash):
            result=sim.smoke(px4_root,runtime,SMOKE_DURATION_S,pymavlink_search_path,
                             phase2=phase,compatibility_manifest=manifest)
    identity_error=None
    try:after=protocol_content(px4_root,manifest)
    except (OSError,ValueError) as error:
        # Preserve nonrepeatable source/send witnesses even on identity failure.
        identity_error=f"{type(error).__name__}: {error}"
        after={"runtime_hashes":runtime_hashes(),"model_hashes":None,
               "checkpoint_hashes":None,"history_hashes":None,"px4":None}
    source={"label":protocol["label"],"source_samples":producer.samples,"sent_samples":sent,
            "summary_sha256":sha256(Path(runtime)/"smoke_summary.json"),
            "protocol_sha256":protocol_hash,"protocol":protocol,
            "runtime_source_sha256":protocol["runtime_hashes"],"runtime_hashes_after":after["runtime_hashes"],
            "runtime_unchanged":protocol["runtime_hashes"]==after["runtime_hashes"],
            "model_hashes_after":after["model_hashes"],"closed_raw_artifact_hashes":closed_raw_hashes(runtime),
            "checkpoint_hashes_before":protocol["checkpoint_hashes"],"checkpoint_hashes_after":after["checkpoint_hashes"],
            "checkpoints_unchanged":protocol["checkpoint_hashes"]==after["checkpoint_hashes"],
            "historical_evidence_unchanged":protocol["history_hashes"]==after["history_hashes"],
            "px4_identity_after":after["px4"],"post_run_identity_error":identity_error,
            "logger":logger_metadata}
    atomic_json(Path(runtime)/"odometry_source.json",source)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-root",type=Path,required=True)
    parser.add_argument("--compatibility-manifest",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--runtime",type=Path)
    parser.add_argument("--freeze-only",action="store_true")
    parser.add_argument("--pymavlink-search-path")
    args=parser.parse_args()
    if args.freeze_only:
        if args.protocol.exists():raise ValueError("do not overwrite frozen protocol")
        atomic_json(args.protocol,protocol_content(args.px4_root,args.compatibility_manifest))
        print(json.dumps({"protocol_sha256":sha256(args.protocol),"SITL_started":False}));return 0
    if args.runtime is None:parser.error("--runtime required for sole run")
    result=run(args.px4_root,args.runtime,args.compatibility_manifest,args.protocol,args.pymavlink_search_path)
    print(json.dumps({k:result.get(k) for k in ("complete","exception","simulation_seconds","messages_sent","px4_exit_code")}))
    return 0 if result.get("complete") and result.get("exception") is None else 1


if __name__=="__main__":raise SystemExit(main())
