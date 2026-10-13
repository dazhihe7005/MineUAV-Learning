"""Single-use <=20s cold-GNSS-denied DISARMED runner; no flight interface."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time

import mujoco

from .. import simulation_runner as sim
from ..actuator_adapter import ActuatorAdapter
from ..allocation_math import effectiveness
from ..diagnostics import atomic_json,sha256
from ..logger_configuration import (LOGGER_TOPICS,CONFIG_RELATIVE,audit_topics,
                                    compiled_topic_names,prepare_private_rootfs,startup_tree_hash)
from ..magnetic_model import field_from_px4
from ..odometry_fusion_runner import SMOKE_DURATION_S
from ..odometry_producer import GroundTruthOdometryProducer
from ..odometry_reverification_runner import protocol_content as retained_protocol
from ..transport import LoopbackTransport,sensor_messages
from .protocol import GnssDeniedProtocol

KIND="gnss_denied_disarmed_verification"
ATTEMPT_PATH=sim.ROOT/"reports/px4_gnss_denied_disarmed_parts/protocol.attempt.json"
PARAMETERS={"EKF2_EV_CTRL":5,"EKF2_GPS_CTRL":0,"EKF2_HGT_REF":0,"SDLOG_PROFILE":0,"SDLOG_MISSION":0}
SOURCE_PARAMETERS={"EKF2_BARO_CTRL":1,"EKF2_MAG_TYPE":0}
EXTRA_TOPICS=("estimator_aid_src_gnss_hgt","estimator_aid_src_gnss_yaw",
              "estimator_aid_src_baro_hgt","estimator_aid_src_mag","vehicle_air_data",
              "vehicle_magnetometer","vehicle_global_position","sensor_gps")
base_environment=sim.px4_environment


def logger_text():
    return "".join(f"{name} {interval} 0\n" for name,interval in LOGGER_TOPICS)+"".join(f"{name} 0 0\n" for name in EXTRA_TOPICS)


def environment(adapter):
    env=base_environment(adapter)
    env.update({"PX4_PARAM_"+key:str(value) for key,value in PARAMETERS.items()})
    return env


def sensor_messages_no_gps(sample,mavlink,producer):
    result=sensor_messages(sample,mavlink,include_gps=False)
    odometry=producer.tick(sample,mavlink)
    if odometry is not None:result.append(odometry)
    return result


class GnssDeniedTransport(LoopbackTransport):
    def __init__(self,*args,sent_samples=None,**kwargs):
        super().__init__(*args,**kwargs);self.sent_samples=sent_samples if sent_samples is not None else []
        self.sent["HIL_GPS"]=0

    def send(self,msg,timeout=None):
        if msg.get_type() not in ("HIL_SENSOR","ODOMETRY"):
            raise ValueError("forbidden outgoing packet in GNSS-denied disarmed mode: "+msg.get_type())
        super().send(msg,timeout=timeout)
        if msg.get_type()=="ODOMETRY":
            self.sent_samples.append({"sample_time_usec":int(msg.time_usec),"send_monotonic_s":time.monotonic(),
                                      "message_fields":msg.to_dict()})


def runtime_hashes():
    from ..odometry_reverification_runner import runtime_hashes as old_hashes
    return {**old_hashes(),**{f"gnss_denied/{p.name}":sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
                           if not p.name.startswith(("test_","analyze"))}}


def history_hashes():
    from ..odometry_reverification_runner import historical_evidence_hashes
    out=historical_evidence_hashes()
    root=sim.ROOT/"reports/px4_odometry_reverification_parts/disarmed_01"
    files=[root/"smoke_summary.json",root/"odometry_source.json",root/"px4_console.bin"]+sorted(root.rglob("*.ulg"))
    if len(files)!=4 or not all(p.is_file() for p in files):raise ValueError("missing historical Phase3C-R evidence")
    out.update({str(p.relative_to(sim.ROOT)):sha256(p) for p in files})
    return out


def protocol_content(px4_root,manifest):
    base=retained_protocol(px4_root,manifest)
    # Recheck the old proof's exact identity without reusing it as new evidence.
    from ..odometry_reverification_runner import verify_protocol as verify_retained
    past=sim.ROOT/"reports/px4_odometry_reverification_parts/protocol.json"
    verify_retained(json.loads(past.read_text()),px4_root,manifest)
    return {**base,"kind":KIND,"base_commit":"8468dfac1137fb7f79ded1718c24eaace09df835",
            "runtime_hashes":runtime_hashes(),"history_hashes":history_hashes(),"parameters":PARAMETERS,
            "required_default_source_parameters":SOURCE_PARAMETERS,
            "logger":audit_topics(logger_text(),compiled_topic_names(base["px4"]["binary"])),
            "expected_messages":{"HIL_SENSOR":5000,"HIL_GPS":0,"ODOMETRY":1000},
            "disarmed_only":True,"cold_start_no_gnss":True,"no_global_origin_commands":True,
            "campaign_attempt_guard":str(ATTEMPT_PATH.resolve()),
            "source_definition_files_sha256":{name:sha256(Path(px4_root)/"src/modules/ekf2"/name)
                for name in ("params_external_vision.yaml","params_gnss.yaml","params_barometer.yaml","params_magnetometer.yaml","module.yaml")}}


def verify_protocol(protocol,px4_root,manifest):
    if protocol!=protocol_content(px4_root,manifest):raise ValueError("frozen GNSS-denied protocol changed")


def reserve_attempt(protocol_path,runtime):
    protocol_path=Path(protocol_path);target=Path(runtime)
    if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
        raise ValueError("new empty runtime required for cold start")
    ATTEMPT_PATH.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(ATTEMPT_PATH,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"w") as stream:
        json.dump({"attempt":1,"authorized_maximum":1,"runtime":str(target.resolve()),
                   "protocol_path":str(protocol_path.resolve()),"protocol_sha256":sha256(protocol_path)},stream)
        stream.flush();os.fsync(stream.fileno())


@contextmanager
def hooks(rootfs,producer,sent,logger_metadata,protocol_hash):
    if threading.active_count()!=1:raise ValueError("single-threaded runner required")
    old=(sim.sensor_messages,sim.px4_environment,sim.prepare_runtime,sim.LoopbackTransport,sim.atomic_json)
    def messages(sample,mv,include_gps=True):return sensor_messages_no_gps(sample,mv,producer)
    def prepare(path):
        runtime=old[2](path);logger_metadata.update(prepare_private_rootfs(runtime,rootfs))
        # Replace only our newly-created private logger file, never the source tree.
        destination=runtime/"etc"/CONFIG_RELATIVE
        temporary=destination.with_suffix(".tmp")
        with temporary.open("x") as stream:stream.write(logger_text());stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,destination)
        logger_metadata["logger_file_sha256"]=sha256(destination)
        if startup_tree_hash(runtime/"etc")!=logger_metadata["startup_tree_sha256"]:
            raise ValueError("private startup files changed")
        return runtime
    class ObservedTransport(GnssDeniedTransport):
        def __init__(self,*args,**kwargs):super().__init__(*args,sent_samples=sent,**kwargs)
    def save(path,value):
        if Path(path).name=="smoke_summary.json":
            if value.get("kind")!="disarmed_communication_smoke" or value.get("flight_command_sent") is not False:
                raise ValueError("cannot save flight evidence in GNSS-denied disarmed mode")
            value.update(kind=KIND,not_flight_proof=True,protocol_sha256=protocol_hash,
                         cold_start_no_gnss=True,gps_hz=0,logger_configuration=dict(logger_metadata))
        atomic_json(path,value)
    try:
        sim.sensor_messages=messages;sim.px4_environment=environment;sim.prepare_runtime=prepare
        sim.LoopbackTransport=ObservedTransport;sim.atomic_json=save
        yield
    finally:
        sim.sensor_messages,sim.px4_environment,sim.prepare_runtime,sim.LoopbackTransport,sim.atomic_json=old


def run(px4_root,runtime,manifest,protocol_path,pymavlink_search_path=None):
    protocol_path=Path(protocol_path);protocol=json.loads(protocol_path.read_text())
    verify_protocol(protocol,px4_root,manifest);sim.require_startup_memory();reserve_attempt(protocol_path,runtime)
    m=mujoco.MjModel.from_xml_path(str(sim.MODEL));a=ActuatorAdapter(m,sim.CONFIG)
    row=effectiveness(a)[2];del a,m
    phase=GnssDeniedProtocol(field_from_px4(px4_root),row)
    producer=GroundTruthOdometryProducer();sent=[];logger={}
    with hooks(protocol["px4"]["rootfs"],producer,sent,logger,sha256(protocol_path)):
        result=sim.smoke(px4_root,runtime,SMOKE_DURATION_S,pymavlink_search_path,
                         phase2=phase,compatibility_manifest=manifest)
    identity_error=None
    try:after=protocol_content(px4_root,manifest)
    except (OSError,ValueError) as error:
        identity_error=str(error);after={"runtime_hashes":runtime_hashes(),"model_hashes":None,"checkpoint_hashes":None,"history_hashes":None,"px4":None}
    from ..odometry_reverification_runner import closed_raw_hashes
    source={"label":"SIMULATED GROUND TRUTH ODOMETRY","protocol":protocol,"protocol_sha256":sha256(protocol_path),
            "summary_sha256":sha256(Path(runtime)/"smoke_summary.json"),"source_samples":producer.samples,"sent_samples":sent,
            "runtime_source_sha256":protocol["runtime_hashes"],"runtime_hashes_after":after["runtime_hashes"],
            "model_hashes_after":after["model_hashes"],"checkpoint_hashes_after":after["checkpoint_hashes"],
            "history_hashes_after":after["history_hashes"],"px4_identity_after":after["px4"],
            "closed_raw_artifact_hashes":closed_raw_hashes(runtime),"post_run_identity_error":identity_error,
            "logger":logger,"cold_start_no_gnss":True}
    atomic_json(Path(runtime)/"odometry_source.json",source)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--px4-root",type=Path,required=True);p.add_argument("--compatibility-manifest",type=Path,required=True)
    p.add_argument("--protocol",type=Path,required=True);p.add_argument("--runtime",type=Path)
    p.add_argument("--freeze-only",action="store_true");p.add_argument("--pymavlink-search-path")
    args=p.parse_args()
    if args.freeze_only:
        if args.protocol.exists():raise ValueError("never overwrite frozen protocol")
        atomic_json(args.protocol,protocol_content(args.px4_root,args.compatibility_manifest))
        print(json.dumps({"protocol_sha256":sha256(args.protocol),"SITL_started":False}));return 0
    if args.runtime is None:p.error("--runtime required")
    result=run(args.px4_root,args.runtime,args.compatibility_manifest,args.protocol,args.pymavlink_search_path)
    print(json.dumps({k:result.get(k) for k in ("complete","exception","simulation_seconds","messages_sent","px4_exit_code")}))
    return 0 if result.get("complete") and result.get("exception") is None else 1


if __name__=="__main__":raise SystemExit(main())
