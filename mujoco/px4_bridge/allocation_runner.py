"""Phase2C explicit isolated build; original safety protocol unchanged."""
import argparse
import json
from pathlib import Path
import re
import mujoco
import numpy as np
from .yaw_telemetry_runner import TelemetryYawProtocol
from .simulation_runner import smoke,source_identity,MODEL,CONFIG
from .actuator_adapter import ActuatorAdapter
from .allocation_math import effectiveness
from .magnetic_model import field_from_px4
from .diagnostics import sha256


def validate_disarmed_evidence(proof,identity,history):
    phase=proof.get("phase2",{});fix=phase.get("telemetry_fix",{});decision=fix.get("last_decision") or {}
    duration=proof.get("simulation_seconds",0);end=proof.get("final_physical_state",{}).get("time_s",0)
    if not (identity and identity.get("custom_sitl") and history and proof.get("px4")==identity
        and proof.get("history_hashes_before")==history and proof.get("history_hashes_after")==history
        and proof.get("history_unchanged") is True and proof.get("complete") is True
        and proof.get("kind")=="disarmed_communication_smoke" and 16-1e-8<=duration<=16.008
        and abs(end-duration)<1e-8 and proof.get("flight_command_sent") is False and phase.get("active") is False
        and proof.get("exception") is None and proof.get("px4_exit_code")==0
        and "fault" in fix and fix["fault"] is None and decision.get("safe_stop") is False
        and decision.get("query_pending") is False
        and set(decision.get("topics",{}))=={"vehicle_local_position","estimator_status_flags","vehicle_status"}
        and proof.get("messages_sent",{}).get("HIL_SENSOR",0)==4000
        and proof.get("messages_sent",{}).get("HIL_GPS",0)==160
        and proof.get("messages_received",{}).get("HIL_ACTUATOR_CONTROLS",0)>=3750):
        raise ValueError("complete matching16s custom disarmed proof required before flight")


def allocation_status(console,expected):
    # PX4 POSIX console appends ANSI color resets to each numeric diagnostic.
    # Strip only recognized SGR color sequences, never missing/invalid values.
    console=re.sub(r"\x1b\[[0-9;]*m","",console)
    lines=re.findall(r"MineUAV yaw axis: ([^\r\n]+)",console)
    if not lines:return False
    try:row=np.array([float(x) for x in lines[-1].split()])
    except ValueError:return False
    return row.shape==(4,) and np.isfinite(row).all() and np.allclose(row,expected,atol=2e-8,rtol=0)


class AllocationYawProtocol(TelemetryYawProtocol):
    def __init__(self,field,expected,active=False,identity=None,history=None):
        super().__init__(field,active);self.expected_yaw=np.array(expected)
        self.identity=identity;self.history=history

    def queries(self):return super().queries()+"control_allocator status\n"

    def before_step(self,data,time_usec,console,wall=None):
        super().before_step(data,time_usec,console,wall)
        if self.active and data.time>=10 and not allocation_status(console,self.expected_yaw):
            raise ValueError("BLOCKED: actual yaw allocation row missing/changed; safe stop")

    def validate_proof(self,proof):
        super().validate_proof(proof)
        if not allocation_status(proof.get("px4_console",""),self.expected_yaw):
            raise ValueError("missing or invalid actual allocator yaw allocation proof; no flight")
        validate_disarmed_evidence(proof,self.identity,self.history)

    def metadata(self):
        return {**super().metadata(),"allocation_fix":{"expected_yaw_row":self.expected_yaw.tolist(),
            "source_patch":"guarded SITL-only independent weak-yaw retention; no physical scaling"}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--px4-root",type=Path,required=True);p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--compatibility-manifest",type=Path,required=True)
    p.add_argument("--mode",choices=("disarmed","yaw"),default="disarmed")
    p.add_argument("--smoke-proof",type=Path);p.add_argument("--pymavlink-search-path")
    args=p.parse_args();identity=source_identity(args.px4_root,args.compatibility_manifest)
    if (args.mode=="yaw")!=(args.smoke_proof is not None):p.error("yaw requires matching disarmed proof; no implicit arming")
    m=mujoco.MjModel.from_xml_path(str(MODEL));a=ActuatorAdapter(m,CONFIG)
    expected=effectiveness(a)[2];del a,m # never keep a second simulator model during the actual run
    history={"model_xml":sha256(MODEL),"rotor_config":sha256(CONFIG),"rotor_actuators":sha256(MODEL.parent/"rotor_actuators.xml")}
    protocol=AllocationYawProtocol(field_from_px4(args.px4_root),expected,active=args.mode=="yaw",identity=identity,history=history)
    result=smoke(args.px4_root,args.runtime,45. if protocol.active else 16.,args.pymavlink_search_path,
        args.smoke_proof,phase2=protocol,compatibility_manifest=args.compatibility_manifest)
    allocation_ok=allocation_status(result.get("px4_console",""),expected)
    print(json.dumps({"complete":result["complete"],"exception":result["exception"],
        "simulation_seconds":result.get("simulation_seconds"),"actual_yaw_axis":allocation_ok,"step_time_s":protocol.step_time}))
    return 0 if result["complete"] and allocation_ok and (not protocol.active or protocol.step_time is not None) else 1


if __name__=="__main__":raise SystemExit(main())
