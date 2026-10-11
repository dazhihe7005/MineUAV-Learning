"""Phase2 fixed-field smoke / one gated alignment ascent and+10deg step."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .closed_loop import check_envelope
from .magnetic_model import field_from_px4
from .yaw_validation import heading_topics_ready,latest_topics,yaw_ned,angle_error
from .simulation_runner import smoke,source_identity

class YawProtocol:
    def __init__(self,magnetic_field,active=False):
        self.field=magnetic_field
        self.sensor_field=np.array(magnetic_field["field_ned_gauss"],dtype=float)
        if self.sensor_field.shape!=(3,) or not np.isfinite(self.sensor_field).all() or np.linalg.norm(self.sensor_field)<=0:
            raise ValueError("finite nonzero NED magnetic field required")
        self.active=active;self.command_yaw=np.pi/2;self.step_time=None;self.gate_evidence=None
        self._console_tail=None;self._topics={}
        self.field_hash=hashlib.sha256(json.dumps(self.sensor_field.tolist(),separators=(",",":")).encode()).hexdigest()

    def metadata(self):
        return {"magnetic_field":self.field,"field_sha256":self.field_hash,"active":self.active,
                "alignment_height_ned_m":-2.,"yaw_start_deg":90.,"yaw_target_deg":100.,
                "planned_step_time_s":32.,"actual_step_time_s":self.step_time,
                "gate_evidence_at_step":self.gate_evidence,"physical_log_hz":50}

    def validate_proof(self,proof):
        if proof.get("phase2",{}).get("field_sha256")!=self.field_hash:
            raise ValueError("Phase2 flight requires smoke with same corrected magnetic field")

    def before_step(self,data,time_usec,console):
        if not self.active or data.time<32:return
        tail=console[-65536:]
        if tail!=self._console_tail:
            self._topics=latest_topics(tail);self._console_tail=tail
        if not heading_topics_ready(self._topics,time_usec):
            action="no yaw maneuver" if self.step_time is None else "abort active yaw"
            raise ValueError(f"BLOCKED: fresh final EKF heading/alignment gate failed; {action}")
        if self.step_time is None:
            local=self._topics["vehicle_local_position"]
            if (abs(data.qpos[2]-2.051)>.15 or np.linalg.norm(data.qpos[:2])>.2
                or np.linalg.norm(data.qvel[:3])>.15
                or abs(angle_error(local["heading"],yaw_ned(data.qpos[3:])))>np.deg2rad(1)):
                raise ValueError("BLOCKED: position hold or estimate-truth yaw check failed; no yaw maneuver")
            topics=self._topics
            self.gate_evidence={"decision_clock_usec":time_usec,
                "topic_timestamps_usec":{name:topics[name]["timestamp"] for name in ("vehicle_local_position","estimator_status_flags","vehicle_status")},
                "heading_good_for_control":local["heading_good_for_control"],
                "cs_mag_aligned_in_flight":topics["estimator_status_flags"]["cs_mag_aligned_in_flight"],
                "heading_estimated_rad":local["heading"],"heading_true_rad":yaw_ned(data.qpos[3:]),
                "console_sha256":hashlib.sha256(console.encode()).hexdigest()}
            self.command_yaw=np.deg2rad(100);self.step_time=float(data.time)

    def setpoint(self,link,time_usec):link.setpoint(time_usec,height=2.,yaw=self.command_yaw)
    def check_envelope(self,data):check_envelope(data,maximum_height=2.6)
    def queries(self):return "listener vehicle_local_position 1\nlistener estimator_status_flags 1\nlistener vehicle_status 1\n"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--px4-root",type=Path,required=True);p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--mode",choices=("disarmed","yaw"),default="disarmed")
    p.add_argument("--smoke-proof",type=Path);p.add_argument("--pymavlink-search-path")
    args=p.parse_args();source_identity(args.px4_root)
    if (args.mode=="yaw")!=(args.smoke_proof is not None):p.error("yaw requires corrected smoke proof; disarmed forbids it")
    protocol=YawProtocol(field_from_px4(args.px4_root),active=args.mode=="yaw")
    result=smoke(args.px4_root,args.runtime,45. if protocol.active else 16.,
                 args.pymavlink_search_path,args.smoke_proof,phase2=protocol)
    print(json.dumps({k:result.get(k) for k in ("complete","exception","simulation_seconds","phase2","messages_received")}))
    return 0 if result["complete"] and (not protocol.active or protocol.step_time is not None) else 1

if __name__=="__main__":raise SystemExit(main())
