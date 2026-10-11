"""Phase2B opt-in source-consistent telemetry, one smoke then one yaw flight."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .yaw_runner import YawProtocol
from .telemetry_freshness import TelemetryFreshness
from .yaw_validation import angle_error,yaw_ned
from .magnetic_model import field_from_px4
from .simulation_runner import smoke,source_identity


class TelemetryYawProtocol(YawProtocol):
    actuator_watchdog_s=TelemetryFreshness.ACTUATOR_S

    def __init__(self,field,active=False):
        super().__init__(field,active)
        self.monitor=TelemetryFreshness();self.query_offset=0;self.console_length=0
        self.last_decision=None

    def query_requested(self,clock,wall,console):
        self.query_offset=len(console);self.console_length=len(console)
        self.monitor.begin_query(clock,wall)

    def on_console(self,console,clock,wall):
        if len(console)<self.console_length:self.monitor._fail("console stream truncated/reset")
        if len(console)!=self.console_length:
            self.monitor.observe(console[self.query_offset:],clock,wall)
            self.console_length=len(console)

    def on_actuator(self,timestamp,wall):self.monitor.actuator(timestamp,wall)

    def before_step(self,data,time_usec,console,wall=None):
        if wall is None:raise ValueError("explicit monotonic receipt/check time required")
        self.on_console(console,time_usec,wall)
        if self.monitor.last_query is None:return
        decision=self.monitor.decision(time_usec,wall,require_flight=self.active and data.time>=32)
        self.last_decision=decision
        # Baseline safety prerequisites throughout controlled flight, while
        # final in-flight magnetic alignment is legitimately pending on ascent.
        if self.active and data.time>=12 and self.monitor.topics:
            topics=self.monitor.topics
            common={"vehicle_local_position":{"xy_valid":True,"z_valid":True,"v_xy_valid":True,"v_z_valid":True},
                    "estimator_status_flags":{"cs_tilt_align":True,"cs_yaw_align":True,"cs_gnss_pos":True,
                        "cs_mag_fault":False,"cs_baro_fault":False,"cs_gnss_fault":False,"cs_gnss_yaw_fault":False,"cs_gnss_hgt_fault":False},
                    "vehicle_status":{"pre_flight_checks_pass":True,"failsafe":False}}
            if any(topics.get(name,{}).get(k) is not v for name,flags in common.items() for k,v in flags.items()):
                self.monitor._fail("flight operational health lost")
                decision=self.monitor.decision(time_usec,wall,require_flight=data.time>=32)
                self.last_decision=decision
        if decision["safe_stop"]:raise ValueError("BLOCKED: telemetry safety stop: "+decision["reason"])
        if not self.active or data.time<32 or self.step_time is not None:return
        if not decision["allow_new_maneuver"]:return # bounded pending query, never new health
        local=self.monitor.topics["vehicle_local_position"]
        if (abs(data.qpos[2]-2.051)>.15 or np.linalg.norm(data.qpos[:2])>.2
            or np.linalg.norm(data.qvel[:3])>.15
            or abs(angle_error(local["heading"],yaw_ned(data.qpos[3:])))>np.deg2rad(1)):
            raise ValueError("BLOCKED: position hold or estimate-truth yaw check failed; no yaw maneuver")
        self.gate_evidence={"decision_clock_usec":time_usec,"freshness":decision,
            "topic_timestamps_usec":{n:f["timestamp"] for n,f in self.monitor.topics.items()},
            "heading_good_for_control":local["heading_good_for_control"],
            "cs_mag_aligned_in_flight":self.monitor.topics["estimator_status_flags"]["cs_mag_aligned_in_flight"],
            "heading_estimated_rad":local["heading"],"heading_true_rad":yaw_ned(data.qpos[3:]),
            "console_sha256":hashlib.sha256(console.encode()).hexdigest()}
        self.command_yaw=np.deg2rad(100);self.step_time=float(data.time)

    def metadata(self):
        out=super().metadata()
        out["telemetry_fix"]={"source_limits_usec":self.monitor.SOURCE_LIMITS,
            "query_period_s":.5,"response_limit_s":.15,"actuator_wall_watchdog_s":.25,
            "last_decision":self.last_decision,"fault":self.monitor.fault,
            "events":self.monitor.events}
        return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--px4-root",type=Path,required=True);p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--mode",choices=("disarmed","yaw"),default="disarmed")
    p.add_argument("--smoke-proof",type=Path);p.add_argument("--pymavlink-search-path")
    args=p.parse_args();source_identity(args.px4_root)
    if (args.mode=="yaw")!=(args.smoke_proof is not None):p.error("yaw requires matching disarmed proof; disarmed forbids it")
    protocol=TelemetryYawProtocol(field_from_px4(args.px4_root),active=args.mode=="yaw")
    result=smoke(args.px4_root,args.runtime,45. if protocol.active else 16.,
                 args.pymavlink_search_path,args.smoke_proof,phase2=protocol)
    print(json.dumps({k:result.get(k) for k in ("complete","exception","simulation_seconds","messages_received")}))
    return 0 if result["complete"] and (not protocol.active or protocol.step_time is not None) else 1


if __name__=="__main__":raise SystemExit(main())
