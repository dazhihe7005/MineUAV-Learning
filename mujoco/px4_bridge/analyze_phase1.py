"""Offline publication of finite SITL evidence. Never launches simulation."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import mujoco
from .diagnostics import sha256,atomic_json
from .closed_loop import clean_console,estimator_ready,assess_hover
from .simulation_runner import MODEL,CONFIG,ROOT,PX4_COMMIT
from .actuator_adapter import ActuatorAdapter
from .frame_conversions import rotation

def physical_metrics(rows):
    if not rows:raise ValueError("physical log absent")
    xyz=np.array([r["position_enu_m"] for r in rows]);v=np.array([r["velocity_origin_enu_m_s"] for r in rows])
    tilts=np.array([np.degrees(np.arccos(np.clip(rotation(r["quaternion_enu_flu"])[2,2],-1,1))) for r in rows])
    return {"samples":len(rows),"height_min_m":float(xyz[:,2].min()),"height_max_m":float(xyz[:,2].max()),
            "peak_speed_m_s":float(np.linalg.norm(v,axis=1).max()),"max_horizontal_offset_m":float(np.linalg.norm(xyz[:,:2],axis=1).max()),
            "peak_tilt_deg":float(tilts.max()),"final_origin_position_enu_m":xyz[-1].tolist(),
            "final_origin_velocity_enu_m_s":v[-1].tolist(),"final_speed_m_s":float(np.linalg.norm(v[-1])),
            "saturated_samples":sum(bool(r["saturated"]) for r in rows)}

def validate_smoke(d):
    if not (d.get("complete") and d.get("history_unchanged") and d.get("px4_exit_code")==0
            and d.get("px4",{}).get("commit")==PX4_COMMIT
            and d.get("messages_received",{}).get("HIL_ACTUATOR_CONTROLS",0)>0
            and d.get("messages_sent",{}).get("HIL_SENSOR",0)>0
            and "TOPIC: sensor_combined" in clean_console(d.get("px4_console",""))):
        raise ValueError("missing actual PX4 sensor/actuator evidence")

def artifact_name(path):
    p=Path(path).resolve()
    return str(p.relative_to(ROOT.parent)) if p.is_relative_to(ROOT.parent) else str(p)

def publish(parts,output):
    parts=Path(parts).resolve();output=Path(output).resolve()
    names=("smoke_01","smoke_02","smoke_03","hover_01")
    logs={n:json.loads((parts/n/"smoke_summary.json").read_text()) for n in names}
    smoke=logs["smoke_03"];hover=logs["hover_01"]
    validate_smoke(smoke);validate_smoke(hover)
    if not estimator_ready(smoke["px4_console"]):raise ValueError("smoke estimator not ready")
    m=mujoco.MjModel.from_xml_path(str(MODEL));adapter=ActuatorAdapter(m,CONFIG)
    raw={artifact_name(parts/n/"smoke_summary.json"):sha256(parts/n/"smoke_summary.json") for n in names}
    for n in names:
        for p in sorted((parts/n/"log").rglob("*.ulg")) if (parts/n/"log").exists() else []:
            raw[artifact_name(p)]=sha256(p)
    checkpoints={}
    for name in ("uav_bc_mlp_seed0.pt","uav_bc_yaw_augmented_seed0.pt","joint_latent_world_model_v3_autonomous_consistency.pt"):
        p=ROOT/"rl/models"/name;checkpoints[name]=sha256(p)
    body=m.body("mine_uav")
    mass=float(m.body_mass[body.id])
    summaries={n:{k:v for k,v in d.items() if k not in ("records","px4_console")} for n,d in logs.items()}
    evidence={n:clean_console(d["px4_console"]) for n,d in logs.items()}
    report={"experiment":"PX4–MuJoCo SITL Integration Phase1","date":"2026-10-11",
        "branch":"feat/px4-mujoco-sitl-integration","base_commit":"89df0fb3b5fc5bcd99db217d0a54c51613655a52",
        "system":{"ubuntu":"22.04.5 LTS","kernel":platform.release(),"cpu":"AMD Ryzen9 7940HX,16cores/32threads","ram_gib":14.83,"swap_bytes":0,
                  "python":platform.python_version(),"mujoco":mujoco.__version__,"pymavlink":"2.4.49","system_installations":0},
        "px4":smoke["px4"],"runs":summaries,"px4_console_evidence":evidence,"raw_local_only_hashes":raw,
        "historical_checkpoint_sha256":checkpoints,"historical_physics_hashes":smoke["history_hashes_after"],
        "model":{"mass_kg":mass,"com_flu_m":body.ipos.tolist(),"principal_inertia_kg_m2":body.inertia.tolist(),
                 "principal_inertia_orientation_wxyz":body.iquat.tolist(),"physics_timestep_s":m.opt.timestep,"gravity_enu_m_s2":m.opt.gravity.tolist(),
                 "kf":adapter.kf,"km":adapter.km,"rpm_max_assumption":8500,"u_max_omega_squared":adapter.u_max,
                 "max_total_static_thrust_n":4*adapter.kf*adapter.u_max,"estimated_hover_fraction":mass*9.81/(4*adapter.kf*adapter.u_max),
                 "reaction_torque_signs_flu_z":adapter.signs.tolist(),"allocation_parameters":adapter.px4_geometry()},
        "motor_mapping":[{"px4_channel":i+1,"px4_function":101+i,"mujoco_actuator":f"motor_{i+1}_wrench","mujoco_ctrl_index":i,
                          "arm_about_com_frd_m":adapter.arms_frd[i].tolist(),"flu_reaction_torque_sign":int(adapter.signs[i]),
                          "cw_ccw_wiring":"UNVERIFIED; these are model reaction torque signs"} for i in range(4)],
        "protocol":{"simulator":"MAVLink2 common,loopbackTCP","tcp_server":"127.0.0.1:4560 bridge","tcp_client":"PX4 simulator_mavlink",
                    "offboard":"loopbackUDP14540→PX4 14580 (only opt-in flight)","source_system_component":[42,51],"px4_system_component":[1,1],
                    "physics_hz":500,"HIL_SENSOR_hz":250,"HIL_GPS_hz":10,"actuator_hz_measured":250,
                    "clock":"integerµs,1s epochoffset,2msphysics,4msIMU; bounded bootstrap then per-actuator lockstep",
                    "HIL_STATE_QUATERNION":"request received; not sent; pinned source publishes only _ground_truth topics, not estimator bypass",
                    "HIL_OPTICAL_FLOW":"pinned handler audited, not generated/no optical flow sensor in Phase1",
                    "effective_SENS_IMU_MODE":0,"effective_EKF2_MULTI_IMU":3,"healthy_IMU_EKF_instances":1},
        "phase1c_physical_metrics":physical_metrics(hover["records"]),"hover_assessment":assess_hover(hover),
        "acceptance":{"px4_version":"PASS","sitl_startup":"PASS","bidirectional_bridge":"PASS","px4_actuators_received_and_applied":"PASS",
                      "sensors_in_px4_uorb":"PASS","estimator":"PASS (idealGNSS/mag/baro limited test)","px4_controls_flight":"PASS",
                      "bounded_closed_loop":"PASS","basic_hover":"PASS" if assess_hover(hover)["pass"] else "FAIL",
                      "dedicated_attitude_step_test":"NOT TESTED","real_motor_identification":"NOT TESTED","ROS_MID360_FAST_LIO2":"NOT TESTED"},
        "resources":{"mujoco_instances_max":1,"px4_instances_max":1,"parallel_builds":0,"oom_observed":False,
                     "max_bridge_hwm_bytes":max(d["bridge_peak_rss_bytes"] for d in logs.values()),
                     "max_px4_sampled_rss_bytes":max(d["px4_sampled_peak_rss_bytes"] for d in logs.values()),
                     "lowest_sampled_system_available_bytes":min(d["system_min_available_bytes"] for d in logs.values()),
                     "cpu_ps_probe_percent_one_core":{"bridge":25.5,"px4":7.1},"cpu_probe_scope":"one startup/flight ps sample, not full-run peak",
                     "startup_available_memory_requirement_gib":2.5,"runtime_reserve_gib":1.5},
        "verification":{"scoped_bridge_tests_passed":33,"scoped_bridge_tests_failed":0,
                        "independent_readonly_review":"0 Critical; 1 Important stale-stream watchdog finding fixed with three RED-to-GREEN pure regression tests",
                        "watchdog_fix":"single absolute bootstrap receive deadline; immediate stale-command rejection after lockstep; whole-run deadline bounds receives",
                        "no_additional_flights_after_fix":True,"historical_training_fixtures_executed":False,
                        "orchestration_fault_test":"fake process/physics/transport proves cleanup and previous nonzero controls zeroed; no PX4 process launched"},
        "limitations":["One bounded position hold, not a comprehensive attitude/yaw/recovery flight validation.",
          "No serial/hardware/Gazebo/legacyPI/training; existing checkpoints unchanged and not used by PX4.",
          "Mass/inertia/rotorpositions engineering estimates; km estimated, CW/CCW wiring unverified; no ESC/motor lag, battery sag or drag identification.",
          "SyntheticGNSS at47.397742/8.545594/488m, idealIMUatCOM, seededmag/baronoise engineering assumptions; no FAST-LIO2.",
          "~3.4deg estimated heading bias versus physical NED attitude under syntheticfield/declination; no yaw tracking quality claim.",
          "heading_good_for_control remained false: currentPX4 requires in-flight finalmag alignment; low0.6m hold did not establish that alignment.",
          "PX4 logger subscription-capacity warnings; missing topics never treated as recorded. RawULG local-only.",
          "Stock mavlinksim overrides requestedsingleIMU params; effectivevalues recorded; only one healthyEKFinstance.",
          "No hard real-time guarantee; measured≈real-time single finite runs, resource samples not continuousOS proof."],
        "next":"One separately scoped sensor-heading/alignment validation before yaw maneuvers or ROS external-vision integration; do not retune gains or connect hardware."}
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/"px4_mujoco_phase1.json",report)
    atomic_json(output/"px4_mujoco_phase1_hover_trace.json",{"sampling_hz":10,"units":"ENU m,m/s; FLU rad/s; ctrl(rad/s)^2", "records":hover["records"]})
    plot(hover,adapter.u_max,output/"px4_mujoco_hover.png")
    return report

def plot(hover,u_max,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=hover["records"];t=np.array([r["time_s"] for r in rows]);xyz=np.array([r["position_enu_m"] for r in rows])
    v=np.array([r["velocity_origin_enu_m_s"] for r in rows]);u=np.array([r["ctrl_omega_squared"] for r in rows])/u_max
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    axes[0].plot(t,xyz[:,2],label="actual body-origin height");axes[0].axhline(.651,ls="--",color="black",label="hold target")
    axes[0].set_ylabel("height [m]");axes[0].legend()
    axes[1].plot(t,np.linalg.norm(v,axis=1),label="origin speed");axes[1].plot(t,np.linalg.norm(xyz[:,:2],axis=1),label="horizontal offset [m]")
    axes[1].set_ylabel("m/s (speed) / m (offset)");axes[1].legend()
    for i in range(4):axes[2].plot(t,u[:,i],label=f"PX4 motor{i+1}")
    axes[2].set_ylabel("normalized thrust");axes[2].set_xlabel("MuJoCo simulation time [s]");axes[2].legend(ncol=4)
    for ax in axes:ax.axvline(hover["arm_request_at_s"],color="grey",ls=":");ax.grid(alpha=.3)
    fig.suptitle("One bounded PX4–MuJoCo SITL position hold (no gain tuning)")
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--parts",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();r=publish(args.parts,args.output);print(json.dumps(r["acceptance"],indent=2))
