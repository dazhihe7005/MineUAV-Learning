"""Finite PX4 SITL smoke, opt-in bounded flight; no serial/old PI imports.

Run with PYTHONPATH=mujoco python -m px4_bridge.simulation_runner --help.
Flight tests require prior matching smoke and fresh estimator evidence.
"""
import argparse
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import time
import traceback
import mujoco
import numpy as np
from .actuator_adapter import ActuatorAdapter
from .sensor_adapter import SensorAdapter
from .transport import LoopbackTransport,load_mavlink,sensor_messages
from .diagnostics import (SimClock,CommandGate,require_memory,require_startup_memory,available_memory,
                          rss_bytes,own_peak_rss_bytes,sha256,atomic_json)
from .closed_loop import OffboardLink,estimator_ready,check_envelope,assess_hover

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/"models/mine_uav_dynamics_v2.xml"
CONFIG=ROOT/"models/rotor_config.json"
PX4_COMMIT="8f9b91b2bdd3a65952601a084c3aa35e96fee59d"

def validate_model(model):
    if abs(model.opt.timestep-.002)>1e-12 or model.nq!=7 or model.nv!=6 or model.nu!=4:
        raise ValueError("phase1 requires unchanged v2 free-body model at500Hz")
    if not np.array_equal(model.opt.gravity,[0,0,-9.81]):raise ValueError("unexpected gravity")

def safe_stop(data):
    data.ctrl[:]=0

def receive_current_actuator(wire,timestamp,lockstep,deadline):
    """Bootstrap may discard old outputs, but never renew its receive budget.

    After lockstep starts a stale response is a protocol fault, not a reason
    to keep the last controls while waiting for more messages.
    """
    stale=0
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError("PX4 actuator absolute watchdog expired")
        msg=wire.receive_until("HIL_ACTUATOR_CONTROLS",timeout=remaining)
        if msg.time_usec>=timestamp:return msg,stale
        if lockstep:raise ValueError("stale PX4 actuator after lockstep started")
        stale+=1

def prepare_runtime(path):
    p=Path(path).resolve()
    if p.exists() and any(p.iterdir()):raise ValueError("runtime must be empty; never reuse historical PX4 parameters/logs")
    p.mkdir(parents=True,exist_ok=True)
    return p

def px4_environment(adapter):
    # 10016_none_iris does not start another simulator. rcS uses this solely
    # for MC startup defaults, then env overrides replace allocator geometry.
    params=adapter.px4_geometry()
    params.update(SIM_GZ_EN=0,IMU_INTEG_RATE=250,EKF2_MULTI_IMU=0,SENS_IMU_MODE=1)
    env={"PX4_SIM_MODEL":"mineuav","PX4_SIMULATOR":"mujoco","PX4_SYS_AUTOSTART":"10016",
         "PX4_SIM_HOST_ADDR":"127.0.0.1","PX4_SIM_SPEED_FACTOR":"1"}
    env.update({"PX4_PARAM_"+k:str(v) for k,v in params.items()})
    return env

def source_identity(px4_root):
    root=Path(px4_root).resolve();build=root/"build/px4_sitl_default"
    commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(root),"diff","--name-only","HEAD"],text=True).strip()
    header=build/"src/lib/version/build_git_version.h"
    if commit!=PX4_COMMIT or dirty or f'"{commit}"' not in header.read_text():
        raise ValueError("PX4 source/build identity mismatch; review new version before running")
    binary=build/"bin/px4"
    if not os.access(binary,os.X_OK):raise ValueError("matching existing PX4 binary unavailable")
    return {"commit":commit,"tag":subprocess.check_output(["git","-C",str(root),"describe","--tags","--always"],text=True).strip(),
            "binary":str(binary),"binary_sha256":sha256(binary),"build_version_header_sha256":sha256(header),
            "rootfs":str(build/"etc"),
            "simulator_source_sha256":sha256(root/"src/modules/simulation/simulator_mavlink/SimulatorMavlink.cpp")}

def smoke(px4_root,runtime,seconds=12.,pymavlink_search_path=None,hover_proof=None):
    if not np.isfinite(seconds) or not 8<=seconds<=30:raise ValueError("finite smoke duration8..30s")
    require_startup_memory();identity=source_identity(px4_root)
    runtime=prepare_runtime(runtime)
    m=mujoco.MjModel.from_xml_path(str(MODEL));validate_model(m)
    d=mujoco.MjData(m)
    # New SITL ground startup, not a modification of the historical model.
    d.qpos[2]=.051;mujoco.mj_forward(m,d)
    require_memory()
    initial_qpos=d.qpos.tolist();a=ActuatorAdapter(m,CONFIG)
    # Perfectly constant mag/baro trip PX4 DataValidator STALE_DATA at rest.
    # Fixed engineering noise assumptions, NOT measured sensor specifications.
    sensors=SensorAdapter(m,noise_seed=0,mag_noise_gauss=.0002,baro_noise_hpa=.01)
    mv=load_mavlink(pymavlink_search_path);clock=SimClock();gate=CommandGate()
    history={"model_xml":sha256(MODEL),"rotor_config":sha256(CONFIG),
             "rotor_actuators":sha256(ROOT/"models/rotor_actuators.xml")}
    if hover_proof:
        proof=json.loads(Path(hover_proof).read_text())
        if not (proof.get("complete") and proof.get("kind")=="disarmed_communication_smoke"
                and proof.get("history_hashes_after")==history and proof.get("px4",{}).get("binary_sha256")==identity["binary_sha256"]
                and estimator_ready(proof.get("px4_console","")) and seconds>=22):
            raise ValueError("flight requires successful matching disarmed smoke + estimator evidence, duration22..30s")
    result={"complete":False,"kind":"bounded_offboard_hover" if hover_proof else "disarmed_communication_smoke","px4":identity,
            "mujoco_version":mujoco.__version__,"python":os.sys.version,"history_hashes_before":history,
            "initial_qpos":initial_qpos,"px4_environment":px4_environment(a),"simulator_port":4560,
            "physics_hz":500,"sensor_hz":250,"gps_hz":10,"motor_assumption":"ideal normalized thrust; THR_MDL_FAC=0; no ESC lag",
            "synthetic_sensor_noise":{"seed":0,"mag_std_gauss":.0002,"baro_std_hpa":.01,"imu_noise":0},
            "workers":1,"flight_command_sent":False,"records":[],"exception":None}
    proc=None;master=None;offboard=None;console=[];wall_start=time.monotonic();cpu_start=time.process_time()
    peak_px4=0;min_memory=available_memory();stale=0;intervals=[];first_actuator_at=None
    def read_console():
        if master is None:return
        while select.select([master],[],[],0)[0]:
            try:chunk=os.read(master,65536)
            except OSError:break
            if not chunk:break
            console.append(chunk.decode("utf-8",errors="replace"))
    try:
        with LoopbackTransport(timeout=5.,mavlink=mv) as wire:
            env=os.environ.copy()
            # Do not inherit any user's hidden PX4 parameter/network override.
            for key in list(env):
                if key.startswith("PX4_"):env.pop(key)
            env.update(px4_environment(a))
            master,slave=pty.openpty()
            proc=subprocess.Popen([identity["binary"],identity["rootfs"],"-w",str(runtime),"-i","0"],
                   cwd=runtime,env=env,stdin=slave,stdout=slave,stderr=slave,start_new_session=True)
            os.close(slave);wire.accept();wire.receive_until("HEARTBEAT")
            if hover_proof:offboard=OffboardLink(mv)
            bootstrap_deadline=time.monotonic()+10
            lockstep=False;last_log=-1;requested=False;flight_requested=False;mode_requested=False;last_setpoint=-1;last_heartbeat=-1
            while d.time<seconds:
                if time.monotonic()-wall_start>seconds+35:raise TimeoutError("finite wall-clock smoke budget")
                if proc.poll() is not None:raise ConnectionError("owned PX4 exited")
                sample=sensors.sample(d,clock.time_usec)
                if offboard:
                    if int(d.time*20)>last_setpoint:
                        last_setpoint=int(d.time*20);offboard.setpoint(clock.time_usec)
                    if int(d.time)>last_heartbeat:
                        last_heartbeat=int(d.time);offboard.heartbeat()
                    offboard.poll()
                    if d.time>=10 and not mode_requested:
                        read_console()
                        if not estimator_ready("".join(console)):raise ValueError("current estimator/preflight gate failed; no arming")
                        offboard.command(mv.MAV_CMD_DO_SET_MODE,1,6);mode_requested=True
                    if d.time>=11 and not flight_requested:
                        offboard.command(mv.MAV_CMD_COMPONENT_ARM_DISARM,1);flight_requested=True
                        result["flight_command_sent"]=True;result["arm_request_at_s"]=float(d.time)
                for msg in sensor_messages(sample,mv,include_gps=(clock.time_usec-1000000)%100000==0):wire.send(msg)
                try:
                    # Startup freewheeling only until the first PX4 output.
                    deadline=min(time.monotonic()+(5. if lockstep else .002),
                                 wall_start+seconds+35,
                                 float("inf") if lockstep else bootstrap_deadline)
                    msg,discarded=receive_current_actuator(wire,clock.time_usec,lockstep,deadline)
                    stale+=discarded
                    if msg.time_usec!=clock.time_usec:raise ValueError("PX4 actuator/sensor clock mismatch")
                    if not msg.flags&1:raise ValueError("matching PX4 build must advertise lockstep")
                    armed=bool(msg.mode&128)
                    if armed and not flight_requested:raise ValueError("unexpected armed output")
                    previous=gate.last;gate.accept(msg.time_usec)
                    if previous>=0:intervals.append(msg.time_usec-previous)
                    d.ctrl[:]=a.apply(msg.controls,armed)
                    if not lockstep:first_actuator_at=float(d.time)
                    lockstep=True
                except TimeoutError:
                    if lockstep or time.monotonic()>bootstrap_deadline:raise
                    safe_stop(d)
                for _ in range(2):mujoco.mj_step(m,d)
                clock.advance(2)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise ValueError("invalid MuJoCo state")
                if offboard:check_envelope(d)
                elif np.linalg.norm(d.qvel[:3])>2 or d.qpos[2]>1:raise ValueError("disarmed physical safety envelope exceeded")
                if int(d.time*10)>last_log:
                    last_log=int(d.time*10)
                    require_memory();min_memory=min(min_memory,available_memory());peak_px4=max(peak_px4,rss_bytes(proc.pid))
                    result["records"].append({"time_s":float(d.time),"clock_usec":clock.time_usec,
                      "position_enu_m":d.qpos[:3].tolist(),"velocity_origin_enu_m_s":d.qvel[:3].tolist(),
                      "quaternion_enu_flu":d.qpos[3:].tolist(),"angular_velocity_flu_rad_s":d.qvel[3:].tolist(),
                      "ctrl_omega_squared":d.ctrl.tolist(),"saturated":bool(np.any(d.ctrl>=a.u_max*(1-1e-6)))})
                    read_console()
                if d.time>=(8 if offboard else seconds-3) and not requested:
                    commands="ver all\nsimulator_mavlink status\nekf2 status\nlistener sensor_combined 1\nlistener vehicle_local_position 1\nlistener vehicle_status 1\nlistener estimator_status_flags 1\nlistener actuator_outputs_sim 1\nlistener vehicle_air_data 1\nlistener vehicle_magnetometer 1\nparam show SENS_IMU_MODE\nparam show EKF2_MULTI_IMU\n"
                    os.write(master,commands.encode());requested=True
                if offboard and d.time>=seconds-3 and requested is True:
                    os.write(master,b"ekf2 status\nlistener vehicle_status 1\nlistener vehicle_local_position 1\nlistener vehicle_attitude 1\nlistener actuator_outputs_sim 1\n")
                    requested="final"
                # Approximately real-time, with lockstep slowing rather than skipping.
                delay=wall_start+float(d.time)-time.monotonic()
                if delay>0:time.sleep(min(delay,.004))
            result.update(complete=True,simulation_seconds=float(d.time),messages_received=wire.counts,
                          messages_sent=wire.sent,first_actuator_at_s=first_actuator_at,
                          stale_bootstrap_messages_discarded=stale,
                          actuator_interval_usec={"min":min(intervals),"max":max(intervals),"median":float(np.median(intervals))} if intervals else None)
    except Exception as e:
        result["exception"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
    finally:
        result["final_physical_state"]={"time_s":float(d.time),"qpos":d.qpos.tolist(),"qvel":d.qvel.tolist(),"ctrl":d.ctrl.tolist()}
        if offboard:
            result["command_acks"]=offboard.acks
            try:offboard.command(mv.MAV_CMD_COMPONENT_ARM_DISARM,0)
            except OSError:pass
            offboard.close()
        safe_stop(d)
        if proc is not None:
            read_console()
            if proc.poll() is None:
                try:os.write(master,b"shutdown\n");proc.wait(timeout=3)
                except (OSError,subprocess.TimeoutExpired):
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=3)
            read_console();result["px4_exit_code"]=proc.returncode
        if master is not None:os.close(master)
        result.update(wall_seconds=time.monotonic()-wall_start,bridge_cpu_seconds=time.process_time()-cpu_start,
                      bridge_peak_rss_bytes=own_peak_rss_bytes(),px4_sampled_peak_rss_bytes=peak_px4,
                      system_min_available_bytes=min_memory,zero_control_on_stop=d.ctrl.tolist(),
                      px4_console="".join(console))
        result["history_hashes_after"]={"model_xml":sha256(MODEL),"rotor_config":sha256(CONFIG),"rotor_actuators":sha256(ROOT/"models/rotor_actuators.xml")}
        result["history_unchanged"]=result["history_hashes_after"]==history
        if hover_proof:result["hover_assessment"]=assess_hover(result)
        atomic_json(runtime/"smoke_summary.json",result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-root",type=Path,required=True)
    parser.add_argument("--runtime",type=Path,required=True,help="NEW empty owned output directory")
    parser.add_argument("--seconds",type=float,default=12.)
    parser.add_argument("--pymavlink-search-path")
    parser.add_argument("--hover-proof",type=Path,help="explicit SITL flight opt-in: matching successful smoke_summary.json")
    args=parser.parse_args()
    result=smoke(args.px4_root,args.runtime,args.seconds,args.pymavlink_search_path,args.hover_proof)
    print(json.dumps({k:result.get(k) for k in ("complete","exception","simulation_seconds","messages_received","messages_sent","bridge_peak_rss_bytes","px4_sampled_peak_rss_bytes")}))
    return 0 if result["complete"] and result.get("hover_assessment",{"pass":True})["pass"] else 1

if __name__=="__main__":raise SystemExit(main())
