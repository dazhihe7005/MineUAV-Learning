import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace
from .test_interfaces import module,fixture,ROOT

def test_px4_effectiveness_equals_model_com_wrench():
    m,d=fixture();a=module("actuator_adapter").ActuatorAdapter(m,ROOT/"models/rotor_config.json")
    p=module("simulation_runner").px4_environment(a)
    assert p["PX4_SYS_AUTOSTART"]=="10016"
    assert p["PX4_PARAM_SIM_GZ_EN"]=="0"
    assert p["PX4_SIMULATOR"]=="mujoco"
    for i in range(4):
        r=np.array([float(p[f"PX4_PARAM_CA_ROTOR{i}_P{axis}"]) for axis in "XYZ"])
        km=float(p[f"PX4_PARAM_CA_ROTOR{i}_KM"])
        torque=np.cross(r,[0,0,-1])-km*np.array([0,0,-1])
        arm=m.site(f"motor_{i+1}_site").pos-m.body("mine_uav").ipos
        world_torque=np.cross(arm,[0,0,1])+np.array([0,0,a.signs[i]*a.km/a.kf])
        np.testing.assert_allclose(torque,world_torque*[1,-1,-1],atol=1e-12)
        assert p[f"PX4_PARAM_PWM_MAIN_FUNC{i+1}"]==str(101+i)

def test_failure_stop_zeros_actual_mujoco_controls():
    m,d=fixture();d.ctrl[:]=123
    module("simulation_runner").safe_stop(d)
    np.testing.assert_equal(d.ctrl,[0,0,0,0])

def test_other_model_timestep_refused():
    m,d=fixture();m.opt.timestep=.001
    with pytest.raises(ValueError):module("simulation_runner").validate_model(m)

def test_fresh_runtime_refuses_old_parameters(tmp_path):
    r=module("simulation_runner")
    (tmp_path/"parameters.bson").write_bytes(b"historical")
    with pytest.raises(ValueError):r.prepare_runtime(tmp_path)
    assert (tmp_path/"parameters.bson").read_bytes()==b"historical"

def test_installed_px4_startup_directory_contract():
    root=Path.home()/"PX4-Autopilot"
    if not (root/"build/px4_sitl_default/bin/px4").exists():pytest.skip("installed binary characterization")
    identity=module("simulation_runner").source_identity(root)
    # PX4 main.cpp symlinks its data-path argument to runtime/etc.
    assert (Path(identity["rootfs"])/"init.d-posix/rcS").is_file()

def test_bootstrap_stale_stream_has_one_absolute_deadline(monkeypatch):
    r=module("simulation_runner");now=[0.];timeouts=[]
    monkeypatch.setattr(r.time,"monotonic",lambda:now[0])
    class Wire:
        def receive_until(self,kind,timeout):
            timeouts.append(timeout);now[0]+=.001
            return SimpleNamespace(time_usec=0)
    with pytest.raises(TimeoutError):
        r.receive_current_actuator(Wire(),1000000,False,.002)
    assert len(timeouts)==2
    np.testing.assert_allclose(timeouts,[.002,.001])

def test_stale_after_lockstep_is_immediately_rejected():
    r=module("simulation_runner");calls=[]
    class Wire:
        def receive_until(self,kind,timeout):
            calls.append(timeout);return SimpleNamespace(time_usec=996000)
    with pytest.raises(ValueError,match="stale"):
        r.receive_current_actuator(Wire(),1000000,True,r.time.monotonic()+5)
    assert len(calls)==1

def test_stale_stream_stops_owned_runner_and_zeroes_controls_without_px4(monkeypatch,tmp_path):
    # Orchestration only: fake process/wire/physics, never start PX4 or fly.
    r=module("simulation_runner");armed=[False];stale_calls=[];writes=[];closed=[]
    model=SimpleNamespace(opt=SimpleNamespace(timestep=.002,gravity=np.array([0,0,-9.81])),nq=7,nv=6,nu=4)
    data=SimpleNamespace(time=0.,qpos=np.array([0.,0.,0.,1.,0.,0.,0.]),qvel=np.zeros(6),ctrl=np.zeros(4))
    class Model:
        @staticmethod
        def from_xml_path(path):return model
    def step(m,d):d.time+=.25 # accelerated fake clock for the orchestration gate
    monkeypatch.setattr(r.mujoco,"MjModel",Model)
    monkeypatch.setattr(r.mujoco,"MjData",lambda m:data)
    monkeypatch.setattr(r.mujoco,"mj_forward",lambda *args:None)
    monkeypatch.setattr(r.mujoco,"mj_step",step)
    monkeypatch.setattr(r,"require_startup_memory",lambda:None)
    monkeypatch.setattr(r,"require_memory",lambda:None)
    monkeypatch.setattr(r,"source_identity",lambda _:dict(binary="fake",rootfs="fake",binary_sha256="hash"))
    monkeypatch.setattr(r,"sha256",lambda _:"hash")
    adapter=SimpleNamespace(u_max=1000.,px4_geometry=lambda:{},apply=lambda controls,is_armed:np.full(4,123. if is_armed else 0.))
    monkeypatch.setattr(r,"ActuatorAdapter",lambda *args:adapter)
    sensor=SimpleNamespace(sample=lambda d,t:SimpleNamespace(time_usec=t))
    monkeypatch.setattr(r,"SensorAdapter",lambda *args,**kwargs:sensor)
    mv=SimpleNamespace(MAV_CMD_DO_SET_MODE=176,MAV_CMD_COMPONENT_ARM_DISARM=400)
    monkeypatch.setattr(r,"load_mavlink",lambda _:mv)
    current=[1000000]
    def messages(sample,*args,**kwargs):current[0]=sample.time_usec;return []
    monkeypatch.setattr(r,"sensor_messages",messages)
    class Wire:
        counts={};sent={}
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):closed.append("wire")
        def accept(self):pass
        def receive_until(self,kind,timeout=None):
            if kind=="HEARTBEAT":return None
            if armed[0] and np.any(data.ctrl):
                stale_calls.append(1)
                # Bound even the old buggy implementation so RED cannot hang.
                if len(stale_calls)>5:raise ConnectionError("test guard: stale stream was not rejected")
                return SimpleNamespace(time_usec=current[0]-4000)
            return SimpleNamespace(time_usec=current[0],flags=1,mode=128 if armed[0] else 0,controls=[.4]*4)
    monkeypatch.setattr(r,"LoopbackTransport",Wire)
    class Process:
        pid=99999999;returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout):self.returncode=0;closed.append("process");return 0
    proc=Process()
    monkeypatch.setattr(r.subprocess,"Popen",lambda *args,**kwargs:proc)
    monkeypatch.setattr(r.pty,"openpty",lambda:(101,102))
    monkeypatch.setattr(r.os,"close",lambda fd:closed.append(fd))
    monkeypatch.setattr(r.os,"write",lambda fd,b:writes.append(b) or len(b))
    monkeypatch.setattr(r.select,"select",lambda *args:([],[],[]))
    monkeypatch.setattr(r.time,"sleep",lambda _:None)
    class Offboard:
        acks=[]
        def __init__(self,*args):pass
        def setpoint(self,*args):pass
        def heartbeat(self):pass
        def poll(self):pass
        def command(self,kind,*args):
            if kind==400:armed[0]=bool(args[0])
        def close(self):closed.append("offboard")
    monkeypatch.setattr(r,"OffboardLink",Offboard)
    monkeypatch.setattr(r,"estimator_ready",lambda _:True)
    monkeypatch.setattr(r,"check_envelope",lambda _:None)
    proof=tmp_path/"proof.json"
    r.atomic_json(proof,{"complete":True,"kind":"disarmed_communication_smoke",
        "history_hashes_after":dict(model_xml="hash",rotor_config="hash",rotor_actuators="hash"),
        "px4":dict(binary_sha256="hash")})
    result=r.smoke(tmp_path,tmp_path/"runtime",26.,hover_proof=proof)
    assert not result["complete"] and result["exception"]["type"]=="ValueError"
    assert "stale" in result["exception"]["message"]
    assert len(stale_calls)==1 and result["final_physical_state"]["ctrl"]==[123.]*4
    assert result["zero_control_on_stop"]==[0.]*4 and not armed[0]
    assert "wire" in closed and "offboard" in closed and "process" in closed
    assert b"shutdown\n" in writes and result["px4_exit_code"]==0
    assert (tmp_path/"runtime/smoke_summary.json").is_file()
