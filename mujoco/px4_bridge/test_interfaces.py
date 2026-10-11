"""Boundary contracts; no PX4 process, model changes or real motors."""
from pathlib import Path
import importlib
import sys
import numpy as np
import mujoco
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def module(name):
    return importlib.import_module(f"px4_bridge.{name}")

def fixture():
    m = mujoco.MjModel.from_xml_path(str(ROOT / "models/mine_uav_dynamics_v2.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d

def test_frames_and_identity_attitude():
    f = module("frame_conversions")
    np.testing.assert_equal(f.enu_to_ned([1,2,3]), [2,1,-3])
    np.testing.assert_equal(f.flu_to_frd([1,2,3]), [1,-2,-3])
    q = f.attitude_to_px4([1,0,0,0])
    np.testing.assert_allclose(q, [2**-.5,0,0,2**-.5], atol=1e-12)
    np.testing.assert_allclose(f.attitude_from_px4(q), [1,0,0,0], atol=1e-12)

@pytest.mark.parametrize("q", [[0,0,0,0],[1,float("nan"),0,0],[2,0,0,0]])
def test_invalid_attitude_rejected(q):
    with pytest.raises(ValueError): module("frame_conversions").attitude_to_px4(q)

def test_freefall_has_zero_specific_force_not_gravity():
    m,d=fixture()
    s=module("sensor_adapter").SensorAdapter(m).sample(d,1000)
    np.testing.assert_allclose(s.accel_frd, [0,0,0], atol=1e-10)
    np.testing.assert_allclose(s.gyro_frd, [0,0,0], atol=1e-12)

def test_force_specific_acceleration_and_body_gyro():
    m,d=fixture()
    # External force at COM, independent of old PI/allocation.
    d.xfrc_applied[1,:3]=[7,14,68.67]
    mujoco.mj_forward(m,d)
    s=module("sensor_adapter").SensorAdapter(m).sample(d,1000)
    np.testing.assert_allclose(s.accel_frd,[1,-2,-9.81],atol=1e-9)
    d.qvel[3:]=[.2,.3,.4]
    mujoco.mj_forward(m,d)
    s=module("sensor_adapter").SensorAdapter(m).sample(d,2000)
    np.testing.assert_allclose(s.gyro_frd,[.2,-.3,-.4],atol=1e-12)

def test_tilted_sensor_rotates_world_force():
    m,d=fixture()
    d.qpos[3:]=[2**-.5,0,2**-.5,0] # FLU +90deg pitch
    d.xfrc_applied[1,:3]=[0,0,68.67]
    mujoco.mj_forward(m,d)
    s=module("sensor_adapter").SensorAdapter(m).sample(d,1000)
    np.testing.assert_allclose(s.accel_frd,[-9.81,0,0],atol=1e-9)

def test_single_rotor_model_force_matches_adapter():
    m,d=fixture()
    a=module("actuator_adapter").ActuatorAdapter(m,ROOT / "models/rotor_config.json")
    d.ctrl[:]=a.apply([.25,0,0,0],True)
    mujoco.mj_forward(m,d)
    s=module("sensor_adapter").SensorAdapter(m).sample(d,1000)
    expected=.25*(8500*2*np.pi/60)**2*4.9977603696661616e-5/7
    np.testing.assert_allclose(s.accel_frd,[0,0,-expected],atol=1e-8)
    np.testing.assert_allclose(np.sqrt(d.ctrl[0]),8500*2*np.pi/60*.5,atol=1e-10)

def test_motor_bounds_disarm_and_exact_index():
    m,d=fixture()
    a=module("actuator_adapter").ActuatorAdapter(m,ROOT / "models/rotor_config.json")
    np.testing.assert_equal(a.apply([.1,.2,.3,.4],False),[0,0,0,0])
    np.testing.assert_allclose(a.apply([.1,.2,.3,.4],True)/a.u_max,[.1,.2,.3,.4])
    for c in ([float("nan"),0,0,0],[-.1,0,0,0],[1.1,0,0,0],[0,0]):
        with pytest.raises(ValueError): a.apply(c,True)

def test_stale_model_gear_rejected():
    m,d=fixture();m.actuator_gear[0,5]*=-1
    with pytest.raises(ValueError):
        module("actuator_adapter").ActuatorAdapter(m,ROOT / "models/rotor_config.json")

def test_sensor_noise_is_seeded_and_prevents_constant_wire_values():
    m,d=fixture();cls=module("sensor_adapter").SensorAdapter
    s1=cls(m,noise_seed=0,mag_noise_gauss=.0002,baro_noise_hpa=.01)
    s2=cls(m,noise_seed=0,mag_noise_gauss=.0002,baro_noise_hpa=.01)
    values=[]
    for i in range(32):
        a=s1.sample(d,1000000+i*4000);b=s2.sample(d,1000000+i*4000)
        np.testing.assert_equal(a.mag_frd_gauss,b.mag_frd_gauss)
        assert a.pressure_hpa==b.pressure_hpa
        values.append(np.r_[a.mag_frd_gauss,a.pressure_hpa])
    # Catches zero-noise wiring (PX4 voter declares identical readings stale).
    std=np.std(values,axis=0)
    assert np.all(std[:3]>.00005) and std[3]>.003

def test_wrong_site_or_dynamic_motor_cannot_silently_pass():
    cls=module("actuator_adapter").ActuatorAdapter
    for mutation in ("site","dynamic"):
        m,d=fixture()
        if mutation=="site":m.actuator_trnid[0,0]=m.site("motor_2_site").id
        else:m.actuator_dyntype[0]=mujoco.mjtDyn.mjDYN_FILTER
        with pytest.raises(ValueError):cls(m,ROOT/"models/rotor_config.json")
