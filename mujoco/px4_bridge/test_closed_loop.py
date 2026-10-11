import pytest
import numpy as np
import socket
from .test_interfaces import module,fixture
from .test_transport import SEARCH

GOOD="""TOPIC: vehicle_local_position
 xy_valid: True
 z_valid: True
 v_xy_valid: True
 v_z_valid: True
TOPIC: vehicle_status
 pre_flight_checks_pass: True
TOPIC: estimator_status_flags
 cs_tilt_align: True
 cs_yaw_align: True
 cs_gnss_pos: True
 cs_mag_fault: False
 cs_baro_fault: False
"""

def test_flight_gate_requires_current_complete_estimator_evidence():
    f=module("closed_loop")
    assert f.estimator_ready(GOOD)
    assert not f.estimator_ready(GOOD.replace("cs_yaw_align: True","cs_yaw_align: False"))
    assert not f.estimator_ready(GOOD.replace("cs_mag_fault: False","cs_mag_fault: True"))
    assert not f.estimator_ready("port connected")

def test_velocity_tilt_and_height_safety_envelope():
    f=module("closed_loop");m,d=fixture()
    f.check_envelope(d)
    d.qvel[0]=1.6
    with pytest.raises(ValueError):f.check_envelope(d)
    d.qvel[:]=0;d.qpos[2]=1.3
    with pytest.raises(ValueError):f.check_envelope(d)
    d.qpos[2]=.7;d.qpos[3:]=[.9238795325,.3826834324,0,0]
    with pytest.raises(ValueError):f.check_envelope(d)

def test_offboard_messages_are_local_and_position_only():
    f=module("closed_loop");mv=module("transport").load_mavlink(str(SEARCH))
    receiver=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);receiver.bind(("127.0.0.1",0));receiver.settimeout(.5)
    try:
        with f.OffboardLink(mv,remote_port=receiver.getsockname()[1],local_port=0) as link:
            link.setpoint(1000000)
            data,addr=receiver.recvfrom(2048);msg=mv.MAVLink(None).parse_char(data)
            assert addr[0]=="127.0.0.1"
            assert msg.get_type()=="SET_POSITION_TARGET_LOCAL_NED"
            assert msg.coordinate_frame==mv.MAV_FRAME_LOCAL_NED
            assert msg.type_mask==2552
            assert [msg.x,msg.y,msg.z]==pytest.approx([0,0,-.6])
            assert msg.target_system==1
            link.command(mv.MAV_CMD_COMPONENT_ARM_DISARM,1)
            data,_=receiver.recvfrom(2048);msg=mv.MAVLink(None).parse_char(data)
            assert msg.command==400 and msg.param1==1 and msg.param2==0 # NEVER force-arm
    finally:receiver.close()

def test_hover_assessment_requires_real_armed_outputs_and_hold():
    f=module("closed_loop")
    record={"position_enu_m":[0,0,.651],"velocity_origin_enu_m_s":[0,0,0],
            "quaternion_enu_flu":[1,0,0,0],"ctrl_omega_squared":[320000]*4}
    result={"complete":True,"kind":"bounded_offboard_hover","initial_qpos":[0,0,.051,1,0,0,0],
        "simulation_seconds":3.,"flight_command_sent":True,"command_acks":[{"command":176,"result":0},{"command":400,"result":0}],
        "records":[dict(record,time_s=1+i*.1) for i in range(21)]}
    assert f.assess_hover(result)["pass"]
    result["records"][-1]["velocity_origin_enu_m_s"]=[1,0,0]
    assert not f.assess_hover(result)["pass"]
    result["records"][-1]["velocity_origin_enu_m_s"]=[0,0,0]
    for r in result["records"]:r["ctrl_omega_squared"]=[0]*4
    assert not f.assess_hover(result)["pass"]
