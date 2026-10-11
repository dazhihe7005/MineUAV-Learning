from pathlib import Path
import numpy as np
import pytest
from .test_interfaces import module,fixture

def safety_console(timestamp=2000000,heading=True,aligned=True):
    return f"""TOPIC: vehicle_local_position
 vehicle_local_position
    timestamp: {timestamp}
    heading: 1.57079632679
    xy_valid: True
    z_valid: True
    v_xy_valid: True
    v_z_valid: True
    heading_good_for_control: {heading}

TOPIC: estimator_status_flags
 estimator_status_flags
    timestamp: {timestamp}
    cs_tilt_align: True
    cs_yaw_align: True
    cs_mag_aligned_in_flight: {aligned}
    cs_in_air: True
    cs_gnss_pos: True
    cs_mag_hdg: False
    cs_mag_3d: True
    cs_mag_fault: False
    cs_baro_fault: False
    cs_gnss_fault: False
    cs_gnss_yaw_fault: False
    cs_gnss_hgt_fault: False

TOPIC: vehicle_status
 vehicle_status
    timestamp: {timestamp}
    arming_state: 2
    nav_state: 14
    failsafe: False
    pre_flight_checks_pass: True

"""

def test_geographic_field_and_handedness():
    f=module("magnetic_model")
    np.testing.assert_allclose(f.field_vector(90.,0.,.5),[0,.5,0],atol=1e-12)
    np.testing.assert_allclose(f.field_vector(0.,60.,.5),[.25,0,.4330127018922193],atol=1e-12)
    np.testing.assert_allclose(f.field_vector(-90.,-30.,.4),[0,-.34641016151377546,-.2],atol=1e-12)

def test_pinned_wmm_matches_logged_px4_declination():
    f=module("magnetic_model")
    state=f.field_from_px4(Path("/home/keyousuke/PX4-Autopilot"))
    assert state["declination_deg"]==pytest.approx(3.4056053161621094,abs=1e-4)
    v=np.array(state["field_ned_gauss"])
    assert np.degrees(np.arctan2(v[1],v[0]))==pytest.approx(state["declination_deg"])
    assert np.linalg.norm(v)==pytest.approx(state["strength_gauss"])

@pytest.mark.parametrize("args",[(float("nan"),0,.5),(0,91,.5),(0,0,0)])
def test_invalid_magnetic_field_rejected(args):
    with pytest.raises(ValueError):module("magnetic_model").field_vector(*args)

def test_ned_yaw_truth_and_rate_sign():
    y=module("yaw_validation")
    assert y.yaw_ned([1,0,0,0])==pytest.approx(np.pi/2)
    assert y.yaw_ned([2**-.5,0,0,2**-.5])==pytest.approx(0,abs=1e-12)
    assert y.angle_error(np.deg2rad(-179),np.deg2rad(179))==pytest.approx(np.deg2rad(2))

def test_live_alignment_gate_requires_fresh_real_complete_topics():
    y=module("yaw_validation")
    assert y.live_heading_ready(safety_console(),2500000)
    assert not y.live_heading_ready(safety_console(),3000001)
    assert not y.live_heading_ready(safety_console(timestamp=3000000),2500000)
    assert not y.live_heading_ready(safety_console(heading=False),2500000)
    assert not y.live_heading_ready(safety_console(aligned=False),2500000)
    assert not y.live_heading_ready(safety_console().replace("    cs_mag_fault: False\n",""),2500000)
    # Do not merge a new partial topic with old healthy fields.
    assert not y.live_heading_ready(safety_console()+"TOPIC: vehicle_local_position\n    timestamp: 2400000\n",2500000)

def test_live_px4_prompt_prefixed_topic_is_not_discarded():
    y=module("yaw_validation")
    console=safety_console().replace("TOPIC:","\x1b[0Kpxh> TOPIC:").replace("TOPIC: estimator_status_flags\n","TOPIC: estimator_status_flags 3 instances\n")
    assert y.live_heading_ready(console,2500000)
    assert set(y.latest_topics(console))=={"vehicle_local_position","estimator_status_flags","vehicle_status"}

def test_blocked_raw_run_has_real_readable_alignment_topics():
    import json
    p=Path(__file__).resolve().parents[1]/"reports/px4_yaw_phase2_parts/yaw_01/smoke_summary.json"
    if not p.exists():pytest.skip("local-only captured instrumentation regression")
    s=json.loads(p.read_text());y=module("yaw_validation")
    clock=round(1000000+s["final_physical_state"]["time_s"]*1000000)
    assert s["phase2"]["actual_step_time_s"] is None
    assert y.live_heading_ready(s["px4_console"],clock)

def test_nonprimary_estimator_listener_cannot_authorize_step():
    console=safety_console().replace("\n estimator_status_flags\n","\nInstance 1:\n estimator_status_flags\n")
    assert not module("yaw_validation").live_heading_ready(console,2500000)

def test_settling_is_sustained_and_time_censored():
    y=module("yaw_validation")
    t=np.arange(0,6.1,.1);err=np.where(t<2.5,.1,.01);rates=np.where(t<3,.1,.01)
    assert y.settling_time(t,err,rates,2.)==pytest.approx(1.)
    rates[-2]=.2
    assert y.settling_time(t,err,rates,2.) is None
    assert y.settling_time(t[:30],err[:30],rates[:30],2.) is None

def test_sensor_configured_field_preserves_frames_and_old_default():
    m,d=fixture();cls=module("sensor_adapter").SensorAdapter
    s=cls(m,field_ned_gauss=[.2,.1,.4]).sample(d,1000000)
    # Identity ENU-FLU means90deg NED heading: B_FRD=(East,-North,Down).
    np.testing.assert_allclose(s.mag_frd_gauss,[.1,-.2,.4],atol=1e-12)
    np.testing.assert_allclose(cls(m).sample(d,1000000).mag_frd_gauss,[0,-.215,.427])
    with pytest.raises(ValueError):cls(m,field_ned_gauss=[0,0,0])

def protocol(active=True):
    return module("yaw_runner").YawProtocol({"field_ned_gauss":[.2,.01,.4]},active=active)

def test_yaw_step_is_gated_not_just_time_triggered():
    from types import SimpleNamespace
    data=SimpleNamespace(time=32.,qpos=np.array([0,0,2.051,1,0,0,0.]),qvel=np.zeros(6))
    for evidence in ("",safety_console(33000000,heading=False),safety_console(33000000,aligned=False)):
        p=protocol()
        with pytest.raises(ValueError):p.before_step(data,33000000,evidence)
        assert p.step_time is None and p.command_yaw==pytest.approx(np.pi/2)
    p=protocol();p.before_step(data,33000000,safety_console(33000000))
    assert p.command_yaw==pytest.approx(np.deg2rad(100)) and p.step_time==32.
    with pytest.raises(ValueError):p.before_step(data,35000000,safety_console(33000000))

def test_no_yaw_step_before_declared_time_or_while_unheld():
    from types import SimpleNamespace
    data=SimpleNamespace(time=31.9,qpos=np.array([0,0,2.051,1,0,0,0.]),qvel=np.zeros(6))
    p=protocol();p.before_step(data,32900000,"")
    assert p.step_time is None
    data.time=32.;data.qvel[0]=.5
    with pytest.raises(ValueError):p.before_step(data,33000000,safety_console(33000000))
    assert p.step_time is None

def test_matching_corrected_field_proof_required():
    p=protocol()
    with pytest.raises(ValueError):p.validate_proof({})
    with pytest.raises(ValueError):p.validate_proof({"phase2":{"field_sha256":"wrong"}})
    p.validate_proof({"phase2":{"field_sha256":p.metadata()["field_sha256"]}})

def test_bounded_height_and_yaw_wire_setpoint():
    from .test_transport import SEARCH
    v=module("transport").load_mavlink(str(SEARCH));link=module("closed_loop").OffboardLink(v,local_port=0)
    messages=[];link.send=messages.append
    try:link.setpoint(33000000,height=2.,yaw=np.deg2rad(100))
    finally:link.close()
    msg=messages[0]
    assert msg.z==-2. and msg.yaw==pytest.approx(np.deg2rad(100))
    assert msg.type_mask==2552 and msg.target_system==1 and msg.target_component==1

def test_phase2_envelope_rejects_unsafe_height():
    from types import SimpleNamespace
    p=protocol();data=SimpleNamespace(qpos=np.array([0,0,2.1,1,0,0,0.]),qvel=np.zeros(6))
    p.check_envelope(data)
    data.qpos[2]=2.601
    with pytest.raises(ValueError):p.check_envelope(data)

def test_gate_caches_parsing_but_never_clock_or_new_flags(monkeypatch):
    from types import SimpleNamespace
    r=module("yaw_runner");real=r.latest_topics;calls=[]
    def counted(text):calls.append(1);return real(text)
    monkeypatch.setattr(r,"latest_topics",counted)
    d=SimpleNamespace(time=32.,qpos=np.array([0,0,2.051,1,0,0,0.]),qvel=np.zeros(6))
    p=protocol();text=safety_console(33000000)
    p.before_step(d,33000000,text);p.before_step(d,33200000,text)
    assert len(calls)==1
    with pytest.raises(ValueError):p.before_step(d,33300000,text.replace("nav_state: 14","nav_state: 15"))
    assert len(calls)==2
    p=protocol();p.before_step(d,33000000,text)
    with pytest.raises(ValueError):p.before_step(d,34000001,text)

def test_gate_abort_after_authorization_does_not_claim_no_step():
    from types import SimpleNamespace
    data=SimpleNamespace(time=32.,qpos=np.array([0,0,2.051,1,0,0,0.]),qvel=np.zeros(6))
    p=protocol();p.before_step(data,33000000,safety_console(33000000))
    with pytest.raises(ValueError,match="abort active yaw"):
        p.before_step(data,34000001,safety_console(33000000))
    assert p.step_time==32.


@pytest.mark.parametrize("fragment",["TOP", "TOPIC:","TOPIC: vehicle_local_pos", "TOPIC: estimator_status_flags 3 inst", "TOPIC: estimator_status_flags 3 inst\n"])
def test_partial_topic_headers_invalidate_old_healthy_snapshot(fragment):
    y=module("yaw_validation")
    assert not y.live_heading_ready(safety_console()+"\n"+fragment,2500000)


def test_unfinished_topic_body_does_not_authorize_with_early_fields():
    assert not module("yaw_validation").live_heading_ready(safety_console().rstrip("\n"),2500000)


@pytest.mark.parametrize("key",["cs_gnss_fault","cs_gnss_yaw_fault","cs_gnss_hgt_fault","failsafe"])
def test_explicit_gnss_and_vehicle_faults_must_be_present_and_false(key):
    y=module("yaw_validation");line=f"    {key}: False\n";text=safety_console()
    assert not y.live_heading_ready(text.replace(line,f"    {key}: True\n"),2500000)
    assert not y.live_heading_ready(text.replace(line,""),2500000)
