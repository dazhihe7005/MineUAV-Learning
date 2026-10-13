"""Flight gates remain intact; new mode never authorizes any maneuver."""
import copy
import pytest

from .protocol import DisarmedFreshness,GnssDeniedProtocol
from ..telemetry_freshness import TelemetryFreshness
from ..yaw_validation import heading_topics_ready


def snapshots(clock=7_000_000):
    out={"vehicle_local_position":{"timestamp":clock,"timestamp_sample":clock,"heading":1.57,
         "xy_valid":True,"z_valid":True,"v_xy_valid":True,"v_z_valid":True,"heading_good_for_control":False},
         "vehicle_status":{"timestamp":clock,"arming_state":1,"nav_state":0,
                           "pre_flight_checks_pass":False,"failsafe":False},
         "estimator_status_flags":{"timestamp":clock,"timestamp_sample":clock}}
    flags=out["estimator_status_flags"]
    for key in DisarmedFreshness.REQUIRED["estimator_status_flags"]:flags[key]=False
    flags.update(cs_tilt_align=True,cs_yaw_align=True,cs_ev_pos=True,cs_ev_vel=True,
                 cs_baro_hgt=True,cs_mag_hdg=True)
    return out


def deliver(m,topics,clock=7_000_000,wall=0.):
    m.begin_query(clock,wall);m.observe_topics(topics,clock,wall,finish=True);m.actuator(clock,wall)
    return m.decision(clock,wall,require_flight=False)


def test_disarmed_health_is_not_flight_authority():
    topics=snapshots();m=DisarmedFreshness();r=deliver(m,topics)
    assert not r["safe_stop"] and not r["allow_new_maneuver"]
    assert not heading_topics_ready(topics,7_000_000)
    assert not m.decision(7_000_000,0.,require_flight=True)["allow_new_maneuver"]
    assert m.fault is not None


@pytest.mark.parametrize("topic,key,value",[
    ("estimator_status_flags","cs_gnss_pos",True),("estimator_status_flags","cs_gnss_vel",True),
    ("estimator_status_flags","cs_gps_hgt",True),("estimator_status_flags","cs_gnss_yaw",True),
    ("estimator_status_flags","cs_mag_fault",True),("estimator_status_flags","cs_baro_fault",True),
    ("estimator_status_flags","fs_bad_hdg",True),("estimator_status_flags","cs_ev_pos",False),
    ("estimator_status_flags","cs_ev_vel",False),("estimator_status_flags","cs_baro_hgt",False),
    ("vehicle_status","arming_state",2),("vehicle_status","failsafe",True),
    ("vehicle_local_position","xy_valid",False),("vehicle_local_position","v_xy_valid",False)])
def test_bad_health_latches_without_grace(topic,key,value):
    m=DisarmedFreshness();bad=snapshots();bad[topic][key]=value
    assert deliver(m,bad)["safe_stop"]
    assert deliver(m,snapshots(7_004_000),7_004_000,.004)["safe_stop"]


def test_complete_but_missing_gnss_height_flag_fails_closed():
    topics=snapshots();del topics["estimator_status_flags"]["cs_gps_hgt"]
    assert deliver(DisarmedFreshness(),topics)["safe_stop"]


def test_no_magnetic_heading_source_fails_closed():
    topics=snapshots();topics["estimator_status_flags"]["cs_mag_hdg"]=False
    assert deliver(DisarmedFreshness(),topics)["safe_stop"]


@pytest.mark.parametrize("fault",["actuator","query","source","rollback"])
def test_original_watchdogs_and_time_order_remain(fault):
    m=DisarmedFreshness();deliver(m,snapshots())
    if fault=="actuator":r=m.decision(7_004_000,.252,require_flight=False)
    elif fault=="query":
        m.begin_query(7_004_000,.004);m.actuator(7_008_000,.008)
        r=m.decision(7_152_000,.156,require_flight=False)
    elif fault=="rollback":r=m.decision(6_996_000,.004,require_flight=False)
    else:
        r=None
        for clock in range(7_004_000,7_656_001,4000):
            wall=(clock-7_000_000)/1e6
            if clock==7_500_000:m.begin_query(clock,wall);m.observe_topics(snapshots(),clock,wall,finish=True)
            m.actuator(clock,wall);r=m.decision(clock,wall,require_flight=False)
            if r["safe_stop"]:break
    assert r["safe_stop"] and not r["allow_new_maneuver"]


def test_protocol_refuses_activation_setpoint_and_old_proof():
    p=GnssDeniedProtocol({"field_ned_gauss":[.2,0.,.4]},[.017,-.017,.017,-.017])
    assert p.active is False
    with pytest.raises(ValueError):p.active=True
    with pytest.raises(ValueError):p.setpoint(None,7_000_000)
    with pytest.raises(ValueError):p.validate_proof({"kind":"disarmed_communication_smoke","complete":True})
    assert isinstance(p.monitor,DisarmedFreshness)
    assert TelemetryFreshness.REQUIRED["estimator_status_flags"]==(
        "cs_tilt_align","cs_yaw_align","cs_mag_aligned_in_flight","cs_in_air","cs_gnss_pos",
        "cs_mag_hdg","cs_mag_3d","cs_mag_fault","cs_baro_fault","cs_gnss_fault","cs_gnss_yaw_fault","cs_gnss_hgt_fault")
