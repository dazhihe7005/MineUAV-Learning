"""Offline adversarial evidence tests; no PX4 process or flight commands."""
import copy
import numpy as np
import pytest

from .odometry_reverification_evidence import (
    assess_boundary, assess_direct, compare_payloads, aggregate_checks)


def boundary(seconds=19.999999999999794):
    return dict(complete=True,simulation_seconds=seconds,
                final_physical_state={"time_s":seconds},
                messages_sent={"HIL_SENSOR":5000,"HIL_GPS":200,"ODOMETRY":1000})


def visual():
    stamps=np.arange(1_000_000,21_000_000,20_000)
    data={"timestamp":stamps.copy(),"timestamp_sample":stamps.copy(),
          "pose_frame":np.ones(1000),"velocity_frame":np.full(1000,3),
          "quality":np.full(1000,100),"reset_counter":np.zeros(1000)}
    for prefix,size in (("position",3),("velocity",3),("angular_velocity",3),
                        ("q",4),("position_variance",3),
                        ("orientation_variance",3),("velocity_variance",3)):
        for i in range(size):
            data[f"{prefix}[{i}]"]=np.full(1000,.01 if "variance" in prefix else 1. if prefix=="q" and i==0 else 0.)
    return data


def test_exact_twenty_second_boundary():
    assert assess_boundary(boundary())["status"]=="PASS"


@pytest.mark.parametrize("seconds",[20.004,21.,20.0000001,19.996,float("nan")])
def test_overshoot_or_missing_cycle_never_passes(seconds):
    assert assess_boundary(boundary(seconds))["status"]=="FAIL"


@pytest.mark.parametrize("topic",["HIL_SENSOR","HIL_GPS","ODOMETRY"])
def test_counts_cannot_be_repaired_by_truncation(topic):
    raw=boundary();raw["messages_sent"][topic]+=1
    assert assess_boundary(raw)["status"]=="FAIL"


def test_complete_direct_evidence():
    assert assess_direct(visual(),end_usec=21_000_000)["status"]=="PASS"


@pytest.mark.parametrize("field",["timestamp_sample","position[0]","q[0]","velocity_frame","velocity_variance[2]"])
def test_missing_direct_fields_are_blocked(field):
    data=visual();del data[field]
    assert assess_direct(data,end_usec=21_000_000)["status"]=="BLOCKED"


@pytest.mark.parametrize("change",["nan","quaternion","frame","gap","duplicate","truncated","covariance","future"])
def test_bad_or_truncated_receipt_never_passes(change):
    data=visual()
    if change=="nan":data["position[0]"][5]=np.nan
    if change=="quaternion":data["q[0]"][5]=2.
    if change=="frame":data["velocity_frame"][5]=1
    if change=="gap":data["timestamp_sample"][500:]+=300_000;data["timestamp"][500:]+=300_000
    if change=="duplicate":data["timestamp_sample"][5]=data["timestamp_sample"][4]
    if change=="truncated":data={k:v[:500] for k,v in data.items()}
    if change=="covariance":data["orientation_variance[0]"][5]=-.01
    if change=="future":data["timestamp_sample"][5]+=1
    assert assess_direct(data,end_usec=21_000_000)["status"]=="FAIL"


def sent_for(data):
    return [{"sample_time_usec":int(t),"message_fields":dict(
        time_usec=int(t),x=0.,y=0.,z=0.,vx=0.,vy=0.,vz=0.,
        rollspeed=0.,pitchspeed=0.,yawspeed=0.,q=[1.,0.,0.,0.],
        pose_covariance=[.01 if i in (0,6,11,15,18,20) else 0. for i in range(21)],
        velocity_covariance=[.01 if i in (0,6,11,15,18,20) else 0. for i in range(21)],
        reset_counter=0,quality=100,frame_id=1,child_frame_id=12)}
        for t in data["timestamp_sample"]]


def test_direct_payload_comparison_and_clock_semantics():
    data=visual();result,_=compare_payloads(sent_for(data),data)
    assert result["status"]=="PASS"
    assert result["paired_samples"]==1000
    assert result["host_transport_latency"]=="NOT TESTED"


def test_payload_mismatch_not_hidden_by_static_position():
    data=visual();sent=sent_for(data);sent[5]["message_fields"]["yawspeed"]=.1
    assert compare_payloads(sent,data)[0]["status"]=="FAIL"


def test_no_future_source_pairing():
    data=visual();sent=sent_for(data);sent[0]["sample_time_usec"]+=1
    assert compare_payloads(sent,data)[0]["status"]!="PASS"


@pytest.mark.parametrize("missing",["C_direct_receipt","D_ev_position","E_ev_velocity","K_log_integrity"])
def test_missing_necessary_evidence_blocks_overall(missing):
    checks={name:"PASS" for name in ("C_direct_receipt","D_ev_position","E_ev_velocity","K_log_integrity")}
    checks[missing]="BLOCKED"
    assert aggregate_checks(checks)=="BLOCKED"


def test_flags_without_advancing_fusion_fail():
    from .analyze_odometry_fusion import assess_aid_source
    data={"timestamp":np.array([20_900_000,20_920_000,20_940_000]),
          "time_last_fuse":np.full(3,20_900_000),"fused":np.ones(3),
          "innovation_rejected":np.zeros(3)}
    assert assess_aid_source(data,end_usec=21_000_000)["status"]=="FAIL"


def test_omitted_acceptance_item_never_means_complete():
    assert aggregate_checks({"A_producer":"PASS"})=="BLOCKED"


def test_all_acceptance_items_required_for_pass():
    from .odometry_reverification_evidence import CHECK_NAMES
    checks={name:"PASS" for name in CHECK_NAMES}
    assert aggregate_checks(checks)=="PASS"
    checks["G_duration"]="FAIL"
    assert aggregate_checks(checks)=="FAIL"


def test_offline_real_physics_and_send_schedule():
    import mujoco
    from . import simulation_runner as sim
    from .odometry_fusion_runner import SMOKE_DURATION_S
    from .odometry_producer import GroundTruthOdometryProducer
    from .sensor_adapter import SensorAdapter
    from .transport import load_mavlink
    from .diagnostics import SimClock
    model=mujoco.MjModel.from_xml_path(str(sim.MODEL));data=mujoco.MjData(model)
    data.qpos[2]=.051;mujoco.mj_forward(model,data)
    sensor=SensorAdapter(model);producer=GroundTruthOdometryProducer();clock=SimClock()
    mavlink=load_mavlink("/home/keyousuke/.local/lib/python3.10/site-packages")
    cycles=gps=odometry=0
    while data.time<SMOKE_DURATION_S:
        sample=sensor.sample(data,clock.time_usec)
        cycles+=1;gps+=int((clock.time_usec-1_000_000)%100_000==0)
        odometry+=int(producer.tick(sample,mavlink) is not None)
        for _ in range(2):mujoco.mj_step(model,data)
        clock.advance(2)
    assert (cycles,gps,odometry)==(5000,200,1000)
    assert data.time<=20. and abs(data.time-20.)<1e-9
    assert clock.time_usec==21_000_000
