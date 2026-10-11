"""Physical motor differential must agree with model-derived FRD yaw torque."""
import numpy as np
import pytest
from .test_interfaces import module


def test_actual_motor_differential_separates_px4_request_units_and_physical_nm():
    a=module("analyze_yaw_allocation")
    t=np.arange(31,34,.02);motors=np.tile([.4,.4,.4,.4],(len(t),1))
    motors[t>=32]+=[-.05,.05,-.05,.05]
    yaw=np.array([-.0172,.0172,-.0172,.0172])*40
    r=a.motor_response(t,motors,yaw,32.)
    assert r["maximum_physical_yaw_torque_nm"]==pytest.approx(.1376)
    assert r["first_positive_physical_torque_s"]==pytest.approx(32.,abs=.021)
    assert r["positive_differential_max"]==pytest.approx(.05)
    assert r["bounds_valid"] and r["motor_saturation_samples"]==0


def test_no_motor_response_is_not_a_yaw_pass():
    a=module("analyze_yaw_allocation")
    r=a.motor_response([32,32.02],np.full((2,4),.4),[-.68,.68,-.68,.68],32.)
    assert r["first_positive_physical_torque_s"] is None
    assert r["positive_differential_max"]==0


def test_analysis_rejects_misaligned_or_nonfinite_physical_arrays():
    a=module("analyze_yaw_allocation")
    with pytest.raises(ValueError):a.motor_response([32,31],np.full((2,4),.4),[-.68,.68,-.68,.68],32)
    with pytest.raises(ValueError):a.motor_response([32],[[np.nan]*4],[-.68,.68,-.68,.68],32)


def test_recording_identity_and_physics_binding_are_required():
    a=module("analyze_yaw_allocation");identity={"binary":"verified"};history={"model":"frozen"}
    raw={"px4":identity,"history_hashes_before":history,"history_hashes_after":history,"history_unchanged":True}
    a.verify_recording(raw,identity,history)
    for key in ("px4","history_hashes_before","history_hashes_after","history_unchanged"):
        bad=dict(raw);bad[key]=None
        with pytest.raises(ValueError):a.verify_recording(bad,identity,history)


def test_phase3_cannot_pass_on_missing_allocation_or_saturation_or_gate_fault():
    a=module("analyze_yaw_allocation")
    good={"identity":"PASS","mapping":"PASS","yaw":"PASS","alignment":"PASS"}
    assert a.entry_status(good)=="PASS"
    for value in ("FAIL","BLOCKED","NOT TESTED"):
        assert a.entry_status({**good,"allocator":value})=="BLOCKED"


def test_no_step_is_not_failed_yaw_or_successful_poststep_allocation():
    a=module("analyze_yaw_allocation")
    r=a.maneuver_status(None,False,False,True,0,0,None,True,0)
    assert r["yaw_direction"]==r["yaw_torque_allocated"]==r["no_motor_saturation"]=="NOT TESTED"
    assert r["flight_completion"]==r["heading_alignment"]=="BLOCKED"
    assert r["observed_no_motor_saturation"]=="PASS"
    assert a.entry_status(r)=="BLOCKED"


def test_shutdown_deadline_identifies_query_not_misleading_actuator_error():
    a=module("analyze_yaw_allocation")
    raw={"phase2":{"telemetry_fix":{"fault":"safety query service stopped","last_decision":{
        "wall_s":16.08525,"last_query":[16000000,15.43584],"last_actuator":[16480000,16.08317],
        "topics":{"vehicle_local_position":{"received_wall_s":15.43804}}}}},
        "exception":{"message":"PX4 actuator absolute watchdog expired"}}
    d=a.shutdown_deadline(raw)
    assert d["limiting_deadline"]=="query_service"
    assert d["deadline_wall_s"]==pytest.approx(16.08584)
    assert d["last_check_actuator_age_s"]==pytest.approx(.00208)
    assert d["monitor_fault"]=="safety query service stopped"


def test_prequery_null_gate_is_not_a_recorded_health_sample():
    a=module("analyze_yaw_allocation")
    valid={"telemetry_gate":{"topics":{"vehicle_status":{}}}}
    assert a.topic_rows([{"telemetry_gate":None},{},valid],"vehicle_status")==[valid]
