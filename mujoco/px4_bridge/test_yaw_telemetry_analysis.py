import numpy as np
import pytest
from .test_interfaces import module


def test_complete_step_statistics_use_truth_wrap_and_declared_settling():
    a=module("analyze_yaw_telemetry")
    t=np.arange(32,45.001,.02)
    yaw=np.deg2rad(np.where(t<33,90,100.5));rate=np.where(t<33,.1,.01)
    r=a.response_metrics(t,yaw,rate,32.,True)
    assert r["settling_time_s"]==pytest.approx(1.,abs=.021)
    assert r["overshoot_deg"]==pytest.approx(.5)
    assert r["max_target_error_deg"]==pytest.approx(10)
    assert r["steady_state_error_deg"]==pytest.approx(-.5)
    assert r["peak_body_z_rate_rad_s"]==pytest.approx(.1)


def test_incomplete_censored_response_has_no_steady_state_or_settling_claim():
    a=module("analyze_yaw_telemetry")
    r=a.response_metrics([32,32.02],[np.pi/2,np.pi/2],[0.,0.],32.,False)
    assert r["status"]=="BLOCKED" and r["settling_time_s"] is None
    assert r["steady_state_error_deg"] is None


def test_bad_or_missing_alignment_does_not_pass_acceptance():
    a=module("analyze_yaw_telemetry")
    r=a.response_metrics([32,33,34,35],[np.pi/2]*4,[0]*4,32.,True)
    assert r["status"]=="FAIL" and r["settling_time_s"] is None


def test_source_time_distribution_keeps_duplicates_separate_from_receipt():
    a=module("analyze_yaw_telemetry")
    rows=[{"time_s":32.,"telemetry_gate":{"topics":{"estimator_status_flags":{"source_age_s":1.004}},"query_pending":False,"safe_stop":False}},
          {"time_s":32.02,"telemetry_gate":{"topics":{"estimator_status_flags":{"source_age_s":1.024}},"query_pending":True,"safe_stop":False}}]
    events=[{"kind":"query","clock_usec":33000000,"wall_s":1.},
            {"kind":"receipt","topic":"estimator_status_flags","clock_usec":33000000,"wall_s":1.004,"publication_usec":31996000,"sample_usec":31844000,"duplicate":True}]
    r=a.freshness_summary(rows,events)
    assert r["source_age_s"]["estimator_status_flags"]["max"]==pytest.approx(1.024)
    assert r["duplicate_receipts"]==1
    assert r["query_pending_samples"]==1
    assert r["safe_stop_samples"]==0


def test_pinned_allocator_weak_axis_rule_explains_existing_geometry_without_changing_it():
    a=module("analyze_yaw_telemetry")
    params={}
    for i in range(4):
        params.update({f"CA_ROTOR{i}_PX":.2*(-1 if i<2 else 1),f"CA_ROTOR{i}_PY":.2*(-1 if i%2==0 else 1),
                       f"CA_ROTOR{i}_PZ":0.,f"CA_ROTOR{i}_CT":1.,f"CA_ROTOR{i}_KM":.0172*(-1 if i%2==0 else 1)})
    r=a.allocation_authority(params)
    assert r["yaw_row_removed"] and r["weak_rows"]==[2,3,4]
    assert r["raw_matrix"][2]==pytest.approx([-.0172,.0172,-.0172,.0172])
    assert r["after_source_weak_row_removal"][2]==[0,0,0,0]
