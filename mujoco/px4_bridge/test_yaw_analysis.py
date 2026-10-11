import numpy as np
import pytest
from .test_interfaces import module


def test_circular_alignment_uses_sample_clock_and_no_extrapolation():
    a=module("analyze_yaw")
    values=a.interpolate([1,2],np.deg2rad([179,-179]),[.9,1,1.5,2,2.1],circular=True)
    assert np.isnan(values[[0,4]]).all()
    assert abs(abs(values[2])-np.pi)<1e-12
    assert a.simulation_time([1004000,1008000]).tolist()==pytest.approx([.004,.008])
    with pytest.raises(ValueError):a.interpolate([2,1],[0,1],[1.5])


def test_final_alignment_excludes_ground_startup_heading_transient():
    a=module("analyze_yaw")
    t=np.arange(6.)
    flags={"cs_in_air":np.array([0,0,1,1,1,1]),"cs_mag_aligned_in_flight":np.array([0,0,0,1,1,1])}
    health=a.alignment_times(t,np.array([1,0,0,0,1,1]),t,flags)
    assert health["first_inflight_mag_alignment_s"]==3.
    assert health["first_final_heading_good_s"]==4.
    assert health["final_heading_good"] is True


def test_incomplete_yaw_run_never_claims_response_pass():
    a=module("analyze_yaw")
    assert a.response_status(False,32.,None)=="BLOCKED"
    assert a.response_status(True,None,None)=="NOT TESTED"
    assert a.response_status(True,32.,None)=="FAIL"
    assert a.response_status(True,32.,1.5)=="PASS"


def test_missing_signal_cannot_fabricate_heading_alignment():
    a=module("analyze_yaw")
    result=a.alignment_times([1,2],[True,True],[],{})
    assert result["first_final_heading_good_s"] is None
    assert result["final_heading_good"] is False


def test_yaw_from_px4_quaternion_is_ned_not_enu():
    a=module("analyze_yaw")
    assert a.px4_yaw([[1,0,0,0],[2**-.5,0,0,2**-.5]]).tolist()==pytest.approx([0,np.pi/2])


def test_offline_acceptance_never_claims_pass_from_bad_inputs():
    a=module("analyze_yaw")
    good={"mean":.003}
    assert a.bias_status(good)=="PASS"
    assert a.bias_status({"mean":3.4})=="FAIL"
    assert a.bias_status(None)=="NOT TESTED"
    with pytest.raises(ValueError):a.verify_preserved({"model":{"unchanged":False}},[])
    with pytest.raises(ValueError):a.verify_preserved({"model":{"unchanged":True}},[{"history_unchanged":False}])


def test_desired_yaw_publications_use_causal_hold_not_future_ramp():
    a=module("analyze_yaw")
    out=a.hold_interpolate([31.9,32.05,32.15],np.deg2rad([90,100,100]),[31.8,32,32.05,32.1,32.16])
    assert np.isnan(out[[0,4]]).all()
    assert np.degrees(out[1:4]).tolist()==pytest.approx([90,100,100])
