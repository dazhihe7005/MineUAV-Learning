"""Evidence must reject flags without advancing, accepted aid-source fusion."""
import numpy as np
from pathlib import Path
import json
import pytest

from .test_interfaces import module


def aid(fused=(1, 1, 1, 1), times=(100, 200, 300, 400), rejected=None):
    if rejected is None:
        rejected = (0,) * len(fused)
    return {"timestamp":np.arange(1, len(fused)+1)*500_000,
            "time_last_fuse":np.asarray(times),
            "fused":np.asarray(fused),
            "innovation_rejected":np.asarray(rejected)}


def test_aid_evidence_requires_multiple_actual_advancing_fusions():
    assess = module("analyze_odometry_fusion").assess_aid_source
    assert assess(aid())["pass"] is True
    assert assess(aid(times=(100, 100, 100, 100)))["pass"] is False
    assert assess(aid(fused=(1, 0, 1, 0)))["pass"] is False
    assert assess(aid(rejected=(0, 0, 1, 0)))["pass"] is False
    assert assess(aid(times=(100, 300, 200, 400)))["pass"] is False
    assert assess(aid(fused=(1,1,1,1,1), times=(100,200,300,300,300)))["pass"] is False
    assert assess(aid(), end_usec=2_000_000)["pass"] is False
    assert assess(aid(times=(500_001,1_000_001,1_500_001,2_000_001)))["pass"] is False
    assert assess(None)["status"] == "BLOCKED"


def test_visual_evidence_rejects_stale_or_nonfinite_received_state():
    assess = module("analyze_odometry_fusion").assess_visual_odometry
    data = {"timestamp":np.array([120, 220, 320]),
            "timestamp_sample":np.array([110, 210, 310]),
            "x":np.zeros(3),"y":np.zeros(3),"z":np.zeros(3),
            "vx":np.zeros(3),"vy":np.zeros(3),"vz":np.zeros(3)}
    assert assess(data)["pass"] is True
    assert assess(data, end_usec=1_000_000)["pass"] is False
    assert assess(data, end_usec=300)["pass"] is False
    data["timestamp_sample"] = np.array([110, 110, 310])
    assert assess(data)["pass"] is False
    data["timestamp_sample"] = np.array([110, 210, 310])
    data["vx"][1] = np.nan
    assert assess(data)["pass"] is False


def test_source_receipt_comparison_never_uses_future_ground_truth():
    analysis = module("analyze_odometry_fusion")
    source = [{"sample_time_usec":time,
               "position_ned_m":[float(i), 0., 0.],
               "velocity_frd_m_s":[0., 0., 0.]}
              for i, time in enumerate((1_000_000, 1_020_000, 1_040_000, 1_060_000))]
    visual = {"timestamp":np.array([1_011_000, 1_031_000, 1_051_000]),
              "timestamp_sample":np.array([1_010_000, 1_030_000, 1_050_000]),
              "x":np.array([0., 1., 2.]), "y":np.zeros(3), "z":np.zeros(3),
              "vx":np.zeros(3), "vy":np.zeros(3), "vz":np.zeros(3)}
    result, paired = analysis._comparison(source, visual)
    assert result["paired_samples"] == 3
    assert result["position_ned_max_error_m"] == 0.
    assert paired["source_pos"][:,0].tolist() == [0., 1., 2.]


def test_old_healthy_ekf_flags_do_not_certify_end_of_run_fusion():
    analysis = module("analyze_odometry_fusion")
    flags = {"timestamp":np.array([1_000_000,1_500_000,2_000_000]),
             "cs_ev_pos":np.ones(3, dtype=int)}
    assert analysis._end_flags(flags, ("cs_ev_pos",), end_usec=2_100_000)
    assert not analysis._end_flags(flags, ("cs_ev_pos",), end_usec=5_000_000)


def test_shutdown_publications_cannot_supply_future_acceptance_evidence():
    analysis = module("analyze_odometry_fusion")
    topics = {"flags":{"timestamp":np.array([100,200,300]),
                       "healthy":np.array([0,0,1])}}
    trimmed, excluded = analysis.cut_at_end(topics, 200)
    assert trimmed["flags"]["healthy"].tolist() == [0,0]
    assert excluded == {"flags":1}


def test_disarmed_evidence_requires_nonempty_current_arming_state():
    assess=module("analyze_odometry_fusion").assess_disarmed
    valid={"timestamp":np.array([1_000_000,1_500_000,2_000_000]),
           "arming_state":np.array([1,1,1])}
    assert assess(valid,end_usec=2_100_000)
    for states in (None, np.array([]), np.array([1,1]), np.array([1,2,1])):
        data=dict(valid)
        if states is None:del data["arming_state"]
        else:data["arming_state"]=states
        assert not assess(data,end_usec=2_100_000)
    assert not assess(valid,end_usec=4_000_000)


def test_saved_single_run_partial_evidence_and_protocol_failure_recompute(tmp_path):
    root=Path(__file__).resolve().parents[2]
    runtime=root/"mujoco/reports/px4_odometry_fusion_parts/disarmed_01"
    if not (runtime/"smoke_summary.json").exists():
        pytest.skip("single-run raw evidence is local-only")
    analysis=module("analyze_odometry_fusion")
    report=analysis.analyze(runtime,tmp_path/"report.json",tmp_path/"figures")
    assert report["identity"]["summary_sha256"] == "35877f9117c836f13563530659eb9964d2c9ed33fea987f401c6ce71d5ffb84d"
    assert report["overall_fusion"] == "BLOCKED"
    assert report["overall_protocol"] == "FAIL"
    assert report["status"]["C_px4_visual_odometry"] == "BLOCKED"
    assert report["status"]["D_ev_position_fusion"] == report["status"]["E_ev_velocity_fusion"] == "PASS"
    assert report["status"]["K_authorized_duration"] == "FAIL"
    assert report["evidence"]["source_sent_count"] == 1001
    assert report["evidence"]["odometry_inside_authorized_window"] == 1000
    assert report["identity"]["final_code_sitl_validation"] == "NOT TESTED"
    recorded=json.loads((root/"mujoco/reports/px4_mujoco_odometry_fusion.json").read_text())
    assert recorded["status"] == report["status"]
    assert recorded["evidence"] == report["evidence"]
