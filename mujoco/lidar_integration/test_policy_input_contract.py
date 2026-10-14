"""Future input layout is separate from the frozen legacy 7D policies."""

import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1] / "reports"
RAW = ROOT / "mid360s_dynamic_scanning_parts" / "run_02" / "x_line" / "frame_00.npz"
REPORT = ROOT / "mujoco_mid360s_dynamic_scanning.json"


def _fresh_read():
    from lidar_integration.observation import AngularRangeEncoder
    from lidar_integration.async_observation import AsynchronousObservationBuffer

    if not RAW.is_file():
        pytest.skip("local-only real MuJoCo frame unavailable")
    sha = json.loads(REPORT.read_text())["trajectories"]["x_line"]["frames"][0]["raw_npz_sha256"]
    obs = AngularRangeEncoder().encode_npz(RAW, sha)
    cache = AsynchronousObservationBuffer(experiment_id="input")
    cache.push(obs, arrival_time_s=0., experiment_id="input")
    return cache.read(.04, experiment_id="input")


def test_future_policy_layout_keeps_scan_state_target_and_age_separate():
    from lidar_integration.policy_input_contract import MotionState, TargetReference, build_policy_input

    state = MotionState(position_enu_m=np.array([0., 0., .7]),
                        velocity_enu_m_s=np.zeros(3), yaw_enu_rad=0.,
                        sample_time_s=0., clock_domain="MUJOCO_SIM_TIME",
                        source="OFFLINE_TEST_STATE")
    target = TargetReference(position_enu_m=np.array([1., 2., 1.]),
                             yaw_enu_rad=0., source="OFFLINE_TEST_TARGET")
    result = build_policy_input(_fresh_read(), state, target, policy_time_s=.04)
    assert result.lidar_tensor.shape == (3, 12, 24)
    assert result.lidar_age_s == pytest.approx(.04)
    assert np.array_equal(result.motion_state.position_enu_m, [0, 0, .7])
    assert np.array_equal(result.target.position_enu_m, [1, 2, 1])
    assert result.motion_state.source == "OFFLINE_TEST_STATE"
    assert result.lidar_source_frame_identity.startswith("SHA256_NPZ:")
    assert not hasattr(result, "legacy_7d_observation")


def test_future_policy_layout_rejects_stale_scan_and_future_motion_state():
    from lidar_integration.policy_input_contract import MotionState, TargetReference, build_policy_input
    from lidar_integration.async_observation import AsynchronousObservationBuffer

    read = _fresh_read()
    target = TargetReference(np.zeros(3), 0., "OFFLINE_TEST_TARGET")
    future_state = MotionState(np.zeros(3), np.zeros(3), 0., .10,
                               "MUJOCO_SIM_TIME", "OFFLINE_TEST_STATE")
    with pytest.raises(ValueError, match="future"):
        build_policy_input(read, future_state, target, policy_time_s=.04)
    cache = AsynchronousObservationBuffer(experiment_id="missing")
    missing = cache.read(.04, experiment_id="missing")
    current = MotionState(np.zeros(3), np.zeros(3), 0., 0.,
                          "MUJOCO_SIM_TIME", "OFFLINE_TEST_STATE")
    with pytest.raises(ValueError, match="FRESH"):
        build_policy_input(missing, current, target, policy_time_s=.04)


def test_motion_state_source_and_frame_are_explicit_not_auto_converted():
    from lidar_integration.policy_input_contract import MotionState

    with pytest.raises(ValueError, match="source"):
        MotionState(np.zeros(3), np.zeros(3), 0., 0., "MUJOCO_SIM_TIME", "UNKNOWN")
    with pytest.raises(ValueError, match="clock"):
        MotionState(np.zeros(3), np.zeros(3), 0., 0., "PX4_BOOT_TIME",
                    "PX4_ESTIMATED_STATE")
    state = MotionState(np.zeros(3), np.zeros(3), 0., 0.,
                        "MUJOCO_SIM_TIME", "PX4_ESTIMATED_STATE")
    assert state.source == "PX4_ESTIMATED_STATE"  # provenance only, not live PX4 evidence


def test_policy_motion_and_target_snapshots_cannot_change_after_construction():
    from lidar_integration.policy_input_contract import MotionState, TargetReference

    state = MotionState(np.array([1., 2., 3.]), np.zeros(3), 0., 0.,
                        "MUJOCO_SIM_TIME", "OFFLINE_TEST_STATE")
    target = TargetReference(np.array([4., 5., 6.]), 0., "OFFLINE_TEST_TARGET")
    for vector in (state.position_enu_m, state.velocity_enu_m_s,
                   target.position_enu_m):
        with pytest.raises(ValueError):
            vector.flags.writeable = True
