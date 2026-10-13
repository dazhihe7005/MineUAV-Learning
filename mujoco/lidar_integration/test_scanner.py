"""Boundary tests against the pinned real MuJoCo-LiDAR CPU backend."""

import hashlib
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mujoco_lidar import MjLidarWrapper

from lidar_integration.scene import make_scene, set_uav_pose


UPSTREAM = Path(os.environ["MINEUAV_LIDAR_UPSTREAM"])


def scanner_api():
    from lidar_integration.scanner import Mid360Scanner
    return Mid360Scanner


def test_pinned_cpu_source_and_pattern_identity():
    assert hashlib.sha256((UPSTREAM / "src/mujoco_lidar/scan_mode/mid360.npy").read_bytes()).hexdigest() == (
        "9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3")
    model, _ = make_scene()
    scanner = scanner_api()(model, UPSTREAM)
    assert scanner.backend == "cpu"
    assert scanner.pattern_samples == 24000
    with pytest.raises((ValueError, RuntimeError)):
        scanner_api()(model, UPSTREAM.parent / "nonexistent")


def test_known_wall_floor_side_and_no_return_distances():
    model, data = make_scene(include_box=False)
    scanner = scanner_api()(model, UPSTREAM)
    frame = scanner.scan_angles(data, [0, math.pi/2, -math.pi/2, 0, 0],
                                [0, 0, 0, -math.pi/2, math.pi/2])
    np.testing.assert_allclose(frame.ranges[:4], [2.9, 2.0, 2.0, 0.8], atol=1e-6)
    assert frame.ranges[4] == -1
    np.testing.assert_array_equal(frame.valid_mask, [True, True, True, True, False])
    assert np.isnan(frame.points_xyz[4]).all()
    np.testing.assert_allclose(frame.world_points[0], [3, 0, .8], atol=1e-6)


def test_box_is_real_occluder_and_does_not_hide_wall_beyond():
    model, data = make_scene(include_box=True)
    theta = math.atan2(.5, 1.4)
    frame = scanner_api()(model, UPSTREAM).scan_angles(data, [theta], [0])
    expected = 1.15 / math.cos(theta)  # box near x face is 1.25; site x is 0.10
    assert frame.ranges[0] == pytest.approx(expected, abs=1e-6)
    no_box_model, no_box_data = make_scene(include_box=False)
    farther = scanner_api()(no_box_model, UPSTREAM).scan_angles(no_box_data, [theta], [0])
    assert farther.ranges[0] > frame.ranges[0] + 1


def test_body_exclusion_removes_self_marker_but_not_external_wall():
    model, data = make_scene(include_box=False, include_self_marker=True)
    raw = MjLidarWrapper(model, "lidar_site", backend="cpu", cutoff_dist=5,
                         args={"bodyexclude": -1})
    raw.trace_rays(data, np.array([0.]), np.array([0.]))
    frame = scanner_api()(model, UPSTREAM).scan_angles(data, [0], [0])
    assert 0 < raw.get_distances()[0] < .3
    assert frame.ranges[0] == pytest.approx(2.9, abs=1e-6)


def test_yaw_pitch_roll_and_translation_rotate_true_rays():
    model, data = make_scene(include_box=False)
    scanner = scanner_api()(model, UPSTREAM)
    q90 = 2**-0.5
    set_uav_pose(model, data, [0, 0, .7], [q90, 0, 0, q90])
    yaw = scanner.scan_angles(data, [0], [0])
    assert yaw.ranges[0] == pytest.approx(1.9, abs=1e-6)
    np.testing.assert_allclose(yaw.world_points[0], [0, 2, .8], atol=1e-6)
    set_uav_pose(model, data, [0, 0, .7], [q90, 0, q90, 0])
    pitch = scanner.scan_angles(data, [0], [0])
    assert pitch.ranges[0] == pytest.approx(.6, abs=1e-6)
    set_uav_pose(model, data, [0, 0, .7], [q90, q90, 0, 0])
    roll = scanner.scan_angles(data, [0], [math.pi/2])
    assert roll.ranges[0] == pytest.approx(1.9, abs=1e-6)
    set_uav_pose(model, data, [1, 0, 1], [1, 0, 0, 0])
    shifted = scanner.scan_angles(data, [0], [0])
    assert shifted.ranges[0] == pytest.approx(1.9, abs=1e-6)
    np.testing.assert_allclose(shifted.world_points[0], [3, 0, 1.1], atol=1e-6)


def test_min_and_max_range_filter_without_fabricating_returns():
    model, data = make_scene(include_box=False)
    near_filtered = scanner_api()(model, UPSTREAM, min_range_m=3.0).scan_angles(data, [0], [0])
    assert near_filtered.ranges[0] == pytest.approx(2.9)
    assert not near_filtered.valid_mask[0]
    assert near_filtered.world_points.shape == (0, 3)
    far_cutoff = scanner_api()(model, UPSTREAM, max_range_m=2.0).scan_angles(data, [0], [0])
    # The pinned CPU backend reports the raw hit even beyond cutoff_dist;
    # the adapter retains that raw measurement but excludes it from the cloud.
    assert far_cutoff.ranges[0] == pytest.approx(2.9)
    assert not far_cutoff.valid_mask[0]
    assert np.isnan(far_cutoff.points_xyz[0]).all()


def test_determinism_and_next_frame_time_monotonicity():
    model, data = make_scene()
    scanner = scanner_api()(model, UPSTREAM)
    first = scanner.scan_angles(data, [0, math.pi/2], [0, 0])
    second = scanner.scan_angles(data, [0, math.pi/2], [0, 0])
    np.testing.assert_array_equal(first.ranges, second.ranges)
    generated = scanner.scan_next(data, downsample=24)
    assert generated.ray_count == 1000
    with pytest.raises(ValueError):
        scanner.scan_next(data, downsample=24)
    mujoco.mj_step(model, data)
    generated2 = scanner.scan_next(data, downsample=24)
    assert generated2.frame_timestamp > generated.frame_timestamp


def test_scan_timing_separates_backend_and_point_conversion():
    model, data = make_scene()
    scanner = scanner_api()(model, UPSTREAM)
    scanner.scan_next(data, downsample=24)
    assert scanner.last_trace_ms >= 0
    assert scanner.last_conversion_ms >= 0


@pytest.mark.parametrize("theta,phi", [([float("nan")], [0]), ([0], [float("inf")]),
                                         ([0, 1], [0]), ([0], [2])])
def test_invalid_ray_angles_rejected(theta, phi):
    model, data = make_scene()
    with pytest.raises(ValueError):
        scanner_api()(model, UPSTREAM).scan_angles(data, theta, phi)


@pytest.mark.parametrize("min_r,max_r", [(0, 5), (2, 1), (float("nan"), 4)])
def test_invalid_range_configuration_rejected(min_r, max_r):
    model, _ = make_scene()
    with pytest.raises(ValueError):
        scanner_api()(model, UPSTREAM, min_range_m=min_r, max_range_m=max_r)
