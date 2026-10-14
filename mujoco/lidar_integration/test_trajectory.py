"""Dynamic replay contracts using actual MuJoCo time and CPU rays."""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from lidar_integration.scene import make_scene
from lidar_integration.scanner import Mid360Scanner


UPSTREAM = Path(os.environ["MINEUAV_LIDAR_UPSTREAM"])


def test_integer_tick_schedule_is_shared_500hz_clock():
    from lidar_integration.trajectory import TickSchedule

    schedule = TickSchedule()
    assert schedule.physics_dt_s == 0.002
    assert schedule.lidar_stride == 50
    assert schedule.imu_stride == 2
    assert [tick for tick in range(101) if schedule.is_lidar_tick(tick)] == [0, 50, 100]
    assert [tick for tick in range(5) if schedule.is_imu_tick(tick)] == [0, 2, 4]
    assert schedule.time_usec(50) == 100_000
    with pytest.raises(ValueError):
        TickSchedule(physics_hz=500, lidar_hz=12)
    with pytest.raises(ValueError):
        schedule.time_usec(-1)


@pytest.mark.parametrize("name,t,xyz,yaw", [
    ("x_line", 1.0, [.4, 0, .7], 0),
    ("y_line", 1.0, [0, .3, .7], 0),
    ("z_triangle", .5, [0, 0, .9], 0),
    ("z_triangle", 1.0, [0, 0, .7], 0),
    ("yaw_spin", 1.0, [0, 0, .7], math.pi / 2),
    ("combined", 1.0, [.25, .1, .8], math.pi / 3),
    ("shaft_vertical", 1.0, [0, 0, 1.7], 0),
])
def test_predeclared_trajectory_pose(name, t, xyz, yaw):
    from lidar_integration.trajectory import trajectory_pose

    position, quat = trajectory_pose(name, t)
    np.testing.assert_allclose(position, xyz, atol=1e-12)
    np.testing.assert_allclose(quat, [math.cos(yaw/2), 0, 0, math.sin(yaw/2)], atol=1e-12)


def test_replay_clock_pose_and_real_wall_distance_are_consistent():
    from lidar_integration.trajectory import replay_scans

    model, data = make_scene(include_box=False)
    scanner = Mid360Scanner(model, UPSTREAM)
    scans = replay_scans(model, data, scanner, "x_line", frames=4, downsample=24)
    assert len(scans) == 4
    np.testing.assert_allclose([entry.frame.frame_timestamp for entry in scans],
                               [0, .1, .2, .3], atol=1e-10)
    np.testing.assert_array_equal([entry.time_usec for entry in scans],
                                  [0, 100_000, 200_000, 300_000])
    np.testing.assert_allclose([entry.body_position_enu_m[0] for entry in scans],
                               [0, .04, .08, .12], atol=1e-10)
    np.testing.assert_allclose([entry.probe_ranges_m["front"] for entry in scans],
                               [2.9, 2.86, 2.82, 2.78], atol=1e-6)
    assert all(entry.frame.ray_count == 1000 for entry in scans)
    assert all(entry.frame.scan_configuration["per_point_time_offset"] == "UNSUPPORTED"
               for entry in scans)
    assert all(entry.provenance == "KINEMATIC_POSE_REPLAY" for entry in scans)
    assert all(entry.scan_total_ms >= entry.scan_trace_ms >= 0 for entry in scans)
    assert not np.array_equal(scans[0].frame.ranges, scans[1].frame.ranges)
    assert data.time == pytest.approx(.3, abs=1e-10)


def test_replay_yaw_updates_world_pose_and_static_wall_hit():
    from lidar_integration.trajectory import replay_scans

    model, data = make_scene(include_box=False)
    scans = replay_scans(model, data, Mid360Scanner(model, UPSTREAM), "yaw_spin",
                         frames=11, downsample=24)
    np.testing.assert_allclose(scans[-1].frame.sensor_pose[:3, :3] @ [1, 0, 0],
                               [0, 1, 0], atol=1e-9)
    assert scans[0].probe_ranges_m["front"] == pytest.approx(2.9, abs=1e-6)
    assert scans[-1].probe_ranges_m["front"] == pytest.approx(1.9, abs=1e-6)
    assert scans[-1].frame.sensor_pose[0, 3] == pytest.approx(0, abs=1e-9)
    assert scans[-1].frame.sensor_pose[1, 3] == pytest.approx(.1, abs=1e-8)


@pytest.mark.parametrize("name,frames,downsample", [
    ("unknown", 2, 24), ("x_line", 0, 24), ("x_line", 12, 24),
    ("x_line", 2, 0),
])
def test_replay_rejects_unplanned_or_invalid_requests(name, frames, downsample):
    from lidar_integration.trajectory import replay_scans

    model, data = make_scene()
    with pytest.raises(ValueError):
        replay_scans(model, data, Mid360Scanner(model, UPSTREAM), name, frames, downsample)
