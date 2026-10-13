"""Contract tests independent of a ROS or PX4 process."""

import numpy as np
import pytest

from lidar_integration.pointcloud import PointCloudFrame


def frame_with_miss():
    pose = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], float)
    return PointCloudFrame(
        frame_timestamp=0.1, clock_domain="MUJOCO_SIM_TIME", sensor_frame_id="lidar_site",
        points_xyz=np.array([[1., 0, 0], [np.nan, np.nan, np.nan], [0, 2, 0]]),
        ranges=np.array([1., -1., 2.]), valid_mask=np.array([True, False, True]),
        ray_directions=np.array([[1., 0, 0], [0, 0, 1], [0, 1, 0]]),
        sensor_pose=pose, T_body_lidar=np.eye(4), extrinsic_provenance="SIM_CONFIG",
        scan_configuration={"mode": "mid360", "ray_count": 3,
                            "per_point_time_offset": "UNSUPPORTED"},
        source_identity="MUJOCO_RAYCAST_APPROX_MID360")


def test_world_coordinates_rotate_only_valid_real_hits():
    frame = frame_with_miss()
    assert frame.ray_count == 3
    assert frame.valid_count == 2
    np.testing.assert_allclose(frame.world_points, [[1, 3, 3], [-1, 2, 3]], atol=1e-12)
    assert frame.world_points.shape == (2, 3)


def test_npz_round_trip_is_pickle_free_and_preserves_mask(tmp_path):
    frame = frame_with_miss()
    file = tmp_path / "scan.npz"
    frame.save_npz(file)
    with np.load(file, allow_pickle=False) as archive:
        assert set(archive.files) >= {"points_xyz", "ranges", "valid_mask", "sensor_pose"}
    restored = PointCloudFrame.load_npz(file)
    np.testing.assert_array_equal(restored.valid_mask, frame.valid_mask)
    np.testing.assert_allclose(restored.world_points, frame.world_points, atol=0)
    assert restored.source_identity == "MUJOCO_RAYCAST_APPROX_MID360"
    assert restored.scan_configuration["per_point_time_offset"] == "UNSUPPORTED"


@pytest.mark.parametrize("change", ["valid_nan", "invalid_finite", "infinite_range", "bad_pose",
                                         "bad_count", "bad_source"])
def test_malformed_frame_fails_closed(change):
    values = vars(frame_with_miss()).copy()
    values = {name: (value.copy() if isinstance(value, np.ndarray) else value)
              for name, value in values.items()}
    if change == "valid_nan":
        values["points_xyz"][0, 0] = np.nan
    elif change == "invalid_finite":
        values["points_xyz"][1, :] = 0
    elif change == "infinite_range":
        values["ranges"][0] = np.inf
    elif change == "bad_pose":
        values["sensor_pose"][0, 0] = 2
    elif change == "bad_count":
        values["valid_mask"] = values["valid_mask"][:2]
    elif change == "bad_source":
        values["source_identity"] = "REAL_MID360S"
    with pytest.raises(ValueError):
        PointCloudFrame(**values)


def test_subminimum_raw_range_remains_recorded_but_masked():
    values = vars(frame_with_miss()).copy()
    values["ranges"] = np.array([0.5, -1., 2.])
    values["valid_mask"] = np.array([False, False, True])
    values["points_xyz"] = np.array([[np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan], [0., 2., 0.]])
    frame = PointCloudFrame(**values)
    assert frame.ranges[0] == 0.5
    assert frame.valid_count == 1
