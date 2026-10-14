"""Angular policy observations from the previous real MuJoCo ray frames."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lidar_integration.pointcloud import PointCloudFrame


ROOT = Path(__file__).resolve().parents[1] / "reports"
RAW = ROOT / "mid360s_dynamic_scanning_parts" / "run_02"
REPORT = ROOT / "mujoco_mid360s_dynamic_scanning.json"


def _historical(name: str, index: int) -> tuple[PointCloudFrame, str, Path]:
    file = RAW / name / f"frame_{index:02d}.npz"
    if not file.is_file():
        pytest.skip("local-only Phase3E-L3 raw ray frames unavailable")
    report = json.loads(REPORT.read_text())
    row = report["trajectories"][name]["frames"][index]
    expected = row["raw_npz_sha256"]
    assert hashlib.sha256(file.read_bytes()).hexdigest() == expected
    return PointCloudFrame.load_npz(file), expected, file


def _meters(normalized: float) -> float:
    return .1 + 4.9 * float(normalized)


def test_actual_scan_encodes_finite_three_channel_tensor_and_distinct_empty_masks():
    from lidar_integration.observation import AngularRangeEncoder

    frame, sha, path = _historical("x_line", 0)
    obs = AngularRangeEncoder().encode_npz(path, sha)
    assert obs.network_tensor.shape == (3, 12, 24)
    assert obs.network_tensor.dtype == np.float32
    assert np.isfinite(obs.network_tensor).all()
    assert obs.sample_count.sum() == 24000
    assert obs.hit_count.sum() == frame.valid_count
    assert np.array_equal(obs.network_tensor[1].astype(bool), obs.sampled_mask)
    assert np.array_equal(obs.network_tensor[2].astype(bool), obs.hit_mask)
    assert obs.source_frame_identity == f"SHA256_NPZ:{sha}"
    assert obs.frame_timestamp == 0
    assert obs.clock_domain == "MUJOCO_SIM_TIME"
    assert obs.encoding_configuration["source_pattern_sha256"] == frame.scan_configuration["pattern_sha256"]
    assert np.count_nonzero(~obs.sampled_mask) > 0
    assert np.count_nonzero(obs.sampled_mask & ~obs.hit_mask) > 0
    assert np.all(obs.range_tensor[~obs.hit_mask] == 1.0)
    assert not np.any((obs.range_tensor == 0) & ~obs.hit_mask)
    assert 1.15 < _meters(obs.range_tensor[1, 0]) < 1.25  # actual box occludes front wall


def test_actual_motion_changes_local_ranges_without_inventing_shaft_height_signal():
    from lidar_integration.observation import AngularRangeEncoder

    encoder = AngularRangeEncoder()
    def take(name, index):
        frame, _, _ = _historical(name, index)
        return encoder.encode_frame(frame)

    x0, x1 = take("x_line", 0), take("x_line", 10)
    y0, y1 = take("y_line", 0), take("y_line", 10)
    yaw0, yaw1 = take("yaw_spin", 0), take("yaw_spin", 10)
    shaft0, shaft1 = take("shaft_vertical", 0), take("shaft_vertical", 10)
    assert _meters(x0.range_tensor[5, 0]) - _meters(x1.range_tensor[5, 0]) > .35
    assert _meters(y0.range_tensor[1, 6]) - _meters(y1.range_tensor[1, 6]) > .25
    assert _meters(y1.range_tensor[1, 18]) - _meters(y0.range_tensor[1, 18]) > .25
    assert _meters(yaw1.range_tensor[1, 0]) - _meters(yaw0.range_tensor[1, 0]) > .65
    assert abs(_meters(shaft1.range_tensor[1, 0]) - _meters(shaft0.range_tensor[1, 0])) < .01
    assert shaft0.sample_count[1, 0] > 0 and shaft1.sample_count[1, 0] > 0


def test_real_z_up_down_scan_changes_encoding_and_returns_near_initial():
    from lidar_integration.observation import AngularRangeEncoder

    encoder = AngularRangeEncoder()
    frames = [encoder.encode_frame(_historical("z_triangle", i)[0]) for i in (0, 5, 10)]
    mid_change = float(np.mean(np.abs(frames[1].network_tensor - frames[0].network_tensor)))
    end_change = float(np.mean(np.abs(frames[2].network_tensor - frames[0].network_tensor)))
    assert mid_change > .01
    assert end_change < .005
    assert frames[0].frame_timestamp < frames[1].frame_timestamp < frames[2].frame_timestamp


def test_real_mujoco_roll_pitch_pose_changes_local_encoding_but_not_world_wall():
    import os
    from lidar_integration.observation import AngularRangeEncoder
    from lidar_integration.scanner import Mid360Scanner
    from lidar_integration.scene import make_scene, set_uav_pose

    root = os.environ.get("MINEUAV_LIDAR_UPSTREAM")
    if not root:
        pytest.skip("pinned upstream MuJoCo-LiDAR checkout unavailable")
    encoder = AngularRangeEncoder()
    frames = []
    for roll_deg, pitch_deg in ((0., 0.), (10., -7.)):
        model, data = make_scene()
        roll, pitch = np.deg2rad([roll_deg, pitch_deg])
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        q = np.array([cr * cp, sr * cp, cr * sp, sr * sp])
        set_uav_pose(model, data, [0, 0, .7], q)
        frame = Mid360Scanner(model, root).scan_next(data, downsample=24)
        frames.append((frame, encoder.encode_frame(frame)))
    assert frames[0][0].ray_count == frames[1][0].ray_count == 1000
    assert float(np.mean(np.abs(frames[1][1].network_tensor - frames[0][1].network_tensor))) > .01
    for frame, _ in frames:
        xyz = frame.world_points
        front = xyz[(xyz[:, 0] > 2.5) & (abs(xyz[:, 1]) < 1.75)
                    & (xyz[:, 2] > .2) & (xyz[:, 2] < 2.9)]
        assert len(front) > 0
        assert float(np.max(abs(front[:, 0] - 3.0))) < 1e-5


def test_encoder_rejects_bad_file_identity_and_mutated_ray_alignment():
    from lidar_integration.observation import AngularRangeEncoder

    frame, sha, path = _historical("x_line", 0)
    encoder = AngularRangeEncoder()
    with pytest.raises(ValueError, match="SHA256"):
        encoder.encode_npz(path, "0" * 64)
    frame.ray_directions[np.flatnonzero(frame.valid_mask)[0]] = [0, 1, 0]
    with pytest.raises(ValueError, match="ray|point|direction"):
        encoder.encode_frame(frame)
    assert len(sha) == 64


def test_npz_decoder_consumes_the_same_bytes_that_were_hashed(monkeypatch):
    import io
    from lidar_integration.observation import AngularRangeEncoder

    _, sha, path = _historical("x_line", 0)
    original = PointCloudFrame.load_npz
    decoded_from = []

    def checked_decode(source):
        decoded_from.append(source)
        return original(source)

    monkeypatch.setattr(PointCloudFrame, "load_npz", checked_decode)
    AngularRangeEncoder().encode_npz(path, sha)
    assert len(decoded_from) == 1
    assert isinstance(decoded_from[0], io.BytesIO)
    assert hashlib.sha256(decoded_from[0].getvalue()).hexdigest() == sha


def test_encoder_rejects_invalid_ranges_config_and_nonfinite_inputs():
    from lidar_integration.observation import AngularRangeEncoder

    frame, _, _ = _historical("x_line", 0)
    encoder = AngularRangeEncoder()
    frame.ranges[0] = np.inf
    with pytest.raises(ValueError):
        encoder.encode_frame(frame)
    frame, _, _ = _historical("x_line", 0)
    frame.scan_configuration["pattern_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pattern"):
        encoder.encode_frame(frame)


def test_encoder_rejects_incomplete_npz_without_fabricating_missing_rays(tmp_path):
    from lidar_integration.observation import AngularRangeEncoder

    incomplete = tmp_path / "incomplete.npz"
    np.savez_compressed(incomplete, ranges=np.array([1.0]))
    sha = hashlib.sha256(incomplete.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="incomplete.*NPZ"):
        AngularRangeEncoder().encode_npz(incomplete, sha)


def test_world_pose_metadata_is_not_a_hidden_encoder_input():
    from lidar_integration.observation import AngularRangeEncoder

    frame, _, _ = _historical("x_line", 0)
    encoder = AngularRangeEncoder()
    before = encoder.encode_frame(frame)
    frame.sensor_pose[:3, 3] += [20., -30., 5.]
    after = encoder.encode_frame(frame)
    np.testing.assert_array_equal(before.network_tensor, after.network_tensor)
    assert before.source_frame_identity == after.source_frame_identity


def test_angular_edges_wrap_positive_x_and_reject_outside_vertical_envelope():
    from lidar_integration.observation import AngularRangeEncoder

    encoder = AngularRangeEncoder()
    directions = np.array([[1., 0., 0.], [0., -1., 0.],
                           [np.cos(np.deg2rad(55)), 0., np.sin(np.deg2rad(55))]])
    rows, cols = encoder.bin_directions(directions)
    assert (rows.tolist(), cols.tolist()) == ([1, 1, 11], [0, 18, 0])
    with pytest.raises(ValueError, match="elevation"):
        encoder.bin_directions(np.array([[0., 0., -1.]]))


def test_range_normalization_endpoints_on_explicit_unit_fixture():
    from lidar_integration.observation import AngularRangeEncoder

    frame, _, _ = _historical("x_line", 0)
    encoder = AngularRangeEncoder()
    rows, cols = encoder.bin_directions(frame.ray_directions)
    selected = (rows == 1) & (cols == 0) & frame.valid_mask
    assert selected.sum() > 0
    for physical_range_m, expected in ((.1, 0.), (5., 1.)):
        ranges = frame.ranges.copy()
        points = frame.points_xyz.copy()
        ranges[selected] = physical_range_m
        points[selected] = frame.ray_directions[selected] * physical_range_m
        # Mathematical boundary fixture only; these edited rays are not a MuJoCo scan.
        unit_fixture = replace(frame, ranges=ranges, points_xyz=points)
        encoded = encoder.encode_frame(unit_fixture)
        assert encoded.range_tensor[1, 0] == pytest.approx(expected, abs=1e-7)
        assert encoded.hit_mask[1, 0]
