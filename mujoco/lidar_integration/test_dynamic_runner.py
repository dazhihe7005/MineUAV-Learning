"""End-to-end dynamic experiment and evidence checks, using real CPU rays."""

import json
import os
from pathlib import Path

import numpy as np
import pytest


UPSTREAM = Path(os.environ["MINEUAV_LIDAR_UPSTREAM"])


def test_dynamic_report_contains_all_paths_real_clock_and_raw_npz(tmp_path):
    from lidar_integration.dynamic_runner import run_dynamic_experiment

    output = tmp_path / "raw"
    report = run_dynamic_experiment(UPSTREAM, output, frames=3,
                                    trajectory_downsample=24,
                                    benchmark_factors=(24,), draw=False)
    assert set(report["trajectories"]) == {
        "x_line", "y_line", "z_triangle", "yaw_spin", "combined", "shaft_vertical"}
    assert report["actual_pattern_scans"] == 6 * 3 + 3
    assert report["sampling"]["physics_hz"] == 500
    assert report["sampling"]["lidar_hz"] == 10
    assert report["sampling"]["imu_schedule_hz"] == 250
    assert report["sampling"]["imu_samples_generated"] == 0
    assert report["imu_audit"]["historical_px4_boot_offset_usec"] == 1_000_000
    assert report["acceptance"]["dynamic_scan_24k"] == "NOT TESTED"
    assert report["status"] == "NOT TESTED"
    times = [entry["time_s"] for entry in report["trajectories"]["x_line"]["frames"]]
    np.testing.assert_allclose(times, [0, .1, .2], atol=1e-10)
    assert report["trajectories"]["x_line"]["end_sim_time_s"] == pytest.approx(.2, abs=1e-10)
    assert report["trajectories"]["x_line"]["wall_range_max_error_m"] < 1e-5
    assert report["trajectories"]["shaft_vertical"]["horizontal_range_spread_m"] < 1e-5
    assert report["trajectories"]["x_line"]["world_front_wall_max_error_m"] < 1e-5
    assert len(list(output.glob("*/frame_*.npz"))) == 18
    assert json.loads((output / "summary.json").read_text())["actual_pattern_scans"] == 21


def test_dynamic_runner_refuses_stale_output_and_invalid_protocol(tmp_path):
    from lidar_integration.dynamic_runner import run_dynamic_experiment

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "old.txt").write_text("old")
    with pytest.raises(ValueError):
        run_dynamic_experiment(UPSTREAM, occupied, frames=3, draw=False)
    assert (occupied / "old.txt").read_text() == "old"
    with pytest.raises(ValueError):
        run_dynamic_experiment(UPSTREAM, tmp_path / "x", frames=1,
                               benchmark_factors=(0,), draw=False)


def test_dynamic_runner_never_overwrites_formal_report_or_figures(tmp_path):
    from lidar_integration.dynamic_runner import run_dynamic_experiment

    report_file = tmp_path / "formal.json"
    report_file.write_text("historic report")
    with pytest.raises(ValueError, match="report"):
        run_dynamic_experiment(UPSTREAM, tmp_path / "new_raw", report_file=report_file,
                               frames=2, trajectory_downsample=24,
                               benchmark_factors=(24,), draw=False)
    assert report_file.read_text() == "historic report"
    assert not (tmp_path / "new_raw").exists()

    figures = tmp_path / "formal_figures"
    figures.mkdir()
    (figures / "historic.png").write_bytes(b"historic figure")
    with pytest.raises(ValueError, match="figure"):
        run_dynamic_experiment(UPSTREAM, tmp_path / "new_raw", figures_dir=figures,
                               frames=2, trajectory_downsample=24,
                               benchmark_factors=(24,), draw=True)
    assert (figures / "historic.png").read_bytes() == b"historic figure"
    assert not (tmp_path / "new_raw").exists()


def test_dynamic_figures_are_generated_from_scanned_frames(tmp_path):
    from lidar_integration.dynamic_runner import run_dynamic_experiment

    figures = tmp_path / "figures"
    report = run_dynamic_experiment(UPSTREAM, tmp_path / "raw", figures_dir=figures,
                                    frames=2, trajectory_downsample=24,
                                    benchmark_factors=(24,), draw=True)
    required = {"dynamic_trajectory.png", "continuous_pointclouds.png",
                "local_vs_world_pointcloud.png", "wall_distance_vs_time.png",
                "shaft_vertical_geometry.png", "scan_timing.png"}
    assert set(report["plot_ms"]) == required
    assert all((figures / name).read_bytes().startswith(b"\x89PNG") for name in required)
