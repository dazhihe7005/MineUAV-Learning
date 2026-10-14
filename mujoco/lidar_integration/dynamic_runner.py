"""Measured dynamic-pose MuJoCo ray experiment, without PX4 or IMU synthesis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import resource
import time
from pathlib import Path

import mujoco
import numpy as np

from .runner import _atomic_json, _available_mib, _stats
from .scanner import Mid360Scanner, PATTERN_SHA256, UPSTREAM_COMMIT
from .scene import BASE_MODEL, BASE_SHA256, make_scene, make_shaft_scene
from .trajectory import TRAJECTORIES, TickSchedule, replay_scans


def _front_wall_error(world_points: np.ndarray) -> tuple[float, int]:
    # Side walls are at |y|=2 and the floor at z=0; exclude both analytically.
    selection = ((world_points[:, 0] > 2.5) & (np.abs(world_points[:, 1]) < 1.75)
                 & (world_points[:, 2] > .2) & (world_points[:, 2] < 2.9))
    values = world_points[selection, 0]
    return (float(np.max(np.abs(values - 3.0))) if values.size else math.nan,
            int(values.size))


def _frame_summary(scan, file_name: str, file_sha: str) -> dict:
    frame = scan.frame
    return {
        "time_s": frame.frame_timestamp, "time_usec_scheduled": scan.time_usec,
        "body_position_enu_m": scan.body_position_enu_m.tolist(),
        "body_quaternion_wxyz": scan.body_quaternion_wxyz.tolist(),
        "sensor_pose_world_enu": scan.sensor_pose_world_enu.tolist(),
        "T_body_lidar": frame.T_body_lidar.tolist(),
        "provenance": scan.provenance,
        "ray_count": frame.ray_count, "valid_count": frame.valid_count,
        "invalid_count": frame.ray_count - frame.valid_count,
        "probe_ranges_m": scan.probe_ranges_m,
        "cpu_trace_ms": scan.scan_trace_ms,
        "point_conversion_ms": scan.scan_conversion_ms,
        "scan_total_ms": scan.scan_total_ms,
        "array_bytes": sum(arr.nbytes for arr in (frame.points_xyz, frame.ranges,
                                                  frame.valid_mask, frame.ray_directions,
                                                  frame.sensor_pose, frame.T_body_lidar)),
        "raw_npz_local_only": file_name, "raw_npz_sha256": file_sha,
    }


def run_dynamic_experiment(upstream_root: Path, output_dir: Path, *,
                           figures_dir: Path | None = None, report_file: Path | None = None,
                           frames: int = 11, trajectory_downsample: int = 1,
                           benchmark_factors: tuple[int, ...] = (24, 6, 1),
                           draw: bool = True) -> dict:
    """Six deterministic paths; all sampled rays use fixed upstream CPU backend."""
    schedule = TickSchedule()
    if (isinstance(frames, bool) or not isinstance(frames, int) or not 2 <= frames <= 11
            or not benchmark_factors or any(isinstance(f, bool) or not isinstance(f, int)
                                            or f < 1 or 24000 % f for f in benchmark_factors)
            or isinstance(trajectory_downsample, bool)
            or not isinstance(trajectory_downsample, int)
            or trajectory_downsample < 1 or 24000 % trajectory_downsample):
        raise ValueError("invalid fixed dynamic scan protocol")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("raw output directory is not empty; refusing to overwrite prior evidence")
    if report_file is not None and Path(report_file).exists():
        raise ValueError("report file already exists; refusing to overwrite prior evidence")
    if draw and figures_dir is not None and Path(figures_dir).exists() and any(Path(figures_dir).iterdir()):
        raise ValueError("figure directory is not empty; refusing to overwrite prior evidence")
    output.mkdir(parents=True, exist_ok=True)
    if importlib.metadata.version("mujoco") != "3.14.0" or not hasattr(mujoco, "MjModel"):
        raise RuntimeError("real MuJoCo 3.14.0 required")
    start_available = _available_mib()
    report: dict = {
        "status": "NOT TESTED", "source_identity": "MUJOCO_RAYCAST_APPROX_MID360",
        "motion_provenance": "KINEMATIC_POSE_REPLAY",
        "environment": {"mujoco_version": importlib.metadata.version("mujoco"),
                        "mujoco_module": str(Path(mujoco.__file__).resolve()),
                        "numpy_version": np.__version__, "worker_count": 1,
                        "px4_runs": 0, "flight_runs": 0, "training_runs": 0},
        "upstream": {"commit": UPSTREAM_COMMIT, "mode": "mid360",
                     "pattern_sha256": PATTERN_SHA256, "backend": "cpu", "license": "MIT"},
        "sampling": {"physics_hz": schedule.physics_hz, "lidar_hz": schedule.lidar_hz,
                     "imu_schedule_hz": schedule.imu_hz, "imu_samples_generated": 0,
                     "clock_domain": "MUJOCO_SIM_TIME", "per_point_time_offset": "UNSUPPORTED",
                     "frame_pose_model": "whole frame frozen at sample tick; no motion distortion"},
        "sensor_extrinsic": {"translation_body_flu_m": [.1, 0, .1],
                             "rpy_body_rad": [0, 0, 0], "provenance": "SIM_CONFIG"},
        "trajectories": {}, "benchmark": {}, "actual_pattern_scans": 0,
    }
    sampled: dict[str, list] = {}
    for name in TRAJECTORIES:
        scene_start = time.perf_counter()
        model, data = make_shaft_scene() if name == "shaft_vertical" else make_scene()
        scanner = Mid360Scanner(model, upstream_root)
        init_ms = (time.perf_counter() - scene_start) * 1000
        replay_start = time.perf_counter()
        scans = replay_scans(model, data, scanner, name, frames=frames,
                             downsample=trajectory_downsample)
        replay_wall_ms = (time.perf_counter() - replay_start) * 1000
        sampled[name] = scans
        rows = []
        front_wall_errors = []
        front_wall_counts = []
        for i, scan in enumerate(scans):
            file = output / name / f"frame_{i:02d}.npz"
            scan.frame.save_npz(file)
            sha = hashlib.sha256(file.read_bytes()).hexdigest()
            rows.append(_frame_summary(scan, file.relative_to(output).as_posix(), sha))
            if name != "shaft_vertical":
                error, count = _front_wall_error(scan.frame.world_points)
                front_wall_errors.append(error)
                front_wall_counts.append(count)
        times = np.array([scan.frame.frame_timestamp for scan in scans])
        periods = np.diff(times)
        entry = {
            "frames": rows, "scan_count": len(scans), "scene_and_mode_init_ms": init_ms,
            "replay_wall_ms_excluding_save_plot": replay_wall_ms,
            "end_sim_time_s": float(data.time), "physics_steps": (frames - 1) * schedule.lidar_stride,
            "scan_period_error_max_s": float(np.max(np.abs(periods - .1))),
            "scan_total_ms": _stats([scan.scan_total_ms for scan in scans]),
            "valid_count_min": min(scan.frame.valid_count for scan in scans),
            "valid_count_max": max(scan.frame.valid_count for scan in scans),
            "sensor_extrinsic_delta_max": float(max(np.max(np.abs(scan.frame.T_body_lidar
                                                 - scans[0].frame.T_body_lidar)) for scan in scans)),
            "world_front_wall_hit_count_min": min(front_wall_counts) if front_wall_counts else None,
            "world_front_wall_max_error_m": float(max(front_wall_errors)) if front_wall_errors else None,
        }
        if name == "x_line":
            entry["wall_range_max_error_m"] = float(max(abs(
                scan.probe_ranges_m["front"] - (2.9 - .4 * scan.frame.frame_timestamp))
                for scan in scans))
        if name == "y_line":
            entry["wall_range_max_error_m"] = float(max(abs(
                scan.probe_ranges_m["left"] - (2 - .3 * scan.frame.frame_timestamp))
                for scan in scans))
        if name == "shaft_vertical":
            horizontal = np.array([[scan.probe_ranges_m[key] for key in ("front", "left", "right")]
                                   for scan in scans])
            entry["horizontal_range_spread_m"] = float(np.max(np.ptp(horizontal, axis=0)))
            entry["horizontal_range_expected_m"] = [2.4, 2.5, 2.5]
            entry["geometry_scope"] = "24-sided uniform vertical walls; horizontal-ray Z invariance only"
        report["trajectories"][name] = entry
        report["actual_pattern_scans"] += len(scans)
    for factor in benchmark_factors:
        model, data = make_scene()
        scanner = Mid360Scanner(model, upstream_root)
        scans = replay_scans(model, data, scanner, "combined", frames=frames, downsample=factor)
        rays = scans[0].frame.ray_count
        report["benchmark"][str(rays)] = {
            "ray_count": rays, "downsample_factor": factor, "frames": frames,
            "sample_sim_time_s": [scan.frame.frame_timestamp for scan in scans],
            "per_frame_total_ms": [scan.scan_total_ms for scan in scans],
            "total_ms": _stats([scan.scan_total_ms for scan in scans]),
            "trace_ms": _stats([scan.scan_trace_ms for scan in scans]),
            "conversion_ms": _stats([scan.scan_conversion_ms for scan in scans]),
            "valid_count_min": min(scan.frame.valid_count for scan in scans),
            "valid_count_max": max(scan.frame.valid_count for scan in scans),
            "within_static_10hz_budget": max(scan.scan_total_ms for scan in scans) < 100,
        }
        report["actual_pattern_scans"] += len(scans)
    report["imu_audit"] = {
        "source": "mujoco/px4_bridge/sensor_adapter.py:SensorAdapter.sample",
        "existing_sitl_sample_hz": 250,
        "historical_px4_boot_offset_usec": 1_000_000,
        "historical_clock_source": "mujoco/px4_bridge/diagnostics.py:SimClock",
        "this_experiment_timestamp_usec": "relative simulation microseconds beginning at 0; not PX4 boot time",
        "existing_method": "mj_forward; COM Jacobian and jacDot with qacc/qvel; gravity subtraction; FLU to FRD",
        "shared_sim_clock_schedule": "500Hz integer ticks; IMU schedule every 2, LiDAR every 50 ticks",
        "physics_consistent_imu_from_kinematic_replay": "NOT TESTED",
        "reason": "direct qpos/qvel overwrite has no physically consistent acceleration",
        "lidar_imu_extrinsic_measured": "NOT TESTED",
        "fast_lio2_synchronization": "NOT TESTED",
    }
    report["plot_ms"] = {}
    if draw:
        from .dynamic_plots import plot_dynamic_evidence
        report["plot_ms"] = plot_dynamic_evidence(sampled, report, figures_dir or output / "figures")
    report["memory"] = {"peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                        "initial_mem_available_mib": start_available,
                        "final_mem_available_mib": _available_mib(), "oom_observed": False}
    report["historical_model_sha256"] = hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest()
    report["acceptance"] = {
        "dynamic_scan_24k": "PASS" if trajectory_downsample == 1 else "NOT TESTED",
        "ten_hz_sim_timestamps": "PASS" if all(
            row["scan_period_error_max_s"] < 1e-9 for row in report["trajectories"].values())
            else "FAIL",
        "geometry_numeric": "PASS" if (
            report["trajectories"]["x_line"]["wall_range_max_error_m"] < 1e-5
            and report["trajectories"]["y_line"]["wall_range_max_error_m"] < 1e-5
            and report["trajectories"]["shaft_vertical"]["horizontal_range_spread_m"] < 1e-5
            and all(row["world_front_wall_hit_count_min"] > 0
                    and row["world_front_wall_max_error_m"] < 1e-5
                    for key, row in report["trajectories"].items() if key != "shaft_vertical"))
            else "FAIL",
        "historical_model_unchanged": "PASS" if report["historical_model_sha256"] == BASE_SHA256
            else "FAIL",
        "full_24k_10hz_scan_only_budget": (
            "NOT TESTED" if "24000" not in report["benchmark"] else
            "PASS" if report["benchmark"]["24000"]["within_static_10hz_budget"] else "FAIL"),
        "physics_consistent_imu": "NOT TESTED", "real_mid360s_point_timing": "NOT TESTED",
        "fast_lio2": "NOT TESTED", "px4_sitl": "NOT TESTED",
    }
    required = list(report["acceptance"].values())[:5]
    report["status"] = "FAIL" if "FAIL" in required else (
        "NOT TESTED" if "NOT TESTED" in required else "PASS")
    _atomic_json(output / "summary.json", report)
    if report_file is not None:
        _atomic_json(Path(report_file), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    parser.add_argument("--report-file", required=True, type=Path)
    args = parser.parse_args()
    result = run_dynamic_experiment(args.upstream, args.output,
                                    figures_dir=args.figures_dir, report_file=args.report_file)
    print(json.dumps({"status": result["status"], "pattern_scans": result["actual_pattern_scans"],
                      "benchmark": result["benchmark"], "memory": result["memory"]}, indent=2))


if __name__ == "__main__":
    main()
