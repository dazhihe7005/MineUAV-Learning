"""Reproducible, static-pose MuJoCo MID360-like CPU ray experiment.

This module never starts PX4, ROS, flight control, or a physical device.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

from .pointcloud import PointCloudFrame
from .scanner import Mid360Scanner, UPSTREAM_COMMIT, PATTERN_SHA256
from .scene import BASE_MODEL, BASE_SHA256, make_scene, set_uav_pose


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()), "mean": float(arr.mean())}


def _available_mib() -> float:
    for row in Path("/proc/meminfo").read_text().splitlines():
        if row.startswith("MemAvailable:"):
            return int(row.split()[1]) / 1024
    raise RuntimeError("MemAvailable unavailable")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".json", delete=False) as file:
        tmp = Path(file.name)
        json.dump(payload, file, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _draw(frame: PointCloudFrame, rotated: PointCloudFrame, output: Path) -> dict[str, float]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output.mkdir(parents=True, exist_ok=True)
    times: dict[str, float] = {}

    def base_axes(title: str):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.add_patch(Rectangle((3, -2), .10, 4, fill=False, edgecolor="black"))
        ax.add_patch(Rectangle((-3, 2), 6, .10, fill=False, edgecolor="black"))
        ax.add_patch(Rectangle((-3, -2.1), 6, .10, fill=False, edgecolor="black"))
        ax.add_patch(Rectangle((1.25, .25), .5, .5, fill=False, edgecolor="red"))
        ax.set(xlim=(-3.3, 3.3), ylim=(-2.4, 2.4), aspect="equal", xlabel="World ENU X (m)",
               ylabel="World ENU Y (m)", title=title)
        return fig, ax

    xyz = frame.world_points
    for name, mask, title in (
        ("wall_pointcloud.png", np.abs(xyz[:, 0] - 3) < .06, "Ray hits on known front wall"),
        ("box_pointcloud.png", (xyz[:, 0] >= 1.24) & (xyz[:, 0] <= 1.76)
         & (xyz[:, 1] >= .24) & (xyz[:, 1] <= .76), "Ray hits on box obstacle"),
    ):
        start = time.perf_counter()
        fig, ax = base_axes(title)
        ax.scatter(xyz[:, 0], xyz[:, 1], s=.35, alpha=.15, c="gray", rasterized=True)
        ax.scatter(xyz[mask, 0], xyz[mask, 1], s=3, alpha=.7, c="tab:blue", rasterized=True)
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
        times[name] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, cloud, title in ((left, frame, "Body yaw 0°"),
                             (right, rotated, "Body yaw +90° (same pattern phase)")):
        points = cloud.world_points
        ax.scatter(points[:, 0], points[:, 1], s=.5, alpha=.45, rasterized=True)
        ax.scatter(cloud.sensor_pose[0, 3], cloud.sensor_pose[1, 3], marker="x", c="red")
        ax.set(xlim=(-3.3, 3.3), ylim=(-2.4, 2.4), aspect="equal", title=title,
               xlabel="World ENU X (m)", ylabel="World ENU Y (m)")
    fig.savefig(output / "rotated_pointcloud_comparison.png", dpi=150)
    plt.close(fig)
    times["rotated_pointcloud_comparison.png"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xyz[:, 0], xyz[:, 2], s=.5, alpha=.4, rasterized=True)
    ax.set(xlim=(-3.3, 3.3), ylim=(-.1, 3.2), xlabel="World ENU X (m)",
           ylabel="World ENU Z (m)", title="Measured vertical ray hits (static scene)")
    fig.savefig(output / "pointcloud_height_profile.png", dpi=150)
    plt.close(fig)
    times["pointcloud_height_profile.png"] = (time.perf_counter() - start) * 1000
    return times


def run_experiment(upstream_root: Path, output_dir: Path, *, repetitions: int = 12,
                   draw: bool = True, downsample_factors: tuple[int, ...] = (24, 6, 1),
                   figures_dir: Path | None = None, report_file: Path | None = None) -> dict:
    if repetitions < 2 or not downsample_factors:
        raise ValueError("at least two repeated scans and one ray scale required")
    if importlib.metadata.version("mujoco") != "3.14.0" or not hasattr(mujoco, "MjModel"):
        raise RuntimeError("real MuJoCo 3.14.0 Python package required")
    if Path(mujoco.__file__).resolve().name != "__init__.py":
        raise RuntimeError("mujoco resolves to unexpected namespace")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    initial_available = _available_mib()
    begin = time.perf_counter()
    model, data = make_scene()
    model_load_ms = (time.perf_counter() - begin) * 1000
    begin = time.perf_counter()
    scanner = Mid360Scanner(model, upstream_root)
    pattern_load_ms = (time.perf_counter() - begin) * 1000
    report: dict = {
        "status": "PASS", "source_identity": "MUJOCO_RAYCAST_APPROX_MID360",
        "sensor_truth_limit": "MID360 pattern approximation; not MID360S hardware validation",
        "environment": {"mujoco_version": importlib.metadata.version("mujoco"),
                        "mujoco_module": str(Path(mujoco.__file__).resolve()),
                        "numpy_version": np.__version__, "scan_frequency_hz_configured": 10,
                        "physics_timestep_s": float(model.opt.timestep),
                        "pose_mode": "static frozen body pose, simulation clock advanced 0.1 s per frame",
                        "worker_count": 1, "px4_runs": 0, "flight_runs": 0},
        "upstream": {"repo_commit": UPSTREAM_COMMIT, "pattern_sha256": PATTERN_SHA256,
                     "backend": "cpu", "pattern_rays": scanner.pattern_samples,
                     "license": "MIT"},
        "extrinsic": {"body_frame": "FLU", "world_frame": "ENU",
                      "T_body_lidar_translation_m": [0.1, 0, 0.1],
                      "T_body_lidar_rpy_rad": [0, 0, 0], "provenance": "SIM_CONFIG"},
        "initialization_ms": {"model_compile": model_load_ms, "pattern_and_adapter": pattern_load_ms},
        "scales": {}, "actual_scans": 0,
    }
    selected_frame: PointCloudFrame | None = None
    for factor in downsample_factors:
        if factor < 1 or scanner.pattern_samples % factor:
            raise ValueError("downsample factor must divide 24000")
        trace_ms: list[float] = []
        conversion_ms: list[float] = []
        total_ms: list[float] = []
        counts: list[int] = []
        for _ in range(repetitions):
            # Static geometry experiment, not a vehicle flight or motion simulation.
            data.time = report["actual_scans"] * 0.1
            start = time.perf_counter()
            frame = scanner.scan_next(data, downsample=factor)
            total_ms.append((time.perf_counter() - start) * 1000)
            trace_ms.append(scanner.last_trace_ms)
            conversion_ms.append(scanner.last_conversion_ms)
            counts.append(frame.valid_count)
            report["actual_scans"] += 1
            selected_frame = frame
        rays = scanner.pattern_samples // factor
        report["scales"][str(rays)] = {
            "downsample_factor": factor, "ray_count": rays, "repetitions": repetitions,
            "valid_count_min": min(counts), "valid_count_max": max(counts),
            "valid_count_mean": float(np.mean(counts)),
            "valid_fraction_mean": float(np.mean(counts) / rays),
            "trace_ms": _stats(trace_ms), "point_conversion_ms": _stats(conversion_ms),
            "frame_total_ms": _stats(total_ms),
            "measured_10hz_scan_budget_pass": _stats(total_ms)["p95"] <= 100.0,
        }
    assert selected_frame is not None
    save_start = time.perf_counter()
    selected_frame.save_npz(output / "local_frame.npz")
    report["save_npz_ms"] = (time.perf_counter() - save_start) * 1000
    report["local_npz_sha256"] = hashlib.sha256((output / "local_frame.npz").read_bytes()).hexdigest()

    # Diagnostic rays use the very same MuJoCo backend, not hand-built point positions.
    model2, data2 = make_scene(include_box=False)
    diag = Mid360Scanner(model2, upstream_root).scan_angles(data2, [0, math.pi/2, -math.pi/2, 0, 0],
                                                             [0, 0, 0, -math.pi/2, math.pi/2])
    expected = np.array([2.9, 2, 2, .8])
    report["geometry_checks"] = {
        "expected_wall_side_ground_range_m": expected.tolist(),
        "actual_range_m": diag.ranges[:4].tolist(),
        "max_abs_range_error_m": float(np.max(np.abs(diag.ranges[:4] - expected))),
        "no_return_raw_range": float(diag.ranges[4]),
        "no_return_valid": bool(diag.valid_mask[4]),
        "no_return_point_all_nan": bool(np.isnan(diag.points_xyz[4]).all()),
        "z_extent_m": float(np.ptp(selected_frame.world_points[:, 2])),
    }
    box_ray = math.atan2(.5, 1.4)
    box = Mid360Scanner(model, upstream_root).scan_angles(data, [box_ray], [0])
    box_expected = 1.15 / math.cos(box_ray)
    report["geometry_checks"]["box_range_error_m"] = float(abs(box.ranges[0] - box_expected))

    q90 = 2**-0.5
    set_uav_pose(model, data, [0, 0, .7], [q90, 0, 0, q90])
    data.time += .1
    rotated_scanner = Mid360Scanner(model, upstream_root)
    rotated = rotated_scanner.scan_next(data, downsample=downsample_factors[-1])
    report["actual_scans"] += 1
    report["coordinate_checks"] = {
        "yaw90_front_range_error_m": abs(float(rotated_scanner.scan_angles(data, [0], [0]).ranges[0]) - 1.9),
        "world_local_roundtrip_max_error_m": float(np.max(np.abs(
            (selected_frame.world_points - selected_frame.sensor_pose[:3, 3])
            @ selected_frame.sensor_pose[:3, :3] - selected_frame.local_points)))
            if selected_frame.valid_count else None,
    }
    report["plot_ms"] = _draw(selected_frame, rotated, figures_dir or output) if draw else {}
    report["memory"] = {"peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                        "initial_mem_available_mib": initial_available,
                        "final_mem_available_mib": _available_mib(), "oom_observed": False}
    report["model_preservation"] = {
        "base_sha256": hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest(),
        "base_sha256_unchanged": hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() == BASE_SHA256,
    }
    report["acceptance"] = {
        "real_mujoco_ray_cloud": "PASS" if selected_frame.valid_count > 0 else "FAIL",
        "known_geometry_distance": "PASS" if (
            report["geometry_checks"]["max_abs_range_error_m"] < 1e-5
            and report["geometry_checks"]["box_range_error_m"] < 1e-5) else "FAIL",
        "local_world_transform": "PASS" if (
            report["coordinate_checks"]["world_local_roundtrip_max_error_m"] < 1e-10
            and report["coordinate_checks"]["yaw90_front_range_error_m"] < 1e-5) else "FAIL",
        "no_return_mask": "PASS" if (
            not report["geometry_checks"]["no_return_valid"]
            and report["geometry_checks"]["no_return_point_all_nan"]) else "FAIL",
        "historical_physics_file_unchanged": "PASS" if
            report["model_preservation"]["base_sha256_unchanged"] else "FAIL",
        "full_24000_ray_static_10hz_budget": (
            "PASS" if report["scales"].get("24000", {}).get("measured_10hz_scan_budget_pass")
            else "NOT TESTED" if "24000" not in report["scales"] else "FAIL"),
        "real_mid360s_fidelity": "NOT TESTED",
        "dynamic_motion_distortion": "NOT TESTED",
        "fast_lio2_compatibility": "NOT TESTED",
        "px4_sitl": "NOT TESTED",
    }
    required = ("real_mujoco_ray_cloud", "known_geometry_distance",
                "local_world_transform", "no_return_mask",
                "historical_physics_file_unchanged", "full_24000_ray_static_10hz_budget")
    required_values = [report["acceptance"][name] for name in required]
    report["status"] = ("FAIL" if "FAIL" in required_values else
                        "NOT TESTED" if "NOT TESTED" in required_values else "PASS")
    _atomic_json(output / "report.json", report)
    if report_file is not None:
        _atomic_json(Path(report_file), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--report-file", type=Path)
    args = parser.parse_args()
    result = run_experiment(args.upstream, args.output, repetitions=args.repetitions,
                            figures_dir=args.figures_dir, report_file=args.report_file)
    print(json.dumps({"status": result["status"], "actual_scans": result["actual_scans"],
                      "scales": result["scales"], "memory": result["memory"]}, indent=2))


if __name__ == "__main__":
    main()
