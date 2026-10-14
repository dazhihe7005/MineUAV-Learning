"""Evidence figures drawn only from real scanned frames and labelled references."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def plot_dynamic_evidence(sampled: dict, report: dict, figures_dir: Path) -> dict[str, float]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(figures_dir)
    output.mkdir(parents=True, exist_ok=True)
    durations: dict[str, float] = {}

    def save(fig, filename: str, start: float) -> None:
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
        durations[filename] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    fig, (xy, height) = plt.subplots(1, 2, figsize=(12, 4.7))
    for name, scans in sampled.items():
        xyz = np.array([scan.body_position_enu_m for scan in scans])
        times = np.array([scan.frame.frame_timestamp for scan in scans])
        xy.plot(xyz[:, 0], xyz[:, 1], marker=".", label=name)
        height.plot(times, xyz[:, 2], marker=".", label=name)
    xy.set(xlabel="Body X, world ENU (m)", ylabel="Body Y, world ENU (m)",
           title="Pre-registered KINEMATIC_POSE_REPLAY tracks")
    height.set(xlabel="MuJoCo simulation time (s)", ylabel="Body Z, world ENU (m)")
    xy.legend(fontsize=7)
    save(fig, "dynamic_trajectory.png", start)

    start = time.perf_counter()
    path = sampled["x_line"]
    picks = sorted({0, len(path) // 2, len(path) - 1})
    fig, axes = plt.subplots(1, len(picks), figsize=(4.5 * len(picks), 4.5), squeeze=False)
    for ax, index in zip(axes[0], picks):
        points = path[index].frame.world_points
        ax.scatter(points[:, 0], points[:, 1], s=.6, alpha=.35, rasterized=True)
        ax.scatter(path[index].frame.sensor_pose[0, 3], path[index].frame.sensor_pose[1, 3],
                   marker="x", c="red")
        ax.set(xlim=(-3.1, 3.2), ylim=(-2.2, 2.2), aspect="equal",
               xlabel="World X (m)", ylabel="World Y (m)",
               title=f"Real rays, t={path[index].frame.frame_timestamp:.1f}s")
    save(fig, "continuous_pointclouds.png", start)

    start = time.perf_counter()
    cloud = sampled["combined"][-1].frame
    local, world = cloud.local_points, cloud.world_points
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(local[:, 0], local[:, 1], s=.6, alpha=.35, rasterized=True)
    axes[1].scatter(world[:, 0], world[:, 1], s=.6, alpha=.35, rasterized=True)
    axes[0].set(xlabel="LiDAR local X (m)", ylabel="LiDAR local Y (m)", title="Local valid ray hits")
    axes[1].set(xlabel="World ENU X (m)", ylabel="World ENU Y (m)", title="Same hits after site transform")
    for ax in axes:
        ax.set_aspect("equal")
    save(fig, "local_vs_world_pointcloud.png", start)

    start = time.perf_counter()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, key, offset, speed in (("x_line", "front", 2.9, .4),
                                     ("y_line", "left", 2.0, .3)):
        scans = sampled[name]
        times = np.array([scan.frame.frame_timestamp for scan in scans])
        measured = [scan.probe_ranges_m[key] for scan in scans]
        ax.plot(times, measured, "o-", label=f"{name}: measured MuJoCo ray")
        ax.plot(times, offset - speed * times, "--", label=f"{name}: analytic plane")
    ax.set(xlabel="MuJoCo simulation time (s)", ylabel="Fixed local ray range (m)",
           title="Wall approach with independent analytic reference")
    ax.legend(fontsize=8)
    save(fig, "wall_distance_vs_time.png", start)

    start = time.perf_counter()
    shaft = sampled["shaft_vertical"]
    fig, (xz, ranges) = plt.subplots(1, 2, figsize=(11, 4.5))
    for index in sorted({0, len(shaft) - 1}):
        points = shaft[index].frame.world_points
        xz.scatter(points[:, 0], points[:, 2], s=.5, alpha=.3,
                   label=f"real hits t={shaft[index].frame.frame_timestamp:.1f}s", rasterized=True)
    xz.set(xlabel="World X (m)", ylabel="World Z (m)",
           title="Uniform shaft: measured world-frame points")
    xz.legend(fontsize=7)
    heights = [scan.body_position_enu_m[2] for scan in shaft]
    for label in ("front", "left", "right"):
        ranges.plot(heights, [scan.probe_ranges_m[label] for scan in shaft], "o-",
                    label=f"{label}: real horizontal ray")
    ranges.set(xlabel="Body world Z (m)", ylabel="Local horizontal range (m)",
               title="Z-invariant walls give zero horizontal range slope")
    ranges.legend(fontsize=7)
    save(fig, "shaft_vertical_geometry.png", start)

    start = time.perf_counter()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for rays, metrics in sorted(report["benchmark"].items(), key=lambda item: int(item[0])):
        ax.plot(metrics["sample_sim_time_s"], metrics["per_frame_total_ms"],
                marker=".", label=f"{rays} rays/frame")
    ax.axhline(100, color="black", linestyle="--", linewidth=.7, label="10Hz scan-only budget")
    ax.set(xlabel="MuJoCo simulation sample time (s)", ylabel="CPU scan + conversion (ms)",
           title="Per-frame measured CPU cost; excludes saving and plotting")
    ax.legend(fontsize=8)
    save(fig, "scan_timing.png", start)
    return durations
