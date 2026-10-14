"""Seven 2D evidence figures from physical Gym steps and actual CPU ray hits."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_navigation_preflight(traces: list[dict], first_cloud: np.ndarray,
                              first_scene, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str, fig) -> None:
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=145)
        plt.close(fig)

    def draw_boxes(ax, scene) -> None:
        for box in scene.obstacle_aabbs:
            x, y = box[0], box[1]
            ax.add_patch(Rectangle((x[0], y[0]), x[1] - x[0], y[1] - y[0],
                                   fill=False, color="crimson", lw=1.8))
        bound = scene.boundary_half_extent_m
        ax.add_patch(Rectangle((-bound, -bound), 2 * bound, 2 * bound,
                               fill=False, color="black", lw=1))
        ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), aspect="equal",
               xlabel="World ENU X (m)", ylabel="World ENU Y (m)")

    fig, axes = plt.subplots(1, len(traces), figsize=(5 * len(traces), 5.2), squeeze=False)
    for ax, trace in zip(axes[0], traces):
        draw_boxes(ax, trace["scene"])
        xyz = trace["position"]
        ax.plot(xyz[:, 0], xyz[:, 1], color="royalblue", label="measured path")
        ax.scatter(trace["target"][0], trace["target"][1], marker="*", s=65,
                   color="goldenrod", label="target")
        ax.set_title(trace["layout"])
        ax.legend(fontsize=8)
    fig.suptitle("Separate physical layouts and measured UAV paths")
    save("obstacle_topdown.png", fig)

    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    draw_boxes(ax, first_scene)
    if len(first_cloud):
        ax.scatter(first_cloud[:, 0], first_cloud[:, 1], s=.3, alpha=.3,
                   rasterized=True, label="actual MuJoCo ray hits")
    ax.legend(loc="upper right", markerscale=12)
    ax.set_title("Physical scene versus actual LiDAR point returns")
    save("real_lidar_pointcloud.png", fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for trace in traces:
        xyz, t = trace["position"], trace["time"]
        axes[0].plot(xyz[:, 0], xyz[:, 1], label=trace["layout"])
        axes[1].plot(t, xyz[:, 2], label=trace["layout"])
    axes[0].set(xlabel="World X (m)", ylabel="World Y (m)", title="Actual XY motion")
    axes[1].set(xlabel="MuJoCo time (s)", ylabel="World Z (m)", title="Actual altitude")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend()
    save("physical_trajectory.png", fig)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    for trace in traces:
        ax.plot(trace["time"], trace["distance"], label=trace["layout"])
    ax.axhline(.10, color="crimson", ls="--", label="success distance threshold")
    ax.set(xlabel="MuJoCo time (s)", ylabel="Target distance (m)",
           title="Measured target distance, including failed episodes")
    ax.grid(alpha=.2)
    ax.legend()
    save("distance_vs_time.png", fig)

    components = ("progress", "action", "brake", "success", "collision",
                  "timeout_or_time_cost", "physical_failure", "sensor_failure")
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(traces))
    positive = np.zeros(len(traces))
    negative = np.zeros(len(traces))
    for component in components:
        vals = np.array([sum(row[component] for row in trace["reward_parts"])
                         for trace in traces])
        bottoms = np.where(vals >= 0, positive, negative)
        ax.bar(xs, vals, bottom=bottoms, label=component)
        positive += np.maximum(vals, 0)
        negative += np.minimum(vals, 0)
    ax.set(xticks=xs, xticklabels=[trace["layout"] for trace in traces],
           ylabel="Cumulative reward contribution", title="Frozen candidate reward breakdown")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(ncol=2, fontsize=8)
    save("reward_breakdown.png", fig)

    selected = traces[0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    source = selected["scan_times"]
    policy = selected["time"]
    ax.scatter(source, np.zeros(len(source)), marker="|", s=110, label="actual LiDAR sample")
    ax.scatter(policy, np.ones(len(policy)), s=7, label="policy observation")
    ax.set(xlabel="Shared MuJoCo simulation time (s)", yticks=[0, 1],
           yticklabels=["10 Hz LiDAR", "25 Hz policy"],
           title=f"Actual event times: {selected['layout']}")
    ax.grid(axis="x", alpha=.2)
    ax.legend()
    save("lidar_policy_timeline.png", fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for trace in traces:
        ax.plot(trace["time"][1:], trace["step_ms"], linewidth=.9,
                label=trace["layout"])
    ax.axhline(40, color="crimson", ls="--", label="25 Hz period (not real-time certification)")
    ax.set(xlabel="MuJoCo time (s)", ylabel="Whole Gym step wall time (ms)",
           title="Physics + PI + due LiDAR scan + encoding")
    ax.grid(alpha=.2)
    ax.legend()
    save("environment_step_time.png", fig)
