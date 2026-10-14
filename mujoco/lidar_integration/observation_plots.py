"""2D evidence plots from measured ray encodings and explicit offline schedules."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .observation import LidarObservation


def plot_observation_evidence(observations: dict[str, list[LidarObservation]],
                              zero_delay: dict, delayed: dict,
                              encode_rows: list[dict], output_dir: Path) -> dict[str, float]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    first = observations["x_line"][0]
    extent = (0, 360, -10, 55)

    def save(name: str, fig) -> None:
        start = time.perf_counter()
        fig.tight_layout()
        fig.savefig(output / name, dpi=145)
        plt.close(fig)
        timings[name] = (time.perf_counter() - start) * 1000

    for name, values, cmap, label, title in (
        ("angular_range.png", first.range_tensor, "viridis", "Normalized nearest range",
         "Actual MuJoCo ray returns: nearest distance (1 = no hit)"),
        ("valid_hit_mask.png", first.hit_mask.astype(float), "Greens", "Valid hit mask",
         "Actual MuJoCo ray returns: bins with valid obstacle hits"),
        ("unscanned_mask.png", (~first.sampled_mask).astype(float), "Reds", "Unscanned mask",
         "Actual scan-pattern bins with no sampled rays"),
    ):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        image = ax.imshow(values, origin="lower", aspect="auto", extent=extent,
                          cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set(xlabel="Local LiDAR azimuth (deg)", ylabel="Local LiDAR elevation (deg)",
               title=title)
        fig.colorbar(image, ax=ax, label=label)
        save(name, fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name, series in observations.items():
        initial = series[0].network_tensor
        change = [float(np.mean(np.abs(obs.network_tensor - initial))) for obs in series]
        ax.plot([obs.frame_timestamp for obs in series], change, marker=".", label=name)
    ax.set(xlabel="MuJoCo sample time (s)", ylabel="Mean |encoded channels - first frame|",
           title="Measured encoding changes along six kinematic pose replays")
    ax.grid(alpha=.25)
    ax.legend(ncol=3, fontsize=8)
    save("dynamic_encoding.png", fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter([obs.frame_timestamp for obs in observations["x_line"]],
               np.zeros(len(observations["x_line"])), marker="|", s=180,
               label="LiDAR source frame (10 Hz)")
    for lane, data, label in ((1, zero_delay, "25 Hz reads: zero arrival delay"),
                              (2, delayed, "25 Hz reads: synthetic +20 ms arrival")):
        records = data["by_trajectory"]["x_line"]
        for row in records:
            if row["status"] == "FRESH":
                ax.plot([row["source_time_s"], row["policy_time_s"]], [lane, lane],
                        color="tab:blue" if lane == 1 else "tab:orange", alpha=.27)
        ax.scatter([row["policy_time_s"] for row in records],
                   np.full(len(records), lane), s=18, label=label)
    ax.set(xlabel="Shared MuJoCo simulation clock (s)", yticks=[0, 1, 2],
           yticklabels=["LiDAR", "zero-delay read", "+20 ms read"],
           title="Causal source-frame selection; horizontal segments show reused old samples")
    ax.set_xlim(-.03, 1.04)
    ax.grid(axis="x", alpha=.2)
    ax.legend(fontsize=8, loc="upper left")
    save("lidar_policy_timeline.png", fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    grouped = {key: [row["encode_ms"] for row in encode_rows if row["trajectory"] == key]
               for key in observations}
    ax.boxplot(list(grouped.values()), showfliers=True)
    ax.set_xticks(range(1, len(grouped) + 1), list(grouped))
    ax.axhline(40, color="crimson", linestyle="--", linewidth=1,
               label="25 Hz period (not a hard real-time certification)")
    ax.set(ylabel="Encoding CPU wall time per actual 24k-ray frame (ms)",
           title="Sequential NPZ load excluded; raycast excluded")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8)
    save("encoding_cpu_time.png", fig)
    return timings
