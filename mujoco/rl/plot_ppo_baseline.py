"""Static scientific figures for the fixed MineUAV PPO baseline report."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_ppo_baseline import REPORT_PATH


PLOT_SPECS = (
    ("ppo_training_reward.png", "rollout_ep_rew_mean", "Mean episode reward", "Training reward"),
    ("ppo_training_episode_length.png", "rollout_ep_len_mean", "Mean episode length (policy steps)",
     "Training episode length"),
    ("ppo_eval_success_rate.png", "success_rate", "Success rate", "Fixed 100-waypoint evaluation"),
    ("ppo_eval_final_distance.png", "mean_final_distance_m", "Mean final distance (m)",
     "Fixed 100-waypoint evaluation"),
)


def plot_report(report_path: Path = REPORT_PATH,
                output_dir: Path | None = None) -> list[Path]:
    """Read measured metrics; never infer unobserved training performance."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_dir = output_dir or report_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    history = report["training_history"]
    milestones = report["milestones"]
    created = []
    for index, (filename, field, ylabel, title) in enumerate(PLOT_SPECS):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        if index < 2:
            points = [(row["timesteps"], row[field]) for row in history
                      if row[field] is not None]
            if not points:
                raise ValueError(f"No measured values for {field}")
            axis.plot([point[0] for point in points], [point[1] for point in points],
                      color="#245db5", linewidth=1.7)
        else:
            ordered = sorted(milestones.items(), key=lambda item: item[1]["actual_timesteps"])
            x = [record["actual_timesteps"] for _, record in ordered]
            y = [record["evaluation"][field] for _, record in ordered]
            axis.plot(x, y, color="#087c6c", marker="o", linewidth=1.8)
            axis.set_xticks(x, [f"{label}\n{step:,}" for (label, _), step in zip(ordered, x)])
            axis.set_ylim(bottom=0)
            if field == "success_rate":
                axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_xlabel("Actual updated policy timesteps")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)
    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MineUAV PPO baseline curves")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for path in plot_report(args.report, args.output_dir):
        print(path)
