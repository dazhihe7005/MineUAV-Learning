"""Terminal state/command summaries for inference-only attribution experiments."""

import math

import numpy as np

from diagnose_ppo_braking import first_entry


def terminal_window_stats(steps: list[dict], target, window_s: float) -> dict:
    """Summarize pre-command policy states in the last available `window_s`."""
    goal = np.asarray(target, dtype=float)
    if (not steps or goal.shape != (3,) or not np.isfinite(goal).all()
            or not math.isfinite(window_s) or window_s <= 0):
        raise ValueError("Need finite target, nonempty steps and positive window")
    terminal_time = float(steps[-1]["post_time_s"])
    selected = [step for step in steps
                if float(step["time_s"]) >= terminal_time - window_s - 1e-10]
    if not selected:
        raise ValueError("No policy step falls in terminal window")
    for step in selected:
        if not np.array_equal(np.asarray(step["target_m"], dtype=float), goal):
            raise ValueError("Terminal step target differs from episode target")

    def mean_vector(key: str) -> list[float]:
        vectors = np.asarray([step[key] for step in selected], dtype=float)
        if vectors.shape != (len(selected), 3) or not np.isfinite(vectors).all():
            raise ValueError(f"Terminal {key} must contain finite 3-vectors")
        return np.mean(vectors, axis=0).tolist()

    error = np.asarray([step["position_error_m"] for step in selected], dtype=float)
    velocity = np.asarray([step["velocity_m_s"] for step in selected], dtype=float)
    command = np.asarray([step["velocity_command_m_s"] for step in selected], dtype=float)
    for matrix in (error, velocity, command):
        if matrix.shape != (len(selected), 3) or not np.isfinite(matrix).all():
            raise ValueError("Terminal vector data must be finite 3-vectors")
    return {
        "requested_window_s": float(window_s),
        "observed_span_s": terminal_time - float(selected[0]["time_s"]),
        "policy_steps": len(selected),
        "mean_position_error_xyz_m": mean_vector("position_error_m"),
        "mean_velocity_xyz_m_s": mean_vector("velocity_m_s"),
        "mean_command_velocity_xyz_m_s": mean_vector("velocity_command_m_s"),
        "mean_position_error_norm_m": float(np.mean(np.linalg.norm(error, axis=1))),
        "mean_speed_m_s": float(np.mean(np.linalg.norm(velocity, axis=1))),
        "mean_command_speed_m_s": float(np.mean(np.linalg.norm(command, axis=1))),
    }


def episode_attribution_summary(trace: dict) -> dict:
    """Compact one episode while retaining success, crossing and terminal bias."""
    steps = trace["steps"]
    target = trace["target_m"]
    entry = first_entry(steps, 0.2) if steps else None
    success = trace["termination_reason"] == "success"
    return {
        "seed": trace["seed"],
        "target_position_m": target,
        "success": success,
        "termination_reason": trace["termination_reason"],
        "episode_length": trace["episode_length"],
        "completion_time_s": float(steps[-1]["post_time_s"]) if success else None,
        "final_distance_m": trace["final_distance_m"],
        "crossed_after_first_0_2_m": bool(
            entry and entry["crossed_target_plane_before_episode_end"]),
        "entered_0_2_m": entry is not None,
        "tail_5_s": terminal_window_stats(steps, target, 5.0),
        "tail_1_s": terminal_window_stats(steps, target, 1.0),
    }
