"""Read-only, paired scripted-vs-Reward-V2 PPO braking diagnosis.

Each row is the state immediately *before* applying the row's command.
Consequently active_braking means the commanded velocity opposes the actual
velocity at decision time, not after the dynamics have already responded.
"""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np


ACTIVE_BRAKING_MIN_SPEED_M_S = 0.1
DECELERATION_MIN_INWARD_SPEED_M_S = 0.15
DECELERATION_COMMAND_MARGIN_M_S = 0.1
CROSSING_HYSTERESIS_M = 0.02
TARGET_DIRECTION_EPSILON_M = 1e-9
DISTANCE_BINS = (
    ("ge_0_5", 0.5, math.inf),
    ("0_3_to_0_5", 0.3, 0.5),
    ("0_2_to_0_3", 0.2, 0.3),
    ("0_1_to_0_2", 0.1, 0.2),
    ("lt_0_1", 0.0, 0.1),
)


def make_step_record(time_s, position, target, velocity, action,
                     velocity_command, yaw_rate_command) -> dict:
    """Derive target-relative velocities from a pre-command policy state."""
    vectors = [np.asarray(value, dtype=float) for value in
               (position, target, velocity, action, velocity_command)]
    p, goal, v, normalized_action, v_cmd = vectors
    if ([value.shape for value in vectors] != [(3,), (3,), (3,), (4,), (3,)]
            or not all(np.isfinite(value).all() for value in vectors)
            or not math.isfinite(float(time_s))
            or not math.isfinite(float(yaw_rate_command))):
        raise ValueError("Policy state, action and command must be finite 3/4-vectors")
    error = goal - p
    distance = float(np.linalg.norm(error))
    speed = float(np.linalg.norm(v))
    command_norm = float(np.linalg.norm(v_cmd))
    if distance > TARGET_DIRECTION_EPSILON_M:
        direction = error / distance
        radial = float(np.dot(v, direction))
        cmd_radial = float(np.dot(v_cmd, direction))
        tangent = float(np.linalg.norm(v - radial * direction))
        cmd_tangent = float(np.linalg.norm(v_cmd - cmd_radial * direction))
    else:
        radial = cmd_radial = 0.0
        tangent, cmd_tangent = speed, command_norm
    velocity_dot_command = float(np.dot(v, v_cmd))
    return {
        "time_s": float(time_s), "position_m": p.tolist(),
        "target_m": goal.tolist(), "position_error_m": error.tolist(),
        "distance_m": distance, "velocity_m_s": v.tolist(),
        "speed_m_s": speed, "action": normalized_action.tolist(),
        "velocity_command_m_s": v_cmd.tolist(),
        "yaw_rate_command_rad_s": float(yaw_rate_command),
        "command_norm_m_s": command_norm, "v_radial_m_s": radial,
        "v_tangent_m_s": tangent, "v_cmd_radial_m_s": cmd_radial,
        "v_cmd_tangent_m_s": cmd_tangent,
        "velocity_dot_command_m2_s2": velocity_dot_command,
        "active_braking": bool(speed > ACTIVE_BRAKING_MIN_SPEED_M_S
                               and velocity_dot_command < 0),
        "deceleration_requested": bool(
            radial > DECELERATION_MIN_INWARD_SPEED_M_S
            and cmd_radial < radial - DECELERATION_COMMAND_MARGIN_M_S),
        "speed_error_m_s": float(np.linalg.norm(v_cmd - v)),
    }


def crosses_target_plane(steps: list[dict], start: int, window_s: float) -> bool:
    """Detect signed-error crossing of the plane normal to entry direction.

    The approach axis is the target direction at `start`; the signed error
    must progress from positive to <= -2 cm within the time window. A local
    distance minimum that remains on the approach side is not a crossing.
    """
    if not steps or not 0 <= start < len(steps) or window_s < 0:
        raise ValueError("Need steps, valid start index and nonnegative window")
    entry_error = np.asarray(steps[start]["position_error_m"], dtype=float)
    entry_distance = float(np.linalg.norm(entry_error))
    if entry_distance <= TARGET_DIRECTION_EPSILON_M:
        return False
    axis = entry_error / entry_distance
    end_time = steps[start]["time_s"] + window_s + 1e-10
    had_positive_projection = False
    for step in steps[start:]:
        if step["time_s"] > end_time:
            break
        signed_error = float(np.dot(step["position_error_m"], axis))
        if signed_error > 0:
            had_positive_projection = True
        if had_positive_projection and signed_error <= -CROSSING_HYSTERESIS_M:
            return True
    return False


def summarize_distance_bins(steps: list[dict]) -> dict:
    """Pooled policy-step means in five mutually exclusive distance regions."""
    measures = {
        "mean_actual_speed_m_s": "speed_m_s",
        "mean_abs_v_radial_m_s": "v_radial_m_s",
        "mean_v_radial_m_s": "v_radial_m_s",
        "mean_v_tangent_m_s": "v_tangent_m_s",
        "mean_command_norm_m_s": "command_norm_m_s",
        "mean_v_cmd_radial_m_s": "v_cmd_radial_m_s",
        "mean_v_cmd_tangent_m_s": "v_cmd_tangent_m_s",
        "mean_speed_error_m_s": "speed_error_m_s",
    }
    result = {}
    for label, lower, upper in DISTANCE_BINS:
        selected = [step for step in steps if lower <= step["distance_m"] < upper]
        stats = {"steps": len(selected)}
        for output_key, input_key in measures.items():
            values = [abs(step[input_key]) if output_key == "mean_abs_v_radial_m_s"
                      else step[input_key] for step in selected]
            stats[output_key] = float(np.mean(values)) if values else None
        stats["active_braking_fraction"] = (
            sum(step["active_braking"] for step in selected) / len(selected)
            if selected else None)
        stats["deceleration_requested_fraction"] = (
            sum(step["deceleration_requested"] for step in selected) / len(selected)
            if selected else None)
        result[label] = stats
    return result


def first_entry(steps: list[dict], threshold: float) -> dict | None:
    """First near-target sample plus available 0.5-second post-entry window."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    index = next((i for i, step in enumerate(steps)
                  if step["distance_m"] < threshold), None)
    if index is None:
        return None
    first = steps[index]
    end_time = first["time_s"] + 0.5 + 1e-10
    follow = [step for step in steps[index:] if step["time_s"] <= end_time]
    return {
        "step_index": index, "time_s": first["time_s"],
        "distance_m": first["distance_m"],
        "speed_m_s": first["speed_m_s"],
        "v_radial_m_s": first["v_radial_m_s"],
        "v_tangent_m_s": first["v_tangent_m_s"],
        "command_norm_m_s": first["command_norm_m_s"],
        "v_cmd_radial_m_s": first["v_cmd_radial_m_s"],
        "v_cmd_tangent_m_s": first["v_cmd_tangent_m_s"],
        "active_braking_at_entry": first["active_braking"],
        "minimum_speed_next_0_5_s_m_s": min(step["speed_m_s"] for step in follow),
        "maximum_speed_next_0_5_s_m_s": max(step["speed_m_s"] for step in follow),
        "crossed_target_plane_next_0_5_s": crosses_target_plane(steps, index, 0.5),
        "crossed_target_plane_before_episode_end": crosses_target_plane(
            steps, index, steps[-1]["time_s"] - first["time_s"]),
        "active_braking_next_0_5_s": any(step["active_braking"] for step in follow),
        "deceleration_requested_next_0_5_s": any(
            step["deceleration_requested"] for step in follow),
    }


def evaluate_episode(env, policy, seed: int) -> dict:
    """Evaluate one policy without changing training/environment internals."""
    observation, _ = env.reset(seed=seed)
    target = np.asarray(env.target_position, dtype=float).copy()
    steps = []
    reward_sum = 0.0
    for step_index in range(env.max_episode_steps):
        action = np.asarray(policy(observation), dtype=float)
        if (action.shape != (4,) or not np.isfinite(action).all()
                or np.any(action < -1) or np.any(action > 1)):
            raise ValueError("Policy returned an invalid normalized action")
        pre_time = float(env.data.time)
        pre_position = np.asarray(env.data.qpos[:3], dtype=float).copy()
        pre_velocity = np.asarray(env.data.qvel[:3], dtype=float).copy()
        observation, reward, terminated, truncated, info = env.step(action)
        if not math.isfinite(float(reward)):
            raise ValueError("Non-finite evaluation reward")
        reward_sum += float(reward)
        record = make_step_record(
            pre_time, pre_position, target, pre_velocity, action,
            info["velocity_command_m_s"], info["yaw_rate_command_rad_s"])
        record["policy_step"] = step_index
        record["post_time_s"] = float(env.data.time)
        post_position = np.asarray(env.data.qpos[:3], dtype=float)
        record["post_position_m"] = (
            post_position.tolist() if np.isfinite(post_position).all() else None)
        record["post_distance_m"] = float(info["distance_m"])
        record["post_speed_m_s"] = (
            float(info["speed_m_s"]) if math.isfinite(float(info["speed_m_s"])) else None)
        record["success_streak"] = int(info["success_streak"])
        steps.append(record)
        if terminated or truncated:
            return {
                "seed": seed, "target_m": target.tolist(), "steps": steps,
                "episode_length": len(steps), "episode_reward": reward_sum,
                "final_distance_m": float(info["distance_m"]),
                "final_speed_m_s": record["post_speed_m_s"],
                "termination_reason": info["termination_reason"],
            }
    raise RuntimeError("Evaluation episode exceeded environment time limit")


def evaluate_paired(policies: dict, env_factory, seeds, progress_callback=None) -> dict:
    """Run both policies in independent headless envs on identical seeds."""
    seed_list = list(seeds)
    if not seed_list or len(policies) != 2:
        raise ValueError("Need exactly two policies and at least one seed")
    result = {}
    for name, policy in policies.items():
        env = env_factory()
        try:
            result[name] = []
            for index, seed in enumerate(seed_list, 1):
                result[name].append(evaluate_episode(env, policy, seed))
                if progress_callback is not None:
                    progress_callback(name, index, len(seed_list))
        finally:
            env.close()
    names = list(result)
    for left, right in zip(result[names[0]], result[names[1]]):
        if left["seed"] != right["seed"] or not np.allclose(
                left["target_m"], right["target_m"], rtol=0, atol=0):
            raise AssertionError("Policies were not evaluated on identical waypoints")
    return result


def _mean(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _step_statistics(steps: list[dict]) -> dict:
    """Policy-step-weighted means, never episode-weighted means."""
    return {
        "steps": len(steps),
        "mean_actual_speed_m_s": _mean(s["speed_m_s"] for s in steps),
        "mean_abs_v_radial_m_s": _mean(abs(s["v_radial_m_s"]) for s in steps),
        "mean_v_radial_m_s": _mean(s["v_radial_m_s"] for s in steps),
        "mean_v_tangent_m_s": _mean(s["v_tangent_m_s"] for s in steps),
        "mean_command_norm_m_s": _mean(s["command_norm_m_s"] for s in steps),
        "mean_v_cmd_radial_m_s": _mean(s["v_cmd_radial_m_s"] for s in steps),
        "mean_v_cmd_tangent_m_s": _mean(s["v_cmd_tangent_m_s"] for s in steps),
        "active_braking_fraction": _mean(float(s["active_braking"]) for s in steps),
        "deceleration_requested_fraction": _mean(
            float(s["deceleration_requested"]) for s in steps),
    }


def _first_approach_braking_onsets(steps: list[dict]) -> dict:
    """Look from first meaningful inbound motion until initial outbound motion."""
    start = next((i for i, step in enumerate(steps)
                  if step["v_radial_m_s"] > DECELERATION_MIN_INWARD_SPEED_M_S), None)
    onset = {"deceleration_request_distance_m": None,
             "reverse_command_distance_m": None}
    if start is None:
        return onset
    inbound_seen = False
    for step in steps[start:]:
        inbound_seen |= step["v_radial_m_s"] > DECELERATION_MIN_INWARD_SPEED_M_S
        if inbound_seen and step["v_radial_m_s"] < -0.05:
            break
        if onset["deceleration_request_distance_m"] is None and step["deceleration_requested"]:
            onset["deceleration_request_distance_m"] = step["distance_m"]
        if onset["reverse_command_distance_m"] is None and step["active_braking"]:
            onset["reverse_command_distance_m"] = step["distance_m"]
    return onset


def summarize_policy(episodes: list[dict]) -> dict:
    if not episodes:
        raise ValueError("Need at least one evaluated episode")
    steps = [step for episode in episodes for step in episode["steps"]]
    if not steps:
        raise ValueError("Evaluated episodes must contain policy steps")
    episode_summaries = []
    for episode in episodes:
        trace = episode["steps"]
        analysis = {
            "first_entry_lt_0_5": first_entry(trace, 0.5),
            "first_entry_lt_0_2": first_entry(trace, 0.2),
            "first_entry_lt_0_1": first_entry(trace, 0.1),
            "first_approach_braking_onsets": _first_approach_braking_onsets(trace),
        }
        episode_summaries.append({
            **{key: value for key, value in episode.items() if key != "steps"},
            "minimum_sampled_distance_m": min(s["distance_m"] for s in trace),
            "analysis": analysis,
        })
    near_target = {
        "lt_0_5": _step_statistics([s for s in steps if s["distance_m"] < 0.5]),
        "lt_0_2": _step_statistics([s for s in steps if s["distance_m"] < 0.2]),
        "lt_0_1": _step_statistics([s for s in steps if s["distance_m"] < 0.1]),
    }
    entries = {}
    for label in ("lt_0_5", "lt_0_2", "lt_0_1"):
        selected = [episode["analysis"][f"first_entry_{label}"]
                    for episode in episode_summaries
                    if episode["analysis"][f"first_entry_{label}"] is not None]
        entries[label] = {
            "episode_count": len(selected),
            "episode_fraction": len(selected) / len(episodes),
            "mean_entry_speed_m_s": _mean(e["speed_m_s"] for e in selected),
            "mean_entry_v_radial_m_s": _mean(e["v_radial_m_s"] for e in selected),
            "mean_entry_v_tangent_m_s": _mean(e["v_tangent_m_s"] for e in selected),
            "mean_entry_command_norm_m_s": _mean(e["command_norm_m_s"] for e in selected),
            "mean_entry_v_cmd_radial_m_s": _mean(e["v_cmd_radial_m_s"] for e in selected),
            "mean_entry_v_cmd_tangent_m_s": _mean(e["v_cmd_tangent_m_s"] for e in selected),
            "active_braking_at_entry_episode_fraction": _mean(
                float(e["active_braking_at_entry"]) for e in selected),
            "mean_minimum_speed_next_0_5_s_m_s": _mean(
                e["minimum_speed_next_0_5_s_m_s"] for e in selected),
            "mean_maximum_speed_next_0_5_s_m_s": _mean(
                e["maximum_speed_next_0_5_s_m_s"] for e in selected),
            "crossing_next_0_5_s_episode_fraction_of_entries": _mean(
                float(e["crossed_target_plane_next_0_5_s"]) for e in selected),
            "crossing_before_episode_end_fraction_of_entries": _mean(
                float(e["crossed_target_plane_before_episode_end"]) for e in selected),
            "active_braking_next_0_5_s_episode_fraction_of_entries": _mean(
                float(e["active_braking_next_0_5_s"]) for e in selected),
            "deceleration_requested_next_0_5_s_episode_fraction_of_entries": _mean(
                float(e["deceleration_requested_next_0_5_s"]) for e in selected),
        }
    onset_rows = [ep["analysis"]["first_approach_braking_onsets"]
                  for ep in episode_summaries if ep["analysis"]["first_entry_lt_0_5"]]
    return {
        "episodes": len(episodes),
        "successes": sum(ep["termination_reason"] == "success" for ep in episodes),
        "mean_final_distance_m": _mean(ep["final_distance_m"] for ep in episodes),
        "mean_episode_length_steps": _mean(ep["episode_length"] for ep in episodes),
        "distance_bins": summarize_distance_bins(steps),
        "near_target": near_target,
        "entries": entries,
        "first_approach_braking_onset": {
            "denominator_episodes_reaching_0_5_m": len(onset_rows),
            "deceleration_request_count": sum(
                row["deceleration_request_distance_m"] is not None for row in onset_rows),
            "median_deceleration_request_distance_m": (
                float(np.median([row["deceleration_request_distance_m"] for row in onset_rows
                                 if row["deceleration_request_distance_m"] is not None]))
                if any(row["deceleration_request_distance_m"] is not None for row in onset_rows)
                else None),
            "reverse_command_count": sum(
                row["reverse_command_distance_m"] is not None for row in onset_rows),
            "median_reverse_command_distance_m": (
                float(np.median([row["reverse_command_distance_m"] for row in onset_rows
                                 if row["reverse_command_distance_m"] is not None]))
                if any(row["reverse_command_distance_m"] is not None for row in onset_rows)
                else None),
        },
        "episode_summaries": episode_summaries,
    }


def select_representative_seed(ppo_episode_summaries: list[dict]) -> int:
    """Median entry-speed crossing episode; fallback to entrants, then all."""
    if not ppo_episode_summaries:
        raise ValueError("Need PPO episode summaries")
    entrants = [ep for ep in ppo_episode_summaries
                if ep["analysis"]["first_entry_lt_0_2"] is not None]
    crossing = [ep for ep in entrants
                if ep["analysis"]["first_entry_lt_0_2"]["crossed_target_plane_next_0_5_s"]]
    candidates = crossing or entrants
    if candidates:
        ranked = sorted(candidates, key=lambda ep: (
            ep["analysis"]["first_entry_lt_0_2"]["speed_m_s"], ep["seed"]))
    else:
        ranked = sorted(ppo_episode_summaries, key=lambda ep: (
            ep["final_distance_m"], ep["seed"]))
    return int(ranked[(len(ranked) - 1) // 2]["seed"])


STEP_LOG_FIELDS = (
    "policy", "seed", "policy_step", "time_s", "x_m", "y_m", "z_m",
    "target_x_m", "target_y_m", "target_z_m", "error_x_m", "error_y_m", "error_z_m",
    "distance_m", "vx_m_s", "vy_m_s", "vz_m_s", "speed_m_s",
    "action_0", "action_1", "action_2", "action_3",
    "vx_cmd_m_s", "vy_cmd_m_s", "vz_cmd_m_s", "yaw_rate_cmd_rad_s",
    "command_norm_m_s", "v_radial_m_s", "v_tangent_m_s",
    "v_cmd_radial_m_s", "v_cmd_tangent_m_s", "velocity_dot_command_m2_s2",
    "active_braking", "deceleration_requested", "speed_error_m_s",
    "post_time_s", "post_x_m", "post_y_m", "post_z_m", "post_distance_m",
    "post_speed_m_s",
)


def write_step_log(paired: dict, path: Path) -> None:
    """Persist every decision step, with post-step state for transition audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STEP_LOG_FIELDS)
        writer.writeheader()
        for policy, episodes in paired.items():
            for episode in episodes:
                for step in episode["steps"]:
                    post_position = step.get("post_position_m")
                    writer.writerow({
                        "policy": policy, "seed": episode["seed"],
                        "policy_step": step.get("policy_step", ""),
                        "time_s": step["time_s"],
                        **dict(zip(("x_m", "y_m", "z_m"), step["position_m"])),
                        **dict(zip(("target_x_m", "target_y_m", "target_z_m"), step["target_m"])),
                        **dict(zip(("error_x_m", "error_y_m", "error_z_m"), step["position_error_m"])),
                        "distance_m": step["distance_m"],
                        **dict(zip(("vx_m_s", "vy_m_s", "vz_m_s"), step["velocity_m_s"])),
                        "speed_m_s": step["speed_m_s"],
                        **dict(zip(("action_0", "action_1", "action_2", "action_3"), step["action"])),
                        **dict(zip(("vx_cmd_m_s", "vy_cmd_m_s", "vz_cmd_m_s"),
                                   step["velocity_command_m_s"])),
                        "yaw_rate_cmd_rad_s": step["yaw_rate_command_rad_s"],
                        "command_norm_m_s": step["command_norm_m_s"],
                        "v_radial_m_s": step["v_radial_m_s"],
                        "v_tangent_m_s": step["v_tangent_m_s"],
                        "v_cmd_radial_m_s": step["v_cmd_radial_m_s"],
                        "v_cmd_tangent_m_s": step["v_cmd_tangent_m_s"],
                        "velocity_dot_command_m2_s2": step["velocity_dot_command_m2_s2"],
                        "active_braking": int(step["active_braking"]),
                        "deceleration_requested": int(step["deceleration_requested"]),
                        "speed_error_m_s": step["speed_error_m_s"],
                        "post_time_s": step.get("post_time_s", ""),
                        **dict(zip(("post_x_m", "post_y_m", "post_z_m"),
                                   post_position if post_position is not None else ("", "", ""))),
                        "post_distance_m": step.get("post_distance_m", ""),
                        "post_speed_m_s": step.get("post_speed_m_s", ""),
                    })


def save_diagnostic_plots(scripted_episode: dict, ppo_episode: dict,
                          output_dir: Path) -> list[Path]:
    """Five static scientific figures from one identical waypoint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    traces = {"Scripted": scripted_episode["steps"], "PPO V2": ppo_episode["steps"]}
    colors = {"Scripted": "#167d65", "PPO V2": "#c25443"}
    filenames = []

    def save_one(fig, name):
        path = output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        filenames.append(path)

    for filename, key, ylabel, reference in (
            ("diagnostic_distance_vs_time.png", "distance_m", "Distance to target (m)", 0.1),
            ("diagnostic_speed_vs_time.png", "speed_m_s", "Actual speed (m/s)", 0.15)):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, steps in traces.items():
            ax.plot([s["time_s"] for s in steps], [s[key] for s in steps],
                    color=colors[name], label=name)
        ax.axhline(reference, color="gray", linestyle=":", label=f"threshold {reference:g}")
        ax.set(xlabel="Simulated time (s)", ylabel=ylabel,
               title=f"Seed {scripted_episode['seed']}: {ylabel}")
        ax.grid(alpha=0.25)
        ax.legend()
        save_one(fig, filename)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, steps in traces.items():
        times = [s["time_s"] for s in steps]
        ax.plot(times, [s["v_radial_m_s"] for s in steps], color=colors[name],
                label=f"{name} radial")
        ax.plot(times, [s["v_tangent_m_s"] for s in steps], color=colors[name],
                linestyle="--", alpha=0.8, label=f"{name} tangent")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(xlabel="Simulated time (s)", ylabel="Actual velocity (m/s)",
           title=f"Seed {scripted_episode['seed']}: target-relative velocity")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    save_one(fig, "diagnostic_radial_velocity.png")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for name, steps in traces.items():
        times = [s["time_s"] for s in steps]
        axes[0].plot(times, [s["command_norm_m_s"] for s in steps],
                     color=colors[name], label=f"{name} command")
        axes[0].plot(times, [s["speed_m_s"] for s in steps], color=colors[name],
                     linestyle="--", alpha=0.7, label=f"{name} actual")
        axes[1].plot(times, [s["v_cmd_radial_m_s"] for s in steps],
                     color=colors[name], label=f"{name} command radial")
        axes[1].plot(times, [s["v_radial_m_s"] for s in steps], color=colors[name],
                     linestyle="--", alpha=0.7, label=f"{name} actual radial")
    axes[0].set_ylabel("Speed norm (m/s)")
    axes[1].set(xlabel="Simulated time (s)", ylabel="Radial velocity (m/s)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    fig.suptitle(f"Seed {scripted_episode['seed']}: commands vs actual motion")
    save_one(fig, "diagnostic_velocity_command.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    for name, steps in traces.items():
        ax.plot([s["position_m"][0] for s in steps],
                [s["position_m"][1] for s in steps],
                color=colors[name], label=name)
    target = scripted_episode["target_m"]
    ax.scatter([target[0]], [target[1]], c="black", marker="*", s=110, label="Waypoint")
    ax.scatter([0], [0], c="gray", marker="o", s=35, label="Start")
    ax.set(xlabel="World X (m)", ylabel="World Y (m)",
           title=f"Seed {scripted_episode['seed']}: XY trajectory")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    save_one(fig, "diagnostic_xy_trajectory.png")
    return filenames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--model", type=Path,
                        default=root / "rl" / "models" / "ppo_waypoint_reward_v2_100k.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output-dir", type=Path, default=root / "reports")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not args.model.is_file():
        parser.error(f"PPO checkpoint not found: {args.model}")

    from stable_baselines3 import PPO
    from mine_uav_env import MineUAVEnv
    from test_env_scripted_policy import scripted_action

    model = PPO.load(args.model, device="cpu")  # Inference only; no .learn().
    policies = {
        "scripted": scripted_action,
        "ppo_v2": lambda observation: model.predict(observation, deterministic=True)[0],
    }
    def progress(name, index, total):
        if index % 10 == 0 or index == total:
            print(f"{name}: {index}/{total} episodes", flush=True)

    paired = evaluate_paired(policies, lambda: MineUAVEnv(reward_version="v2"),
                             range(args.seed, args.seed + args.episodes), progress)
    summaries = {name: summarize_policy(episodes) for name, episodes in paired.items()}
    selected_seed = select_representative_seed(summaries["ppo_v2"]["episode_summaries"])
    representative = {name: next(ep for ep in episodes if ep["seed"] == selected_seed)
                      for name, episodes in paired.items()}
    figures = save_diagnostic_plots(representative["scripted"],
                                    representative["ppo_v2"], args.output_dir)
    step_log = args.output_dir / "ppo_braking_policy_steps.csv.gz"
    write_step_log(paired, step_log)
    report = {
        "experiment": "Read-only scripted vs deterministic Reward V2 100k braking diagnosis",
        "model_path": str(args.model), "reward_version": "v2",
        "evaluation_seed_start": args.seed, "evaluation_episodes": args.episodes,
        "paired_seed_and_target_equality_verified": True,
        "time_sampling": "pre-command state at each policy decision; post-step state also logged",
        "sample_weighting": "distance-zone means pool all qualifying policy steps",
        "definitions": {
            "target_direction": "(target_position - position) / distance if distance > 1e-9 m",
            "v_radial_m_s": "dot(world velocity, target direction); positive means approaching",
            "v_tangent_m_s": "norm(velocity - v_radial * target_direction)",
            "v_cmd_radial_m_s": "dot(mapped world velocity command, target direction)",
            "active_braking": "actual speed > 0.1 m/s and dot(actual velocity, velocity command) < 0",
            "deceleration_requested": "v_radial > 0.15 m/s and v_cmd_radial < v_radial - 0.1 m/s",
            "crossing": "After first threshold entry, signed position error along entry target direction changes from positive to <= -0.02 m within 0.5 s; a distance minimum alone does not count",
            "first_approach_braking_onset": "First deceleration/reverse command after first v_radial>0.15 m/s, before first v_radial<-0.05 after inbound motion; medians condition on episodes reaching 0.5 m",
        },
        "representative_seed": selected_seed,
        "representative_selection": "Among PPO episodes entering <0.2m and crossing within 0.5s, median first-entry speed; fallback entrants then median final distance",
        "step_log": str(step_log),
        "figures": [str(path) for path in figures],
        "policies": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "ppo_braking_diagnostic.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {report_path}; step log: {step_log}; representative seed: {selected_seed}")


if __name__ == "__main__":
    main()
