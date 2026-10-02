"""Common deterministic waypoint diagnostics for original and Reward V2 PPO."""

import math

import numpy as np


EVAL_SEED = 20261001  # Same fixed waypoint set as the original 200k baseline.
DISTANCE_THRESHOLDS_M = (0.5, 0.2, 0.1)
SPEED_THRESHOLD_M_S = 0.15
ACTION_BOUNDARY = 0.95


def summarize_episodes(episodes: list[dict], policy_dt: float) -> dict:
    """Summarize policy-step traces; speed means are pooled over qualifying steps."""
    if not episodes or policy_dt <= 0:
        raise ValueError("Need episodes and a positive policy_dt")
    count = len(episodes)
    steps = [step for episode in episodes for step in episode["steps"]]
    if not steps:
        raise ValueError("Each evaluated episode must have policy steps")
    actions = np.asarray([step["action"] for step in steps], dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 4 or not np.isfinite(actions).all():
        raise ValueError("Evaluation actions must be finite four-vectors")
    successful = [episode for episode in episodes if episode["termination_reason"] == "success"]
    result = {
        "episodes": count,
        "seeds": [episode["seed"] for episode in episodes],
        "successes": len(successful),
        "success_rate": len(successful) / count,
        "mean_final_distance_m": float(np.mean([episode["final_distance_m"] for episode in episodes])),
        "mean_episode_length": float(np.mean([episode["episode_length"] for episode in episodes])),
        "mean_reward": float(np.mean([episode["episode_reward"] for episode in episodes])),
        "mean_successful_completion_time_s": (
            float(np.mean([episode["episode_length"] * policy_dt for episode in successful]))
            if successful else None),
        "ever_speed_below_0_15_m_s_fraction": sum(
            any(step["speed_m_s"] < SPEED_THRESHOLD_M_S for step in episode["steps"])
            for episode in episodes) / count,
        "ever_distance_and_speed_fraction": sum(
            any(step["distance_m"] < 0.1 and step["speed_m_s"] < SPEED_THRESHOLD_M_S
                for step in episode["steps"]) for episode in episodes) / count,
        "max_success_streak_policy_steps": max(
            step["success_streak"] for step in steps),
        "action_near_boundary_fraction_per_axis": np.mean(
            np.abs(actions) >= ACTION_BOUNDARY, axis=0).tolist(),
        "action_near_boundary_fraction_overall": float(np.mean(np.abs(actions) >= ACTION_BOUNDARY)),
        "action_max_abs_per_axis": np.max(np.abs(actions), axis=0).tolist(),
        "action_boundary": ACTION_BOUNDARY,
    }
    for threshold, suffix in zip(DISTANCE_THRESHOLDS_M, ("0_5", "0_2", "0_1")):
        result[f"ever_within_{suffix}_m_fraction"] = sum(
            any(step["distance_m"] < threshold for step in episode["steps"])
            for episode in episodes) / count
        near_speeds = [step["speed_m_s"] for step in steps
                       if step["distance_m"] < threshold and math.isfinite(step["speed_m_s"])]
        result[f"mean_speed_within_{suffix}_m_s"] = (
            float(np.mean(near_speeds)) if near_speeds else None)
        result[f"speed_samples_within_{suffix}_m"] = len(near_speeds)
    result["episode_summaries"] = [
        {key: value for key, value in episode.items() if key != "steps"}
        | {"minimum_distance_m": min(step["distance_m"] for step in episode["steps"]),
           "max_success_streak_policy_steps": max(step["success_streak"] for step in episode["steps"])}
        for episode in episodes]
    return result


def evaluate_diagnostics(model, env, episodes: int = 100, seed: int = EVAL_SEED) -> dict:
    """Evaluate a checkpoint without training or changing waypoint sampling."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    base_env = env.unwrapped
    records = []
    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        observation, _ = env.reset(seed=episode_seed)
        target = base_env.target_position.tolist()
        trace = []
        episode_reward = 0.0
        for policy_step in range(base_env.max_episode_steps):
            action, _ = model.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=float)
            if action.shape != (4,) or not np.isfinite(action).all():
                raise ValueError("Evaluation policy returned a non-finite or non-4D action")
            observation, reward, terminated, truncated, info = env.step(action)
            if not math.isfinite(float(reward)):
                raise ValueError("Evaluation produced a non-finite reward")
            episode_reward += float(reward)
            trace.append({
                "distance_m": float(info["distance_m"]),
                "speed_m_s": float(info["speed_m_s"]),
                "success_streak": int(info["success_streak"]),
                "action": action.tolist(),
            })
            if terminated or truncated:
                records.append({
                    "seed": episode_seed,
                    "target_position_m": target,
                    "final_distance_m": float(info["distance_m"]),
                    "final_speed_m_s": float(info["speed_m_s"]),
                    "episode_length": policy_step + 1,
                    "episode_reward": episode_reward,
                    "termination_reason": info["termination_reason"],
                    "steps": trace,
                })
                break
        else:
            raise RuntimeError("Evaluation episode exceeded environment time limit")
    result = summarize_episodes(records, policy_dt=base_env.policy_dt)
    result["reward_version"] = base_env.reward_version
    return result
