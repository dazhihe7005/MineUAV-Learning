"""Waypoint heuristic exercising Gymnasium action → controller → MuJoCo."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from mine_uav_env import MineUAVEnv


REPORT = Path(__file__).resolve().parents[1] / "reports" / "rl_scripted_policy.json"


def scripted_action(observation: np.ndarray) -> np.ndarray:
    """v_cmd proportional to position error; yaw_rate proportional to yaw error."""
    velocity = np.clip((1.0 / 3.0) * observation[:3], [-1.5, -1.5, -1.0],
                       [1.5, 1.5, 1.0])
    yaw_rate = float(np.clip(1.0 * observation[6], -1.0, 1.0))
    return np.clip(np.r_[velocity / [1.5, 1.5, 1.0], yaw_rate], -1, 1).astype(np.float32)


def run_scripted_episodes(episodes: int = 20, seed: int = 20260929) -> dict:
    env = MineUAVEnv()
    reasons = Counter()
    lengths = []
    final_distances = []
    max_motor_rpm = 0.0
    total_saturation_updates = 0
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            for step_index in range(env.max_episode_steps):
                action = scripted_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                assert env.observation_space.contains(observation)
                assert np.isfinite(observation).all() and np.isfinite(reward)
                max_motor_rpm = max(max_motor_rpm, info["max_motor_rpm"])
                total_saturation_updates += info["allocator_saturation_count"]
                if terminated or truncated:
                    assert not (terminated and truncated)
                    reasons[info["termination_reason"]] += 1
                    final_distances.append(info["distance_m"])
                    lengths.append(step_index + 1)
                    break
            else:
                raise AssertionError("Episode did not end by time limit")
        result = {
            "episodes": episodes,
            "seed": seed,
            "successes": reasons["success"],
            "success_rate": reasons["success"] / episodes,
            "termination_reasons": dict(reasons),
            "mean_episode_steps": float(np.mean(lengths)),
            "mean_final_distance_m": float(np.mean(final_distances)),
            "max_motor_rpm": max_motor_rpm,
            "allocator_saturation_updates": total_saturation_updates,
        }
        REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        env.close()


if __name__ == "__main__":
    print(json.dumps(run_scripted_episodes(), ensure_ascii=False, indent=2))
