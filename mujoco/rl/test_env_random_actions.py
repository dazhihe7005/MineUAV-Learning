"""Run at least 20 headless random-action episodes through the public Gym API."""

import json
from collections import Counter
from pathlib import Path

from gymnasium.utils.env_checker import check_env
import numpy as np

from mine_uav_env import MineUAVEnv


REPORT = Path(__file__).resolve().parents[1] / "reports" / "rl_random_actions.json"


def run_random_episodes(episodes: int = 20, seed: int = 20260929) -> dict:
    if episodes < 20:
        raise ValueError("Random-action validation requires at least 20 episodes")
    env = MineUAVEnv()
    rng = np.random.default_rng(seed)
    reasons = Counter()
    rewards = []
    lengths = []
    total_saturation_updates = 0
    try:
        check_env(env, skip_render_check=True)
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            assert observation.shape == (7,) and observation.dtype == np.float32
            assert env.observation_space.contains(observation)
            assert np.allclose(env.data.qpos[:3], [0, 0, 1])
            assert np.allclose(env.data.qvel, 0)
            assert env.velocity_controller.yaw_target == 0.0
            episode_reward = 0.0
            for step_index in range(env.max_episode_steps):
                action = rng.uniform(-1, 1, size=4).astype(np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                assert observation.shape == (7,) and observation.dtype == np.float32
                assert env.observation_space.contains(observation)
                assert np.isfinite(observation).all() and np.isfinite(reward)
                assert np.isclose(reward, sum(info["reward_breakdown"].values()))
                np.testing.assert_allclose(info["velocity_command_m_s"],
                                           action[:3] * [1.5, 1.5, 1.0], atol=1e-7)
                assert np.isclose(info["yaw_rate_command_rad_s"], action[3], atol=1e-7)
                assert not (terminated and truncated)
                total_saturation_updates += info["allocator_saturation_count"]
                episode_reward += reward
                if terminated or truncated:
                    assert info["termination_reason"] is not None
                    reasons[info["termination_reason"]] += 1
                    rewards.append(episode_reward)
                    lengths.append(step_index + 1)
                    break
            else:
                raise AssertionError("Episode did not terminate or truncate by time limit")
        result = {
            "episodes": episodes,
            "seed": seed,
            "check_env": "passed (headless render check skipped)",
            "termination_reasons": dict(reasons),
            "mean_episode_steps": float(np.mean(lengths)),
            "min_episode_steps": min(lengths),
            "max_episode_steps_observed": max(lengths),
            "mean_episode_reward": float(np.mean(rewards)),
            "allocator_saturation_updates": total_saturation_updates,
            "observation_action_reward_finite": True,
            "reset_state_cleared": True,
        }
        REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        env.close()


if __name__ == "__main__":
    print(json.dumps(run_random_episodes(), ensure_ascii=False, indent=2))
