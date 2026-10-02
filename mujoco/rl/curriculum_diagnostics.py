"""Fixed-seed Reward V2 evaluations for local and original full waypoints."""

import math

from diagnose_ppo_braking import evaluate_episode, summarize_policy
from reward_v2_diagnostics import summarize_episodes
from reward_v4_diagnostics import crossing_counts


def evaluate_curriculum(model, env, episodes: int = 100, seed: int = 20261001) -> dict:
    """One headless deterministic rollout per seed with existing V2 metrics."""
    base_env = env.unwrapped
    if (base_env.reward_version != "v2" or base_env.render_mode is not None
            or episodes <= 0):
        raise ValueError("Need a headless Reward V2 environment and positive episode count")
    traces = [evaluate_episode(
        base_env, lambda observation: model.predict(observation, deterministic=True)[0],
        seed + index) for index in range(episodes)]
    task_records = [{
        "seed": episode["seed"],
        "target_position_m": episode["target_m"],
        "final_distance_m": episode["final_distance_m"],
        "final_speed_m_s": episode["final_speed_m_s"],
        "episode_length": episode["episode_length"],
        "episode_reward": episode["episode_reward"],
        "termination_reason": episode["termination_reason"],
        "steps": [{
            "distance_m": step["post_distance_m"],
            "speed_m_s": (step["post_speed_m_s"] if step["post_speed_m_s"] is not None
                          else math.inf),
            "success_streak": step["success_streak"],
            "action": step["action"],
        } for step in episode["steps"]],
    } for episode in traces]
    result = summarize_episodes(task_records, policy_dt=base_env.policy_dt)
    braking = summarize_policy(traces)
    result.update({
        "reward_version": "v2",
        "target_distribution": base_env.target_distribution,
        "braking_diagnostic": braking,
        "crossing_after_first_0_2_m": crossing_counts(braking),
        "sampling_convention": {
            "task_metrics": "post-step, identical to prior Reward V2 evaluation",
            "braking_metrics": "pre-command, identical to prior braking diagnostic",
        },
    })
    return result
