"""Fixed-seed V4 task metrics plus the existing pre-command braking diagnosis."""

import math

from diagnose_ppo_braking import evaluate_episode, summarize_policy
from reward_v2_diagnostics import EVAL_SEED, summarize_episodes


def crossing_counts(braking_summary: dict) -> dict:
    """Count exactly the signed-target-plane events from braking diagnostic."""
    episodes = braking_summary["episode_summaries"]
    entries = [ep["analysis"]["first_entry_lt_0_2"] for ep in episodes]
    entered = [entry for entry in entries if entry is not None]
    within = sum(entry["crossed_target_plane_next_0_5_s"] for entry in entered)
    before_end = sum(entry["crossed_target_plane_before_episode_end"] for entry in entered)
    return {
        "entered_0_2_m_count": len(entered),
        "within_0_5_s_count": within,
        "before_episode_end_count": before_end,
        "within_0_5_s_fraction_all_episodes": within / len(episodes),
        "before_episode_end_fraction_all_episodes": before_end / len(episodes),
    }


def evaluate_reward_v4(model, env, episodes: int = 100, seed: int = EVAL_SEED) -> dict:
    """One deterministic rollout per waypoint; no duplicate evaluation run."""
    if env.reward_version != "v4" or env.render_mode is not None or episodes <= 0:
        raise ValueError("Need a headless V4 environment and positive episode count")
    traces = [evaluate_episode(
        env, lambda observation: model.predict(observation, deterministic=True)[0],
        seed + index) for index in range(episodes)]
    task_records = []
    for episode in traces:
        task_records.append({
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
        })
    result = summarize_episodes(task_records, policy_dt=env.policy_dt)
    braking = summarize_policy(traces)
    result["reward_version"] = "v4"
    result["braking_diagnostic"] = braking
    result["crossing_after_first_0_2_m"] = crossing_counts(braking)
    result["sampling_convention"] = {
        "task_metrics": "post-step, identical to Reward V2 evaluation",
        "braking_metrics": "pre-command, identical to scripted-vs-PPO braking diagnostic",
    }
    return result


def validate_fixed_waypoints(v2_evaluation: dict, v2_braking_summary: dict,
                             v4_evaluation: dict) -> bool:
    """Require exact ordered seed and target-coordinate identity across runs."""
    expected = list(range(EVAL_SEED, EVAL_SEED + 100))
    for label, evaluation in (("V2", v2_evaluation), ("V4", v4_evaluation)):
        if evaluation["seeds"] != expected:
            raise ValueError(f"{label} seeds differ from fixed 100-waypoint set")
        if [ep["seed"] for ep in evaluation["episode_summaries"]] != expected:
            raise ValueError(f"{label} episode seed order differs")
    braking_episodes = v2_braking_summary["episode_summaries"]
    if [ep["seed"] for ep in braking_episodes] != expected:
        raise ValueError("V2 braking episode seed order differs")
    target_sets = [
        [ep["target_position_m"] for ep in v2_evaluation["episode_summaries"]],
        [ep["target_m"] for ep in braking_episodes],
        [ep["target_position_m"] for ep in v4_evaluation["episode_summaries"]],
    ]
    if target_sets[0] != target_sets[1] or target_sets[0] != target_sets[2]:
        raise ValueError("V2 and V4 waypoint coordinates differ")
    return True
