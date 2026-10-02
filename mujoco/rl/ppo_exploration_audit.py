"""Read-only audit of the saved Reward V2 PPO action distribution and rollouts."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution

from diagnose_ppo_braking import evaluate_episode, summarize_policy
from mine_uav_env import MineUAVEnv
from reward_v2_alignment_audit import fixed_waypoints
from reward_v2_diagnostics import summarize_episodes
from reward_v4_diagnostics import crossing_counts


RL_DIR = Path(__file__).resolve().parent
CHECKPOINT = RL_DIR / "models" / "ppo_waypoint_reward_v2_100k.zip"
REPORT_PATH = RL_DIR.parent / "reports" / "ppo_v2_exploration_audit_raw_mean.json"
ACTION_AXES = ["vx", "vy", "vz", "yaw_rate"]
COMMAND_SCALES = np.array([1.5, 1.5, 1.0, 1.0], dtype=float)
STOCHASTIC_REPEATS = 3


def inspect_policy_distribution(model: PPO) -> dict:
    """Read the global trainable Gaussian parameter from the saved policy."""
    policy = model.policy
    if not isinstance(policy.action_dist, DiagGaussianDistribution) or model.use_sde:
        raise TypeError("Expected standard DiagGaussianDistribution without gSDE")
    parameter = dict(policy.named_parameters()).get("log_std")
    if parameter is None or parameter.ndim != 1 or parameter.shape[0] != 4:
        raise ValueError("Expected one state-independent four-axis log_std parameter")
    if not parameter.requires_grad:
        raise ValueError("Saved log_std is not trainable")
    log_std = parameter.detach().cpu().numpy().astype(float)
    sigma = np.exp(log_std)
    if not np.isfinite(sigma).all():
        raise ValueError("Saved action standard deviation is not finite")
    return {
        "checkpoint": str(CHECKPOINT),
        "num_timesteps": int(model.num_timesteps),
        "distribution": type(policy.action_dist).__name__,
        "state_independent_trainable_log_std": True,
        "axes": ACTION_AXES,
        "log_std_init": float(policy.log_std_init),
        "log_std": log_std.tolist(),
        "normalized_std": sigma.tolist(),
        "command_scale": COMMAND_SCALES.tolist(),
        "physical_command_std": (sigma * COMMAND_SCALES).tolist(),
        "physical_units": ["m/s", "m/s", "m/s", "rad/s"],
        "note": "Raw Gaussian sigma before SB3 clips action to [-1,1].",
    }


def raw_gaussian_mean(model: PPO, observation: np.ndarray) -> np.ndarray:
    """Return the unbounded Gaussian mean before SB3 clips environment action."""
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        mean = distribution.distribution.mean.detach().cpu().numpy()
    if mean.shape != (1, 4) or not np.isfinite(mean).all():
        raise ValueError("Expected a finite four-dimensional Gaussian mean")
    return mean[0].astype(float)


def evaluate_exploration(model: PPO, env: MineUAVEnv,
                         waypoints: list[tuple[int, list[float]]],
                         deterministic: bool, repeats: int = 1) -> dict:
    """Evaluate on identical fixed targets, measuring means at visited states."""
    if env.reward_version != "v2" or env.render_mode is not None or not waypoints:
        raise ValueError("Need nonempty waypoints and a headless Reward V2 environment")
    if repeats <= 0 or (deterministic and repeats != 1):
        raise ValueError("Use positive repeats; deterministic needs one per target")
    traces = []
    clipped_mean_actions = []
    raw_mean_actions = []
    target_sequence = []
    for repeat in range(repeats):
        for seed, target in waypoints:
            if not deterministic:
                torch.manual_seed(917_000 + 10_000 * repeat + seed)

            def policy(observation):
                raw_mean_actions.append(raw_gaussian_mean(model, observation))
                mean_action, _ = model.predict(observation, deterministic=True)
                action = (mean_action if deterministic else
                          model.predict(observation, deterministic=False)[0])
                clipped_mean_actions.append(np.asarray(mean_action, dtype=float).copy())
                return action

            trace = evaluate_episode(env, policy, seed)
            if trace["target_m"] != target:
                raise ValueError(f"Waypoint mismatch for fixed seed {seed}")
            traces.append(trace)
            target_sequence.append(target)
    actions = np.asarray([step["action"] for ep in traces for step in ep["steps"]], dtype=float)
    clipped_means = np.asarray(clipped_mean_actions, dtype=float)
    raw_means = np.asarray(raw_mean_actions, dtype=float)
    if actions.shape != clipped_means.shape or actions.shape != raw_means.shape or actions.shape[1:] != (4,):
        raise RuntimeError("Sampled action and policy-mean accounting is misaligned")
    braking = summarize_policy(traces)
    task_records = [{
        "seed": ep["seed"], "target_position_m": ep["target_m"],
        "final_distance_m": ep["final_distance_m"],
        "final_speed_m_s": ep["final_speed_m_s"],
        "episode_length": ep["episode_length"],
        "episode_reward": ep["episode_reward"],
        "termination_reason": ep["termination_reason"],
        "steps": [{
            "distance_m": step["post_distance_m"],
            "speed_m_s": (step["post_speed_m_s"] if step["post_speed_m_s"] is not None
                          else math.inf),
            "success_streak": step["success_streak"],
            "action": step["action"],
        } for step in ep["steps"]],
    } for ep in traces]
    return {
        "deterministic": deterministic,
        "repeats_per_waypoint": repeats,
        "episodes": len(traces),
        "seeds": [ep["seed"] for ep in traces],
        "targets": target_sequence,
        "raw_gaussian_mean_action_norm": float(np.linalg.norm(raw_means, axis=1).mean()),
        "clipped_deterministic_action_norm": float(np.linalg.norm(clipped_means, axis=1).mean()),
        "raw_mean_outside_action_box_fraction_per_axis":
            np.mean(np.abs(raw_means) > 1.0, axis=0).tolist(),
        "mean_action_norm": float(np.linalg.norm(clipped_means, axis=1).mean()),
        "executed_action_norm": float(np.linalg.norm(actions, axis=1).mean()),
        "executed_action_near_boundary_fraction_per_axis":
            np.mean(np.abs(actions) >= 0.95, axis=0).tolist(),
        "executed_action_near_boundary_fraction_overall":
            float(np.mean(np.abs(actions) >= 0.95)),
        "task_metrics": summarize_episodes(task_records, policy_dt=env.policy_dt),
        "braking_diagnostic": braking,
        "crossing_after_first_0_2_m": crossing_counts(braking),
        "sampling_convention": "Raw Gaussian mean, clipped deterministic action and "
                               "executed action at pre-command state; braking diagnostics "
                               "pre-command; task metrics post-step. mean_action_norm is "
                               "the clipped deterministic action norm (legacy alias).",
    }


def run_exploration_audit() -> dict:
    if REPORT_PATH.exists():
        raise FileExistsError(f"Preserving existing audit: {REPORT_PATH}")
    model = PPO.load(CHECKPOINT, device="cpu")
    parameters = inspect_policy_distribution(model)
    waypoints = fixed_waypoints("full")
    env = MineUAVEnv(render_mode=None, reward_version="v2")
    try:
        deterministic = evaluate_exploration(model, env, waypoints, True)
        print("Existing V2 deterministic: 100/100 episodes evaluated", flush=True)
        stochastic = evaluate_exploration(model, env, waypoints, False,
                                          repeats=STOCHASTIC_REPEATS)
        print("Existing V2 stochastic: 300/300 episodes evaluated", flush=True)
    finally:
        env.close()
    if (deterministic["targets"] != stochastic["targets"][:100]
            or any(stochastic["targets"][i * 100:(i + 1) * 100] !=
                   deterministic["targets"] for i in range(STOCHASTIC_REPEATS))):
        raise RuntimeError("Stochastic and deterministic target sets differ")
    result = {"policy_distribution": parameters,
              "fixed_full_waypoints": len(waypoints),
              "stochastic_repeats_per_waypoint": STOCHASTIC_REPEATS,
              "deterministic": deterministic, "stochastic": stochastic}
    REPORT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run_exploration_audit()
    print(json.dumps({"policy_distribution": result["policy_distribution"],
                      "deterministic_success": result["deterministic"]["task_metrics"]["successes"],
                      "stochastic_success": result["stochastic"]["task_metrics"]["successes"],
                      "report": str(REPORT_PATH)}, indent=2))
