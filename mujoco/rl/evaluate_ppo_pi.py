"""Deterministic PPO-on-PI evaluation with separate hidden-integral logging."""

import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from attribution_audit import write_condition
from ppo_exploration_audit import evaluate_exploration
from ppo_pi_env import MineUAVPIEnv, PIRobustnessEnv
from train_ppo_lowstd_multiseed import compact_evaluation


def nominal_gate(benchmark: dict, holdout: dict) -> bool:
    """Permit robustness only if both distinct nominal fixed-100 sets reach 90%."""
    for evaluation in (benchmark, holdout):
        task = evaluation["task_metrics"]
        episodes = task["episodes"]
        successes = task["successes"]
        if (episodes != 100 or not isinstance(successes, int)
                or not 0 <= successes <= episodes):
            raise ValueError("Nominal gate requires complete fixed-100 evaluations")
    return (benchmark["task_metrics"]["successes"] >= 90
            and holdout["task_metrics"]["successes"] >= 90)


def _integral_diagnostics(captures: list[dict], evaluation: dict) -> dict:
    episodes = evaluation["braking_diagnostic"]["episode_summaries"]
    if len(captures) != len(episodes) or not captures:
        raise ValueError("Integral captures do not match evaluation episodes")
    per_episode = []
    all_error = []
    all_accel = []
    for capture, episode in zip(captures, episodes):
        if (capture["seed"] != episode["seed"]
                or capture["target_m"] != episode["target_m"]
                or len(capture["steps"]) != episode["episode_length"]
                or not np.array_equal(capture["initial_integral_error_m"], [0, 0, 0])):
            raise ValueError("Integral episode/target/reset alignment differs")
        if not capture["steps"]:
            raise ValueError("Integral episode has no policy steps")
        if capture["steps"][-1]["termination_reason"] != episode["termination_reason"]:
            raise ValueError("Integral trace terminal reason differs")
        state = np.asarray([row["integral_error_m"] for row in capture["steps"]],
                           dtype=float)
        accel = np.asarray([row["integral_acceleration_m_s2"]
                            for row in capture["steps"]], dtype=float)
        if (state.shape != (len(capture["steps"]), 3)
                or accel.shape != state.shape
                or not np.isfinite(state).all() or not np.isfinite(accel).all()
                or not np.allclose(accel, state * [0.5, 0.5, 0.8], atol=1e-9)
                or np.max(np.linalg.norm(accel[:, :2], axis=1)) > 1.5 + 1e-8
                or np.max(np.abs(accel[:, 2])) > 1.5 + 1e-8):
            raise ValueError("Integral state/contribution is invalid or unbounded")
        all_error.append(state)
        all_accel.append(accel)
        per_episode.append({
            "seed": episode["seed"],
            "termination_reason": episode["termination_reason"],
            "success": episode["termination_reason"] == "success",
            "final_integral_error_xyz_m": state[-1].tolist(),
            "final_integral_acceleration_xyz_m_s2": accel[-1].tolist(),
            "max_integral_error_norm_m": float(np.max(np.linalg.norm(state, axis=1))),
            "max_integral_acceleration_norm_m_s2":
                float(np.max(np.linalg.norm(accel, axis=1))),
            "max_integral_acceleration_xy_m_s2":
                float(np.max(np.linalg.norm(accel[:, :2], axis=1))),
            "max_integral_acceleration_z_m_s2": float(np.max(np.abs(accel[:, 2]))),
        })
    all_error = np.concatenate(all_error)
    all_accel = np.concatenate(all_accel)

    def mean_or_none(values):
        return float(np.mean(values)) if values else None

    return {
        "episode_count": len(per_episode),
        "policy_step_count": len(all_error),
        "all_resets_zero": True,
        "max_abs_integral_error_xyz_m": np.max(np.abs(all_error), axis=0).tolist(),
        "max_abs_integral_acceleration_xyz_m_s2":
            np.max(np.abs(all_accel), axis=0).tolist(),
        "max_xy_accel_m_s2": float(np.max(np.linalg.norm(all_accel[:, :2], axis=1))),
        "max_z_accel_m_s2": float(np.max(np.abs(all_accel[:, 2]))),
        "mean_episode_peak_accel_success_m_s2": mean_or_none([
            row["max_integral_acceleration_norm_m_s2"] for row in per_episode
            if row["success"]]),
        "mean_episode_peak_accel_failure_m_s2": mean_or_none([
            row["max_integral_acceleration_norm_m_s2"] for row in per_episode
            if not row["success"]]),
        "per_episode": per_episode,
    }


def evaluate_pi_policy(model: PPO, env: MineUAVPIEnv | PIRobustnessEnv,
                       waypoints: list[tuple[int, list[float]]],
                       trace_path: Path) -> dict:
    """Evaluate once per exact target; publish the full PI trace separately."""
    if trace_path.exists():
        raise FileExistsError(f"Preserving existing PI integral trace: {trace_path}")
    controller = env.velocity_controller
    if (env.reward_version != "v2" or env.render_mode is not None
            or not math.isclose(controller.ki_xy, 0.5, abs_tol=1e-12)
            or not math.isclose(controller.ki_z, 0.8, abs_tol=1e-12)
            or not math.isclose(controller.integral_accel_limit_xy, 1.5, abs_tol=1e-12)
            or not math.isclose(controller.integral_accel_limit_z, 1.5, abs_tol=1e-12)
            or not waypoints):
        raise ValueError("Need headless Reward V2 with frozen PI and targets")
    env.begin_integral_capture()
    try:
        evaluation = evaluate_exploration(model, env, waypoints, deterministic=True)
    finally:
        captures = env.take_integral_capture()
    if (evaluation["seeds"] != [seed for seed, _ in waypoints]
            or evaluation["targets"] != [target for _, target in waypoints]):
        raise ValueError("PI evaluation targets differ from requested targets")
    summary = _integral_diagnostics(captures, evaluation)
    write_condition(trace_path, {
        "waypoint_seeds": [seed for seed, _ in waypoints],
        "targets": [target for _, target in waypoints],
        "policy_steps": summary["policy_step_count"],
        "episodes": captures,
    })
    evaluation["integral_summary"] = summary
    evaluation["integral_trace_path"] = str(trace_path)
    return evaluation


def compact_pi_evaluation(evaluation: dict) -> dict:
    """Task, crossing, near-target and PI-state metrics for checkpoint tables."""
    result = compact_evaluation(evaluation)
    result["integral"] = evaluation["integral_summary"]
    result["integral_trace_path"] = evaluation["integral_trace_path"]
    return result
