"""Independent Reward V2 low-std training seeds and fixed/holdout evaluation.

This wraps the existing experiment; the environment, reward, PPO settings and
checkpoint evaluator remain those of train_ppo_v2_lowstd.py.
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from mine_uav_env import MineUAVEnv
from ppo_exploration_audit import evaluate_exploration
from reward_v2_alignment_audit import fixed_waypoints
from train_ppo_v2_lowstd import (LOG_STD_INIT, LowStdCallback,
                                 checkpoint_schedule, read_policy_std)
from train_ppo_waypoint import (N_ENVS, N_STEPS, make_ppo,
                                make_training_envs)


RL_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = RL_DIR.parent
PRIOR_TRAIN_SEED = 20260929
# SB3 gives VecEnv rank r the seed (training_seed + r). Spacing fresh
# seeds by N_ENVS makes *all* per-rank target RNG streams disjoint.
FRESH_TRAIN_SEEDS = (0, 8, 16, 24)
ALL_TRAIN_SEEDS = (PRIOR_TRAIN_SEED, *FRESH_TRAIN_SEEDS)
HOLDOUT_FIRST_SEED = 20271001
HOLDOUT_COUNT = 100
PRIOR_REPORT = MUJOCO_DIR / "reports" / "ppo_waypoint_v2_lowstd.json"
SUMMARY_REPORT = MUJOCO_DIR / "reports" / "ppo_waypoint_v2_lowstd_multiseed.json"


def seed_paths(seed: int, root: Path = MUJOCO_DIR) -> dict[str, Path]:
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Training seed must be a nonnegative integer")
    name = f"seed_{seed}"
    return {
        "model_dir": root / "rl" / "models" / "ppo_waypoint_v2_lowstd_multiseed" / name,
        "log_dir": root / "rl" / "logs" / "ppo_waypoint_v2_lowstd_multiseed" / name,
        "tensorboard_dir": root / "rl" / "tensorboard" / "ppo_waypoint_v2_lowstd_multiseed" / name,
        "report": root / "reports" / "ppo_waypoint_v2_lowstd_multiseed" / f"{name}.json",
    }


def require_fresh_seed_outputs(paths: dict[str, Path]) -> None:
    existing = []
    for key in ("model_dir", "log_dir", "tensorboard_dir"):
        directory = paths[key]
        if directory.exists() and any(directory.iterdir()):
            existing.append(directory)
    if paths["report"].exists():
        existing.append(paths["report"])
    if existing:
        raise FileExistsError(f"Preserving existing per-seed outputs: {existing}")


def seed_training_envs(vec_env, seed: int) -> list[int]:
    """Schedule rank-specific Gymnasium seeds for the next VecEnv reset."""
    expected = [seed + rank for rank in range(vec_env.num_envs)]
    actual = vec_env.seed(seed)
    if actual != expected:
        raise RuntimeError(f"Vector environment seed mismatch: {actual} != {expected}")
    return actual


def build_holdout_waypoints() -> list[tuple[int, list[float]]]:
    """Use the untouched full-task sampler and disjoint evaluation seeds."""
    env = MineUAVEnv(render_mode=None, reward_version="v2", target_distribution="full")
    try:
        result = []
        for seed in range(HOLDOUT_FIRST_SEED, HOLDOUT_FIRST_SEED + HOLDOUT_COUNT):
            _, info = env.reset(seed=seed)
            result.append((seed, info["target_position_m"].tolist()))
        return result
    finally:
        env.close()


def first_observed_thresholds(milestones: dict) -> dict[str, int | None]:
    """Earliest evaluated checkpoint meeting each fixed100 full success rate."""
    thresholds = {"50pct": 0.5, "90pct": 0.9, "100pct": 1.0}
    result = dict.fromkeys(thresholds)
    last_step = -1
    for label in ("20k", "50k", "100k"):
        point = milestones[label]
        step = int(point["actual_timesteps"])
        rate = float(point["evaluation"]["full"]["task_metrics"]["success_rate"])
        if step <= last_step or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Invalid milestone timestep or full-task success rate")
        last_step = step
        for name, threshold in thresholds.items():
            if result[name] is None and rate >= threshold:
                result[name] = step
    return result


def aggregate_numeric(values: list[float | None]) -> dict:
    """Mean/sample SD/min/max, keeping undefined no-success times missing."""
    present = [float(value) for value in values if value is not None]
    if not all(math.isfinite(value) for value in present):
        raise ValueError("Aggregate requires finite values")
    return {
        "count": len(present), "missing_count": len(values) - len(present),
        "mean": float(np.mean(present)) if present else None,
        "std": float(np.std(present, ddof=1)) if len(present) > 1 else None,
        "min": min(present) if present else None,
        "max": max(present) if present else None,
        "std_definition": "sample standard deviation (ddof=1)",
    }


def compact_evaluation(evaluation: dict) -> dict:
    """Extract comparable task/near-target metrics without episode-level traces."""
    task = evaluation["task_metrics"]
    near = evaluation["braking_diagnostic"]["near_target"]["lt_0_1"]
    crossing = evaluation["crossing_after_first_0_2_m"]
    return {
        "episodes": task["episodes"],
        "successes": task["successes"],
        "success_rate": task["success_rate"],
        "mean_final_distance_m": task["mean_final_distance_m"],
        "mean_successful_completion_time_s": task["mean_successful_completion_time_s"],
        "mean_episode_length": task["mean_episode_length"],
        "ever_within_0_1_m_fraction": task["ever_within_0_1_m_fraction"],
        "ever_distance_and_speed_fraction": task["ever_distance_and_speed_fraction"],
        "near_target_actual_speed_m_s": near["mean_actual_speed_m_s"],
        "near_target_command_speed_m_s": near["mean_command_norm_m_s"],
        "near_target_tangential_speed_m_s": near["mean_v_tangent_m_s"],
        "crossing_rate": crossing["before_episode_end_fraction_all_episodes"],
        "action_saturation_fraction":
            evaluation["executed_action_near_boundary_fraction_overall"],
    }


class SeededLowStdCallback(LowStdCallback):
    """Observe the actual first training targets without altering the rollout."""

    def _on_training_start(self) -> None:
        self.first_training_targets = [
            wrapped.unwrapped.target_position.astype(float).tolist()
            for wrapped in self.training_env.envs
        ]


def initial_actor_hash(model: PPO) -> str:
    """Compact proof of independent, reproducible policy initialization."""
    weight = model.policy.action_net.weight.detach().cpu().numpy()
    return hashlib.sha256(weight.tobytes()).hexdigest()


def expected_training_targets(seed: int) -> list[list[float]]:
    env = MineUAVEnv(render_mode=None, reward_version="v2",
                     target_distribution="full")
    try:
        return [env.reset(seed=seed + rank)[1]["target_position_m"].tolist()
                for rank in range(N_ENVS)]
    finally:
        env.close()


def train_one_seed(seed: int) -> dict:
    """Run one fresh 100,352-transition PPO experiment; never resume/overwrite."""
    if seed not in FRESH_TRAIN_SEEDS:
        raise ValueError(f"Fresh training seeds are {FRESH_TRAIN_SEEDS}")
    paths = seed_paths(seed)
    require_fresh_seed_outputs(paths)
    for key in ("model_dir", "log_dir", "tensorboard_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_env = make_training_envs(paths["log_dir"] / "monitor", n_envs=N_ENVS,
                                   reward_version="v2", target_distribution="full")
    evaluation_envs = {
        "local": MineUAVEnv(render_mode=None, reward_version="v2",
                             target_distribution="local_a"),
        "full": MineUAVEnv(render_mode=None, reward_version="v2",
                            target_distribution="full"),
    }
    holdout_env = MineUAVEnv(render_mode=None, reward_version="v2",
                             target_distribution="full")
    started_at = time.perf_counter()
    try:
        rank_seeds = seed_training_envs(train_env, seed)
        model = make_ppo(train_env, paths["tensorboard_dir"],
                         log_std_init=LOG_STD_INIT, seed=seed)
        model.verbose = 0
        if model.seed != seed or model.n_envs != N_ENVS:
            raise RuntimeError("SB3 did not receive the requested seed/env count")
        initial_std = read_policy_std(model)
        if initial_std["log_std"] != [LOG_STD_INIT] * 4:
            raise RuntimeError("Wrong initial policy standard deviation")
        actor_hash = initial_actor_hash(model)
        waypoints = {task: fixed_waypoints(task) for task in ("local", "full")}
        callback = SeededLowStdCallback(evaluation_envs, waypoints,
                                        paths["model_dir"], paths["log_dir"])
        budget = checkpoint_schedule()[-1]["actual_timesteps"]
        model.learn(total_timesteps=budget, callback=callback,
                    tb_log_name="PPO", log_interval=1)
        if model.num_timesteps != budget:
            raise RuntimeError(f"Wrong timestep count: {model.num_timesteps} != {budget}")
        callback.record_updated_state()
        if set(callback.milestones) != {"20k", "50k", "100k"}:
            raise RuntimeError("Missing one or more checkpoint evaluations")
        if callback.first_training_targets != expected_training_targets(seed):
            raise RuntimeError("First Gymnasium target samples do not match rank seeds")
        training_wall_s = time.perf_counter() - started_at
        holdout_waypoints = build_holdout_waypoints()
        holdout = evaluate_exploration(model, holdout_env,
                                       holdout_waypoints, deterministic=True)
        if holdout["targets"] != [target for _, target in holdout_waypoints]:
            raise RuntimeError("Holdout waypoint mismatch")
        model.dump_logs(iteration=budget // (N_ENVS * N_STEPS))
        result = {
            "reward_version": "v2", "initialization": "from_scratch",
            "only_changed_ppo_parameter_vs_default_v2":
                {"log_std_init": LOG_STD_INIT},
            "seed": seed,
            "randomness": {
                "python_random_seed": seed, "numpy_seed": seed,
                "pytorch_seed": seed, "sb3_seed": model.seed,
                "gymnasium_initial_reset_rank_seeds": rank_seeds,
                "first_training_targets_m": callback.first_training_targets,
                "initial_actor_action_weight_sha256": actor_hash,
            },
            "n_envs": N_ENVS, "n_steps": N_STEPS,
            "rollout_size": N_ENVS * N_STEPS,
            "requested_timesteps": 100_000,
            "actual_timesteps": model.num_timesteps,
            "ppo_n_updates": model._n_updates,
            "initial_policy_std": initial_std,
            "training_history": callback.training_history,
            "std_history": callback.std_history,
            "milestones": callback.milestones,
            "first_observed_full_success_threshold_timesteps":
                first_observed_thresholds(callback.milestones),
            "holdout_evaluation": holdout,
            "training_wall_s": training_wall_s,
            "total_wall_s_including_holdout": time.perf_counter() - started_at,
            "tensorboard_dir": str(paths["tensorboard_dir"]),
            "log_dir": str(paths["log_dir"]),
        }
        paths["report"].write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"seed": seed, "actual_timesteps": model.num_timesteps,
                          "fixed_full_successes":
                            callback.milestones["100k"]["evaluation"]["full"]
                            ["task_metrics"]["successes"],
                          "holdout_successes": holdout["task_metrics"]["successes"],
                          "report": str(paths["report"])}, indent=2), flush=True)
        return result
    finally:
        train_env.close()
        holdout_env.close()
        for env in evaluation_envs.values():
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=FRESH_TRAIN_SEEDS, required=True)
    arguments = parser.parse_args()
    train_one_seed(arguments.seed)


if __name__ == "__main__":
    main()
