"""One fresh Reward-V2/low-std PPO seed trained against fixed velocity PI."""

import json
import hashlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from attribution_audit import write_condition
from audit_ppo_robustness import MUJOCO_DIR, waypoint_sha256
from evaluate_ppo_pi import (compact_pi_evaluation, evaluate_pi_policy,
                             nominal_gate)
from ppo_pi_env import MineUAVPIEnv, make_pi_training_envs
from reward_v2_alignment_audit import fixed_waypoints
from train_ppo_lowstd_multiseed import (build_holdout_waypoints,
                                        expected_training_targets,
                                        initial_actor_hash,
                                        seed_training_envs)
from train_ppo_v2_lowstd import read_policy_std
from train_ppo_waypoint import N_ENVS, N_STEPS, TRAIN_METRICS, make_ppo


SEED = 0
LOG_STD_INIT = -2.0
MODEL_PREFIX = "ppo_waypoint_pi_lowstd"


def checkpoint_schedule() -> list[dict]:
    size = N_ENVS * N_STEPS
    return [{"label": label, "requested_timesteps": requested,
             "actual_timesteps": math.ceil(requested / size) * size}
            for label, requested in (("20k", 20_000), ("50k", 50_000),
                                     ("100k", 100_000))]


def output_paths(root: Path = MUJOCO_DIR, *, model_prefix: str = MODEL_PREFIX,
                 trace_scope: str = "ppo_pi_compatibility") -> dict[str, Path]:
    return {
        "model_dir": root / "rl" / "models",
        "log_dir": root / "rl" / "logs" / f"{model_prefix}_seed{SEED}",
        "tensorboard_dir": root / "rl" / "tensorboard" / f"{model_prefix}_seed{SEED}",
        "trace_dir": (root / "reports" / trace_scope /
                      "integral_traces" / f"train_seed_{SEED}"),
        "report": root / "reports" / f"{model_prefix}_seed{SEED}.json",
    }


def require_fresh_outputs(model_dir: Path, log_dir: Path,
                          tensorboard_dir: Path, trace_dir: Path,
                          report_path: Path, *,
                          model_prefix: str = MODEL_PREFIX) -> None:
    existing = [model_dir / f"{model_prefix}_{point['label']}.zip"
                for point in checkpoint_schedule()
                if (model_dir / f"{model_prefix}_{point['label']}.zip").exists()]
    if report_path.exists():
        existing.append(report_path)
    for directory in (log_dir, tensorboard_dir, trace_dir):
        if directory.exists() and any(directory.iterdir()):
            existing.append(directory)
    if existing:
        raise FileExistsError(f"Preserving PPO/PI experiment outputs: {existing}")


class PIMilestoneCallback(BaseCallback):
    """Observe each completed update and evaluate only updated checkpoints."""

    def __init__(self, evaluation_envs: dict, waypoints: dict, paths: dict,
                 model_prefix: str = MODEL_PREFIX):
        super().__init__()
        self.evaluation_envs = evaluation_envs
        self.waypoints = waypoints
        self.paths = paths
        self.model_prefix = model_prefix
        self.schedule = checkpoint_schedule()
        self.training_history = []
        self.milestones = {}
        self.recorded_steps = set()
        self.started_at = time.perf_counter()

    def _on_training_start(self) -> None:
        self.first_training_targets = [
            wrapped.unwrapped.target_position.astype(float).tolist()
            for wrapped in self.training_env.envs]

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self.record_updated_state()

    def record_updated_state(self) -> None:
        step = self.model.num_timesteps
        if step == 0 or step in self.recorded_steps:
            return
        rollout_size = N_ENVS * N_STEPS
        if step % rollout_size:
            raise RuntimeError(f"Checkpoint would follow incomplete rollout: {step}")
        if self.model._n_updates < step // rollout_size * self.model.n_epochs:
            raise RuntimeError(f"Checkpoint at {step} precedes PPO update")
        values = self.model.logger.name_to_value
        metrics = {name: float(values[f"train/{name}"]) for name in TRAIN_METRICS}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"Non-finite PPO metrics at {step}: {metrics}")
        recent = list(self.model.ep_info_buffer or [])
        record = {
            "timesteps": step,
            "rollout_ep_rew_mean": (float(np.mean([ep["r"] for ep in recent]))
                                    if recent else None),
            "rollout_ep_len_mean": (float(np.mean([ep["l"] for ep in recent]))
                                    if recent else None),
            "episode_count_in_window": len(recent),
            "train": metrics,
            "policy_std": read_policy_std(self.model),
        }
        self.training_history.append(record)
        with (self.paths["log_dir"] / "training_history.jsonl").open(
                "a", encoding="utf-8") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")
        self.recorded_steps.add(step)
        for point in self.schedule:
            if point["actual_timesteps"] != step:
                continue
            checkpoint = (self.paths["model_dir"] /
                          f"{self.model_prefix}_{point['label']}.zip")
            if checkpoint.exists():
                raise FileExistsError(checkpoint)
            self.model.save(checkpoint)
            evaluation = {}
            for task in ("benchmark", "holdout"):
                trace = (self.paths["trace_dir"] /
                         f"{point['label']}_{task}_integral_trace.json")
                full = evaluate_pi_policy(self.model, self.evaluation_envs[task],
                                          self.waypoints[task], trace)
                evaluation[task] = compact_pi_evaluation(full)
                if (full["seeds"] != [seed for seed, _ in self.waypoints[task]]
                        or full["targets"] != [target for _, target in self.waypoints[task]]):
                    raise RuntimeError(f"{task} waypoint pairing differs")
                self.logger.record(f"eval/{task}_success_rate",
                                   evaluation[task]["success_rate"])
            milestone = {
                **point, "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()).hexdigest(),
                "ppo_n_updates": self.model._n_updates,
                "elapsed_wall_s": time.perf_counter() - self.started_at,
                "policy_std": record["policy_std"],
                "evaluation": evaluation,
            }
            self.milestones[point["label"]] = milestone
            write_condition(self.paths["log_dir"] / f"{point['label']}_milestone.json",
                            milestone)
            print(f"PPO/PI {point['label']} at {step:,}: benchmark "
                  f"{evaluation['benchmark']['successes']}/100, holdout "
                  f"{evaluation['holdout']['successes']}/100", flush=True)


def run_training(output_root: Path = MUJOCO_DIR, *,
                 env_type=MineUAVPIEnv, model_prefix: str = MODEL_PREFIX,
                 trace_scope: str = "ppo_pi_compatibility",
                 expected_observation_shape: tuple[int, ...] = (7,),
                 experiment_description: str =
                 "Reward V2 PPO trained from scratch with fixed velocity PI") -> dict:
    """Train one fresh seed-0 PPO with fixed PI and isolated output names."""
    paths = output_paths(output_root, model_prefix=model_prefix,
                         trace_scope=trace_scope)
    require_fresh_outputs(paths["model_dir"], paths["log_dir"],
                          paths["tensorboard_dir"], paths["trace_dir"],
                          paths["report"], model_prefix=model_prefix)
    for key in ("model_dir", "log_dir", "tensorboard_dir", "trace_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_env = make_pi_training_envs(paths["log_dir"] / "monitor", n_envs=N_ENVS,
                                      env_type=env_type)
    evaluation_envs = {
        "benchmark": env_type(render_mode=None, reward_version="v2",
                              target_distribution="full"),
        "holdout": env_type(render_mode=None, reward_version="v2",
                            target_distribution="full"),
    }
    waypoints = {"benchmark": fixed_waypoints("full"),
                 "holdout": build_holdout_waypoints()}
    if len(waypoints["benchmark"]) != 100 or len(waypoints["holdout"]) != 100:
        raise RuntimeError("Need exact fixed 100 benchmark and holdout targets")
    started_at = time.perf_counter()
    try:
        if (train_env.observation_space.shape != expected_observation_shape
                or any(env.observation_space.shape != expected_observation_shape
                       for env in evaluation_envs.values())):
            raise ValueError("Training/evaluation observation shape differs")
        rank_seeds = seed_training_envs(train_env, SEED)
        model = make_ppo(train_env, paths["tensorboard_dir"],
                         log_std_init=LOG_STD_INIT, seed=SEED)
        model.verbose = 0
        if model.seed != SEED or model.n_envs != N_ENVS or model.num_timesteps != 0:
            raise RuntimeError("PPO must be new, seed 0 and eight environments")
        initial_std = read_policy_std(model)
        if initial_std["log_std"] != [LOG_STD_INIT] * 4:
            raise RuntimeError("Unexpected initial exploration variance")
        actor_hash = initial_actor_hash(model)
        callback = PIMilestoneCallback(evaluation_envs, waypoints, paths,
                                       model_prefix=model_prefix)
        budget = checkpoint_schedule()[-1]["actual_timesteps"]
        model.learn(total_timesteps=budget, callback=callback,
                    tb_log_name="PPO", log_interval=1)
        if model.num_timesteps != budget:
            raise RuntimeError(f"Wrong timestep count: {model.num_timesteps}")
        callback.record_updated_state()
        if set(callback.milestones) != {"20k", "50k", "100k"}:
            raise RuntimeError("Missing 20k/50k/100k PI checkpoint")
        if callback.first_training_targets != expected_training_targets(SEED):
            raise RuntimeError("Initial rank-specific Gymnasium targets differ")
        model.dump_logs(iteration=budget // (N_ENVS * N_STEPS))
        final = callback.milestones["100k"]["evaluation"]
        gate = nominal_gate(
            {"task_metrics": final["benchmark"]},
            {"task_metrics": final["holdout"]})
        result = {
            "experiment": experiment_description,
            "seed": SEED, "initialization": "random_from_scratch",
            "randomness": {
                "python_random_seed": SEED, "numpy_seed": SEED,
                "pytorch_seed": SEED, "sb3_seed": model.seed,
                "gymnasium_initial_reset_rank_seeds": rank_seeds,
                "first_training_targets_m": callback.first_training_targets,
                "initial_actor_action_weight_sha256": actor_hash,
            },
            "controller": {
                "mode": "velocity_pi", "ki_xy_s2": 0.5, "ki_z_s2": 0.8,
                "integral_accel_limit_xy_m_s2": 1.5,
                "integral_accel_limit_z_m_s2": 1.5,
            },
            "reward_version": "v2",
            "observation_shape": list(expected_observation_shape),
            "action_shape": [4], "physics_hz": 500, "controller_hz": 100,
            "policy_hz": 25, "n_envs": N_ENVS, "n_steps": N_STEPS,
            "rollout_buffer_size": N_ENVS * N_STEPS,
            "requested_timesteps": 100_000, "actual_timesteps": budget,
            "ppo_n_updates": model._n_updates,
            "initial_policy_std": initial_std,
            "waypoint_sha256": {task: waypoint_sha256(values)
                                    for task, values in waypoints.items()},
            "training_history": callback.training_history,
            "milestones": callback.milestones,
            "nominal_100k_gate_requires_both_sets_90pct": gate,
            "training_wall_s_including_milestone_evaluations":
                time.perf_counter() - started_at,
            "tensorboard_dir": str(paths["tensorboard_dir"]),
            "log_dir": str(paths["log_dir"]),
        }
        write_condition(paths["report"], result)
        return result
    finally:
        train_env.close()
        for env in evaluation_envs.values():
            env.close()


if __name__ == "__main__":
    report = run_training()
    print(json.dumps({
        "actual_timesteps": report["actual_timesteps"],
        "benchmark_successes": report["milestones"]["100k"]
            ["evaluation"]["benchmark"]["successes"],
        "holdout_successes": report["milestones"]["100k"]
            ["evaluation"]["holdout"]["successes"],
        "nominal_gate_passed": report["nominal_100k_gate_requires_both_sets_90pct"],
        "report": str(output_paths()["report"]),
    }, indent=2), flush=True)
