"""Reward V2 PPO single-variable experiment: initial Gaussian log_std=-2."""

import json
import math
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from mine_uav_env import MineUAVEnv
from ppo_exploration_audit import COMMAND_SCALES, evaluate_exploration
from reward_v2_alignment_audit import fixed_waypoints
from train_ppo_waypoint import (N_ENVS, N_STEPS, SEED, TRAIN_METRICS,
                                make_ppo, make_training_envs)


RL_DIR = Path(__file__).resolve().parent
MODEL_DIR = RL_DIR / "models"
LOG_DIR = RL_DIR / "logs" / "ppo_waypoint_v2_lowstd"
TENSORBOARD_DIR = RL_DIR / "tensorboard" / "ppo_waypoint_v2_lowstd"
REPORT_PATH = RL_DIR.parent / "reports" / "ppo_waypoint_v2_lowstd.json"
BASELINE_CHECKPOINT = MODEL_DIR / "ppo_waypoint_reward_v2_100k.zip"
LOG_STD_INIT = -2.0


def checkpoint_schedule() -> list[dict]:
    rollout_size = N_ENVS * N_STEPS
    return [{"label": label, "requested_timesteps": requested,
             "actual_timesteps": math.ceil(requested / rollout_size) * rollout_size}
            for label, requested in (("20k", 20_000), ("50k", 50_000),
                                     ("100k", 100_000))]


def require_fresh_outputs(model_dir: Path, report_path: Path,
                          log_dir: Path, tensorboard_dir: Path) -> None:
    existing = [model_dir / f"ppo_waypoint_v2_lowstd_{point['label']}.zip"
                for point in checkpoint_schedule()
                if (model_dir / f"ppo_waypoint_v2_lowstd_{point['label']}.zip").exists()]
    if report_path.exists():
        existing.append(report_path)
    for directory in (log_dir, tensorboard_dir):
        if directory.exists() and any(directory.iterdir()):
            existing.append(directory)
    if existing:
        raise FileExistsError(f"Preserving existing low-std outputs: {existing}")


def read_policy_std(model: PPO) -> dict:
    """Record the learned global Gaussian standard deviation, not entropy."""
    parameter = dict(model.policy.named_parameters()).get("log_std")
    if parameter is None or parameter.shape != (4,) or not parameter.requires_grad:
        raise ValueError("Expected four trainable state-independent log_std values")
    values = parameter.detach().cpu().numpy().astype(float)
    std = np.exp(values)
    if not np.isfinite(std).all():
        raise ValueError("Non-finite policy standard deviation")
    return {"log_std": values.tolist(), "std": std.tolist(),
            "physical_command_std": (std * COMMAND_SCALES).tolist()}


class LowStdCallback(BaseCallback):
    """Record each completed update; evaluate/save at updated milestones."""

    def __init__(self, evaluation_envs: dict, waypoints: dict,
                 model_dir: Path, log_dir: Path):
        super().__init__()
        self.evaluation_envs = evaluation_envs
        self.waypoints = waypoints
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.schedule = checkpoint_schedule()
        self.training_history = []
        self.std_history = []
        self.milestones = {}
        self.recorded_steps = set()
        self.started_at = time.perf_counter()

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
            raise RuntimeError(f"Incomplete PPO rollout at {step}")
        if self.model._n_updates < step // rollout_size * self.model.n_epochs:
            raise RuntimeError(f"Checkpoint at {step} precedes PPO update")
        values = self.model.logger.name_to_value
        metrics = {name: float(values[f"train/{name}"]) for name in TRAIN_METRICS}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"Non-finite PPO metrics at {step}: {metrics}")
        recent = list(self.model.ep_info_buffer or [])
        std = read_policy_std(self.model)
        record = {
            "timesteps": step,
            "rollout_ep_rew_mean": (float(np.mean([ep["r"] for ep in recent]))
                                    if recent else None),
            "rollout_ep_len_mean": (float(np.mean([ep["l"] for ep in recent]))
                                    if recent else None),
            "episode_count_in_window": len(recent),
            "train": metrics,
            "policy_std": std,
        }
        self.training_history.append(record)
        self.std_history.append({"timesteps": step, **std})
        with (self.log_dir / "training_history.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")
        self.recorded_steps.add(step)
        for point in self.schedule:
            if point["actual_timesteps"] != step:
                continue
            checkpoint = self.model_dir / f"ppo_waypoint_v2_lowstd_{point['label']}.zip"
            if checkpoint.exists():
                raise FileExistsError(checkpoint)
            self.model.save(checkpoint)
            print(f"Low-std {point['label']}: updated {step:,}; evaluating local/full 100 each",
                  flush=True)
            evaluation = {}
            for task in ("local", "full"):
                evaluation[task] = evaluate_exploration(
                    self.model, self.evaluation_envs[task], self.waypoints[task], True)
                if evaluation[task]["targets"] != [target for _, target in self.waypoints[task]]:
                    raise RuntimeError(f"{task} fixed-waypoint mismatch")
            self.milestones[point["label"]] = {
                **point, "checkpoint": str(checkpoint), "ppo_n_updates": self.model._n_updates,
                "elapsed_wall_s": time.perf_counter() - self.started_at,
                "policy_std": std, "evaluation": evaluation,
            }
            (self.log_dir / "milestones.json").write_text(
                json.dumps(self.milestones, indent=2, allow_nan=False) + "\n",
                encoding="utf-8")
            for task in ("local", "full"):
                self.logger.record(f"eval/{task}_success_rate",
                                   evaluation[task]["task_metrics"]["success_rate"])
            print(f"Low-std {point['label']}: local "
                  f"{evaluation['local']['task_metrics']['successes']}/100, "
                  f"full {evaluation['full']['task_metrics']['successes']}/100",
                  flush=True)


def run_lowstd_training() -> dict:
    require_fresh_outputs(MODEL_DIR, REPORT_PATH, LOG_DIR, TENSORBOARD_DIR)
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(BASELINE_CHECKPOINT)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    waypoints = {task: fixed_waypoints(task) for task in ("local", "full")}
    train_env = make_training_envs(LOG_DIR / "monitor", n_envs=N_ENVS,
                                   reward_version="v2", target_distribution="full")
    evaluation_envs = {
        "local": MineUAVEnv(render_mode=None, reward_version="v2",
                            target_distribution="local_a"),
        "full": MineUAVEnv(render_mode=None, reward_version="v2",
                           target_distribution="full"),
    }
    started_at = time.perf_counter()
    try:
        model = make_ppo(train_env, TENSORBOARD_DIR, log_std_init=LOG_STD_INIT)
        model.verbose = 0
        initial_std = read_policy_std(model)
        if not np.allclose(initial_std["log_std"], [LOG_STD_INIT] * 4, rtol=0, atol=0):
            raise RuntimeError("Low-std initial policy differs from requested value")
        callback = LowStdCallback(evaluation_envs, waypoints, MODEL_DIR, LOG_DIR)
        budget = checkpoint_schedule()[-1]["actual_timesteps"]
        model.learn(total_timesteps=budget, callback=callback,
                    tb_log_name="PPO", log_interval=1)
        if model.num_timesteps != budget:
            raise RuntimeError(f"Training timestep budget mismatch: {model.num_timesteps}")
        callback.record_updated_state()
        if set(callback.milestones) != {"20k", "50k", "100k"}:
            raise RuntimeError("Missing low-std milestone")
        model.dump_logs(iteration=budget // (N_ENVS * N_STEPS))
        experiment_wall_s = time.perf_counter() - started_at
        baseline_model = PPO.load(BASELINE_CHECKPOINT, device="cpu")
        baseline_evaluation = {
            task: evaluate_exploration(baseline_model, evaluation_envs[task],
                                       waypoints[task], True)
            for task in ("local", "full")}
        for task in ("local", "full"):
            if baseline_evaluation[task]["targets"] != (
                    callback.milestones["100k"]["evaluation"][task]["targets"]):
                raise RuntimeError(f"Baseline/{task} low-std waypoint mismatch")
        result = {
            "reward_version": "v2", "initialization": "from_scratch",
            "single_changed_ppo_parameter": {"log_std_init": LOG_STD_INIT},
            "seed": SEED, "n_envs": N_ENVS, "n_steps": N_STEPS,
            "rollout_size": N_ENVS * N_STEPS,
            "requested_timesteps": 100_000, "actual_timesteps": model.num_timesteps,
            "ppo_n_updates": model._n_updates,
            "initial_policy_std": initial_std,
            "training_history": callback.training_history,
            "std_history": callback.std_history,
            "milestones": callback.milestones,
            "baseline_v2_100k": {"checkpoint": str(BASELINE_CHECKPOINT),
                                 "policy_std": read_policy_std(baseline_model),
                                 "evaluation": baseline_evaluation},
            "experiment_wall_s": experiment_wall_s,
            "total_wall_s_including_baseline_evaluation": time.perf_counter() - started_at,
            "tensorboard_dir": str(TENSORBOARD_DIR), "log_dir": str(LOG_DIR),
        }
        REPORT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        return result
    finally:
        for env in evaluation_envs.values():
            env.close()
        train_env.close()


if __name__ == "__main__":
    report = run_lowstd_training()
    print(json.dumps({
        "actual_timesteps": report["actual_timesteps"],
        "experiment_wall_s": report["experiment_wall_s"],
        "local_successes": report["milestones"]["100k"]["evaluation"]["local"]["task_metrics"]["successes"],
        "full_successes": report["milestones"]["100k"]["evaluation"]["full"]["task_metrics"]["successes"],
        "report": str(REPORT_PATH),
    }, indent=2))
