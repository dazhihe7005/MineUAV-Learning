"""Independent, fixed-configuration 100k PPO experiment with Reward V2.

All PPO settings and the training seed come from the baseline factory. The
only environment difference is the explicitly selected braking reward.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from mine_uav_env import MineUAVEnv
from reward_v2_diagnostics import EVAL_SEED, evaluate_diagnostics
from train_ppo_waypoint import N_ENVS, N_STEPS, SEED, TRAIN_METRICS, make_ppo, make_training_envs


RL_DIR = Path(__file__).resolve().parent
MODEL_DIR = RL_DIR / "models"
LOG_DIR = RL_DIR / "logs" / "ppo_waypoint_reward_v2"
TENSORBOARD_DIR = RL_DIR / "tensorboard" / "ppo_waypoint_reward_v2"
REPORT_PATH = RL_DIR.parent / "reports" / "ppo_waypoint_reward_v2.json"
BASELINE_100K = MODEL_DIR / "ppo_waypoint_100k.zip"


def checkpoint_schedule() -> list[dict]:
    rollout_size = N_ENVS * N_STEPS
    return [{"label": label, "requested_timesteps": requested,
             "actual_timesteps": math.ceil(requested / rollout_size) * rollout_size}
            for label, requested in (("20k", 20_000), ("50k", 50_000),
                                     ("100k", 100_000))]


def require_fresh_v2_artifacts(model_dir: Path, report_path: Path,
                               log_dir: Path, tensorboard_dir: Path) -> None:
    """Do not overwrite either prior V2 runs or the original baseline."""
    existing = [model_dir / f"ppo_waypoint_reward_v2_{entry['label']}.zip"
                for entry in checkpoint_schedule()
                if (model_dir / f"ppo_waypoint_reward_v2_{entry['label']}.zip").exists()]
    if report_path.exists():
        existing.append(report_path)
    for directory in (log_dir, tensorboard_dir):
        if directory.exists() and any(directory.iterdir()):
            existing.append(directory)
    if existing:
        raise FileExistsError(f"Preserving existing Reward V2 artifacts: {existing}")


class RewardV2Callback(BaseCallback):
    """Save policies and evaluate only after complete rollout + PPO update."""

    def __init__(self, eval_env, model_dir: Path, log_dir: Path):
        super().__init__()
        self.eval_env = eval_env
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.schedule = checkpoint_schedule()
        self.training_history = []
        self.milestones = {}
        self.recorded_steps = set()
        self.started_at = time.perf_counter()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self.record_updated_state()

    def record_evaluation_metrics(self, evaluation: dict) -> None:
        self.logger.record("eval/success_rate", evaluation["success_rate"])
        self.logger.record("eval/mean_final_distance_m", evaluation["mean_final_distance_m"])
        self.logger.record("eval/speed_samples_within_0_1_m",
                           evaluation["speed_samples_within_0_1_m"])
        near_speed = evaluation["mean_speed_within_0_1_m_s"]
        if near_speed is not None:
            self.logger.record("eval/mean_speed_within_0_1_m_s", near_speed)

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
        recent_episodes = list(self.model.ep_info_buffer or [])
        record = {
            "timesteps": step,
            "rollout_ep_rew_mean": (float(np.mean([item["r"] for item in recent_episodes]))
                                    if recent_episodes else None),
            "rollout_ep_len_mean": (float(np.mean([item["l"] for item in recent_episodes]))
                                    if recent_episodes else None),
            "episode_count_in_window": len(recent_episodes),
            "train": metrics,
        }
        self.training_history.append(record)
        with (self.log_dir / "training_history.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")
        self.recorded_steps.add(step)
        for milestone in self.schedule:
            if milestone["actual_timesteps"] != step:
                continue
            label = milestone["label"]
            checkpoint = self.model_dir / f"ppo_waypoint_reward_v2_{label}.zip"
            if checkpoint.exists():
                raise FileExistsError(checkpoint)
            self.model.save(checkpoint)
            print(f"Reward V2 {label}: updated {step:,} timesteps; evaluating 100 fixed seeds",
                  flush=True)
            evaluation = evaluate_diagnostics(self.model, self.eval_env,
                                              episodes=100, seed=EVAL_SEED)
            self.milestones[label] = {
                **milestone,
                "checkpoint": str(checkpoint),
                "ppo_n_updates": self.model._n_updates,
                "elapsed_wall_s": time.perf_counter() - self.started_at,
                "evaluation": evaluation,
            }
            (self.log_dir / "milestones.json").write_text(
                json.dumps(self.milestones, indent=2) + "\n", encoding="utf-8")
            self.record_evaluation_metrics(evaluation)
            print(f"Reward V2 {label}: {evaluation['successes']}/100 success, "
                  f"mean final distance {evaluation['mean_final_distance_m']:.3f} m",
                  flush=True)


def run_reward_v2_training() -> dict:
    require_fresh_v2_artifacts(MODEL_DIR, REPORT_PATH, LOG_DIR, TENSORBOARD_DIR)
    if not BASELINE_100K.is_file():
        raise FileNotFoundError(f"Baseline 100k checkpoint for comparison: {BASELINE_100K}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_env = make_training_envs(LOG_DIR / "monitor", n_envs=N_ENVS,
                                   reward_version="v2")
    eval_env = Monitor(MineUAVEnv(render_mode=None, reward_version="v2"),
                       filename=str(LOG_DIR / "evaluation"),
                       info_keywords=("distance_m", "termination_reason"))
    baseline_eval_env = MineUAVEnv(render_mode=None, reward_version="v1")
    started_at = time.perf_counter()
    try:
        model = make_ppo(train_env, TENSORBOARD_DIR)  # Fresh weights; unchanged PPO configuration.
        model.verbose = 0
        callback = RewardV2Callback(eval_env, MODEL_DIR, LOG_DIR)
        total_timesteps = checkpoint_schedule()[-1]["actual_timesteps"]
        model.learn(total_timesteps=total_timesteps, callback=callback,
                    tb_log_name="PPO", log_interval=1)
        if model.num_timesteps != total_timesteps:
            raise RuntimeError(f"Training timestep budget mismatch: {model.num_timesteps}")
        callback.record_updated_state()  # Last PPO update has no next rollout_start hook.
        if set(callback.milestones) != {"20k", "50k", "100k"}:
            raise RuntimeError("Missing Reward V2 milestone")
        model.dump_logs(iteration=total_timesteps // (N_ENVS * N_STEPS))
        train_and_v2_evaluation_wall_s = time.perf_counter() - started_at
        baseline_model = PPO.load(BASELINE_100K, device="cpu")
        baseline_eval = evaluate_diagnostics(baseline_model, baseline_eval_env,
                                             episodes=100, seed=EVAL_SEED)
        if baseline_eval["seeds"] != callback.milestones["100k"]["evaluation"]["seeds"]:
            raise RuntimeError("Baseline and Reward V2 evaluation seeds differ")
        result = {
            "reward_version": "v2",
            "brake_lambda": 0.5,
            "initialization": "from_scratch",
            "seed": SEED,
            "evaluation_seed": EVAL_SEED,
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "rollout_size": N_ENVS * N_STEPS,
            "requested_timesteps": 100_000,
            "actual_timesteps": model.num_timesteps,
            "ppo_n_updates": model._n_updates,
            "train_and_v2_evaluation_wall_s": train_and_v2_evaluation_wall_s,
            "total_wall_s_including_baseline_reevaluation": time.perf_counter() - started_at,
            "training_history": callback.training_history,
            "milestones": callback.milestones,
            "baseline_100k_reevaluation": baseline_eval,
            "baseline_100k_checkpoint": str(BASELINE_100K),
            "tensorboard_dir": str(TENSORBOARD_DIR),
            "log_dir": str(LOG_DIR),
        }
        REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        baseline_eval_env.close()
        eval_env.close()
        train_env.close()


if __name__ == "__main__":
    report = run_reward_v2_training()
    print(json.dumps({
        "actual_timesteps": report["actual_timesteps"],
        "train_and_v2_evaluation_wall_s": report["train_and_v2_evaluation_wall_s"],
        "v2_100k_success_rate": report["milestones"]["100k"]["evaluation"]["success_rate"],
        "baseline_100k_success_rate": report["baseline_100k_reevaluation"]["success_rate"],
        "report": str(REPORT_PATH),
    }, indent=2))
