"""Independent, from-scratch 100k PPO experiment using Reward V3.

V3 changes only the environment reward: V2 plus -0.02 * current distance.
The PPO factory, seeds, vectorization and deterministic diagnostics are
shared with the baseline/V2 experiments.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from mine_uav_env import MineUAVEnv
from reward_v2_diagnostics import EVAL_SEED, evaluate_diagnostics
from train_ppo_waypoint import N_ENVS, N_STEPS, SEED, TRAIN_METRICS, make_ppo, make_training_envs


RL_DIR = Path(__file__).resolve().parent
MODEL_DIR = RL_DIR / "models"
LOG_DIR = RL_DIR / "logs" / "ppo_waypoint_reward_v3"
TENSORBOARD_DIR = RL_DIR / "tensorboard" / "ppo_waypoint_reward_v3"
REPORT_PATH = RL_DIR.parent / "reports" / "ppo_waypoint_reward_v3.json"
PRIOR_REPORT_PATH = RL_DIR.parent / "reports" / "ppo_waypoint_reward_v2.json"
BASELINE_100K = MODEL_DIR / "ppo_waypoint_100k.zip"
V2_100K = MODEL_DIR / "ppo_waypoint_reward_v2_100k.zip"


def checkpoint_schedule() -> list[dict]:
    rollout_size = N_ENVS * N_STEPS
    return [{"label": label, "requested_timesteps": requested,
             "actual_timesteps": math.ceil(requested / rollout_size) * rollout_size}
            for label, requested in (("20k", 20_000), ("50k", 50_000),
                                     ("100k", 100_000))]


def require_fresh_v3_artifacts(model_dir: Path, report_path: Path,
                               log_dir: Path, tensorboard_dir: Path) -> None:
    """Never overwrite a V3 run; baseline and V2 paths are never targets."""
    existing = [model_dir / f"ppo_waypoint_reward_v3_{entry['label']}.zip"
                for entry in checkpoint_schedule()
                if (model_dir / f"ppo_waypoint_reward_v3_{entry['label']}.zip").exists()]
    if report_path.exists():
        existing.append(report_path)
    for directory in (log_dir, tensorboard_dir):
        if directory.exists() and any(directory.iterdir()):
            existing.append(directory)
    if existing:
        raise FileExistsError(f"Preserving existing Reward V3 artifacts: {existing}")


def validated_prior_comparison(prior_report_path: Path, v3_evaluation: dict) -> dict:
    """Use prior 100k diagnostics only when all 100 seeds and targets match."""
    report = json.loads(prior_report_path.read_text(encoding="utf-8"))
    baseline = report["baseline_100k_reevaluation"]
    reward_v2 = report["milestones"]["100k"]["evaluation"]
    expected_seeds = list(range(EVAL_SEED, EVAL_SEED + 100))
    for label, evaluation in (("baseline", baseline), ("Reward V2", reward_v2),
                              ("Reward V3", v3_evaluation)):
        if evaluation["seeds"] != expected_seeds:
            raise ValueError(f"{label} evaluation seeds differ from fixed 100-waypoint set")
        if len(evaluation["episode_summaries"]) != 100:
            raise ValueError(f"{label} evaluation is not 100 episodes")
        if [entry["seed"] for entry in evaluation["episode_summaries"]] != expected_seeds:
            raise ValueError(f"{label} episode seed order differs from fixed waypoint set")
    targets = [[entry["target_position_m"] for entry in evaluation["episode_summaries"]]
               for evaluation in (baseline, reward_v2, v3_evaluation)]
    if targets[0] != targets[1] or targets[0] != targets[2]:
        raise ValueError("Baseline, V2 and V3 waypoint coordinates differ")
    return {
        "fixed_evaluation_seed": EVAL_SEED,
        "same_seeds_and_waypoints": True,
        "prior_report": str(prior_report_path),
        "baseline_100k": baseline,
        "reward_v2_100k": reward_v2,
        "reward_v3_100k": v3_evaluation,
    }


class RewardV3Callback(BaseCallback):
    """Capture updated policies after complete rollouts and fixed evaluations."""

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
            checkpoint = self.model_dir / f"ppo_waypoint_reward_v3_{label}.zip"
            if checkpoint.exists():
                raise FileExistsError(checkpoint)
            self.model.save(checkpoint)
            print(f"Reward V3 {label}: updated {step:,} timesteps; evaluating 100 fixed seeds",
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
            print(f"Reward V3 {label}: {evaluation['successes']}/100 success, "
                  f"mean final distance {evaluation['mean_final_distance_m']:.3f} m",
                  flush=True)


def run_reward_v3_training() -> dict:
    require_fresh_v3_artifacts(MODEL_DIR, REPORT_PATH, LOG_DIR, TENSORBOARD_DIR)
    for required in (BASELINE_100K, V2_100K, PRIOR_REPORT_PATH):
        if not required.is_file():
            raise FileNotFoundError(f"Required prior comparison artifact: {required}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_env = make_training_envs(LOG_DIR / "monitor", n_envs=N_ENVS,
                                   reward_version="v3")
    eval_env = Monitor(MineUAVEnv(render_mode=None, reward_version="v3"),
                       filename=str(LOG_DIR / "evaluation"),
                       info_keywords=("distance_m", "termination_reason"))
    started_at = time.perf_counter()
    try:
        model = make_ppo(train_env, TENSORBOARD_DIR)  # Fresh weights, unchanged PPO config.
        model.verbose = 0
        callback = RewardV3Callback(eval_env, MODEL_DIR, LOG_DIR)
        total_timesteps = checkpoint_schedule()[-1]["actual_timesteps"]
        model.learn(total_timesteps=total_timesteps, callback=callback,
                    tb_log_name="PPO", log_interval=1)
        if model.num_timesteps != total_timesteps:
            raise RuntimeError(f"Training timestep budget mismatch: {model.num_timesteps}")
        callback.record_updated_state()  # Last update has no next rollout-start hook.
        if set(callback.milestones) != {"20k", "50k", "100k"}:
            raise RuntimeError("Missing Reward V3 milestone")
        model.dump_logs(iteration=total_timesteps // (N_ENVS * N_STEPS))
        comparison = validated_prior_comparison(
            PRIOR_REPORT_PATH, callback.milestones["100k"]["evaluation"])
        result = {
            "reward_version": "v3",
            "brake_lambda": 0.5,
            "distance_lambda": 0.02,
            "initialization": "from_scratch",
            "seed": SEED,
            "evaluation_seed": EVAL_SEED,
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "rollout_size": N_ENVS * N_STEPS,
            "requested_timesteps": 100_000,
            "actual_timesteps": model.num_timesteps,
            "ppo_n_updates": model._n_updates,
            "train_and_v3_evaluation_wall_s": time.perf_counter() - started_at,
            "training_history": callback.training_history,
            "milestones": callback.milestones,
            "comparison_100k": comparison,
            "tensorboard_dir": str(TENSORBOARD_DIR),
            "log_dir": str(LOG_DIR),
        }
        REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        eval_env.close()
        train_env.close()


if __name__ == "__main__":
    report = run_reward_v3_training()
    print(json.dumps({
        "actual_timesteps": report["actual_timesteps"],
        "train_and_v3_evaluation_wall_s": report["train_and_v3_evaluation_wall_s"],
        "v3_100k_success_rate": report["milestones"]["100k"]["evaluation"]["success_rate"],
        "report": str(REPORT_PATH),
    }, indent=2))
