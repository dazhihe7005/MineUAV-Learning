"""Fixed-configuration, from-scratch MineUAV PPO baseline through 200k.

The four checkpoints are taken only after complete PPO rollouts/updates.
With 8 x 256 = 2,048 transitions per update, the final full update not
exceeding 200,000 timesteps is at 198,656.
"""

import json
import math
import sys
import time
from pathlib import Path

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from mine_uav_env import MineUAVEnv
from train_ppo_waypoint import (N_ENVS, N_STEPS, SEED, TRAIN_METRICS,
                                make_ppo, make_training_envs)


EVAL_SEED = 20261001
FINAL_EVAL_SEED = 20271001
RL_DIR = Path(__file__).resolve().parent
MODEL_DIR = RL_DIR / "models"
LOG_DIR = RL_DIR / "logs" / "ppo_waypoint_baseline"
TENSORBOARD_DIR = RL_DIR / "tensorboard" / "ppo_waypoint_baseline"
REPORT_PATH = RL_DIR.parent / "reports" / "ppo_waypoint_baseline.json"


def checkpoint_schedule() -> list[dict]:
    """Map requested milestones to complete rollouts without exceeding 200k."""
    rollout_size = N_ENVS * N_STEPS
    requests = (("20k", 20_000), ("50k", 50_000),
                ("100k", 100_000), ("200k", 200_000))
    return [
        {
            "label": label,
            "requested_timesteps": requested,
            "actual_timesteps": ((requested // rollout_size if label == "200k"
                                  else math.ceil(requested / rollout_size)) * rollout_size),
        }
        for label, requested in requests
    ]


def evaluate_baseline(model, env: MineUAVEnv, episodes: int = 100,
                      seed: int = EVAL_SEED) -> dict:
    """Evaluate deterministic policy on seeded waypoints and log behavior."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    base_env = env.unwrapped
    summaries = []
    all_actions = []
    rewards = []
    for episode in range(episodes):
        episode_seed = seed + episode
        observation, _ = env.reset(seed=episode_seed)
        target = base_env.target_position.tolist()
        initial_distance = float(np.linalg.norm(base_env.target_position - base_env.data.qpos[:3]))
        total_reward = 0.0
        min_distance = initial_distance
        max_streak = 0
        within_distance_steps = 0
        within_both_steps = 0
        speeds_near_target = []
        for step in range(base_env.max_episode_steps):
            action, _ = model.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=float)
            if action.shape != (4,) or not np.isfinite(action).all():
                raise ValueError("Evaluation policy returned a non-finite or non-4D action")
            all_actions.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            if not math.isfinite(reward):
                raise ValueError("Evaluation produced a non-finite reward")
            total_reward += float(reward)
            distance = float(info["distance_m"])
            speed = float(info["speed_m_s"])
            min_distance = min(min_distance, distance)
            max_streak = max(max_streak, int(info["success_streak"]))
            if distance < 0.10:
                within_distance_steps += 1
                if speed < 0.15:
                    within_both_steps += 1
            if distance < 0.30:
                speeds_near_target.append(speed)
            if terminated or truncated:
                reason = info["termination_reason"]
                summary = {
                    "seed": episode_seed,
                    "target_position_m": target,
                    "initial_distance_m": initial_distance,
                    "final_distance_m": distance,
                    "min_distance_m": min_distance,
                    "final_speed_m_s": speed,
                    "episode_length": step + 1,
                    "episode_reward": total_reward,
                    "termination_reason": reason,
                    "max_success_streak": max_streak,
                    "steps_within_0_10_m": within_distance_steps,
                    "steps_within_distance_and_speed": within_both_steps,
                    "mean_speed_within_0_30_m_s": (float(np.mean(speeds_near_target))
                                                   if speeds_near_target else None),
                }
                summaries.append(summary)
                rewards.append(total_reward)
                break
        else:
            raise RuntimeError("Evaluation episode exceeded environment time limit")
    actions = np.asarray(all_actions)
    successes = [entry for entry in summaries if entry["termination_reason"] == "success"]
    return {
        "episodes": episodes,
        "seeds": list(range(seed, seed + episodes)),
        "successes": len(successes),
        "success_rate": len(successes) / episodes,
        "mean_final_distance_m": float(np.mean([entry["final_distance_m"] for entry in summaries])),
        "max_final_distance_m": float(max(entry["final_distance_m"] for entry in summaries)),
        "mean_episode_length": float(np.mean([entry["episode_length"] for entry in summaries])),
        "mean_reward": float(np.mean(rewards)),
        "mean_completion_time_s": (float(np.mean([entry["episode_length"] * base_env.policy_dt
                                                   for entry in successes])) if successes else None),
        "ever_within_0_10_m_count": sum(entry["steps_within_0_10_m"] > 0 for entry in summaries),
        "ever_distance_and_speed_count": sum(entry["steps_within_distance_and_speed"] > 0
                                             for entry in summaries),
        "actions": {
            "p05_per_axis": np.quantile(actions, 0.05, axis=0).tolist(),
            "p50_per_axis": np.quantile(actions, 0.50, axis=0).tolist(),
            "p95_per_axis": np.quantile(actions, 0.95, axis=0).tolist(),
            "max_abs_per_axis": np.max(np.abs(actions), axis=0).tolist(),
            "max_abs_overall": float(np.max(np.abs(actions))),
            "near_boundary_fraction_per_axis": np.mean(np.abs(actions) >= 0.95, axis=0).tolist(),
            "near_boundary_fraction_overall": float(np.mean(np.abs(actions) >= 0.95)),
            "near_boundary_threshold": 0.95,
        },
        "episode_summaries": summaries,
    }


def require_fresh_baseline_artifacts(model_dir: Path) -> None:
    """Never silently overwrite checkpoints from an existing baseline run."""
    existing = [model_dir / f"ppo_waypoint_{entry['label']}.zip"
                for entry in checkpoint_schedule()
                if (model_dir / f"ppo_waypoint_{entry['label']}.zip").exists()]
    if existing:
        raise FileExistsError(f"Preserving existing baseline checkpoint(s): {existing}")


class BaselineCallback(BaseCallback):
    """Record trained-policy metrics and evaluate after completed PPO updates."""

    def __init__(self, eval_env: MineUAVEnv, model_dir: Path, log_dir: Path,
                 eval_episodes: int = 100):
        super().__init__()
        self.eval_env = eval_env
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.eval_episodes = eval_episodes
        self.schedule = checkpoint_schedule()
        self.training_history = []
        self.milestone_results = {}
        self.recorded_steps = set()
        self.started_at = time.perf_counter()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        # This hook occurs after the previous completed rollout's train().
        self.record_updated_state()

    def record_updated_state(self) -> None:
        step = self.model.num_timesteps
        if step == 0 or step in self.recorded_steps:
            return
        rollout_size = N_ENVS * N_STEPS
        if step % rollout_size:
            raise RuntimeError(f"Incomplete rollout at {step} timesteps")
        expected_updates = (step // rollout_size) * self.model.n_epochs
        if self.model._n_updates < expected_updates:
            raise RuntimeError(f"Checkpoint at {step} would be before PPO update")
        values = self.model.logger.name_to_value
        train = {name: float(values[f"train/{name}"]) for name in TRAIN_METRICS}
        if not all(math.isfinite(value) for value in train.values()):
            raise RuntimeError(f"Non-finite PPO metrics at step {step}: {train}")
        episodes = list(self.model.ep_info_buffer or [])
        record = {
            "timesteps": step,
            "rollout_ep_rew_mean": (float(np.mean([entry["r"] for entry in episodes]))
                                    if episodes else None),
            "rollout_ep_len_mean": (float(np.mean([entry["l"] for entry in episodes]))
                                    if episodes else None),
            "episode_count_in_window": len(episodes),
            "train": train,
        }
        self.training_history.append(record)
        with (self.log_dir / "training_history.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")
        self.recorded_steps.add(step)
        for milestone in self.schedule:
            label = milestone["label"]
            if milestone["actual_timesteps"] != step or label in self.milestone_results:
                continue
            checkpoint = self.model_dir / f"ppo_waypoint_{label}.zip"
            if checkpoint.exists():
                raise FileExistsError(checkpoint)
            self.model.save(checkpoint)
            print(f"{label} checkpoint saved at {step:,} updated timesteps; "
                  f"evaluating {self.eval_episodes} fixed waypoints", flush=True)
            evaluation = evaluate_baseline(self.model, self.eval_env,
                                           episodes=self.eval_episodes, seed=EVAL_SEED)
            self.milestone_results[label] = {
                **milestone,
                "checkpoint": str(checkpoint),
                "ppo_n_updates": self.model._n_updates,
                "evaluation": evaluation,
                "elapsed_wall_s": time.perf_counter() - self.started_at,
            }
            (self.log_dir / "milestones.json").write_text(
                json.dumps(self.milestone_results, indent=2) + "\n", encoding="utf-8")
            self.logger.record("eval/success_rate", evaluation["success_rate"])
            self.logger.record("eval/mean_final_distance_m", evaluation["mean_final_distance_m"])
            self.logger.record("eval/mean_reward", evaluation["mean_reward"])
            print(f"{label} evaluation: {evaluation['successes']}/{evaluation['episodes']} "
                  f"success, mean distance {evaluation['mean_final_distance_m']:.3f} m",
                  flush=True)


def run_baseline_training() -> dict:
    """Train one new PPO curve, evaluate four milestones, and stop below 200k."""
    require_fresh_baseline_artifacts(MODEL_DIR)
    if REPORT_PATH.exists():
        raise FileExistsError(f"Preserving existing baseline report: {REPORT_PATH}")
    for directory in (LOG_DIR, TENSORBOARD_DIR):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Preserving existing baseline logs: {directory}")
    schedule = checkpoint_schedule()
    total_timesteps = schedule[-1]["actual_timesteps"]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_env = make_training_envs(LOG_DIR / "monitor", n_envs=N_ENVS)
    eval_env = Monitor(MineUAVEnv(render_mode=None),
                       filename=str(LOG_DIR / "evaluation"),
                       info_keywords=("distance_m", "termination_reason"))
    started_at = time.perf_counter()
    try:
        # make_ppo constructs a fresh MlpPolicy; no smoke checkpoint is loaded.
        model = make_ppo(train_env, TENSORBOARD_DIR)
        model.verbose = 0  # Console output only; all PPO settings remain fixed.
        callback = BaselineCallback(eval_env, MODEL_DIR, LOG_DIR, eval_episodes=100)
        model.learn(total_timesteps=total_timesteps, callback=callback,
                    tb_log_name="PPO", log_interval=1)
        learn_wall_s = time.perf_counter() - started_at
        if model.num_timesteps != total_timesteps:
            raise RuntimeError(f"Baseline exceeded or missed budget: {model.num_timesteps}")
        # The final PPO train() ran after its on_rollout_end; there is no next
        # rollout_start, so capture the updated 200k-labelled policy here.
        callback.record_updated_state()
        if set(callback.milestone_results) != {entry["label"] for entry in schedule}:
            raise RuntimeError("Missing baseline milestone checkpoint or evaluation")
        model.dump_logs(iteration=total_timesteps // (N_ENVS * N_STEPS))
        through_milestones_wall_s = time.perf_counter() - started_at
        final_eval_started_at = time.perf_counter()
        final_eval = evaluate_baseline(model, eval_env, episodes=100,
                                       seed=FINAL_EVAL_SEED)
        final_eval_wall_s = time.perf_counter() - final_eval_started_at
        result = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "initialization": "from_scratch; smoke checkpoint not loaded",
            "seed": SEED,
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "rollout_size": N_ENVS * N_STEPS,
            "requested_timesteps": 200_000,
            "actual_timesteps": model.num_timesteps,
            "ppo_n_updates": model._n_updates,
            "learn_wall_s_including_earlier_milestone_evals": learn_wall_s,
            "through_200k_milestone_wall_s": through_milestones_wall_s,
            "final_independent_eval_wall_s": final_eval_wall_s,
            "schedule": schedule,
            "training_history": callback.training_history,
            "milestones": callback.milestone_results,
            "final_independent_evaluation": final_eval,
            "tensorboard_dir": str(TENSORBOARD_DIR),
            "monitor_dir": str(LOG_DIR / "monitor"),
        }
        REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        eval_env.close()
        train_env.close()


if __name__ == "__main__":
    result = run_baseline_training()
    print(json.dumps({
        "actual_timesteps": result["actual_timesteps"],
        "through_200k_milestone_wall_s": result["through_200k_milestone_wall_s"],
        "final_independent_evaluation": {
            key: value for key, value in result["final_independent_evaluation"].items()
            if key != "episode_summaries"
        },
        "report": str(REPORT_PATH),
    }, indent=2))
