"""Independent 100k PPO run: Reward V2 plus near-target tangential penalty.

No distance penalty from V3. All PPO configuration is reused unchanged from
the baseline factory; only MineUAVEnv(reward_version="v4") differs.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from mine_uav_env import MineUAVEnv
from reward_v2_diagnostics import EVAL_SEED
from reward_v4_diagnostics import (crossing_counts, evaluate_reward_v4,
                                   validate_fixed_waypoints)
from train_ppo_waypoint import (N_ENVS, N_STEPS, SEED, TRAIN_METRICS, make_ppo,
                                make_training_envs)


RL_DIR = Path(__file__).resolve().parent
MODEL_DIR = RL_DIR / "models"
LOG_DIR = RL_DIR / "logs" / "ppo_waypoint_reward_v4"
TENSORBOARD_DIR = RL_DIR / "tensorboard" / "ppo_waypoint_reward_v4"
REPORT_PATH = RL_DIR.parent / "reports" / "ppo_waypoint_reward_v4.json"
V2_REPORT_PATH = RL_DIR.parent / "reports" / "ppo_waypoint_reward_v2.json"
V2_BRAKING_PATH = RL_DIR.parent / "reports" / "ppo_braking_diagnostic.json"
V2_100K = MODEL_DIR / "ppo_waypoint_reward_v2_100k.zip"


def checkpoint_schedule() -> list[dict]:
    rollout_size = N_ENVS * N_STEPS
    return [{"label": label, "requested_timesteps": requested,
             "actual_timesteps": math.ceil(requested / rollout_size) * rollout_size}
            for label, requested in (("20k", 20_000), ("50k", 50_000),
                                     ("100k", 100_000))]


def require_fresh_v4_artifacts(model_dir: Path, report_path: Path,
                               log_dir: Path, tensorboard_dir: Path) -> None:
    """Reject pre-existing V4 outputs; never target older versions."""
    existing = [model_dir / f"ppo_waypoint_reward_v4_{entry['label']}.zip"
                for entry in checkpoint_schedule()
                if (model_dir / f"ppo_waypoint_reward_v4_{entry['label']}.zip").exists()]
    if report_path.exists():
        existing.append(report_path)
    for directory in (log_dir, tensorboard_dir):
        if directory.exists() and any(directory.iterdir()):
            existing.append(directory)
    if existing:
        raise FileExistsError(f"Preserving existing Reward V4 artifacts: {existing}")


class RewardV4Callback(BaseCallback):
    """Save and evaluate at the first complete updated rollout past each mark."""

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
        record = {
            "timesteps": step,
            "rollout_ep_rew_mean": (float(np.mean([item["r"] for item in recent]))
                                    if recent else None),
            "rollout_ep_len_mean": (float(np.mean([item["l"] for item in recent]))
                                    if recent else None),
            "episode_count_in_window": len(recent),
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
            checkpoint = self.model_dir / f"ppo_waypoint_reward_v4_{label}.zip"
            if checkpoint.exists():
                raise FileExistsError(checkpoint)
            self.model.save(checkpoint)
            print(f"Reward V4 {label}: updated {step:,} timesteps; evaluating 100 fixed seeds",
                  flush=True)
            evaluation = evaluate_reward_v4(self.model, self.eval_env,
                                            episodes=100, seed=EVAL_SEED)
            self.milestones[label] = {
                **milestone, "checkpoint": str(checkpoint),
                "ppo_n_updates": self.model._n_updates,
                "elapsed_wall_s": time.perf_counter() - self.started_at,
                "evaluation": evaluation,
            }
            (self.log_dir / "milestones.json").write_text(
                json.dumps(self.milestones, indent=2) + "\n", encoding="utf-8")
            self.logger.record("eval/success_rate", evaluation["success_rate"])
            self.logger.record("eval/mean_final_distance_m",
                               evaluation["mean_final_distance_m"])
            near = evaluation["braking_diagnostic"]["near_target"]["lt_0_1"]
            if near["mean_v_tangent_m_s"] is not None:
                self.logger.record("eval/mean_tangent_speed_within_0_1_m_s",
                                   near["mean_v_tangent_m_s"])
            print(f"Reward V4 {label}: {evaluation['successes']}/100 success, "
                  f"mean final distance {evaluation['mean_final_distance_m']:.3f} m",
                  flush=True)


def run_reward_v4_training() -> dict:
    require_fresh_v4_artifacts(MODEL_DIR, REPORT_PATH, LOG_DIR, TENSORBOARD_DIR)
    for required in (V2_100K, V2_REPORT_PATH, V2_BRAKING_PATH):
        if not required.is_file():
            raise FileNotFoundError(f"Required V2 comparison artifact: {required}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_env = make_training_envs(LOG_DIR / "monitor", n_envs=N_ENVS,
                                   reward_version="v4")
    eval_env = MineUAVEnv(render_mode=None, reward_version="v4")
    started_at = time.perf_counter()
    try:
        model = make_ppo(train_env, TENSORBOARD_DIR)
        model.verbose = 0
        callback = RewardV4Callback(eval_env, MODEL_DIR, LOG_DIR)
        total_timesteps = checkpoint_schedule()[-1]["actual_timesteps"]
        model.learn(total_timesteps=total_timesteps, callback=callback,
                    tb_log_name="PPO", log_interval=1)
        if model.num_timesteps != total_timesteps:
            raise RuntimeError(f"Training timestep budget mismatch: {model.num_timesteps}")
        callback.record_updated_state()  # Final PPO update lacks a next rollout-start hook.
        if set(callback.milestones) != {"20k", "50k", "100k"}:
            raise RuntimeError("Missing Reward V4 milestone")
        model.dump_logs(iteration=total_timesteps // (N_ENVS * N_STEPS))

        v2_report = json.loads(V2_REPORT_PATH.read_text(encoding="utf-8"))
        v2_braking_report = json.loads(V2_BRAKING_PATH.read_text(encoding="utf-8"))
        v2_eval = v2_report["milestones"]["100k"]["evaluation"]
        v2_braking = v2_braking_report["policies"]["ppo_v2"]
        v4_eval = callback.milestones["100k"]["evaluation"]
        validate_fixed_waypoints(v2_eval, v2_braking, v4_eval)
        result = {
            "reward_version": "v4", "derived_from": "v2",
            "brake_lambda": 0.5, "tangent_lambda": 0.5,
            "distance_penalty_present": False,
            "initialization": "from_scratch", "seed": SEED,
            "evaluation_seed": EVAL_SEED,
            "n_envs": N_ENVS, "n_steps": N_STEPS,
            "rollout_size": N_ENVS * N_STEPS,
            "requested_timesteps": 100_000,
            "actual_timesteps": model.num_timesteps,
            "ppo_n_updates": model._n_updates,
            "training_and_evaluation_wall_s": time.perf_counter() - started_at,
            "training_history": callback.training_history,
            "milestones": callback.milestones,
            "comparison_100k": {
                "same_fixed_seeds_and_targets": True,
                "v2_evaluation": v2_eval,
                "v2_braking_diagnostic": v2_braking,
                "v2_crossing_after_first_0_2_m": crossing_counts(v2_braking),
                "v4_evaluation": v4_eval,
            },
            "tensorboard_dir": str(TENSORBOARD_DIR),
            "log_dir": str(LOG_DIR),
        }
        REPORT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        return result
    finally:
        eval_env.close()
        train_env.close()


if __name__ == "__main__":
    report = run_reward_v4_training()
    print(json.dumps({
        "actual_timesteps": report["actual_timesteps"],
        "wall_s": report["training_and_evaluation_wall_s"],
        "v4_100k_success_rate": report["milestones"]["100k"]["evaluation"]["success_rate"],
        "report": str(REPORT_PATH),
    }, indent=2))
