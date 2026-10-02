"""One continuous Reward V2 PPO run through local A, local B, full targets.

Only the reset target distribution changes. The same PPO object, policy,
optimizer and cumulative timestep counter survive all three stages.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from curriculum_diagnostics import evaluate_curriculum
from mine_uav_env import MineUAVEnv
from reward_v2_diagnostics import EVAL_SEED
from train_ppo_waypoint import (N_ENVS, N_STEPS, SEED, TRAIN_METRICS,
                                make_ppo, make_training_envs)


RL_DIR = Path(__file__).resolve().parent
MODEL_DIR = RL_DIR / "models"
LOG_DIR = RL_DIR / "logs" / "ppo_waypoint_curriculum"
TENSORBOARD_DIR = RL_DIR / "tensorboard" / "ppo_waypoint_curriculum"
REPORT_PATH = RL_DIR.parent / "reports" / "ppo_waypoint_curriculum.json"
V2_100K = MODEL_DIR / "ppo_waypoint_reward_v2_100k.zip"
V2_REPORT = RL_DIR.parent / "reports" / "ppo_waypoint_reward_v2.json"
LOCAL_EVAL_SEED = 20261101
STAGES = (("a", "local_a", 50_000),
          ("b", "local_b", 50_000),
          ("c", "full", 100_000))


def stage_schedule() -> list[dict]:
    rollout_size = N_ENVS * N_STEPS
    cumulative = 0
    schedule = []
    for label, distribution, requested in STAGES:
        stage_steps = math.ceil(requested / rollout_size) * rollout_size
        cumulative += stage_steps
        schedule.append({
            "label": label, "target_distribution": distribution,
            "requested_stage_timesteps": requested,
            "stage_timesteps": stage_steps,
            "cumulative_timesteps": cumulative,
        })
    return schedule


def require_fresh_curriculum_artifacts(model_dir: Path, report_path: Path,
                                       log_dir: Path, tensorboard_dir: Path) -> None:
    existing = [model_dir / f"ppo_waypoint_curriculum_stage_{stage['label']}.zip"
                for stage in stage_schedule()
                if (model_dir / f"ppo_waypoint_curriculum_stage_{stage['label']}.zip").exists()]
    if report_path.exists():
        existing.append(report_path)
    for directory in (log_dir, tensorboard_dir):
        if directory.exists() and any(directory.iterdir()):
            existing.append(directory)
    if existing:
        raise FileExistsError(f"Preserving curriculum artifacts: {existing}")


def train_stage(model: PPO, stage_env, stage_timesteps: int, *, first: bool) -> None:
    """Complete whole rollouts and PPO updates using one persistent PPO object."""
    rollout_size = model.n_envs * model.n_steps
    if stage_timesteps <= 0 or stage_timesteps % rollout_size:
        raise ValueError("Stage budget must contain complete PPO rollouts")
    if first and model.num_timesteps != 0:
        raise ValueError("First stage must start from a fresh PPO model")
    before_steps = model.num_timesteps
    before_updates = model._n_updates
    if not first:
        model.set_env(stage_env, force_reset=True)
    model.learn(total_timesteps=stage_timesteps, reset_num_timesteps=first,
                tb_log_name="PPO", log_interval=1)
    expected_steps = before_steps + stage_timesteps
    expected_updates = before_updates + stage_timesteps // rollout_size * model.n_epochs
    if model.num_timesteps != expected_steps or model._n_updates != expected_updates:
        raise RuntimeError("PPO stage did not complete all expected samples and updates: "
                           f"steps={model.num_timesteps}/{expected_steps}, "
                           f"updates={model._n_updates}/{expected_updates}")


def _target_rows(evaluation: dict) -> list[list[float]]:
    return [episode["target_position_m"] for episode in evaluation["episode_summaries"]]


def _assert_same_waypoints(candidate: dict, reference: dict, label: str) -> None:
    if (candidate["seeds"] != reference["seeds"]
            or _target_rows(candidate) != _target_rows(reference)):
        raise RuntimeError(f"{label}: fixed evaluation waypoints differ")


def _write_report(report: dict) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")


def run_curriculum_training() -> dict:
    require_fresh_curriculum_artifacts(MODEL_DIR, REPORT_PATH, LOG_DIR, TENSORBOARD_DIR)
    if not V2_100K.is_file() or not V2_REPORT.is_file():
        raise FileNotFoundError("Reward V2 100k checkpoint and report are required")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    local_eval_env = MineUAVEnv(render_mode=None, reward_version="v2",
                                target_distribution="local_a")
    full_eval_env = MineUAVEnv(render_mode=None, reward_version="v2",
                               target_distribution="full")
    training_env = None
    started_at = time.perf_counter()
    try:
        old_report = json.loads(V2_REPORT.read_text(encoding="utf-8"))
        old_full = old_report["milestones"]["100k"]["evaluation"]
        baseline_model = PPO.load(V2_100K, device="cpu")
        print("Evaluating original Reward V2 100k on fixed local and full sets", flush=True)
        baseline_local = evaluate_curriculum(baseline_model, local_eval_env,
                                             episodes=100, seed=LOCAL_EVAL_SEED)
        baseline_full = evaluate_curriculum(baseline_model, full_eval_env,
                                            episodes=100, seed=EVAL_SEED)
        _assert_same_waypoints(baseline_full, old_full, "Original V2 full evaluation")
        report = {
            "experiment": "Reward V2 target-distribution curriculum, no other task changes",
            "reward_version": "v2", "initialization": "fresh PPO weights",
            "stage_continuity": "same in-memory PPO policy and optimizer across A, B, C",
            "seed": SEED, "local_evaluation_seed": LOCAL_EVAL_SEED,
            "full_evaluation_seed": EVAL_SEED,
            "n_envs": N_ENVS, "n_steps": N_STEPS,
            "rollout_size": N_ENVS * N_STEPS,
            "schedule": stage_schedule(),
            "baseline_v2_100k_checkpoint": str(V2_100K),
            "baseline_v2_100k": {"local": baseline_local, "full": baseline_full},
            "stages": {},
            "tensorboard_dir": str(TENSORBOARD_DIR),
            "log_dir": str(LOG_DIR),
        }
        model = None
        for index, stage in enumerate(stage_schedule()):
            label = stage["label"]
            next_env = make_training_envs(
                LOG_DIR / f"stage_{label}" / "monitor", n_envs=N_ENVS,
                reward_version="v2", target_distribution=stage["target_distribution"])
            next_env.seed(SEED + index * 1000)
            if model is None:
                model = make_ppo(next_env, TENSORBOARD_DIR)
                model.verbose = 0
            previous_env = training_env
            training_env = next_env
            stage_started = time.perf_counter()
            print(f"Stage {label.upper()}: {stage['target_distribution']}, "
                  f"{stage['stage_timesteps']:,} transitions", flush=True)
            train_stage(model, training_env, stage["stage_timesteps"], first=index == 0)
            if previous_env is not None:
                previous_env.close()
            if model.num_timesteps != stage["cumulative_timesteps"]:
                raise RuntimeError("Curriculum cumulative timestep mismatch")
            metrics = {key: float(model.logger.name_to_value[f"train/{key}"])
                       for key in TRAIN_METRICS}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"Non-finite PPO training metric: {metrics}")
            checkpoint = MODEL_DIR / f"ppo_waypoint_curriculum_stage_{label}.zip"
            model.save(checkpoint)
            print(f"Stage {label.upper()} updated; evaluating fixed local/full sets",
                  flush=True)
            local = evaluate_curriculum(model, local_eval_env, episodes=100,
                                        seed=LOCAL_EVAL_SEED)
            full = evaluate_curriculum(model, full_eval_env, episodes=100,
                                       seed=EVAL_SEED)
            _assert_same_waypoints(local, baseline_local, f"Stage {label} local")
            _assert_same_waypoints(full, baseline_full, f"Stage {label} full")
            report["stages"][label] = {
                **stage,
                "checkpoint": str(checkpoint),
                "ppo_n_updates": model._n_updates,
                "stage_and_evaluation_wall_s": time.perf_counter() - stage_started,
                "train_metrics": metrics,
                "local": local, "full": full,
            }
            report["elapsed_wall_s"] = time.perf_counter() - started_at
            _write_report(report)  # Preserve each stage's evaluation before the next stage.
            print(f"Stage {label.upper()}: local {local['successes']}/100, "
                  f"full {full['successes']}/100 success", flush=True)
        report["actual_timesteps"] = model.num_timesteps
        report["ppo_n_updates"] = model._n_updates
        report["total_wall_s"] = time.perf_counter() - started_at
        _write_report(report)
        return report
    finally:
        local_eval_env.close()
        full_eval_env.close()
        if training_env is not None:
            training_env.close()


if __name__ == "__main__":
    result = run_curriculum_training()
    print(json.dumps({
        "actual_timesteps": result["actual_timesteps"],
        "total_wall_s": result["total_wall_s"],
        "report": str(REPORT_PATH),
    }, indent=2))
