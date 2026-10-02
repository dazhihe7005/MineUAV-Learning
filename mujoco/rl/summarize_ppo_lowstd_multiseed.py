"""Aggregate the five unchanged Reward V2 low-std PPO seeds.

The original 20260929 run is reused. Four new runs have separate model/log
directories. The second full-task benchmark uses disjoint, preselected seeds.
"""

import json
from pathlib import Path

from stable_baselines3 import PPO

from mine_uav_env import MineUAVEnv
from ppo_exploration_audit import evaluate_exploration
from reward_v2_alignment_audit import fixed_waypoints
from train_ppo_lowstd_multiseed import (
    ALL_TRAIN_SEEDS, FRESH_TRAIN_SEEDS, HOLDOUT_COUNT, HOLDOUT_FIRST_SEED,
    PRIOR_REPORT, PRIOR_TRAIN_SEED, SUMMARY_REPORT, aggregate_numeric,
    build_holdout_waypoints, compact_evaluation, first_observed_thresholds,
    seed_paths,
)
from train_ppo_v2_lowstd import LOG_STD_INIT


COMPARISON_METRICS = (
    "success_rate", "mean_final_distance_m",
    "mean_successful_completion_time_s", "mean_episode_length",
    "ever_within_0_1_m_fraction", "ever_distance_and_speed_fraction",
    "near_target_actual_speed_m_s", "near_target_command_speed_m_s",
    "near_target_tangential_speed_m_s", "crossing_rate",
    "action_saturation_fraction",
)


def aggregate_seed_rows(rows: list[dict]) -> dict:
    if len(rows) != 5:
        raise ValueError("Need exactly five independent training seeds")
    return {
        **{
            task: {
                key: aggregate_numeric([row[task][key] for row in rows])
                for key in COMPARISON_METRICS
            }
            for task in ("fixed", "holdout")
        },
        "first_observed_full_success_threshold_timesteps": {
            key: aggregate_numeric([
                row["first_observed_full_success_threshold_timesteps"][key]
                for row in rows
            ])
            for key in ("50pct", "90pct", "100pct")
        },
    }


def _validate_fixed_evaluations(report: dict, waypoints: list) -> None:
    expected_seeds = [seed for seed, _ in waypoints]
    expected_targets = [target for _, target in waypoints]
    for label in ("20k", "50k", "100k"):
        evaluation = report["milestones"][label]["evaluation"]["full"]
        if (evaluation["seeds"] != expected_seeds
                or evaluation["targets"] != expected_targets
                or not evaluation["deterministic"]):
            raise RuntimeError(f"{report['seed']} {label} is not the fixed benchmark")


def _validate_holdout(evaluation: dict, waypoints: list) -> None:
    if (evaluation["seeds"] != [seed for seed, _ in waypoints]
            or evaluation["targets"] != [target for _, target in waypoints]
            or not evaluation["deterministic"]):
        raise RuntimeError("Holdout evaluation used different waypoints")


def _seed_row(report: dict, holdout: dict, source: str) -> dict:
    milestones = report["milestones"]
    return {
        "seed": report["seed"], "source": source,
        "actual_timesteps": report["actual_timesteps"],
        "training_wall_s": report.get("training_wall_s", report.get("experiment_wall_s")),
        "checkpoint": milestones["100k"]["checkpoint"],
        "fixed": compact_evaluation(milestones["100k"]["evaluation"]["full"]),
        "holdout": compact_evaluation(holdout),
        "first_observed_full_success_threshold_timesteps":
            first_observed_thresholds(milestones),
        "sigma_by_checkpoint": {
            "initial": report["initial_policy_std"],
            **{label: milestones[label]["policy_std"]
               for label in ("20k", "50k", "100k")},
        },
        "final_training_metrics": report["training_history"][-1],
    }


def run_summary() -> dict:
    if SUMMARY_REPORT.exists():
        raise FileExistsError(f"Preserving existing summary: {SUMMARY_REPORT}")
    fixed = fixed_waypoints("full")
    holdout = build_holdout_waypoints()
    if (len(fixed) != 100 or len(holdout) != HOLDOUT_COUNT
            or set(seed for seed, _ in fixed) & set(seed for seed, _ in holdout)):
        raise RuntimeError("Fixed and holdout evaluations are not disjoint 100-task sets")

    reports = {}
    prior = json.loads(PRIOR_REPORT.read_text(encoding="utf-8"))
    if prior["seed"] != PRIOR_TRAIN_SEED:
        raise RuntimeError("Original low-std run has unexpected seed")
    reports[PRIOR_TRAIN_SEED] = prior
    for seed in FRESH_TRAIN_SEEDS:
        path = seed_paths(seed)["report"]
        reports[seed] = json.loads(path.read_text(encoding="utf-8"))
        if reports[seed]["seed"] != seed:
            raise RuntimeError(f"Wrong seed in {path}")
    if set(reports) != set(ALL_TRAIN_SEEDS):
        raise RuntimeError("Expected five unique training seeds")
    for seed, report in reports.items():
        if (report["reward_version"] != "v2"
                or report["initialization"] != "from_scratch"
                or report["actual_timesteps"] != 100352
                or report["n_envs"] != 8 or report["n_steps"] != 256
                or report["initial_policy_std"]["log_std"] != [LOG_STD_INIT] * 4):
            raise RuntimeError(f"Seed {seed} differs from the fixed experiment")
        _validate_fixed_evaluations(report, fixed)
        if not Path(report["milestones"]["100k"]["checkpoint"]).is_file():
            raise FileNotFoundError(report["milestones"]["100k"]["checkpoint"])

    original_checkpoint = Path(prior["milestones"]["100k"]["checkpoint"])
    original_model = PPO.load(original_checkpoint, device="cpu")
    original_env = MineUAVEnv(render_mode=None, reward_version="v2",
                              target_distribution="full")
    try:
        original_holdout = evaluate_exploration(original_model, original_env,
                                                holdout, deterministic=True)
    finally:
        original_env.close()
    _validate_holdout(original_holdout, holdout)
    rows = []
    for seed in ALL_TRAIN_SEEDS:
        report = reports[seed]
        holdout_eval = (original_holdout if seed == PRIOR_TRAIN_SEED
                        else report["holdout_evaluation"])
        _validate_holdout(holdout_eval, holdout)
        source = ("preexisting_lowstd_run" if seed == PRIOR_TRAIN_SEED
                  else "new_independent_run")
        rows.append(_seed_row(report, holdout_eval, source))

    result = {
        "experiment": "Reward V2, PPO log_std_init=-2.0, five training seeds",
        "training_seed_order": list(ALL_TRAIN_SEEDS),
        "original_seed_provenance": "The old run records SB3 seed 20260929; "
            "Python/NumPy/Torch and env seeding follow SB3 set_random_seed in "
            "the unchanged old runner but first training targets were not logged then.",
        "new_seed_provenance": "Python, NumPy, PyTorch and SB3 seeds explicitly "
            "recorded; rank-specific Gymnasium first-reset seeds and actual first "
            "targets verified and recorded in individual reports.",
        "fixed_benchmark_seeds": [seed for seed, _ in fixed],
        "holdout_seeds": [seed for seed, _ in holdout],
        "holdout_first_seed": HOLDOUT_FIRST_SEED,
        "holdout_seed_range_predeclared_before_training": True,
        "holdout_targets_generated_after_training_without_model_selection": True,
        "convergence_resolution": "Only checkpoint evaluations at 20,480, 51,200 "
            "and 100,352 transitions; threshold crossing between checkpoints "
            "is unknown, and none means not observed by 100,352.",
        "rows": rows,
        "aggregate": aggregate_seed_rows(rows),
        "per_seed_report_paths": {
            str(seed): str(PRIOR_REPORT if seed == PRIOR_TRAIN_SEED
                           else seed_paths(seed)["report"])
            for seed in ALL_TRAIN_SEEDS
        },
    }
    SUMMARY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_REPORT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
    print(json.dumps({"rows": [{"seed": row["seed"],
                                "fixed_success": row["fixed"]["successes"],
                                "holdout_success": row["holdout"]["successes"],
                                "convergence": row[
                                    "first_observed_full_success_threshold_timesteps"]}
                               for row in rows],
                      "summary": str(SUMMARY_REPORT)}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run_summary()
