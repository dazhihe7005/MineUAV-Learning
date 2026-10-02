"""Inference-only, five-policy/100-holdout sensitivity evaluation.

Run one factor in a separate process, for example:
  .venv/bin/python mujoco/rl/audit_ppo_robustness.py --factor mass

Every condition is saved immediately in a new file. Neither checkpoints nor
the nominal MJCF are modified. The factor=nominal run is the gate for all
other sweeps and is reused as each sweep's zero/factor-one reference.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from stable_baselines3 import PPO

from diagnose_ppo_braking import evaluate_episode, summarize_policy
from reward_v2_diagnostics import summarize_episodes
from reward_v4_diagnostics import crossing_counts
from robustness_dynamics import RobustnessEnv, Scenario
from train_ppo_lowstd_multiseed import build_holdout_waypoints


RL_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = RL_DIR.parent
FIVE_SEED_REPORT = MUJOCO_DIR / "reports" / "ppo_waypoint_v2_lowstd_multiseed.json"
PARTS_DIR = MUJOCO_DIR / "reports" / "ppo_baseline_v1_robustness_parts"
EXPECTED_TRAIN_SEEDS = (20260929, 0, 8, 16, 24)
FACTOR_LEVELS = {
    "mass": (0.8, 0.9, 0.95, 1.05, 1.1, 1.2),
    "inertia": (0.8, 0.9, 0.95, 1.05, 1.1, 1.2),
    "thrust": (0.8, 0.9, 0.95, 1.05, 1.1, 1.2),
    "motor_lag_s": (0.02, 0.04, 0.06, 0.08, 0.10),
    "force_x_n": (-20.0, -15.0, -10.0, -5.0, 5.0, 10.0, 15.0, 20.0),
}


def waypoint_sha256(waypoints: list[tuple[int, list[float]]]) -> str:
    if not waypoints:
        raise ValueError("Need at least one waypoint")
    encoded = json.dumps(waypoints, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_frozen_policies() -> list[tuple[int, PPO]]:
    report = json.loads(FIVE_SEED_REPORT.read_text(encoding="utf-8"))
    if tuple(report["training_seed_order"]) != EXPECTED_TRAIN_SEEDS:
        raise RuntimeError("Five-seed report contains unexpected policies")
    if [row["seed"] for row in report["rows"]] != list(EXPECTED_TRAIN_SEEDS):
        raise RuntimeError("Five-seed checkpoint order differs from report")
    policies = []
    for row in report["rows"]:
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = PPO.load(checkpoint, device="cpu")
        if (model.num_timesteps != 100352 or model.policy.log_std_init != -2.0
                or model.observation_space.shape != (7,)
                or model.action_space.shape != (4,)):
            raise RuntimeError(f"Checkpoint is not frozen baseline v1: {checkpoint}")
        policies.append((row["seed"], model))
    return policies


def evaluate_policy(model: PPO, env: RobustnessEnv,
                    waypoints: list[tuple[int, list[float]]]) -> dict:
    """One deterministic rollout per target using existing diagnostic definitions."""
    if not waypoints or env.reward_version != "v2" or env.render_mode is not None:
        raise ValueError("Need headless Reward V2 and nonempty fixed waypoints")
    traces = []
    warning_episode_count = 0
    warning_total = 0
    for seed, expected_target in waypoints:
        trace = evaluate_episode(
            env, lambda obs: model.predict(obs, deterministic=True)[0], seed)
        if trace["target_m"] != expected_target:
            raise RuntimeError(f"Target changed for holdout seed {seed}")
        traces.append(trace)
        warnings = sum(int(item.number) for item in env.data.warning)
        warning_episode_count += int(warnings > 0)
        warning_total += warnings

    task_records = [{
        "seed": ep["seed"], "target_position_m": ep["target_m"],
        "final_distance_m": ep["final_distance_m"],
        "final_speed_m_s": ep["final_speed_m_s"],
        "episode_length": ep["episode_length"],
        "episode_reward": ep["episode_reward"],
        "termination_reason": ep["termination_reason"],
        "steps": [{
            "distance_m": step["post_distance_m"],
            "speed_m_s": (step["post_speed_m_s"] if step["post_speed_m_s"] is not None
                          else math.inf),
            "success_streak": step["success_streak"],
            "action": step["action"],
        } for step in ep["steps"]],
    } for ep in traces]
    task = summarize_episodes(task_records, policy_dt=env.policy_dt)
    braking = summarize_policy(traces)
    crossing = crossing_counts(braking)
    reasons = Counter(ep["termination_reason"] for ep in traces)
    failures = sum(count for reason, count in reasons.items()
                   if reason not in ("success", "time_limit"))
    metrics = {
        "episodes": task["episodes"], "successes": task["successes"],
        "success_rate": task["success_rate"],
        "mean_final_distance_m": task["mean_final_distance_m"],
        "mean_successful_completion_time_s": task["mean_successful_completion_time_s"],
        "mean_episode_length": task["mean_episode_length"],
        "crossing_rate": crossing["before_episode_end_fraction_all_episodes"],
        "near_target_speed_m_s": braking["near_target"]["lt_0_1"]
            ["mean_actual_speed_m_s"],
        "near_target_policy_steps": braking["near_target"]["lt_0_1"]["steps"],
        "action_saturation": task["action_near_boundary_fraction_overall"],
        "failure_count": failures, "time_limit_count": reasons["time_limit"],
        "nonfinite_state_count": reasons["nonfinite_state"],
        "mujoco_warning_episode_count": warning_episode_count,
        "mujoco_warning_total": warning_total,
        "termination_counts": dict(reasons),
    }
    if task["successes"] + failures + reasons["time_limit"] != len(traces):
        raise RuntimeError("Terminal outcome accounting does not sum to episodes")
    return {
        "metrics": metrics,
        "episodes": [{
            "seed": ep["seed"], "target_position_m": ep["target_m"],
            "success": ep["termination_reason"] == "success",
            "termination_reason": ep["termination_reason"],
            "episode_length": ep["episode_length"],
            "final_distance_m": ep["final_distance_m"],
            "final_speed_m_s": ep["final_speed_m_s"],
        } for ep in traces],
    }


def evaluate_scenario(scenario: Scenario, policies: list[tuple[int, PPO]],
                      waypoints: list[tuple[int, list[float]]]) -> dict:
    if [seed for seed, _ in policies] != list(EXPECTED_TRAIN_SEEDS):
        raise ValueError("Every scenario must evaluate the same five seeds")
    if len(waypoints) != 100 or waypoints != build_holdout_waypoints():
        raise ValueError("Every scenario must use the exact holdout 100")
    result = {
        "scenario": {"kind": scenario.kind, "value": scenario.value},
        "waypoint_sha256": waypoint_sha256(waypoints),
        "waypoint_seeds": [seed for seed, _ in waypoints],
        "policy_seeds": list(EXPECTED_TRAIN_SEEDS),
        "evaluations": [],
        "nominal_model_path": str(MUJOCO_DIR / "models" / "mine_uav_dynamics_v2.xml"),
        "motor_model": "u_cmd=omega_cmd^2; tau>0: omega_actual exact first-order "
                       "ZOH at dt=0.002s, starts at 0; u_actual=omega_actual^2",
    }
    for training_seed, model in policies:
        env = RobustnessEnv(scenario)
        try:
            evaluation = evaluate_policy(model, env, waypoints)
            evaluation["training_seed"] = training_seed
            result["evaluations"].append(evaluation)
        finally:
            env.close()
        print(f"{scenario.kind}={scenario.value:g} seed={training_seed}: "
              f"{evaluation['metrics']['successes']}/100", flush=True)
    return result


def scenario_output_path(scenario: Scenario, root: Path = PARTS_DIR) -> Path:
    if scenario.kind == "nominal":
        return root / "nominal.json"
    return root / scenario.kind / f"{scenario.value:+.3f}.json"


def write_new_result(path: Path, result: dict) -> None:
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(serialized)


def validate_nominal_reference(result: dict) -> None:
    """Require the altered evaluation harness to reproduce prior holdout."""
    previous = json.loads(FIVE_SEED_REPORT.read_text(encoding="utf-8"))
    if [item["training_seed"] for item in result["evaluations"]] != (
            [row["seed"] for row in previous["rows"]]):
        raise RuntimeError("Nominal policy order differs from prior report")
    mapping = {
        "success_rate": "success_rate",
        "mean_final_distance_m": "mean_final_distance_m",
        "mean_successful_completion_time_s": "mean_successful_completion_time_s",
        "mean_episode_length": "mean_episode_length",
        "crossing_rate": "crossing_rate",
        "near_target_speed_m_s": "near_target_actual_speed_m_s",
        "action_saturation": "action_saturation_fraction",
    }
    for evaluation, row in zip(result["evaluations"], previous["rows"]):
        for new_key, old_key in mapping.items():
            observed = evaluation["metrics"][new_key]
            expected = row["holdout"][old_key]
            if observed is None or expected is None:
                if observed is not None or expected is not None:
                    raise RuntimeError(f"Nominal {row['seed']} {new_key} null mismatch")
            elif not math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-10):
                raise RuntimeError(f"Nominal {row['seed']} {new_key} mismatch: "
                                   f"{observed} vs {expected}")


def run_part(factor: str) -> list[Path]:
    if factor not in FACTOR_LEVELS and factor != "nominal":
        raise ValueError(f"Unknown audit factor: {factor}")
    scenarios = ([Scenario("nominal", 0.0)] if factor == "nominal" else
                 [Scenario(factor, value) for value in FACTOR_LEVELS[factor]])
    paths = [scenario_output_path(scenario) for scenario in scenarios]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Preserving existing robustness outputs: {existing}")
    if factor != "nominal" and not scenario_output_path(Scenario("nominal", 0)).is_file():
        raise RuntimeError("Validate and save nominal baseline before sensitivity sweeps")
    policies = load_frozen_policies()
    waypoints = build_holdout_waypoints()
    started_at = time.perf_counter()
    for scenario, path in zip(scenarios, paths):
        result = evaluate_scenario(scenario, policies, waypoints)
        if factor == "nominal":
            validate_nominal_reference(result)
        result["elapsed_wall_s_since_factor_start"] = time.perf_counter() - started_at
        write_new_result(path, result)
        print(f"saved {path}", flush=True)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", choices=("nominal", *FACTOR_LEVELS), required=True)
    arguments = parser.parse_args()
    run_part(arguments.factor)
