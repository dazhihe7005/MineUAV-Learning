"""Conditional fixed-PI robustness for one newly trained PPO seed; no training."""

import hashlib
import json
import math
from pathlib import Path

from stable_baselines3 import PPO

from attribution_audit import write_condition
from audit_ppo_robustness import MUJOCO_DIR, waypoint_sha256
from evaluate_ppo_pi import (compact_pi_evaluation, evaluate_pi_policy,
                             nominal_gate)
from ppo_pi_env import PIRobustnessEnv
from robustness_dynamics import Scenario
from train_ppo_lowstd_multiseed import build_holdout_waypoints
from train_ppo_pi_lowstd import output_paths


OLD_P_REPORT = MUJOCO_DIR / "reports" / "ppo_baseline_v1_attribution.json"
OLD_PI_REPORT = MUJOCO_DIR / "reports" / "velocity_pi_robustness_v2.json"


def disturbance_scenarios() -> list[Scenario]:
    """Exactly ±5/±2 percent or newtons; nominal is reused, not rerun."""
    return ([Scenario(kind, value)
             for kind in ("mass", "thrust")
             for value in (0.95, 0.98, 1.02, 1.05)]
            + [Scenario("force_x_n", value) for value in (-5.0, -2.0, 2.0, 5.0)])


def condition_path(root: Path, scenario: Scenario) -> Path:
    return (root / "reports" / "ppo_pi_compatibility" / "condition_parts" /
            scenario.kind / f"{scenario.value:+.3f}.json")


def integral_trace_path(root: Path, scenario: Scenario) -> Path:
    return (root / "reports" / "ppo_pi_compatibility" / "integral_traces" /
            "robustness" / scenario.kind / f"{scenario.value:+.3f}.json")


def _historical_point(report: dict, kind: str, value: float) -> dict:
    if kind == "nominal":
        return report["nominal"]
    return next(row for row in report["curves"][kind]
                if math.isclose(row["value"], value, abs_tol=1e-12))


def _seed_zero_metrics(point: dict) -> dict:
    selected = [row["metrics"] for row in point["ppo_per_seed"]
                if row["training_seed"] == 0]
    if len(selected) != 1 or selected[0]["episodes"] != 100:
        raise ValueError("Historical seed-0 PPO result is missing")
    metric = selected[0]
    return {key: metric[key] for key in (
        "episodes", "successes", "success_rate", "mean_final_distance_m",
        "mean_successful_completion_time_s", "mean_episode_length",
        "crossing_rate", "termination_counts")}


def compare_abc(old_p: dict, old_pi: dict, kind: str, value: float,
                new_pi: dict, waypoints: list[tuple[int, list[float]]]) -> dict:
    """Pair old seed 0 under P/PI with new seed 0 under PI on exact targets."""
    digest = waypoint_sha256(waypoints)
    if (len(waypoints) != 100 or old_p["waypoint_sha256"] != digest
            or old_pi["waypoint_sha256"] != digest):
        raise ValueError("Historical P/PI and new PPO waypoints differ")
    if (new_pi["episodes"] != 100 or not 0 <= new_pi["successes"] <= 100
            or not math.isclose(new_pi["success_rate"],
                                new_pi["successes"] / 100, abs_tol=1e-12)):
        raise ValueError("New PI-trained PPO evaluation is incomplete")
    p_point = _historical_point(old_p, kind, value)
    pi_point = _historical_point(old_pi, kind, value)["new_pi"]
    return {
        "scenario": {"kind": kind, "value": value},
        "waypoint_sha256": digest,
        "training_seed": 0,
        "A_old_ppo_with_P": _seed_zero_metrics(p_point),
        "B_same_old_ppo_with_PI": _seed_zero_metrics(pi_point),
        "C_new_ppo_trained_with_PI": new_pi,
    }


def _validated_existing_part(path: Path, scenario: Scenario,
                             digest: str, checkpoint_sha: str) -> dict:
    part = json.loads(path.read_text(encoding="utf-8"))
    if (part["scenario"] != {"kind": scenario.kind, "value": scenario.value}
            or part["waypoint_sha256"] != digest
            or part["checkpoint_sha256"] != checkpoint_sha
            or part["evaluation"]["episodes"] != 100
            or not Path(part["evaluation"]["integral_trace_path"]).is_file()):
        raise ValueError(f"Existing PI condition differs: {path}")
    return part


def run_conditional_robustness(train_report: dict,
                               output_root: Path = MUJOCO_DIR) -> dict:
    """Skip all disturbed work unless both 100k nominal sets pass 90/100."""
    final = train_report["milestones"]["100k"]["evaluation"]
    gate = nominal_gate({"task_metrics": final["benchmark"]},
                        {"task_metrics": final["holdout"]})
    if gate != train_report["nominal_100k_gate_requires_both_sets_90pct"]:
        raise ValueError("Stored nominal gate disagrees with actual fixed-set results")
    if train_report["seed"] != 0 or train_report["reward_version"] != "v2":
        raise ValueError("Need the one-seed Reward-V2 PPO/PI training report")
    waypoints = build_holdout_waypoints()
    digest = waypoint_sha256(waypoints)
    if train_report["waypoint_sha256"]["holdout"] != digest:
        raise ValueError("New training report used different holdout waypoints")
    old_p = json.loads(OLD_P_REPORT.read_text(encoding="utf-8"))
    old_pi = json.loads(OLD_PI_REPORT.read_text(encoding="utf-8"))
    checkpoint = Path(train_report["milestones"]["100k"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    recorded_sha = train_report["milestones"]["100k"].get("checkpoint_sha256")
    if recorded_sha is not None and recorded_sha != checkpoint_sha:
        raise ValueError("Saved checkpoint hash differs from training report")
    if gate and recorded_sha is None:
        raise ValueError("A passing nominal gate requires a training-time checkpoint hash")
    nominal = compare_abc(old_p, old_pi, "nominal", 0.0,
                          final["holdout"], waypoints)
    nominal_report = {
        "nominal_gate_passed": gate,
        "disturbed_conditions_evaluated": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_hash_recorded_at_training": recorded_sha is not None,
        "waypoint_sha256": digest,
        "nominal": nominal,
        "source_old_p_report": str(OLD_P_REPORT),
        "source_direct_pi_report": str(OLD_PI_REPORT),
    }
    nominal_path = (output_root / "reports" / "ppo_pi_compatibility" /
                    "nominal_comparison.json")
    if nominal_path.exists():
        if json.loads(nominal_path.read_text(encoding="utf-8")) != nominal_report:
            raise ValueError("Existing nominal comparison differs from checkpoint/report")
    else:
        write_condition(nominal_path, nominal_report)
    if not gate:
        return {**nominal_report,
                "reason": "At least one nominal fixed100 set is below 90/100"}
    model = PPO.load(checkpoint, device="cpu")
    if (model.num_timesteps != 100_352 or model.policy.log_std_init != -2.0
            or model.observation_space.shape != (7,)
            or model.action_space.shape != (4,)):
        raise ValueError("100k checkpoint is not the fixed low-std PPO baseline")
    curves = {kind: [] for kind in ("mass", "thrust", "force_x_n")}
    for kind in curves:
        curves[kind].append({**nominal, "scenario": {
            "kind": kind, "value": 0.0 if kind == "force_x_n" else 1.0},
            "reused_nominal_evaluation": True})
    for scenario in disturbance_scenarios():
        path = condition_path(output_root, scenario)
        if path.exists():
            part = _validated_existing_part(path, scenario, digest, checkpoint_sha)
        else:
            env = PIRobustnessEnv(scenario)
            try:
                trace_path = integral_trace_path(output_root, scenario)
                evaluation = evaluate_pi_policy(model, env, waypoints, trace_path)
                compact = compact_pi_evaluation(evaluation)
            finally:
                env.close()
            part = {
                "scenario": {"kind": scenario.kind, "value": scenario.value},
                "waypoint_sha256": digest,
                "waypoint_seeds": [seed for seed, _ in waypoints],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "controller": train_report["controller"],
                "evaluation": compact,
            }
            write_condition(path, part)
            print(f"PPO trained with PI {scenario.kind}={scenario.value:g}: "
                  f"{compact['successes']}/100", flush=True)
        curves[scenario.kind].append(compare_abc(
            old_p, old_pi, scenario.kind, scenario.value,
            part["evaluation"], waypoints))
    for rows in curves.values():
        rows.sort(key=lambda row: row["scenario"]["value"])
    result = {
        "experiment": "One-seed PPO(P) vs same PPO switched to PI vs fresh PPO(PI)",
        "nominal_gate_passed": True,
        "disturbed_conditions_evaluated": len(disturbance_scenarios()),
        "training_seed": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "waypoint_sha256": digest,
        "nominal": nominal,
        "curves": curves,
        "source_training_report": str(output_paths(output_root)["report"]),
        "source_old_p_report": str(OLD_P_REPORT),
        "source_direct_pi_report": str(OLD_PI_REPORT),
        "note": "Nominal point is reused for all three curves; C is one training seed "
                "and must not be read as a five-seed robustness estimate.",
    }
    destination = (output_root / "reports" / "ppo_pi_compatibility" /
                   "ppo_pi_compatibility_report.json")
    write_condition(destination, result)
    return result


if __name__ == "__main__":
    train_path = output_paths()["report"]
    result = run_conditional_robustness(
        json.loads(train_path.read_text(encoding="utf-8")))
    print(json.dumps({"nominal_gate_passed": result["nominal_gate_passed"],
                      "disturbed_conditions_evaluated":
                          result["disturbed_conditions_evaluated"]}, indent=2),
          flush=True)
