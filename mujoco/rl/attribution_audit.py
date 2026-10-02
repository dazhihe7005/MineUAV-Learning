"""Inference-only scripted/PPO and zero-command dynamics attribution audit."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from attribution_diagnostics import episode_attribution_summary, terminal_window_stats
from audit_ppo_robustness import (EXPECTED_TRAIN_SEEDS, MUJOCO_DIR,
                                  PARTS_DIR as PREVIOUS_ROBUSTNESS_PARTS,
                                  load_frozen_policies, waypoint_sha256)
from diagnose_ppo_braking import evaluate_episode
from robustness_dynamics import RobustnessEnv, Scenario
from test_env_scripted_policy import scripted_action
from train_ppo_lowstd_multiseed import build_holdout_waypoints


PARTS_DIR = MUJOCO_DIR / "reports" / "ppo_baseline_v1_attribution_parts"
FACTOR_LEVELS = {
    "mass": (0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10),
    "thrust": (0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10),
    "force_x_n": (-10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0),
}
HOLD_SCENARIOS = (Scenario("nominal", 0.0),
                  Scenario("mass", 0.95), Scenario("mass", 1.05),
                  Scenario("thrust", 0.95), Scenario("thrust", 1.05),
                  Scenario("force_x_n", -5.0), Scenario("force_x_n", 5.0))


def condition_path(scenario: Scenario, root: Path = PARTS_DIR) -> Path:
    if scenario.kind == "nominal":
        return root / "nominal.json"
    return root / scenario.kind / f"{scenario.value:+.3f}.json"


def hold_path(scenario: Scenario, root: Path = PARTS_DIR) -> Path:
    label = ("nominal" if scenario.kind == "nominal" else
             f"{scenario.kind}_{scenario.value:+.3f}")
    return root / "hold" / f"{label}.json"


def write_condition(path: Path, result: dict) -> None:
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        # Hard-link publication is atomic and refuses to replace existing data.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _mean(values) -> float | None:
    observed = list(values)
    return float(np.mean(observed)) if observed else None


def _tail_mean(records: list[dict], key: str, *, timeouts_only=False) -> dict:
    selected = [row[key] for row in records
                if not timeouts_only or row["termination_reason"] == "time_limit"]
    if not selected:
        return {"episodes": 0, "mean_position_error_xyz_m": None,
                "mean_velocity_xyz_m_s": None,
                "mean_command_velocity_xyz_m_s": None,
                "mean_position_error_norm_m": None, "mean_speed_m_s": None,
                "mean_command_speed_m_s": None, "mean_observed_span_s": None}
    return {
        "episodes": len(selected),
        **{field: np.mean(np.asarray([row[field] for row in selected]), axis=0).tolist()
           for field in ("mean_position_error_xyz_m", "mean_velocity_xyz_m_s",
                         "mean_command_velocity_xyz_m_s")},
        **{field: _mean(row[field] for row in selected)
           for field in ("mean_position_error_norm_m", "mean_speed_m_s",
                         "mean_command_speed_m_s")},
        "mean_observed_span_s": _mean(row["observed_span_s"] for row in selected),
    }


def summarize_records(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Need at least one completed episode")
    reasons = Counter(row["termination_reason"] for row in records)
    successes = reasons["success"]
    return {
        "episodes": len(records), "successes": successes,
        "success_rate": successes / len(records),
        "mean_final_distance_m": _mean(row["final_distance_m"] for row in records),
        "mean_successful_completion_time_s": _mean(
            row["completion_time_s"] for row in records if row["success"]),
        "mean_episode_length": _mean(row["episode_length"] for row in records),
        "crossing_rate": sum(row["crossed_after_first_0_2_m"] for row in records)
                         / len(records),
        "entered_0_2_m_rate": sum(row["entered_0_2_m"] for row in records)
                              / len(records),
        "termination_counts": dict(reasons),
        "tail_5_s": _tail_mean(records, "tail_5_s"),
        "tail_1_s": _tail_mean(records, "tail_1_s"),
        "timeout_tail_5_s": _tail_mean(records, "tail_5_s", timeouts_only=True),
        "timeout_tail_1_s": _tail_mean(records, "tail_1_s", timeouts_only=True),
    }


def evaluate_single_policy(env: RobustnessEnv, policy,
                           waypoints: list[tuple[int, list[float]]]) -> dict:
    """No training; one deterministic closed-loop episode per exact target."""
    if not waypoints or env.reward_version != "v2" or env.render_mode is not None:
        raise ValueError("Need headless Reward V2 and nonempty fixed targets")
    records = []
    for seed, target in waypoints:
        trace = evaluate_episode(env, policy, seed)
        if trace["target_m"] != target:
            raise RuntimeError(f"Holdout target changed at seed {seed}")
        records.append(episode_attribution_summary(trace))
    return {"metrics": summarize_records(records), "episodes": records}


def evaluate_waypoint_condition(scenario: Scenario, policies,
                                waypoints: list[tuple[int, list[float]]]) -> dict:
    if len(waypoints) != 100 or waypoints != build_holdout_waypoints():
        raise ValueError("Every condition must use the exact 100 holdout targets")
    if [seed for seed, _ in policies] != list(EXPECTED_TRAIN_SEEDS):
        raise ValueError("Every condition must use all five frozen PPO seeds")
    result = {
        "scenario": {"kind": scenario.kind, "value": scenario.value},
        "waypoint_sha256": waypoint_sha256(waypoints),
        "waypoint_seeds": [seed for seed, _ in waypoints],
        "policy_seeds": list(EXPECTED_TRAIN_SEEDS),
        "deterministic": True, "evaluations": [],
    }
    named_policies = [("scripted", None, scripted_action)] + [
        ("ppo", seed, lambda obs, trained=model: trained.predict(
            obs, deterministic=True)[0]) for seed, model in policies]
    for name, seed, action_function in named_policies:
        env = RobustnessEnv(scenario)
        try:
            evaluation = evaluate_single_policy(env, action_function, waypoints)
        finally:
            env.close()
        evaluation.update({"policy": name, "training_seed": seed})
        result["evaluations"].append(evaluation)
        print(f"{scenario.kind}={scenario.value:g} {name} seed={seed}: "
              f"{evaluation['metrics']['successes']}/100", flush=True)
    return result


def validate_nominal_condition(result: dict) -> None:
    """Require prior frozen PPO performance and a successful scripted reference."""
    previous = json.loads((PREVIOUS_ROBUSTNESS_PARTS / "nominal.json").read_text(
        encoding="utf-8"))
    evaluations = result["evaluations"]
    if (len(evaluations) != 6 or evaluations[0]["policy"] != "scripted"
            or evaluations[0]["metrics"]["successes"] < 95):
        raise RuntimeError("Scripted nominal reference is not near 100/100")
    metric_keys = ("successes", "mean_final_distance_m",
                   "mean_successful_completion_time_s", "mean_episode_length",
                   "crossing_rate")
    for current, old in zip(evaluations[1:], previous["evaluations"]):
        if (current["policy"] != "ppo"
                or current["training_seed"] != old["training_seed"]):
            raise RuntimeError("Nominal checkpoint order changed")
        for key in metric_keys:
            observed, expected = current["metrics"][key], old["metrics"][key]
            if observed is None or expected is None:
                matches = observed is None and expected is None
            else:
                matches = math.isclose(observed, expected, rel_tol=1e-10,
                                       abs_tol=1e-10)
            if not matches:
                raise RuntimeError(f"Nominal PPO {current['training_seed']} {key} changed")


def _positive_step_count(seconds: float, dt: float) -> int:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Probe duration must be finite and positive")
    count = round(seconds / dt)
    if count < 1 or not math.isclose(count * dt, seconds, abs_tol=1e-9):
        raise ValueError("Probe duration must be a multiple of policy timestep")
    return count


def _current_state(env: RobustnessEnv) -> dict:
    return {"position_m": env.data.qpos[:3].tolist(),
            "velocity_m_s": env.data.qvel[:3].tolist(),
            "time_s": float(env.data.time)}


def evaluate_hold_condition(scenario: Scenario, *, warmup_s: float = 2.0,
                            observe_s: float = 15.0) -> dict:
    """Zero velocity command, nominal settle, then one plant-only perturbation."""
    # The hold probe lasts 2+15 s; this local limit does not change the
    # frozen 15-second waypoint task used for scripted/PPO comparisons.
    env = RobustnessEnv(Scenario("nominal", 0.0),
                        max_episode_seconds=warmup_s + observe_s + 1.0)
    hover_point = np.array([0.0, 0.0, 1.0])
    zero_action = np.zeros(4, dtype=np.float32)
    try:
        warmup_steps = _positive_step_count(warmup_s, env.policy_dt)
        observe_steps = _positive_step_count(observe_s, env.policy_dt)
        if warmup_steps + observe_steps >= env.max_episode_steps:
            raise ValueError("Hold probe duration must fit inside episode time limit")
        env.reset(seed=20271001, options={"target_position": [2.0, 2.0, 1.0]})
        for _ in range(warmup_steps):
            _, _, terminated, truncated, _ = env.step(zero_action)
            if terminated or truncated:
                raise RuntimeError("Nominal hold ended before perturbation")
        injection_state = _current_state(env)
        if scenario.kind != "nominal":
            env.activate_scenario(scenario)
        if not math.isclose(float(env.data.time), warmup_s, abs_tol=1e-8):
            raise RuntimeError("Plant injection happened at the wrong time")
        records = []
        terminal_reason = "probe_complete"
        for _ in range(observe_steps):
            before = _current_state(env)
            _, _, terminated, truncated, info = env.step(zero_action)
            records.append({
                "time_s": before["time_s"], "post_time_s": float(env.data.time),
                "position_m": before["position_m"],
                "target_m": hover_point.tolist(),
                "position_error_m": (hover_point - before["position_m"]).tolist(),
                "velocity_m_s": before["velocity_m_s"],
                "velocity_command_m_s": info["velocity_command_m_s"].tolist(),
            })
            if terminated or truncated:
                terminal_reason = info["termination_reason"]
                break
        result = {
            "scenario": {"kind": scenario.kind, "value": scenario.value},
            "warmup_s": warmup_s, "requested_observation_s": observe_s,
            "injection_time_s": injection_state["time_s"],
            "state_at_injection": injection_state,
            "plant_mass_after_injection_kg": float(env.model.body_mass[env.uav_body_id]),
            "controller_mass_kg": env.velocity_controller.attitude_controller.mass_kg,
            "policy_steps_after_injection": len(records),
            "terminal_reason": terminal_reason,
            "terminal_state": _current_state(env),
            "final_position_error_xyz_m": (hover_point - env.data.qpos[:3]).tolist(),
            "tail_5_s": terminal_window_stats(records, hover_point, 5.0),
            "tail_1_s": terminal_window_stats(records, hover_point, 1.0),
        }
        return result
    finally:
        env.close()


def run_factor(factor: str) -> list[Path]:
    if factor not in FACTOR_LEVELS and factor != "nominal":
        raise ValueError(f"Unknown factor: {factor}")
    scenarios = ([Scenario("nominal", 0.0)] if factor == "nominal" else
                 [Scenario(factor, value) for value in FACTOR_LEVELS[factor]])
    if factor != "nominal" and not condition_path(Scenario("nominal", 0.0)).is_file():
        raise RuntimeError("Run and validate shared nominal before factor sweeps")
    paths = [condition_path(scenario) for scenario in scenarios]
    if all(path.exists() for path in paths):
        raise FileExistsError(f"Factor already evaluated: {factor}")
    policies = load_frozen_policies()
    waypoints = build_holdout_waypoints()
    start = time.perf_counter()
    for scenario, path in zip(scenarios, paths):
        if path.exists():
            from summarize_attribution_audit import validate_waypoint_part
            saved = json.loads(path.read_text(encoding="utf-8"))
            validate_waypoint_part(saved, scenario, waypoints)
            print(f"validated existing {path}", flush=True)
            continue
        result = evaluate_waypoint_condition(scenario, policies, waypoints)
        if factor == "nominal":
            validate_nominal_condition(result)
        result["factor_elapsed_wall_s"] = time.perf_counter() - start
        write_condition(path, result)
        print(f"saved {path}", flush=True)
    return paths


def run_hold(root: Path = PARTS_DIR) -> list[Path]:
    saved = []
    for scenario in HOLD_SCENARIOS:
        path = hold_path(scenario, root)
        if path.exists():
            from summarize_attribution_audit import validate_hold_result
            existing = json.loads(path.read_text(encoding="utf-8"))
            validate_hold_result(existing, scenario)
            saved.append(path)
            print(f"validated existing {path}", flush=True)
            continue
        result = evaluate_hold_condition(scenario)
        write_condition(path, result)
        saved.append(path)
        print(f"hold {scenario.kind}={scenario.value:g}: "
              f"end={result['terminal_reason']}, "
              f"error={result['final_position_error_xyz_m']}", flush=True)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", choices=("nominal", *FACTOR_LEVELS, "hold"),
                        required=True)
    args = parser.parse_args()
    if args.factor == "hold":
        run_hold()
    else:
        run_factor(args.factor)
