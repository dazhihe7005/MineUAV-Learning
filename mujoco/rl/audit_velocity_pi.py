"""Inference-only velocity-PI disturbance and frozen-policy audit."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from attribution_audit import (HOLD_SCENARIOS, evaluate_single_policy,
                               write_condition)
from attribution_diagnostics import terminal_window_stats
from audit_ppo_robustness import (EXPECTED_TRAIN_SEEDS, MUJOCO_DIR,
                                  load_frozen_policies, waypoint_sha256)
from robustness_dynamics import RobustnessEnv, Scenario
from test_env_scripted_policy import scripted_action
from train_ppo_lowstd_multiseed import build_holdout_waypoints
from velocity_command_controller import VelocityCommandController


PARTS_DIR = MUJOCO_DIR / "reports" / "velocity_pi_parts"
KI_XY = 0.5  # s^-2; fixed conservative baseline, no search
KI_Z = 0.8   # s^-2
FACTOR_LEVELS = {
    "mass": (0.95, 0.98, 0.99, 1.01, 1.02, 1.05),
    "thrust": (0.95, 0.98, 0.99, 1.01, 1.02, 1.05),
    "force_x_n": (-5.0, -2.0, -1.0, 1.0, 2.0, 5.0),
}


def configure_pi_env(env: RobustnessEnv) -> None:
    """Replace only the velocity outer loop; retain attitude/Kp/limits/plant."""
    old = env.velocity_controller
    env.velocity_controller = VelocityCommandController(
        old.attitude_controller, env.model.opt.gravity,
        kv=old.kv, horizontal_accel_limit=old.horizontal_accel_limit,
        vertical_accel_limit=old.vertical_accel_limit,
        ki_xy=KI_XY, ki_z=KI_Z)


def hold_output_path(scenario: Scenario, root: Path = PARTS_DIR) -> Path:
    label = ("nominal" if scenario.kind == "nominal" else
             f"{scenario.kind}_{scenario.value:+.3f}")
    return root / "hold" / f"{label}.json"


def waypoint_output_path(scenario: Scenario, root: Path = PARTS_DIR) -> Path:
    if scenario.kind == "nominal":
        return root / "waypoint" / "nominal.json"
    return root / "waypoint" / scenario.kind / f"{scenario.value:+.3f}.json"


def write_new_part(path: Path, result: dict) -> None:
    """Atomically publish a distinct part without replacing old results."""
    write_condition(path, result)


def _positive_step_count(seconds: float, dt: float) -> int:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Duration must be finite and positive")
    count = round(seconds / dt)
    if count < 1 or not math.isclose(count * dt, seconds, abs_tol=1e-9):
        raise ValueError("Duration must be a multiple of policy timestep")
    return count


def _state(env: RobustnessEnv) -> dict:
    return {"position_m": env.data.qpos[:3].tolist(),
            "velocity_m_s": env.data.qvel[:3].tolist(),
            "time_s": float(env.data.time)}


def _pi_tail(steps: list[dict], terminal_time: float, seconds: float) -> dict:
    selected = [row for row in steps
                if row["time_s"] >= terminal_time - seconds - 1e-10]
    if not selected:
        raise ValueError("No PI samples in requested terminal window")
    return {
        "policy_steps": len(selected),
        "mean_integral_error_m": np.mean(
            [row["integral_error_m"] for row in selected], axis=0).tolist(),
        "mean_integral_acceleration_m_s2": np.mean(
            [row["integral_acceleration_m_s2"] for row in selected], axis=0).tolist(),
        "mean_desired_acceleration_m_s2": np.mean(
            [row["desired_acceleration_m_s2"] for row in selected], axis=0).tolist(),
        "max_motor_rpm": max(row["max_motor_rpm"] for row in selected),
        "allocator_saturation_count": sum(
            row["allocator_saturation_count"] for row in selected),
    }


def evaluate_pi_hold(scenario: Scenario, *, warmup_s: float = 2.0,
                     observe_s: float = 15.0) -> dict:
    """Settle nominal at zero velocity, inject one plant mismatch, observe PI."""
    env = RobustnessEnv(Scenario("nominal", 0.0),
                        max_episode_seconds=warmup_s + observe_s + 1.0)
    configure_pi_env(env)
    hover_point = np.array([0.0, 0.0, 1.0])
    zero_action = np.zeros(4, dtype=np.float32)
    try:
        warmup_steps = _positive_step_count(warmup_s, env.policy_dt)
        observe_steps = _positive_step_count(observe_s, env.policy_dt)
        if warmup_steps + observe_steps >= env.max_episode_steps:
            raise ValueError("Hold probe duration exceeds local episode limit")
        env.reset(seed=20271001, options={"target_position": [2.0, 2.0, 1.0]})
        for _ in range(warmup_steps):
            _, _, terminated, truncated, _ = env.step(zero_action)
            if terminated or truncated:
                raise RuntimeError("Nominal hover ended before mismatch injection")
        injection_state = _state(env)
        if scenario.kind != "nominal":
            env.activate_scenario(scenario)
        steps = []
        terminal_reason = "probe_complete"
        max_integral_xy = 0.0
        max_integral_z = 0.0
        max_motor_rpm = 0.0
        for _ in range(observe_steps):
            before = _state(env)
            controller = env.velocity_controller
            integral = controller.integral_error
            integral_accel = controller.integral_acceleration_world
            desired_accel = controller.last_desired_acceleration_world
            _, _, terminated, truncated, info = env.step(zero_action)
            steps.append({
                "time_s": before["time_s"], "post_time_s": float(env.data.time),
                "position_m": before["position_m"],
                "target_m": hover_point.tolist(),
                "position_error_m": (hover_point - before["position_m"]).tolist(),
                "velocity_m_s": before["velocity_m_s"],
                "velocity_command_m_s": info["velocity_command_m_s"].tolist(),
                "integral_error_m": integral.tolist(),
                "integral_acceleration_m_s2": integral_accel.tolist(),
                "desired_acceleration_m_s2": desired_accel.tolist(),
                "max_motor_rpm": float(info["max_motor_rpm"]),
                "allocator_saturation_count": int(info["allocator_saturation_count"]),
            })
            current_integral_accel = controller.integral_acceleration_world
            max_integral_xy = max(max_integral_xy, float(np.linalg.norm(
                current_integral_accel[:2])))
            max_integral_z = max(max_integral_z, abs(float(current_integral_accel[2])))
            max_motor_rpm = max(max_motor_rpm, float(info["max_motor_rpm"]))
            if terminated or truncated:
                terminal_reason = info["termination_reason"]
                break
        terminal_state = _state(env)
        result = {
            "scenario": {"kind": scenario.kind, "value": scenario.value},
            "ki_xy_s2": KI_XY, "ki_z_s2": KI_Z,
            "integral_accel_limit_xy_m_s2": controller.integral_accel_limit_xy,
            "integral_accel_limit_z_m_s2": controller.integral_accel_limit_z,
            "warmup_s": warmup_s, "requested_observation_s": observe_s,
            "injection_time_s": injection_state["time_s"],
            "actual_observation_s": terminal_state["time_s"] - injection_state["time_s"],
            "state_at_injection": injection_state,
            "plant_mass_after_injection_kg": float(
                env.model.body_mass[env.uav_body_id]),
            "controller_mass_kg": controller.attitude_controller.mass_kg,
            "policy_steps_after_injection": len(steps),
            "terminal_reason": terminal_reason,
            "terminal_state": terminal_state,
            "final_position_error_xyz_m": (hover_point - env.data.qpos[:3]).tolist(),
            "final_integral_error_m": controller.integral_error.tolist(),
            "final_integral_acceleration_m_s2":
                controller.integral_acceleration_world.tolist(),
            "anti_windup_freeze_count_xy": controller.anti_windup_freeze_count_xy,
            "anti_windup_freeze_count_z": controller.anti_windup_freeze_count_z,
            "max_integral_accel_xy_m_s2": max_integral_xy,
            "max_integral_accel_z_m_s2": max_integral_z,
            "max_motor_rpm": max_motor_rpm,
            "allocator_saturation_count": sum(
                row["allocator_saturation_count"] for row in steps),
            "tail_5_s": terminal_window_stats(steps, hover_point, 5.0),
            "tail_1_s": terminal_window_stats(steps, hover_point, 1.0),
            "pi_tail_5_s": _pi_tail(steps, terminal_state["time_s"], 5.0),
            "pi_tail_1_s": _pi_tail(steps, terminal_state["time_s"], 1.0),
            "steps": steps,
        }
        return result
    finally:
        env.close()


def run_holds(root: Path = PARTS_DIR) -> list[Path]:
    paths = []
    for scenario in HOLD_SCENARIOS:
        path = hold_output_path(scenario, root)
        if path.exists():
            from summarize_velocity_pi import validate_pi_hold
            validate_pi_hold(json.loads(path.read_text(encoding="utf-8")), scenario)
            paths.append(path)
            print(f"validated existing {path}", flush=True)
            continue
        result = evaluate_pi_hold(scenario)
        write_new_part(path, result)
        paths.append(path)
        print(f"hold {scenario.kind}={scenario.value:g}: "
              f"end={result['terminal_reason']}, "
              f"error={result['final_position_error_xyz_m']}", flush=True)
    return paths


def evaluate_pi_single_policy(env: RobustnessEnv, policy,
                              waypoints: list[tuple[int, list[float]]]) -> dict:
    """Use the established episode, crossing and 1/5-s tail definitions."""
    if env.velocity_controller.ki_xy != KI_XY or env.velocity_controller.ki_z != KI_Z:
        raise ValueError("PI gains were not enabled on this private environment")
    return evaluate_single_policy(env, policy, waypoints)


def evaluate_pi_waypoint_condition(scenario: Scenario, policies,
                                   waypoints: list[tuple[int, list[float]]]) -> dict:
    if len(waypoints) != 100 or waypoints != build_holdout_waypoints():
        raise ValueError("Every PI condition must use exact fixed holdout 100")
    if [seed for seed, _ in policies] != list(EXPECTED_TRAIN_SEEDS):
        raise ValueError("Every PI condition must use all five frozen PPO seeds")
    result = {
        "scenario": {"kind": scenario.kind, "value": scenario.value},
        "waypoint_sha256": waypoint_sha256(waypoints),
        "waypoint_seeds": [seed for seed, _ in waypoints],
        "policy_seeds": list(EXPECTED_TRAIN_SEEDS),
        "deterministic": True,
        "controller": {"mode": "velocity_pi", "ki_xy_s2": KI_XY,
                       "ki_z_s2": KI_Z},
        "evaluations": [],
    }
    named_policies = [("scripted", None, scripted_action)] + [
        ("ppo", seed, lambda obs, trained=model: trained.predict(
            obs, deterministic=True)[0]) for seed, model in policies]
    for name, seed, action_function in named_policies:
        env = RobustnessEnv(scenario)
        try:
            configure_pi_env(env)
            evaluation = evaluate_pi_single_policy(env, action_function, waypoints)
        finally:
            env.close()
        evaluation.update({"policy": name, "training_seed": seed})
        result["evaluations"].append(evaluation)
        print(f"PI {scenario.kind}={scenario.value:g} {name} seed={seed}: "
              f"{evaluation['metrics']['successes']}/100", flush=True)
    return result


def run_factor(factor: str) -> list[Path]:
    if factor not in FACTOR_LEVELS and factor != "nominal":
        raise ValueError(f"Unknown PI factor: {factor}")
    scenarios = ([Scenario("nominal", 0.0)] if factor == "nominal" else
                 [Scenario(factor, value) for value in FACTOR_LEVELS[factor]])
    if (factor != "nominal"
            and not waypoint_output_path(Scenario("nominal", 0.0)).is_file()):
        raise RuntimeError("Validate and save PI nominal before sensitivity sweeps")
    paths = [waypoint_output_path(scenario) for scenario in scenarios]
    policies = load_frozen_policies()
    waypoints = build_holdout_waypoints()
    started = time.perf_counter()
    for scenario, path in zip(scenarios, paths):
        if path.exists():
            from summarize_attribution_audit import validate_waypoint_part
            saved = json.loads(path.read_text(encoding="utf-8"))
            validate_waypoint_part(saved, scenario, waypoints)
            if saved.get("controller") != {"mode": "velocity_pi",
                                           "ki_xy_s2": KI_XY, "ki_z_s2": KI_Z}:
                raise ValueError(f"Existing condition is not this PI experiment: {path}")
            print(f"validated existing {path}", flush=True)
            continue
        result = evaluate_pi_waypoint_condition(scenario, policies, waypoints)
        result["factor_elapsed_wall_s"] = time.perf_counter() - started
        write_new_part(path, result)
        print(f"saved {path}", flush=True)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold", action="store_true", help="Run seven zero-command holds")
    parser.add_argument("--factor", choices=("nominal", *FACTOR_LEVELS),
                        help="Evaluate scripted and five frozen PPO checkpoints")
    arguments = parser.parse_args()
    if arguments.hold and arguments.factor:
        parser.error("Select only one of --hold or --factor")
    if arguments.hold:
        run_holds()
    elif arguments.factor:
        run_factor(arguments.factor)
    else:
        parser.error("Select --hold or --factor")
