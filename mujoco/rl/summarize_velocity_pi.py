"""Compare the opt-in velocity PI audit against frozen P-controller results."""

import hashlib
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from attribution_audit import HOLD_SCENARIOS, write_condition
from attribution_diagnostics import terminal_window_stats
from audit_ppo_robustness import MUJOCO_DIR, waypoint_sha256
from audit_velocity_pi import (FACTOR_LEVELS, KI_XY, KI_Z, PARTS_DIR,
                               _pi_tail, hold_output_path, waypoint_output_path)
from robustness_dynamics import Scenario
from summarize_attribution_audit import (describe_hold_observation,
                                         point_from_part, validate_waypoint_part,
                                         _check_finite_tree)
from train_ppo_lowstd_multiseed import build_holdout_waypoints


OLD_P_REPORT = MUJOCO_DIR / "reports" / "ppo_baseline_v1_attribution.json"
# The first generated report is retained as an audit artifact. V2 adds
# stricter part validation and an explicit paired-policy interpretation.
REPORT_PATH = MUJOCO_DIR / "reports" / "velocity_pi_robustness_v2.json"
FIGURES = {
    "mass": MUJOCO_DIR / "reports" / "velocity_pi_mass_comparison_v2.png",
    "thrust": MUJOCO_DIR / "reports" / "velocity_pi_thrust_comparison_v2.png",
    "force_x_n": MUJOCO_DIR / "reports" / "velocity_pi_force_comparison_v2.png",
}


def expected_waypoint_scenarios() -> list[Scenario]:
    return [Scenario("nominal", 0.0)] + [
        Scenario(factor, value) for factor, values in FACTOR_LEVELS.items()
        for value in values]


def _same_value(actual, expected) -> bool:
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_same_value(actual[key], item)
                        for key, item in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_same_value(left, right)
                        for left, right in zip(actual, expected)))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9))
    return actual == expected


def validate_pi_hold(row: dict, scenario: Scenario) -> None:
    """Recompute both tail windows and check every PI/controller sample."""
    try:
        _check_finite_tree(row)
        if row["scenario"] != {"kind": scenario.kind, "value": scenario.value}:
            raise ValueError("PI hold scenario differs")
        if (not math.isclose(row["ki_xy_s2"], KI_XY, abs_tol=1e-12)
                or not math.isclose(row["ki_z_s2"], KI_Z, abs_tol=1e-12)
                or not math.isclose(row["integral_accel_limit_xy_m_s2"],
                                    1.5, abs_tol=1e-12)
                or not math.isclose(row["integral_accel_limit_z_m_s2"],
                                    1.5, abs_tol=1e-12)
                or not math.isclose(row["warmup_s"], 2.0, abs_tol=1e-8)
                or not math.isclose(row["requested_observation_s"], 15.0,
                                    abs_tol=1e-8)
                or not math.isclose(row["controller_mass_kg"], 7.0,
                                    abs_tol=1e-9)):
            raise ValueError("PI hold configuration differs")
        observation = describe_hold_observation(row)
        if not math.isclose(row["actual_observation_s"],
                            observation["actual_observation_s"], abs_tol=1e-8):
            raise ValueError("PI observation duration differs")
        steps = row["steps"]
        if len(steps) != row["policy_steps_after_injection"] or not steps:
            raise ValueError("PI hold step count differs")
        for state_key in ("state_at_injection", "terminal_state"):
            for vector_key in ("position_m", "velocity_m_s"):
                vector = np.asarray(row[state_key][vector_key], dtype=float)
                if vector.shape != (3,) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid {state_key}.{vector_key}")
        expected_mass = 7.0 * scenario.value if scenario.kind == "mass" else 7.0
        if not math.isclose(row["plant_mass_after_injection_kg"], expected_mass,
                            abs_tol=1e-8):
            raise ValueError("PI plant mass differs")
        for step in steps:
            for key in ("position_m", "position_error_m", "velocity_m_s",
                        "velocity_command_m_s", "integral_error_m",
                        "integral_acceleration_m_s2", "desired_acceleration_m_s2"):
                value = np.asarray(step[key], dtype=float)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(f"Bad PI step vector: {key}")
            if (not np.allclose(step["velocity_command_m_s"], [0, 0, 0],
                                atol=1e-12)
                    or not np.array_equal(step["target_m"], [0, 0, 1])):
                raise ValueError("PI hold command/target differs")
            if (np.linalg.norm(step["integral_acceleration_m_s2"][:2]) > 1.5 + 1e-8
                    or abs(step["integral_acceleration_m_s2"][2]) > 1.5 + 1e-8):
                raise ValueError("PI integrator exceeded contribution limit")
            if not np.allclose(step["integral_acceleration_m_s2"],
                               np.asarray(step["integral_error_m"])
                               * [KI_XY, KI_XY, KI_Z], atol=1e-9):
                raise ValueError("PI step contribution differs from Ki times state")
            if (step["max_motor_rpm"] < 0
                    or step["allocator_saturation_count"] < 0):
                raise ValueError("PI motor/saturation data invalid")
        expected_max_xy = max(np.linalg.norm(
            step["integral_acceleration_m_s2"][:2]) for step in steps)
        expected_max_z = max(abs(step["integral_acceleration_m_s2"][2])
                             for step in steps)
        final_extra = np.asarray(row["final_integral_acceleration_m_s2"])
        if final_extra.shape != (3,) or not np.isfinite(final_extra).all():
            raise ValueError("PI final contribution invalid")
        expected_max_xy = max(expected_max_xy, np.linalg.norm(final_extra[:2]))
        expected_max_z = max(expected_max_z, abs(final_extra[2]))
        if (expected_max_xy > 1.5 + 1e-8 or expected_max_z > 1.5 + 1e-8
                or not math.isclose(row["max_integral_accel_xy_m_s2"],
                                    expected_max_xy, rel_tol=1e-9, abs_tol=1e-9)
                or not math.isclose(row["max_integral_accel_z_m_s2"],
                                    expected_max_z, rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError("PI reported contribution maxima differ from samples")
        terminal_time = row["terminal_state"]["time_s"]
        if not math.isclose(steps[0]["time_s"], row["injection_time_s"],
                            abs_tol=1e-8):
            raise ValueError("PI hold starts at wrong injection time")
        if not math.isclose(steps[-1]["post_time_s"], terminal_time,
                            abs_tol=1e-8):
            raise ValueError("PI hold ends at wrong terminal time")
        for left, right in zip(steps, steps[1:]):
            if not math.isclose(left["post_time_s"], right["time_s"], abs_tol=1e-8):
                raise ValueError("PI hold time sequence has a gap")
        for seconds, key, pi_key in ((5.0, "tail_5_s", "pi_tail_5_s"),
                                     (1.0, "tail_1_s", "pi_tail_1_s")):
            if not _same_value(row[key], terminal_window_stats(
                    steps, [0, 0, 1], seconds)):
                raise ValueError(f"PI {key} differs from raw steps")
            if not _same_value(row[pi_key], _pi_tail(steps, terminal_time, seconds)):
                raise ValueError(f"PI {pi_key} differs from raw steps")
        if (not _same_value(row["max_motor_rpm"],
                            max(step["max_motor_rpm"] for step in steps))
                or row["allocator_saturation_count"] != sum(
                    step["allocator_saturation_count"] for step in steps)):
            raise ValueError("PI motor/saturation summary differs")
        if (final_extra.shape != (3,) or not np.isfinite(final_extra).all()
                or np.linalg.norm(final_extra[:2]) > 1.5 + 1e-8
                or abs(final_extra[2]) > 1.5 + 1e-8
                or not np.allclose(final_extra,
                                   np.asarray(row["final_integral_error_m"])
                                   * [KI_XY, KI_XY, KI_Z], atol=1e-9)):
            raise ValueError("PI final integral state invalid")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Incomplete PI hold part") from exc


def compare_point(old_p_point: dict, pi_part: dict, scenario: Scenario,
                  waypoints: list[tuple[int, list[float]]]) -> dict:
    validate_waypoint_part(pi_part, scenario, waypoints)
    if pi_part["controller"] != {"mode": "velocity_pi", "ki_xy_s2": KI_XY,
                                  "ki_z_s2": KI_Z}:
        raise ValueError("Waypoint part did not use prescribed PI gains")
    value = 0.0 if scenario.kind == "nominal" else scenario.value
    if not math.isclose(old_p_point["value"], value, abs_tol=1e-12):
        raise ValueError("Historical P point has wrong perturbation level")
    return {"value": value, "old_p": old_p_point,
            "new_pi": point_from_part(pi_part, value)}


def derive_findings(nominal: dict, curves: dict) -> dict:
    """Surface the paired frozen-policy tradeoff, not just plotted raw rates."""
    old = nominal["old_p"]
    new = nominal["new_pi"]
    old_success = old["ppo_aggregate"]["success_rate"]["mean"]
    new_success = new["ppo_aggregate"]["success_rate"]["mean"]
    old_crossing = old["ppo_aggregate"]["crossing_rate"]["mean"]
    new_crossing = new["ppo_aggregate"]["crossing_rate"]["mean"]
    selected = {}
    for factor, values in (("mass", (0.95, 1.05)),
                           ("thrust", (0.95, 1.05)),
                           ("force_x_n", (-5.0, 5.0))):
        selected[factor] = {}
        for value in values:
            point = next(row for row in curves[factor]
                         if math.isclose(row["value"], value, abs_tol=1e-12))
            selected[factor][str(value)] = {
                "scripted_success_old_new": [point[which]["scripted"]["success_rate"]
                                             for which in ("old_p", "new_pi")],
                "ppo_mean_success_old_new": [point[which]["ppo_aggregate"]
                                             ["success_rate"]["mean"]
                                             for which in ("old_p", "new_pi")],
            }
    extreme_improvement = all(
        values["scripted_success_old_new"][1]
        > values["scripted_success_old_new"][0]
        and values["ppo_mean_success_old_new"][1]
        > values["ppo_mean_success_old_new"][0]
        for factor_rows in selected.values() for values in factor_rows.values())
    if extreme_improvement and new_success < old_success and new_crossing > old_crossing:
        verdict = ("PI improves static-disturbance rejection, but is not a drop-in "
                   "replacement for these frozen PPO policies: nominal success "
                   "decreases and waypoint crossing increases. Stable zero-command "
                   "holds do not imply stable waypoint control.")
    else:
        verdict = ("Interpret paired nominal success and crossing together with "
                   "disturbed-condition performance; zero-command holds alone do "
                   "not establish waypoint robustness.")
    return {
        "nominal_scripted_success_old_new": [old["scripted"]["success_rate"],
                                             new["scripted"]["success_rate"]],
        "nominal_ppo_success_old_new": [old_success, new_success],
        "nominal_ppo_success_change_percentage_points":
            100.0 * (new_success - old_success),
        "nominal_ppo_crossing_old_new": [old_crossing, new_crossing],
        "nominal_ppo_crossing_change_percentage_points":
            100.0 * (new_crossing - old_crossing),
        "plus_5_n_force_ppo_success_old_new":
            selected["force_x_n"]["5.0"]["ppo_mean_success_old_new"],
        "paired_extreme_conditions": selected,
        "all_six_extreme_conditions_improved_for_scripted_and_ppo":
            extreme_improvement,
        "verdict": verdict,
    }


def plot_comparison(factor: str, rows: list[dict], target: Path) -> None:
    if factor not in FACTOR_LEVELS or not rows:
        raise ValueError("Known factor and nonempty comparison required")
    if target.exists():
        raise FileExistsError(f"Preserving existing figure: {target}")
    ordered = sorted(rows, key=lambda row: row["value"])
    x = ([row["value"] for row in ordered] if factor == "force_x_n" else
         [(row["value"] - 1.0) * 100 for row in ordered])
    xlabel = ("Constant world-X force (N)" if factor == "force_x_n" else
              "Plant parameter change from nominal (%)")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for controller, policy, color, marker, line in (
            ("old_p", "scripted", "tab:orange", "s", "--"),
            ("new_pi", "scripted", "tab:red", "o", "-"),
            ("old_p", "ppo_aggregate", "tab:blue", "s", "--"),
            ("new_pi", "ppo_aggregate", "tab:green", "o", "-")):
        y = [(row[controller][policy]["success_rate"] if policy == "scripted"
              else row[controller][policy]["success_rate"]["mean"])
             for row in ordered]
        axis.plot(x, y, color=color, marker=marker, linestyle=line, linewidth=2,
                  label=f"{'P' if controller == 'old_p' else 'PI'} "
                        f"{'scripted' if policy == 'scripted' else 'PPO mean (5 seeds)'}")
    axis.set(xlabel=xlabel, ylabel="Holdout waypoint success rate", ylim=(-0.02, 1.02),
             title=f"Velocity P vs PI — {factor.replace('_', ' ')}")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    payload = io.BytesIO()
    fig.savefig(payload, dpi=150, format="png")
    plt.close(fig)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        output.write(payload.getvalue())


def _oscillation_diagnostic(steps: list[dict], terminal_time: float) -> dict:
    selected = [row for row in steps if row["time_s"] >= terminal_time - 5.0 - 1e-10]
    velocity = np.asarray([row["velocity_m_s"] for row in selected])
    position = np.asarray([row["position_m"] for row in selected])
    reversals = []
    for axis in range(3):
        significant = np.sign(velocity[:, axis][np.abs(velocity[:, axis]) >= 0.02])
        reversals.append(int(np.sum(significant[1:] != significant[:-1])))
    return {"velocity_std_xyz_m_s": np.std(velocity, axis=0).tolist(),
            "velocity_direction_reversals_xyz": reversals,
            "position_displacement_xyz_m": (position[-1] - position[0]).tolist()}


def summarize_parts() -> dict:
    if REPORT_PATH.exists() or any(path.exists() for path in FIGURES.values()):
        raise FileExistsError("Preserving existing PI comparison outputs")
    scenarios = expected_waypoint_scenarios()
    expected = ({waypoint_output_path(item) for item in scenarios}
                | {hold_output_path(item) for item in HOLD_SCENARIOS})
    found = set(PARTS_DIR.rglob("*.json")) if PARTS_DIR.exists() else set()
    if expected != found:
        raise RuntimeError(f"PI parts mismatch: missing={expected-found}, "
                           f"unexpected={found-expected}")
    old = json.loads(OLD_P_REPORT.read_text(encoding="utf-8"))
    waypoints = build_holdout_waypoints()
    if old["waypoint_sha256"] != waypoint_sha256(waypoints):
        raise ValueError("Historical P targets differ from PI targets")
    model_path = MUJOCO_DIR / "models" / "mine_uav_dynamics_v2.xml"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if old["nominal_model_sha256"] != model_hash:
        raise ValueError("Nominal dynamics XML changed since P reference")
    nominal_scenario = Scenario("nominal", 0.0)
    nominal = compare_point(old["nominal"], json.loads(
        waypoint_output_path(nominal_scenario).read_text(encoding="utf-8")),
        nominal_scenario, waypoints)
    curves = {}
    for factor, values in FACTOR_LEVELS.items():
        zero_value = 0.0 if factor == "force_x_n" else 1.0
        points = [dict(nominal, value=zero_value)]
        for value in values:
            scenario = Scenario(factor, value)
            old_point = next(row for row in old["curves"][factor]
                             if math.isclose(row["value"], value, abs_tol=1e-12))
            part = json.loads(waypoint_output_path(scenario).read_text(encoding="utf-8"))
            points.append(compare_point(old_point, part, scenario, waypoints))
        curves[factor] = sorted(points, key=lambda row: row["value"])
    holds = {}
    for scenario in HOLD_SCENARIOS:
        key = f"{scenario.kind}:{scenario.value:g}"
        row = json.loads(hold_output_path(scenario).read_text(encoding="utf-8"))
        validate_pi_hold(row, scenario)
        old_hold = old["low_level_zero_command_hold"][key]
        holds[key] = {
            "old_p": {"terminal_reason": old_hold["terminal_reason"],
                      "actual_observation_s": old["hold_observation"][key]
                          ["actual_observation_s"],
                      "tail_5_s": old_hold["tail_5_s"],
                      "tail_1_s": old_hold["tail_1_s"],
                      "final_position_error_xyz_m": old_hold[
                          "final_position_error_xyz_m"]},
            "new_pi": {field: row[field] for field in (
                "terminal_reason", "actual_observation_s", "tail_5_s", "tail_1_s",
                "pi_tail_5_s", "pi_tail_1_s", "final_position_error_xyz_m",
                "final_integral_error_m", "final_integral_acceleration_m_s2",
                "max_integral_accel_xy_m_s2", "max_integral_accel_z_m_s2",
                "max_motor_rpm", "allocator_saturation_count",
                "anti_windup_freeze_count_xy", "anti_windup_freeze_count_z")},
            "pi_last_5_s_oscillation": _oscillation_diagnostic(
                row["steps"], row["terminal_state"]["time_s"]),
            "pi_part_path": str(hold_output_path(scenario)),
        }
    result = {
        "audit": "Velocity PI static-disturbance rejection, inference only",
        "report_version": 2,
        "old_p_report": str(OLD_P_REPORT),
        "nominal_model_path": str(model_path), "nominal_model_sha256": model_hash,
        "waypoint_sha256": waypoint_sha256(waypoints),
        "controller": {"kp_xyz_s1": [1.5, 1.5, 2.0],
                       "ki_xy_s2": KI_XY, "ki_z_s2": KI_Z,
                       "integral_accel_limit_xy_m_s2": 1.5,
                       "integral_accel_limit_z_m_s2": 1.5,
                       "anti_windup": "Conditional integration freeze when raw "
                                      "acceleration saturates and increment pushes "
                                      "further into saturation; unwind remains allowed."},
        "nominal": nominal, "curves": curves, "zero_command_holds": holds,
        "findings": derive_findings(nominal, curves),
        "waypoint_conditions": len(scenarios),
        "waypoint_episodes": len(scenarios) * 6 * len(waypoints),
        "hold_conditions": len(holds),
        "figures": {factor: str(path) for factor, path in FIGURES.items()},
        "interpretation_caveat": "Zero-velocity hold checks stopping drift, not "
                                 "returning to the original position. Safety-ended "
                                 "holds are late-stage observations, not equilibrium. "
                                 "Waypoint tail windows include final approach of "
                                 "early successful episodes.",
    }
    for factor, rows in curves.items():
        plot_comparison(factor, rows, FIGURES[factor])
    write_condition(REPORT_PATH, result)
    print(json.dumps({"report": str(REPORT_PATH),
                      "waypoint_conditions": result["waypoint_conditions"],
                      "waypoint_episodes": result["waypoint_episodes"],
                      "hold_conditions": result["hold_conditions"]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    summarize_parts()
