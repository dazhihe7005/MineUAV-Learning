"""Validate paired attribution parts and produce three headless comparison curves."""

import hashlib
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from attribution_audit import (FACTOR_LEVELS, HOLD_SCENARIOS, PARTS_DIR,
                               condition_path, hold_path, summarize_records,
                               validate_nominal_condition)
from audit_ppo_robustness import EXPECTED_TRAIN_SEEDS, MUJOCO_DIR, waypoint_sha256
from robustness_dynamics import Scenario
from train_ppo_lowstd_multiseed import aggregate_numeric, build_holdout_waypoints


REPORT_PATH = MUJOCO_DIR / "reports" / "ppo_baseline_v1_attribution.json"
FIGURE_NAMES = {
    "mass": "mass_attribution.png",
    "thrust": "thrust_attribution.png",
    "force_x_n": "constant_force_attribution.png",
}


def expected_waypoint_scenarios() -> list[Scenario]:
    return [Scenario("nominal", 0.0)] + [
        Scenario(factor, value) for factor, values in FACTOR_LEVELS.items()
        for value in values
    ]


def _check_finite_tree(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _check_finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite_tree(item)
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise ValueError("Part contains a nonfinite number")


def _matches_derived(observed, expected) -> bool:
    if isinstance(expected, dict):
        return (isinstance(observed, dict) and observed.keys() == expected.keys()
                and all(_matches_derived(observed[key], item)
                        for key, item in expected.items()))
    if isinstance(expected, list):
        return (isinstance(observed, list) and len(observed) == len(expected)
                and all(_matches_derived(left, right)
                        for left, right in zip(observed, expected)))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9))
    return observed == expected


def validate_waypoint_part(part: dict, scenario: Scenario,
                           waypoints: list[tuple[int, list[float]]]) -> None:
    if part["scenario"] != {"kind": scenario.kind, "value": scenario.value}:
        raise ValueError("Scenario metadata differs")
    if (part["waypoint_sha256"] != waypoint_sha256(waypoints)
            or part["waypoint_seeds"] != [seed for seed, _ in waypoints]
            or part["policy_seeds"] != list(EXPECTED_TRAIN_SEEDS)
            or part["deterministic"] is not True):
        raise ValueError("Fixed targets/policy list/determinism differ")
    evaluations = part["evaluations"]
    if (len(evaluations) != 6
            or [(row["policy"], row["training_seed"]) for row in evaluations]
            != [("scripted", None)] + [("ppo", seed) for seed in EXPECTED_TRAIN_SEEDS]):
        raise ValueError("Need scripted and exactly five frozen PPO policies")
    for row in evaluations:
        episodes = row["episodes"]
        metrics = row["metrics"]
        if (len(episodes) != len(waypoints)
                or [(ep["seed"], ep["target_position_m"]) for ep in episodes]
                   != waypoints
                or metrics["episodes"] != len(waypoints)
                or metrics["successes"] != sum(ep["success"] for ep in episodes)
                or not math.isclose(metrics["success_rate"],
                                    metrics["successes"] / len(waypoints), abs_tol=1e-12)):
            raise ValueError("Episode target pairing or outcome accounting differs")
        if not all(ep["success"] == (ep["termination_reason"] == "success")
                   for ep in episodes):
            raise ValueError("Episode success disagrees with terminal reason")
        _check_finite_tree(row)
        if not _matches_derived(metrics, summarize_records(episodes)):
            raise ValueError("Reported metrics differ from paired episode records")


def describe_hold_observation(row: dict) -> dict:
    actual = row["terminal_state"]["time_s"] - row["injection_time_s"]
    requested = row["requested_observation_s"]
    if actual <= 0 or actual > requested + 1e-6:
        raise ValueError("Invalid hold observation duration")
    early = actual < requested - 1e-6
    if early and row["terminal_reason"] == "probe_complete":
        raise ValueError("Probe cannot complete early")
    return {
        "actual_observation_s": actual,
        "requested_observation_s": requested,
        "ended_early": early,
        "status": "safety_terminated_during_drift" if early else "full_observation",
    }


def validate_hold_result(row: dict, scenario: Scenario) -> None:
    """Check the saved probe's timing, both tail windows and zero command."""
    try:
        _check_finite_tree(row)
        if row["scenario"] != {"kind": scenario.kind, "value": scenario.value}:
            raise ValueError("Hold scenario metadata differs")
        if (not math.isclose(row["warmup_s"], 2.0, abs_tol=1e-8)
                or not math.isclose(row["requested_observation_s"], 15.0,
                                    abs_tol=1e-8)
                or not math.isclose(row["injection_time_s"], 2.0, abs_tol=1e-8)
                or not math.isclose(row["state_at_injection"]["time_s"],
                                    row["injection_time_s"], abs_tol=1e-8)
                or not math.isclose(row["controller_mass_kg"], 7.0, abs_tol=1e-9)):
            raise ValueError("Hold setup/timing/controller differs")
        expected_mass = 7.0 * scenario.value if scenario.kind == "mass" else 7.0
        if not math.isclose(row["plant_mass_after_injection_kg"], expected_mass,
                            abs_tol=1e-8):
            raise ValueError("Hold plant mass differs")
        observation = describe_hold_observation(row)
        policy_steps = row["policy_steps_after_injection"]
        if (not isinstance(policy_steps, int) or policy_steps < 1
                or observation["actual_observation_s"] <= (policy_steps - 1) * 0.04 - 1e-7
                or observation["actual_observation_s"] > policy_steps * 0.04 + 1e-7):
            raise ValueError("Hold policy-step count disagrees with elapsed time")
        if not np.allclose(row["final_position_error_xyz_m"],
                           np.array([0.0, 0.0, 1.0])
                           - row["terminal_state"]["position_m"], atol=1e-8):
            raise ValueError("Hold final position error differs from state")
        for seconds, key in ((5.0, "tail_5_s"), (1.0, "tail_1_s")):
            tail = row[key]
            count = min(policy_steps, round(seconds / 0.04))
            if (not math.isclose(tail["requested_window_s"], seconds, abs_tol=1e-8)
                    or tail["policy_steps"] != count
                    or tail["observed_span_s"] <= (count - 1) * 0.04 - 1e-7
                    or tail["observed_span_s"] > count * 0.04 + 1e-7):
                raise ValueError(f"{key} span/sample count differs")
            for field in ("mean_position_error_xyz_m", "mean_velocity_xyz_m_s",
                          "mean_command_velocity_xyz_m_s"):
                if len(tail[field]) != 3:
                    raise ValueError(f"{key} vector shape differs")
            if not np.allclose(tail["mean_command_velocity_xyz_m_s"], [0, 0, 0],
                               atol=1e-12):
                raise ValueError("Hold command was not zero")
            for field in ("mean_position_error_norm_m", "mean_speed_m_s",
                          "mean_command_speed_m_s"):
                if tail[field] < 0:
                    raise ValueError(f"{key} contains negative norm")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Incomplete hold result") from exc


def derive_attribution_interpretation(curves: dict, holds: dict) -> dict:
    """Classify only the measured six ±5%/±5 N stress levels."""
    key_levels = {"mass": (0.95, 1.05), "thrust": (0.95, 1.05),
                  "force_x_n": (-5.0, 5.0)}
    shared = 0
    hold_failed = 0
    tracking_gap = 0
    near_zero_actual = 0
    waypoint_tail_evidence = []
    for factor, values in key_levels.items():
        for value in values:
            point = next(row for row in curves[factor]
                         if math.isclose(row["value"], value, abs_tol=1e-12))
            shared += (point["scripted"]["success_rate"] < 0.1
                       and point["ppo_aggregate"]["success_rate"]["mean"] < 0.1)
            hold = holds[f"{factor}:{value:g}"]
            hold_failed += (hold["tail_5_s"]["mean_position_error_norm_m"] > 0.5
                            or hold["terminal_reason"] != "probe_complete")
            axis = 0 if factor == "force_x_n" else 2
            script_tail = point["scripted"]["tail_5_s"]
            ppo_tail = point["ppo_aggregate"]["tail_5_s"]
            script_axis = {
                "position_error_m": script_tail["mean_position_error_xyz_m"][axis],
                "actual_velocity_m_s": script_tail["mean_velocity_xyz_m_s"][axis],
                "command_velocity_m_s": script_tail[
                    "mean_command_velocity_xyz_m_s"][axis],
            }
            ppo_axis = {
                "position_error_m": ppo_tail[
                    "mean_position_error_xyz_m"][axis]["mean"],
                "actual_velocity_m_s": ppo_tail[
                    "mean_velocity_xyz_m_s"][axis]["mean"],
                "command_velocity_m_s": ppo_tail[
                    "mean_command_velocity_xyz_m_s"][axis]["mean"],
            }
            tracking_gap += all(abs(item["command_velocity_m_s"]
                                    - item["actual_velocity_m_s"]) > 0.15
                                for item in (script_axis, ppo_axis))
            near_zero_actual += all(abs(item["actual_velocity_m_s"]) < 0.05
                                    for item in (script_axis, ppo_axis))
            waypoint_tail_evidence.append({
                "factor": factor, "value": value,
                "axis": "x" if axis == 0 else "z",
                "scripted": script_axis, "ppo_five_seed_mean": ppo_axis,
            })
    ppo_worse = sum(
        row["scripted"]["success_rate"]
        - row["ppo_aggregate"]["success_rate"]["mean"] > 0.05
        for factor, points in curves.items() for row in points
        if not math.isclose(row["value"],
                            0.0 if factor == "force_x_n" else 1.0,
                            abs_tol=1e-12))
    case = ("B" if shared == 6 and hold_failed == 6 and tracking_gap == 6
            and near_zero_actual == 6 and ppo_worse == 0
            else "C" if shared >= 4 and hold_failed >= 4 and ppo_worse > 0
            else "A" if shared <= 1 and ppo_worse >= 4
            else "undetermined")
    return {
        "case": case,
        "shared_collapse_levels": shared,
        "hold_failed_levels": hold_failed,
        "tracking_gap_levels": tracking_gap,
        "near_zero_actual_velocity_levels": near_zero_actual,
        "waypoint_tail_evidence": waypoint_tail_evidence,
        "ppo_mean_below_scripted_nonnominal_levels": ppo_worse,
        "policy_seed_effects_remain": True,
        "interpretation": (
            "Both high-level policies issue persistent nonzero compensating velocity "
            "commands at all six stress levels, while actual axial velocity is near "
            "zero and a large position error remains. Together with zero-command "
            "hold drift and shared task failures, this supports inadequate low-level "
            "static-disturbance rejection as the primary bottleneck; policy-dependent "
            "success margins still exist."
            if case == "B" else
            "See measured curves and hold probes; classification is not decisive."
        ),
        "scope": "Only the tested waypoint distribution, five frozen PPO seeds, "
                 "scripted policy, and six ±5%/±5 N levels are classified.",
    }


def _aggregate_ppo_metrics(metrics: list[dict]) -> dict:
    if len(metrics) != 5:
        raise ValueError("Need five policy-seed metrics")
    scalar_fields = ("success_rate", "mean_final_distance_m",
                     "mean_successful_completion_time_s", "mean_episode_length",
                     "crossing_rate", "entered_0_2_m_rate")
    result = {key: aggregate_numeric([row[key] for row in metrics])
              for key in scalar_fields if all(key in row for row in metrics)}
    for window in ("tail_5_s", "tail_1_s", "timeout_tail_5_s", "timeout_tail_1_s"):
        if not all(window in row for row in metrics):
            continue
        result[window] = {}
        for field in ("mean_position_error_norm_m", "mean_speed_m_s",
                      "mean_command_speed_m_s", "mean_observed_span_s"):
            result[window][field] = aggregate_numeric(
                [row[window][field] for row in metrics])
        for field in ("mean_position_error_xyz_m", "mean_velocity_xyz_m_s",
                      "mean_command_velocity_xyz_m_s"):
            result[window][field] = [aggregate_numeric(
                [row[window][field][axis] if row[window][field] is not None else None
                 for row in metrics]) for axis in range(3)]
    return result


def point_from_part(part: dict, value: float) -> dict:
    evaluations = part["evaluations"]
    if len(evaluations) != 6:
        raise ValueError("Need scripted plus five PPO evaluations")
    script = evaluations[0]["metrics"]
    ppo = evaluations[1:]
    return {
        "value": value,
        "scripted": script,
        "ppo_aggregate": _aggregate_ppo_metrics([row["metrics"] for row in ppo]),
        "ppo_per_seed": [{"training_seed": row["training_seed"],
                          "metrics": row["metrics"]} for row in ppo],
    }


def plot_success_comparison(factor: str, points: list[dict], target: Path) -> None:
    if factor not in FIGURE_NAMES or not points:
        raise ValueError("Need known factor and nonempty comparison")
    if target.exists():
        raise FileExistsError(f"Preserving figure: {target}")
    ordered = sorted(points, key=lambda row: row["value"])
    if factor in {"mass", "thrust"}:
        x = [(row["value"] - 1) * 100 for row in ordered]
        xlabel = "Plant parameter change from nominal (%)"
    else:
        x = [row["value"] for row in ordered]
        xlabel = "Constant world-X force (N)"
    scripted = [row["scripted"]["success_rate"] for row in ordered]
    means = [row["ppo_aggregate"]["success_rate"]["mean"] for row in ordered]
    stds = [row["ppo_aggregate"]["success_rate"]["std"] or 0
            for row in ordered]
    lows = [row["ppo_aggregate"]["success_rate"]["min"] for row in ordered]
    highs = [row["ppo_aggregate"]["success_rate"]["max"] for row in ordered]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.fill_between(x, lows, highs, color="tab:blue", alpha=0.15,
                      label="PPO min–max across five seeds")
    axis.errorbar(x, means, yerr=stds, color="tab:blue", marker="o",
                  capsize=3, linewidth=2, label="PPO mean ± sample SD")
    axis.plot(x, scripted, color="tab:orange", marker="s", linewidth=2,
              label="Scripted")
    axis.axhline(0.9, color="tab:green", linestyle="--", linewidth=1,
                 label="90%")
    axis.set(xlabel=xlabel, ylabel="Holdout waypoint success rate",
             ylim=(-0.02, 1.02), title=f"{factor.replace('_', ' ').title()} attribution")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    fig.tight_layout()
    payload = io.BytesIO()
    fig.savefig(payload, dpi=150, format="png")
    plt.close(fig)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        output.write(payload.getvalue())


def summarize_parts() -> dict:
    if REPORT_PATH.exists():
        raise FileExistsError(f"Preserving existing report: {REPORT_PATH}")
    figure_paths = {factor: MUJOCO_DIR / "reports" / name
                    for factor, name in FIGURE_NAMES.items()}
    if any(path.exists() for path in figure_paths.values()):
        raise FileExistsError("Preserving existing attribution figure")
    scenarios = expected_waypoint_scenarios()
    expected_paths = {condition_path(scenario) for scenario in scenarios} | {
        hold_path(scenario) for scenario in HOLD_SCENARIOS}
    found_paths = set(PARTS_DIR.rglob("*.json")) if PARTS_DIR.exists() else set()
    if expected_paths != found_paths:
        raise RuntimeError(f"Parts mismatch; missing={expected_paths-found_paths}, "
                           f"unexpected={found_paths-expected_paths}")
    waypoints = build_holdout_waypoints()
    parts = {}
    for scenario in scenarios:
        part = json.loads(condition_path(scenario).read_text(encoding="utf-8"))
        validate_waypoint_part(part, scenario, waypoints)
        parts[(scenario.kind, scenario.value)] = part
    validate_nominal_condition(parts[("nominal", 0.0)])

    holds = {}
    hold_observation = {}
    for scenario in HOLD_SCENARIOS:
        row = json.loads(hold_path(scenario).read_text(encoding="utf-8"))
        validate_hold_result(row, scenario)
        key = f"{scenario.kind}:{scenario.value:g}"
        holds[key] = row
        hold_observation[key] = describe_hold_observation(row)

    nominal = point_from_part(parts[("nominal", 0.0)], 0.0)
    curves = {}
    for factor, values in FACTOR_LEVELS.items():
        zero_value = 0.0 if factor == "force_x_n" else 1.0
        points = [dict(nominal, value=zero_value)] + [
            point_from_part(parts[(factor, value)], value) for value in values]
        curves[factor] = sorted(points, key=lambda row: row["value"])

    model_path = MUJOCO_DIR / "models" / "mine_uav_dynamics_v2.xml"
    result = {
        "audit": "MineUAV RL Baseline v1 robustness attribution; inference only",
        "nominal_model_path": str(model_path),
        "nominal_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "waypoint_sha256": waypoint_sha256(waypoints),
        "waypoint_seeds": [seed for seed, _ in waypoints],
        "training_seeds": list(EXPECTED_TRAIN_SEEDS),
        "deterministic_ppo": True,
        "waypoint_episode_limit_s": 15.0,
        "hold_warmup_s": 2.0, "hold_post_injection_observation_s": 15.0,
        "tail_definition": "Equal weight per episode: pre-command world-frame states "
                           "in the last available 5 s or 1 s. Position error = "
                           "target minus position. Actual span and sample count "
                           "are stored per episode; short successes are not padded.",
        "nominal": nominal,
        "curves": curves,
        "low_level_zero_command_hold": holds,
        "hold_observation": hold_observation,
        "attribution_interpretation": derive_attribution_interpretation(curves, holds),
        "waypoint_conditions": len(parts),
        "waypoint_episodes": sum(len(item["episodes"]) for part in parts.values()
                                 for item in part["evaluations"]),
        "hold_conditions": len(holds),
        "figures": {factor: str(path) for factor, path in figure_paths.items()},
        "metric_caveat": "Some hold probes end on safety limits before 15 s; their "
                         "tail windows describe late-stage drift, not equilibrium. "
                         "A zero crossing rate can mean the aircraft never entered "
                         "the near-target region. Tail means from successful episodes "
                         "may include their final approach, so timeout-only tail "
                         "metrics are provided separately. Ground contact can make "
                         "vertical speed small despite a large position error.",
    }
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    for factor, curve in curves.items():
        plot_success_comparison(factor, curve, figure_paths[factor])
    with REPORT_PATH.open("x", encoding="utf-8") as output:
        output.write(payload)
    print(json.dumps({"report": str(REPORT_PATH),
                      "waypoint_conditions": result["waypoint_conditions"],
                      "waypoint_episodes": result["waypoint_episodes"],
                      "hold_conditions": result["hold_conditions"]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    summarize_parts()
