"""Validate all frozen-policy sensitivity parts, aggregate and plot results."""

import hashlib
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit_ppo_robustness import (
    EXPECTED_TRAIN_SEEDS, FACTOR_LEVELS, FIVE_SEED_REPORT, MUJOCO_DIR,
    PARTS_DIR, scenario_output_path, waypoint_sha256,
)
from robustness_dynamics import Scenario
from train_ppo_lowstd_multiseed import aggregate_numeric, build_holdout_waypoints


REPORT_PATH = MUJOCO_DIR / "reports" / "ppo_baseline_v1_robustness.json"
FIGURE_NAMES = {
    "mass": "mass_sensitivity.png",
    "inertia": "inertia_sensitivity.png",
    "thrust": "thrust_coefficient_sensitivity.png",
    "motor_lag_s": "motor_lag_sensitivity.png",
    "force_x_n": "external_force_sensitivity.png",
}
METRIC_KEYS = (
    "success_rate", "mean_final_distance_m", "mean_successful_completion_time_s",
    "mean_episode_length", "crossing_rate", "near_target_speed_m_s",
    "action_saturation", "failure_count", "time_limit_count",
    "nonfinite_state_count", "mujoco_warning_episode_count", "mujoco_warning_total",
)


def expected_scenarios() -> list[Scenario]:
    return [Scenario("nominal", 0.0)] + [
        Scenario(factor, value)
        for factor, values in FACTOR_LEVELS.items()
        for value in values
    ]


def success_band(success_rate: float) -> str:
    if not math.isfinite(success_rate) or not 0 <= success_rate <= 1:
        raise ValueError("Success rate must be finite and between zero and one")
    if success_rate >= 0.90:
        return "ge_90"
    if success_rate >= 0.70:
        return "70_to_90"
    return "lt_70"


def success_band_for_policy_rows(rows: list[dict]) -> str:
    """Classify pooled exact episode outcomes, avoiding mean-rate rounding."""
    successes = sum(int(row["successes"]) for row in rows)
    episodes = sum(int(row["episodes"]) for row in rows)
    if episodes <= 0 or not 0 <= successes <= episodes:
        raise ValueError("Invalid pooled success/episode counts")
    if 10 * successes >= 9 * episodes:
        return "ge_90"
    if 10 * successes >= 7 * episodes:
        return "70_to_90"
    return "lt_70"


def aggregate_policy_metrics(rows: list[dict]) -> dict:
    if len(rows) != 5:
        raise ValueError("Each sensitivity point needs exactly five policies")
    common = set.intersection(*(set(row) for row in rows))
    keys = [key for key in METRIC_KEYS if key in common]
    if not keys:
        raise ValueError("No scalar robustness metrics to aggregate")
    return {key: aggregate_numeric([row[key] for row in rows]) for key in keys}


def validate_part(part: dict, scenario: Scenario,
                  waypoints: list[tuple[int, list[float]]]) -> None:
    if part["scenario"] != {"kind": scenario.kind, "value": scenario.value}:
        raise ValueError("Scenario metadata differs from requested factor/value")
    if part["waypoint_sha256"] != waypoint_sha256(waypoints):
        raise ValueError("Waypoint target hash differs between conditions")
    if (part["waypoint_seeds"] != [seed for seed, _ in waypoints]
            or part["policy_seeds"] != list(EXPECTED_TRAIN_SEEDS)):
        raise ValueError("Waypoint or policy seed order differs")
    evaluations = part["evaluations"]
    if ([item["training_seed"] for item in evaluations]
            != list(EXPECTED_TRAIN_SEEDS)):
        raise ValueError("Missing or reordered policy evaluation")
    for item in evaluations:
        metrics = item["metrics"]
        episodes = item["episodes"]
        if (len(episodes) != len(waypoints) or metrics["episodes"] != len(waypoints)
                or metrics["successes"] != sum(ep["success"] for ep in episodes)
                or metrics["failure_count"] + metrics["time_limit_count"]
                    + metrics["successes"] != len(waypoints)):
            raise ValueError("Episode count or terminal accounting differs")
        if [(ep["seed"], ep["target_position_m"]) for ep in episodes] != waypoints:
            raise ValueError("Policy episode targets differ from fixed holdout")
        if not all((ep["success"] == (ep["termination_reason"] == "success"))
                   for ep in episodes):
            raise ValueError("Episode success disagrees with terminal reason")
        for key in METRIC_KEYS:
            value = metrics[key]
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"Non-finite {key}")


def rank_factor_sensitivity(curves: dict[str, list[dict]],
                            nominal_rate: float) -> list[dict]:
    ranking = []
    for factor, points in curves.items():
        worst = min(points, key=lambda point: point["aggregate"]["success_rate"]["mean"])
        minimum = worst["aggregate"]["success_rate"]["mean"]
        ranking.append({
            "factor": factor,
            "worst_tested_value": worst["value"],
            "worst_mean_success_rate": minimum,
            "largest_success_drop": max(0.0, nominal_rate - minimum),
        })
    return sorted(ranking, key=lambda row: (-row["largest_success_drop"], row["factor"]))


def plot_success_curve(factor: str, curve: list[dict], target: Path) -> None:
    if factor not in FIGURE_NAMES or not curve:
        raise ValueError("Need a known factor and at least one curve point")
    if target.exists():
        raise FileExistsError(f"Preserving existing sensitivity figure: {target}")
    ordered = sorted(curve, key=lambda point: point["value"])
    values = [point["value"] for point in ordered]
    if factor in {"mass", "inertia", "thrust"}:
        x = [(value - 1.0) * 100 for value in values]
        xlabel = "Change from nominal (%)"
    elif factor == "motor_lag_s":
        x = [value * 1000 for value in values]
        xlabel = "Motor time constant (ms)"
    else:
        x = values
        xlabel = "Constant world-X force (N)"
    means = [point["aggregate"]["success_rate"]["mean"] for point in ordered]
    stds = [point["aggregate"]["success_rate"]["std"] or 0.0
            for point in ordered]
    minimums = [point["aggregate"]["success_rate"]["min"] for point in ordered]
    maximums = [point["aggregate"]["success_rate"]["max"] for point in ordered]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.fill_between(x, minimums, maximums, color="tab:blue", alpha=0.12,
                      label="min–max across 5 policies")
    axis.errorbar(x, means, yerr=stds, color="tab:blue", marker="o", linewidth=2,
                  capsize=3, label="mean ± sample SD")
    axis.axhline(0.90, color="tab:green", linestyle="--", linewidth=1,
                 label="90%")
    axis.axhline(0.70, color="tab:orange", linestyle="--", linewidth=1,
                 label="70%")
    axis.set(xlabel=xlabel, ylabel="Waypoint success rate", ylim=(-0.02, 1.02),
             title=f"{factor.replace('_', ' ').title()} sensitivity")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    png = io.BytesIO()
    figure.savefig(png, dpi=150, format="png")
    plt.close(figure)
    with target.open("xb") as output:
        output.write(png.getvalue())


def summarize_parts() -> dict:
    if REPORT_PATH.exists():
        raise FileExistsError(f"Preserving existing audit: {REPORT_PATH}")
    expected_figures = [MUJOCO_DIR / "reports" / name for name in FIGURE_NAMES.values()]
    existing_figures = [str(path) for path in expected_figures if path.exists()]
    if existing_figures:
        raise FileExistsError(f"Preserving existing figures: {existing_figures}")

    waypoints = build_holdout_waypoints()
    scenarios = expected_scenarios()
    expected_paths = {scenario_output_path(scenario) for scenario in scenarios}
    found_paths = set(PARTS_DIR.rglob("*.json")) if PARTS_DIR.exists() else set()
    if found_paths != expected_paths:
        raise RuntimeError(f"Robustness parts mismatch: missing={sorted(map(str, expected_paths-found_paths))}, "
                           f"unexpected={sorted(map(str, found_paths-expected_paths))}")
    parts = {}
    for scenario in scenarios:
        part = json.loads(scenario_output_path(scenario).read_text(encoding="utf-8"))
        validate_part(part, scenario, waypoints)
        parts[(scenario.kind, scenario.value)] = part

    def curve_point(scenario: Scenario, factor: str, value: float) -> dict:
        part = parts[(scenario.kind, scenario.value)]
        rows = [evaluation["metrics"] for evaluation in part["evaluations"]]
        aggregate = aggregate_policy_metrics(rows)
        return {
            "value": value, "source_part": str(scenario_output_path(scenario)),
            "aggregate": aggregate,
            "success_band_by_mean": success_band_for_policy_rows(rows),
            "per_policy": [{"training_seed": item["training_seed"],
                            "metrics": item["metrics"]}
                           for item in part["evaluations"]],
        }

    nominal = curve_point(Scenario("nominal", 0.0), "nominal", 0.0)
    curves = {}
    band_levels = {}
    for factor, values in FACTOR_LEVELS.items():
        nominal_value = 0.0 if factor in {"motor_lag_s", "force_x_n"} else 1.0
        points = [dict(nominal, value=nominal_value)]
        points.extend(curve_point(Scenario(factor, value), factor, value)
                      for value in values)
        curves[factor] = sorted(points, key=lambda point: point["value"])
        band_levels[factor] = {band: [point["value"] for point in curves[factor]
                                        if point["success_band_by_mean"] == band]
                               for band in ("ge_90", "70_to_90", "lt_70")}

    nominal_rate = nominal["aggregate"]["success_rate"]["mean"]
    ranking = rank_factor_sensitivity(curves, nominal_rate)
    original = json.loads(FIVE_SEED_REPORT.read_text(encoding="utf-8"))
    model_xml = MUJOCO_DIR / "models" / "mine_uav_dynamics_v2.xml"
    result = {
        "audit": "MineUAV RL Baseline v1 dynamics robustness, inference only",
        "nominal_model_path": str(model_xml),
        "nominal_model_sha256_at_summary": hashlib.sha256(model_xml.read_bytes()).hexdigest(),
        "policy_seeds": list(EXPECTED_TRAIN_SEEDS),
        "policy_checkpoints": [row["checkpoint"] for row in original["rows"]],
        "waypoint_seeds": [seed for seed, _ in waypoints],
        "waypoint_targets_m": [target for _, target in waypoints],
        "waypoint_sha256": waypoint_sha256(waypoints),
        "deterministic": True,
        "physics_hz": 500, "controller_hz": 100, "policy_hz": 25,
        "group_std_definition": "sample standard deviation across five policies, ddof=1",
        "near_target_definition": "pooled pre-command policy steps with distance <0.1 m",
        "crossing_definition": "signed target-plane crossing after first entry <0.2 m, "
                               "fraction of all 100 episodes",
        "action_saturation_definition": "fraction of executed normalized action components "
                                        "with absolute value >=0.95",
        "metric_caveat": "Near-target speed is null when no policy steps enter <0.1 m. "
                         "A zero crossing rate can also mean no approach to <0.2 m, "
                         "not good braking.",
        "success_bands": {
            "ge_90": "mean success >= 0.90",
            "70_to_90": "0.70 <= mean success < 0.90",
            "lt_70": "mean success < 0.70",
        },
        "nominal": nominal,
        "curves": curves,
        "tested_levels_by_band": band_levels,
        "sensitivity_ranking_by_largest_success_drop": ranking,
        "nonfinite_state_episodes_all_unique_conditions": sum(
            evaluation["metrics"]["nonfinite_state_count"]
            for part in parts.values() for evaluation in part["evaluations"]),
        "mujoco_warning_episodes_all_unique_conditions": sum(
            evaluation["metrics"]["mujoco_warning_episode_count"]
            for part in parts.values() for evaluation in part["evaluations"]),
        "unique_conditions": len(parts),
        "total_episodes": len(parts) * len(EXPECTED_TRAIN_SEEDS) * len(waypoints),
        "figures": {factor: str(MUJOCO_DIR / "reports" / filename)
                    for factor, filename in FIGURE_NAMES.items()},
        "note": "Ranges are tested grid points, not interpolated real-aircraft guarantees. "
                "Mass is measured; COM/inertia, thrust fit, rotor signs, motor lag values "
                "and external force levels have the provenance documented in the baseline.",
    }
    report_payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    for factor, curve in curves.items():
        plot_success_curve(factor, curve, Path(result["figures"][factor]))
    with REPORT_PATH.open("x", encoding="utf-8") as output:
        output.write(report_payload)
    print(json.dumps({
        "nominal_success_mean": nominal_rate,
        "factor_worst_success": {row["factor"]: row["worst_mean_success_rate"]
                                 for row in ranking},
        "nonfinite_state_episodes": result["nonfinite_state_episodes_all_unique_conditions"],
        "mujoco_warning_episodes": result["mujoco_warning_episodes_all_unique_conditions"],
        "unique_conditions": result["unique_conditions"],
        "total_episodes": result["total_episodes"],
        "report": str(REPORT_PATH),
    }, indent=2), flush=True)
    return result


if __name__ == "__main__":
    summarize_parts()
