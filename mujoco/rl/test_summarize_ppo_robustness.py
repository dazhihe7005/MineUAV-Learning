"""Five-seed sensitivity curves must not conceal missing or weak policies."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ppo_lowstd_multiseed import build_holdout_waypoints  # noqa: E402


def summary_module():
    try:
        return importlib.import_module("summarize_ppo_robustness")
    except ImportError as error:
        raise AssertionError("Robustness summary implementation is missing") from error


class RobustnessSummaryTests(unittest.TestCase):
    def test_mean_sample_sd_min_max_and_missing_completion(self):
        """Catches std=0 for one observed completion or zero-imputed failures."""
        summary = summary_module()
        rows = [{"success_rate": value,
                 "mean_successful_completion_time_s": time}
                for value, time in ((1.0, 6.0), (0.9, 8.0), (0.8, None),
                                    (0.7, None), (0.6, None))]
        result = summary.aggregate_policy_metrics(rows)
        self.assertAlmostEqual(result["success_rate"]["mean"], 0.8)
        self.assertAlmostEqual(result["success_rate"]["std"], 0.15811388300841897)
        self.assertEqual((result["success_rate"]["min"],
                          result["success_rate"]["max"]), (0.6, 1.0))
        self.assertEqual(result["mean_successful_completion_time_s"]["count"], 2)
        self.assertEqual(result["mean_successful_completion_time_s"]["missing_count"], 3)
        self.assertAlmostEqual(result["mean_successful_completion_time_s"]["mean"], 7.0)

    def test_success_bands_use_exact_90_and_70_percent_boundaries(self):
        """Catches classifying exactly 90% or 70% in the wrong band."""
        summary = summary_module()
        self.assertEqual(summary.success_band(0.90), "ge_90")
        self.assertEqual(summary.success_band(0.70), "70_to_90")
        self.assertEqual(summary.success_band(0.699), "lt_70")

    def test_success_band_uses_integer_episode_counts_after_aggregation(self):
        """Means of five exact rates can round below a 70/90% boundary."""
        summary = summary_module()
        for successes, expected in (
            ([60, 70, 72, 72, 76], "70_to_90"),
            ([60, 97, 99, 97, 97], "ge_90"),
        ):
            rows = [{"episodes": 100, "successes": count,
                     "success_rate": count / 100} for count in successes]
            aggregated = summary.aggregate_policy_metrics(rows)
            self.assertLess(aggregated["success_rate"]["mean"],
                            sum(successes) / 500 + 1e-12)
            self.assertEqual(summary.success_band_for_policy_rows(rows), expected)

    def test_factor_ranking_uses_measured_worst_drop_from_nominal(self):
        """Catches choosing a presumed factor instead of observed sensitivity."""
        summary = summary_module()
        curves = {
            "mass": [{"value": 1.0, "aggregate": {"success_rate": {"mean": 0.956}}},
                     {"value": 0.8, "aggregate": {"success_rate": {"mean": 0.80}}}],
            "thrust": [{"value": 1.0, "aggregate": {"success_rate": {"mean": 0.956}}},
                       {"value": 0.8, "aggregate": {"success_rate": {"mean": 0.30}}}],
        }
        ranking = summary.rank_factor_sensitivity(curves, nominal_rate=0.956)
        self.assertEqual(ranking[0]["factor"], "thrust")
        self.assertAlmostEqual(ranking[0]["largest_success_drop"], 0.656)

    def test_expected_grid_has_one_shared_nominal_and_31_perturbations(self):
        """Catches omitting a signed force or repeating nominal independently."""
        summary = summary_module()
        grid = summary.expected_scenarios()
        self.assertEqual(len(grid), 32)
        self.assertEqual(sum(s.kind == "nominal" for s in grid), 1)
        self.assertEqual(sum(s.kind == "force_x_n" for s in grid), 8)
        self.assertEqual(sum(s.kind == "motor_lag_s" for s in grid), 5)
        self.assertEqual(sum(s.kind in {"mass", "inertia", "thrust"}
                             for s in grid), 18)

    def test_validate_part_rejects_wrong_target_even_with_correct_seed_count(self):
        """Catches silent target mismatch across parameter conditions."""
        summary = summary_module()
        from robustness_dynamics import Scenario
        from audit_ppo_robustness import waypoint_sha256
        waypoints = build_holdout_waypoints()
        episodes = [{"seed": seed, "target_position_m": target,
                     "success": True, "termination_reason": "success",
                     "episode_length": 10, "final_distance_m": 0.05,
                     "final_speed_m_s": 0.05} for seed, target in waypoints]
        metrics = {"episodes": 100, "successes": 100, "success_rate": 1.0,
                   "mean_final_distance_m": 0.05,
                   "mean_successful_completion_time_s": 0.4,
                   "mean_episode_length": 10.0, "crossing_rate": 0.0,
                   "near_target_speed_m_s": 0.05,
                   "near_target_policy_steps": 1000,
                   "action_saturation": 0.0, "failure_count": 0,
                   "time_limit_count": 0, "nonfinite_state_count": 0,
                   "mujoco_warning_episode_count": 0,
                   "mujoco_warning_total": 0,
                   "termination_counts": {"success": 100}}
        evaluations = [{"training_seed": seed,
                        "metrics": metrics,
                        "episodes": episodes} for seed in (20260929, 0, 8, 16, 24)]
        part = {"scenario": {"kind": "mass", "value": 0.8},
                "waypoint_sha256": waypoint_sha256(waypoints),
                "waypoint_seeds": [seed for seed, _ in waypoints],
                "policy_seeds": [20260929, 0, 8, 16, 24],
                "evaluations": evaluations}
        summary.validate_part(part, Scenario("mass", 0.8), waypoints)
        part["evaluations"][2]["episodes"] = [dict(ep) for ep in episodes]
        part["evaluations"][2]["episodes"][0]["target_position_m"] = [0, 0, 1]
        with self.assertRaises(ValueError):
            summary.validate_part(part, Scenario("mass", 0.8), waypoints)

    def test_success_curve_writes_one_headless_png(self):
        """Catches plot omission or a GUI-dependent plotting backend."""
        summary = summary_module()
        curve = [{"value": 0.8, "aggregate": {"success_rate": {
                     "mean": 0.6, "std": 0.1, "min": 0.4, "max": 0.8}}},
                 {"value": 1.0, "aggregate": {"success_rate": {
                     "mean": 0.95, "std": 0.03, "min": 0.90, "max": 1.0}}}]
        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / "mass_sensitivity.png"
            summary.plot_success_curve("mass", curve, target)
            self.assertTrue(target.is_file())
            self.assertEqual(target.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                summary.plot_success_curve("mass", curve, target)
            self.assertEqual(target.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
