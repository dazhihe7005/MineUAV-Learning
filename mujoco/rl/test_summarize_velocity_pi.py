"""PI/P sensitivity report must preserve paired data and hold interpretation."""

import copy
import importlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_dynamics import Scenario  # noqa: E402
from train_ppo_lowstd_multiseed import build_holdout_waypoints  # noqa: E402


def summary_module():
    return importlib.import_module("summarize_velocity_pi")


class VelocityPISummaryTests(unittest.TestCase):
    def test_grid_has_one_nominal_and_six_levels_per_factor(self):
        """Dropping a ±1/±2/±5 level must fail the audit's grid test."""
        summary = summary_module()
        scenarios = summary.expected_waypoint_scenarios()
        self.assertEqual(len(scenarios), 19)
        self.assertEqual(sum(item.kind == "nominal" for item in scenarios), 1)
        for factor in ("mass", "thrust", "force_x_n"):
            self.assertEqual(sum(item.kind == factor for item in scenarios), 6)

    def test_real_nominal_point_extracts_p_and_pi_five_seed_means(self):
        """A comparison must not silently use only the best PPO seed."""
        summary = summary_module()
        old = json.loads(summary.OLD_P_REPORT.read_text())
        part = json.loads(summary.waypoint_output_path(
            Scenario("nominal", 0.0)).read_text())
        result = summary.compare_point(old["nominal"], part,
                                       Scenario("nominal", 0.0),
                                       build_holdout_waypoints())
        self.assertAlmostEqual(result["old_p"]["scripted"]["success_rate"], 1.0)
        self.assertAlmostEqual(result["old_p"]["ppo_aggregate"]
                               ["success_rate"]["mean"], 0.956)
        self.assertAlmostEqual(result["new_pi"]["scripted"]["success_rate"], 1.0)
        self.assertAlmostEqual(result["new_pi"]["ppo_aggregate"]
                               ["success_rate"]["mean"], 0.702)
        self.assertEqual(result["new_pi"]["ppo_aggregate"]
                         ["success_rate"]["count"], 5)

    def test_pi_hold_rejects_missing_integral_or_impossible_window(self):
        """A parseable hold record with fake PI diagnostics cannot enter report."""
        summary = summary_module()
        scenario = Scenario("force_x_n", 5.0)
        original = json.loads(summary.hold_output_path(scenario).read_text())
        summary.validate_pi_hold(original, scenario)
        missing = copy.deepcopy(original)
        del missing["steps"][0]["integral_error_m"]
        with self.assertRaises(ValueError):
            summary.validate_pi_hold(missing, scenario)
        impossible = copy.deepcopy(original)
        impossible["tail_5_s"]["policy_steps"] = 124
        with self.assertRaises(ValueError):
            summary.validate_pi_hold(impossible, scenario)

    def test_pi_hold_rejects_corrupt_state_integrator_and_maxima(self):
        """Finite terminal state and every saved Ki contribution must be genuine."""
        summary = summary_module()
        scenario = Scenario("force_x_n", 5.0)
        original = json.loads(summary.hold_output_path(scenario).read_text())
        mutants = []
        bad_terminal = copy.deepcopy(original)
        bad_terminal["terminal_state"]["position_m"][0] = math.nan
        mutants.append(bad_terminal)
        bad_terminal_shape = copy.deepcopy(original)
        bad_terminal_shape["terminal_state"]["velocity_m_s"] = [0.0, 0.0]
        mutants.append(bad_terminal_shape)
        false_contribution = copy.deepcopy(original)
        false_contribution["steps"][0]["integral_acceleration_m_s2"][0] = 0.1
        mutants.append(false_contribution)
        false_maximum = copy.deepcopy(original)
        false_maximum["max_integral_accel_xy_m_s2"] = 999.0
        mutants.append(false_maximum)
        underreported_maximum = copy.deepcopy(original)
        underreported_maximum["max_integral_accel_xy_m_s2"] = 0.0
        mutants.append(underreported_maximum)
        for index, mutant in enumerate(mutants):
            with self.subTest(mutant=index):
                with self.assertRaises(ValueError):
                    summary.validate_pi_hold(mutant, scenario)

    def test_curve_renderer_keeps_old_figure_and_is_headless_png(self):
        summary = summary_module()
        def point(value, p_script, p_ppo, pi_script, pi_ppo):
            return {"value": value,
                    "old_p": {"scripted": {"success_rate": p_script},
                              "ppo_aggregate": {"success_rate": {"mean": p_ppo}}},
                    "new_pi": {"scripted": {"success_rate": pi_script},
                               "ppo_aggregate": {"success_rate": {"mean": pi_ppo}}}}
        rows = [point(0.95, 0, 0, 1, 0.8),
                point(1.0, 1, 0.956, 1, 0.702),
                point(1.05, 0, 0, 1, 0.75)]
        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / "comparison.png"
            summary.plot_comparison("mass", rows, target)
            image = target.read_bytes()
            self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
            with self.assertRaises(FileExistsError):
                summary.plot_comparison("mass", rows, target)
            self.assertEqual(target.read_bytes(), image)

    def test_report_findings_disclose_nominal_frozen_ppo_regression(self):
        """Disturbance gains must not hide nominal success/crossing regressions."""
        summary = summary_module()
        report = json.loads(summary.REPORT_PATH.read_text())
        findings = summary.derive_findings(report["nominal"], report["curves"])
        self.assertEqual(report["findings"], findings)
        self.assertAlmostEqual(findings["nominal_ppo_success_change_percentage_points"],
                               -25.4)
        self.assertAlmostEqual(findings["nominal_ppo_crossing_change_percentage_points"],
                               35.2)
        self.assertEqual(findings["nominal_scripted_success_old_new"], [1.0, 1.0])
        self.assertIn("not a drop-in", findings["verdict"])
        self.assertAlmostEqual(findings["plus_5_n_force_ppo_success_old_new"][1],
                               0.386)
        no_disturbance_gain = copy.deepcopy(report["curves"])
        for row in no_disturbance_gain["force_x_n"]:
            if row["value"] == 5.0:
                row["new_pi"]["ppo_aggregate"]["success_rate"]["mean"] = 0.0
        false_claim = summary.derive_findings(report["nominal"],
                                              no_disturbance_gain)
        self.assertNotIn("improves static-disturbance rejection",
                         false_claim["verdict"])


if __name__ == "__main__":
    unittest.main()
