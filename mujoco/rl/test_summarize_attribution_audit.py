"""Attribution report must not lose paired targets, seeds or tail diagnostics."""

import copy
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ppo_lowstd_multiseed import build_holdout_waypoints  # noqa: E402


def summary_module():
    return importlib.import_module("summarize_attribution_audit")


class AttributionSummaryTests(unittest.TestCase):
    def test_grid_has_one_shared_nominal_and_twenty_four_perturbations(self):
        """Catches omitting ±1%/±2% or evaluating nominal three times."""
        summary = summary_module()
        scenarios = summary.expected_waypoint_scenarios()
        self.assertEqual(len(scenarios), 25)
        self.assertEqual(sum(item.kind == "nominal" for item in scenarios), 1)
        self.assertEqual(sum(item.kind == "mass" for item in scenarios), 8)
        self.assertEqual(sum(item.kind == "thrust" for item in scenarios), 8)
        self.assertEqual(sum(item.kind == "force_x_n" for item in scenarios), 8)

    def test_real_nominal_part_rejects_one_ppo_target_swap(self):
        """Catches a false policy comparison caused by a single changed target."""
        summary = summary_module()
        path = (Path(__file__).resolve().parents[1] / "reports" /
                "ppo_baseline_v1_attribution_parts" / "nominal.json")
        part = json.loads(path.read_text())
        waypoints = build_holdout_waypoints()
        summary.validate_waypoint_part(part, summary.expected_waypoint_scenarios()[0],
                                       waypoints)
        changed = copy.deepcopy(part)
        changed["evaluations"][4]["episodes"][3]["target_position_m"] = [0, 0, 1]
        with self.assertRaises(ValueError):
            summary.validate_waypoint_part(
                changed, summary.expected_waypoint_scenarios()[0], waypoints)

    def test_real_nominal_part_rejects_finite_but_wrong_derived_metric(self):
        """A parseable part must not silently corrupt the final comparison."""
        summary = summary_module()
        path = (Path(__file__).resolve().parents[1] / "reports" /
                "ppo_baseline_v1_attribution_parts" / "nominal.json")
        part = json.loads(path.read_text())
        changed = copy.deepcopy(part)
        changed["evaluations"][1]["metrics"]["mean_final_distance_m"] += 0.1
        with self.assertRaises(ValueError):
            summary.validate_waypoint_part(
                changed, summary.expected_waypoint_scenarios()[0],
                build_holdout_waypoints())

    def test_hold_observation_labels_early_safety_stop(self):
        summary = summary_module()
        row = {"injection_time_s": 2.0,
               "requested_observation_s": 15.0,
               "terminal_state": {"time_s": 14.6},
               "terminal_reason": "outside_flight_area"}
        observed = summary.describe_hold_observation(row)
        self.assertAlmostEqual(observed["actual_observation_s"], 12.6)
        self.assertTrue(observed["ended_early"])
        self.assertEqual(observed["status"], "safety_terminated_during_drift")

    def test_real_hold_rejects_missing_tail_or_impossible_sample_count(self):
        summary = summary_module()
        path = (Path(__file__).resolve().parents[1] / "reports" /
                "ppo_baseline_v1_attribution_parts" / "hold" / "force_x_n_+5.000.json")
        row = json.loads(path.read_text())
        scenario = summary.HOLD_SCENARIOS[-1]
        summary.validate_hold_result(row, scenario)
        partial = json.loads((path.parent / "mass_+0.950.json").read_text())
        summary.validate_hold_result(partial, summary.HOLD_SCENARIOS[1])
        missing = copy.deepcopy(row)
        del missing["tail_1_s"]
        with self.assertRaises(ValueError):
            summary.validate_hold_result(missing, scenario)
        impossible = copy.deepcopy(row)
        impossible["tail_5_s"]["policy_steps"] = 124
        with self.assertRaises(ValueError):
            summary.validate_hold_result(impossible, scenario)

    def test_interpretation_uses_shared_failure_and_hold_data(self):
        summary = summary_module()
        def point(value, scripted, ppo):
            tail_script = {"mean_position_error_xyz_m": [1.0, 0, 1.0],
                           "mean_velocity_xyz_m_s": [0.0, 0, 0.0],
                           "mean_command_velocity_xyz_m_s": [0.48, 0, 0.24]}
            tail_ppo = {key: [{"mean": x} for x in vector]
                        for key, vector in tail_script.items()}
            return {"value": value,
                    "scripted": {"success_rate": scripted, "tail_5_s": tail_script},
                    "ppo_aggregate": {"success_rate": {"mean": ppo},
                                      "tail_5_s": tail_ppo}}
        curves = {
            "mass": [point(0.95, 0.0, 0.0), point(1.0, 1.0, 0.956),
                     point(1.05, 0.0, 0.0)],
            "thrust": [point(0.95, 0.0, 0.0), point(1.0, 1.0, 0.956),
                       point(1.05, 0.0, 0.0)],
            "force_x_n": [point(-5.0, 0.0, 0.0), point(0.0, 1.0, 0.956),
                          point(5.0, 0.0, 0.0)],
        }
        holds = {f"{factor}:{value:g}": {
            "tail_5_s": {"mean_position_error_norm_m": 1.0},
            "terminal_reason": "outside_flight_area"}
            for factor, values in (("mass", (0.95, 1.05)),
                                   ("thrust", (0.95, 1.05)),
                                   ("force_x_n", (-5.0, 5.0)))
            for value in values}
        result = summary.derive_attribution_interpretation(curves, holds)
        self.assertEqual(result["case"], "B")
        self.assertEqual(result["shared_collapse_levels"], 6)
        self.assertEqual(result["hold_failed_levels"], 6)
        self.assertEqual(result["ppo_mean_below_scripted_nonnominal_levels"], 0)
        self.assertEqual(result["tracking_gap_levels"], 6)
        self.assertEqual(len(result["waypoint_tail_evidence"]), 6)

    def test_five_policy_stats_include_sample_sd_and_extremes(self):
        """Catches reporting only a best checkpoint or hiding a weak seed."""
        summary = summary_module()
        part = {"evaluations": [{"policy": "scripted", "training_seed": None,
                                 "metrics": {"success_rate": 1.0}},
                                *[{"policy": "ppo", "training_seed": seed,
                                   "metrics": {"success_rate": rate}}
                                  for seed, rate in zip(
                                      (20260929, 0, 8, 16, 24),
                                      (1.0, 1.0, 0.9, 0.8, 0.7))]]}
        point = summary.point_from_part(part, 0.99)
        rate = point["ppo_aggregate"]["success_rate"]
        self.assertAlmostEqual(rate["mean"], 0.88)
        self.assertAlmostEqual(rate["std"], 0.130384048104053)
        self.assertEqual((rate["min"], rate["max"]), (0.7, 1.0))
        self.assertEqual(point["scripted"]["success_rate"], 1.0)

    def test_headless_curve_is_png_and_preserves_existing_file(self):
        """Catches missing/overwritten sensitivity figures or GUI dependency."""
        summary = summary_module()
        points = [
            {"value": 0.99, "scripted": {"success_rate": 0.9},
             "ppo_aggregate": {"success_rate": {"mean": 0.8, "std": 0.1,
                                                 "min": 0.6, "max": 1.0}}},
            {"value": 1.0, "scripted": {"success_rate": 1.0},
             "ppo_aggregate": {"success_rate": {"mean": 0.95, "std": 0.04,
                                                 "min": 0.86, "max": 1.0}}},
        ]
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "mass_attribution.png"
            summary.plot_success_comparison("mass", points, path)
            original = path.read_bytes()
            self.assertEqual(original[:8], b"\x89PNG\r\n\x1a\n")
            with self.assertRaises(FileExistsError):
                summary.plot_success_comparison("mass", points, path)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
