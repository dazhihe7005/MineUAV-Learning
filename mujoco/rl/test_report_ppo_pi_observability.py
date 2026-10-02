"""Nominal A/B/C/D comparison must use one paired holdout target set."""

import sys
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_ppo_pi_observability import (build_paired_comparison,
                                         validate_reference_replay,
                                         verify_report_artifacts)  # noqa: E402
from reward_v2_alignment_audit import fixed_waypoints  # noqa: E402
from train_ppo_lowstd_multiseed import build_holdout_waypoints  # noqa: E402


def compact(successes, crossing, speed):
    return {
        "episodes": 100, "successes": successes,
        "success_rate": successes / 100,
        "mean_final_distance_m": 0.1,
        "mean_successful_completion_time_s": 8.0,
        "mean_episode_length": 250.0,
        "near_target_actual_speed_m_s": speed,
        "near_target_command_speed_m_s": speed / 2,
        "crossing_rate": crossing,
        "action_saturation_fraction": 0.0,
    }


class ObservablePIComparisonTests(unittest.TestCase):
    def setUp(self):
        self.references = {
            "A_7d_ppo_p": {"waypoint_sha256": "paired", "training_seed": 0,
                           "evaluation": compact(100, 0.0, 0.1)},
            "B_7d_ppo_direct_pi": {"waypoint_sha256": "paired", "training_seed": 0,
                                   "evaluation": compact(78, 0.53, 0.2)},
        }
        self.c = {
            "seed": 0, "reward_version": "v2", "observation_shape": [7],
            "actual_timesteps": 100_352,
            "waypoint_sha256": {"holdout": "paired", "benchmark": "bench"},
            "milestones": {"100k": {"evaluation": {
                "holdout": compact(82, 0.20, 0.206)}}},
        }
        self.d = deepcopy(self.c)
        self.d["observation_shape"] = [10]
        self.d["milestones"]["100k"]["evaluation"]["holdout"] = compact(
            94, 0.05, 0.12)

    def test_paired_comparison_reports_all_four_with_common_metrics(self):
        result = build_paired_comparison(self.references, self.c, self.d,
                                         expected_digest="paired")
        rows = result["holdout"]
        self.assertEqual([rows[key]["successes"] for key in (
            "A_7d_ppo_p", "B_7d_ppo_direct_pi", "C_7d_ppo_trained_pi",
            "D_10d_ppo_trained_pi")], [100, 78, 82, 94])
        self.assertEqual(rows["D_10d_ppo_trained_pi"]["near_target_actual_speed_m_s"],
                         0.12)
        self.assertEqual(result["holdout_digest"], "paired")
        self.assertEqual(result["d_minus_c_success_percentage_points"], 12)

    def test_mismatched_targets_or_shapes_are_rejected(self):
        wrong = deepcopy(self.d)
        wrong["waypoint_sha256"]["holdout"] = "different"
        with self.assertRaisesRegex(ValueError, "holdout"):
            build_paired_comparison(self.references, self.c, wrong,
                                    expected_digest="paired")
        wrong = deepcopy(self.d)
        wrong["observation_shape"] = [7]
        with self.assertRaisesRegex(ValueError, "observation"):
            build_paired_comparison(self.references, self.c, wrong,
                                    expected_digest="paired")
        wrong = deepcopy(self.references)
        wrong["B_7d_ppo_direct_pi"]["training_seed"] = 8
        with self.assertRaisesRegex(ValueError, "seed"):
            build_paired_comparison(wrong, self.c, self.d,
                                    expected_digest="paired")

    def test_saved_ten_dimensional_report_checks_models_and_integral_traces(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "reports" /
                             "ppo_waypoint_pi_observable_lowstd_seed0.json").read_text())
        targets = {"benchmark": fixed_waypoints("full"),
                   "holdout": build_holdout_waypoints()}
        result = verify_report_artifacts(report, expected_shape=(10,),
                                         waypoints=targets,
                                         require_training_hash=True)
        self.assertEqual(set(result["checkpoint_sha256"]),
                         {"20k", "50k", "100k"})
        self.assertEqual(result["trace_count"], 6)
        wrong = deepcopy(report)
        wrong["milestones"]["20k"]["checkpoint_sha256"] = "tampered"
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            verify_report_artifacts(wrong, expected_shape=(10,),
                                    waypoints=targets,
                                    require_training_hash=True)
        wrong = deepcopy(report)
        wrong["milestones"]["20k"]["evaluation"]["benchmark"][
            "integral_trace_path"] = "/nonexistent/integral_trace.json"
        with self.assertRaises(FileNotFoundError):
            verify_report_artifacts(wrong, expected_shape=(10,),
                                    waypoints=targets,
                                    require_training_hash=True)
        wrong = deepcopy(report)
        evaluation = wrong["milestones"]["20k"]["evaluation"]["benchmark"]
        trace = json.loads(Path(evaluation["integral_trace_path"]).read_text())
        trace["episodes"].pop()
        trace["policy_steps"] = sum(len(row["steps"]) for row in trace["episodes"])
        evaluation["integral"]["policy_step_count"] = trace["policy_steps"]
        with tempfile.TemporaryDirectory() as scratch:
            fake_trace = Path(scratch) / "missing_episode.json"
            fake_trace.write_text(json.dumps(trace))
            evaluation["integral_trace_path"] = str(fake_trace)
            with self.assertRaisesRegex(ValueError, "trace summary"):
                verify_report_artifacts(wrong, expected_shape=(10,),
                                        waypoints=targets,
                                        require_training_hash=True)

    def test_cached_reference_metrics_require_fresh_replay_agreement(self):
        saved = compact(100, 0.0, 0.1)
        validate_reference_replay(saved, deepcopy(saved))
        changed = deepcopy(saved)
        changed["near_target_actual_speed_m_s"] = 0.9
        with self.assertRaisesRegex(ValueError, "reference replay"):
            validate_reference_replay(saved, changed)


if __name__ == "__main__":
    unittest.main()
