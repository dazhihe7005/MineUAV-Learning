"""Conditional PPO/PI robustness must pair historical and new policies."""

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_ppo_pi_compatibility import (compare_abc, disturbance_scenarios,
                                        run_conditional_robustness)  # noqa: E402
from train_ppo_lowstd_multiseed import build_holdout_waypoints  # noqa: E402
from audit_ppo_robustness import waypoint_sha256  # noqa: E402


OLD_P = Path(__file__).resolve().parents[1] / "reports" / "ppo_baseline_v1_attribution.json"
OLD_PI = Path(__file__).resolve().parents[1] / "reports" / "velocity_pi_robustness_v2.json"


class PPOPICompatibilityTests(unittest.TestCase):
    def test_sweep_has_only_twelve_non_nominal_conditions(self):
        scenarios = disturbance_scenarios()
        self.assertEqual(len(scenarios), 12)
        self.assertEqual([(item.kind, item.value) for item in scenarios], [
            ("mass", 0.95), ("mass", 0.98), ("mass", 1.02), ("mass", 1.05),
            ("thrust", 0.95), ("thrust", 0.98), ("thrust", 1.02), ("thrust", 1.05),
            ("force_x_n", -5.0), ("force_x_n", -2.0),
            ("force_x_n", 2.0), ("force_x_n", 5.0),
        ])

    def test_abc_nominal_uses_same_seed_zero_old_p_and_direct_pi(self):
        old_p = json.loads(OLD_P.read_text())
        old_pi = json.loads(OLD_PI.read_text())
        holdout = build_holdout_waypoints()
        new_pi = {
            "episodes": 100, "successes": 94, "success_rate": 0.94,
            "mean_final_distance_m": 0.1,
            "mean_successful_completion_time_s": 3.0,
            "mean_episode_length": 80.0, "crossing_rate": 0.01,
        }
        result = compare_abc(old_p, old_pi, "nominal", 0.0, new_pi, holdout)
        self.assertEqual(result["A_old_ppo_with_P"]["successes"], 100)
        self.assertEqual(result["B_same_old_ppo_with_PI"]["successes"], 78)
        self.assertEqual(result["C_new_ppo_trained_with_PI"]["successes"], 94)
        self.assertEqual(result["training_seed"], 0)
        changed = json.loads(OLD_PI.read_text())
        changed["waypoint_sha256"] = "wrong-targets"
        with self.assertRaises(ValueError):
            compare_abc(old_p, changed, "nominal", 0.0, new_pi, holdout)

    def test_failed_nominal_gate_does_not_create_disturbance_outputs(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            checkpoint = root / "checkpoint.zip"
            checkpoint.write_bytes(b"saved nominal policy")
            fake = {
                "seed": 0, "reward_version": "v2",
                "waypoint_sha256": {
                    "holdout": waypoint_sha256(build_holdout_waypoints())},
                "nominal_100k_gate_requires_both_sets_90pct": False,
                "milestones": {"100k": {
                    "checkpoint": str(checkpoint),
                    "evaluation": {
                        "benchmark": {"episodes": 100, "successes": 89},
                        "holdout": {
                            "episodes": 100, "successes": 82,
                            "success_rate": 0.82,
                            "mean_final_distance_m": 0.1,
                            "mean_successful_completion_time_s": 8.0,
                            "mean_episode_length": 250.0,
                            "crossing_rate": 0.2,
                        },
                    },
                }},
            }
            result = run_conditional_robustness(fake, output_root=root)
            self.assertFalse(result["nominal_gate_passed"])
            self.assertEqual(result["disturbed_conditions_evaluated"], 0)
            self.assertEqual(result["nominal"]["A_old_ppo_with_P"]["successes"], 100)
            self.assertEqual(result["nominal"]["B_same_old_ppo_with_PI"]["successes"], 78)
            self.assertEqual(result["nominal"]["C_new_ppo_trained_with_PI"]["successes"], 82)
            self.assertEqual(result["checkpoint_sha256"], hashlib.sha256(
                checkpoint.read_bytes()).hexdigest())
            self.assertEqual(list(root.rglob("*.json")), [
                root / "reports" / "ppo_pi_compatibility" / "nominal_comparison.json"])
            self.assertEqual(list(root.rglob("*condition_parts*")), [])
            self.assertEqual(run_conditional_robustness(fake, output_root=root), result)
            checkpoint.write_bytes(b"substituted policy")
            with self.assertRaisesRegex(ValueError, "nominal comparison"):
                run_conditional_robustness(fake, output_root=root)

    def test_passing_gate_refuses_missing_checkpoint_hash_provenance(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            checkpoint = root / "checkpoint.zip"
            checkpoint.write_bytes(b"saved policy")
            fake = {
                "seed": 0, "reward_version": "v2",
                "waypoint_sha256": {
                    "holdout": waypoint_sha256(build_holdout_waypoints())},
                "nominal_100k_gate_requires_both_sets_90pct": True,
                "milestones": {"100k": {
                    "checkpoint": str(checkpoint),
                    "evaluation": {
                        "benchmark": {"episodes": 100, "successes": 90},
                        "holdout": {"episodes": 100, "successes": 90,
                                    "success_rate": 0.9},
                    },
                }},
            }
            with self.assertRaisesRegex(ValueError, "checkpoint hash"):
                run_conditional_robustness(fake, output_root=root)
            self.assertEqual(list(root.rglob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
