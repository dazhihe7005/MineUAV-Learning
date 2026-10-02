"""Frozen-policy attribution evaluations must stay paired and inference-only."""

import importlib
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_ppo_braking import evaluate_episode  # noqa: E402
from robustness_dynamics import RobustnessEnv, Scenario  # noqa: E402
from test_env_scripted_policy import scripted_action  # noqa: E402
from train_ppo_lowstd_multiseed import build_holdout_waypoints  # noqa: E402


def audit_module():
    return importlib.import_module("attribution_audit")


class AttributionAuditTests(unittest.TestCase):
    def test_one_waypoint_scripted_parity_and_tail_command(self):
        """Catches replacing the verified scripted action or metric conventions."""
        audit = audit_module()
        waypoint = build_holdout_waypoints()[:1]
        left = RobustnessEnv(Scenario("nominal", 0.0))
        right = RobustnessEnv(Scenario("nominal", 0.0))
        try:
            previous = evaluate_episode(left, scripted_action, waypoint[0][0])
            result = audit.evaluate_single_policy(right, scripted_action, waypoint)
            self.assertEqual(result["metrics"]["successes"], 1)
            self.assertAlmostEqual(result["episodes"][0]["final_distance_m"],
                                   previous["final_distance_m"])
            self.assertEqual(result["episodes"][0]["target_position_m"], waypoint[0][1])
            self.assertEqual(result["episodes"][0]["tail_5_s"]["policy_steps"] > 0,
                             True)
        finally:
            left.close()
            right.close()

    def test_pairing_rejects_nonholdout_and_missing_policy_seed(self):
        """Catches silently comparing different targets or fewer than five PPO seeds."""
        audit = audit_module()
        holdout = build_holdout_waypoints()
        with self.assertRaises(ValueError):
            audit.evaluate_waypoint_condition(Scenario("mass", 0.99), [], holdout[:1])
        with self.assertRaises(ValueError):
            audit.evaluate_waypoint_condition(Scenario("mass", 0.99), [], holdout)

    def test_hold_injects_after_warmup_and_keeps_zero_command(self):
        """Catches changing the plant before settling or injecting a waypoint policy."""
        audit = audit_module()
        result = audit.evaluate_hold_condition(
            Scenario("mass", 1.05), warmup_s=0.2, observe_s=0.4)
        self.assertAlmostEqual(result["injection_time_s"], 0.2, places=8)
        self.assertAlmostEqual(result["plant_mass_after_injection_kg"], 7.35)
        self.assertAlmostEqual(result["controller_mass_kg"], 7.0)
        self.assertLess(abs(result["state_at_injection"]["position_m"][2] - 1.0), 1e-6)
        np.testing.assert_array_equal(result["tail_5_s"]["mean_command_velocity_xyz_m_s"],
                                      [0, 0, 0])
        self.assertGreater(result["tail_5_s"]["policy_steps"], 0)

    def test_hold_force_is_applied_only_after_settling(self):
        """Catches an external force contaminating the nominal pre-hold phase."""
        audit = audit_module()
        result = audit.evaluate_hold_condition(
            Scenario("force_x_n", 5.0), warmup_s=0.2, observe_s=0.4)
        self.assertAlmostEqual(result["state_at_injection"]["position_m"][0], 0,
                               places=8)
        self.assertGreater(result["terminal_state"]["position_m"][0], 0)
        np.testing.assert_array_equal(result["tail_5_s"]["mean_command_velocity_xyz_m_s"],
                                      [0, 0, 0])

    def test_default_hold_observes_fifteen_seconds_after_two_second_warmup(self):
        """Catches truncating the probe at the waypoint task's 15 s total limit."""
        audit = audit_module()
        result = audit.evaluate_hold_condition(Scenario("nominal", 0.0))
        self.assertEqual(result["terminal_reason"], "probe_complete")
        self.assertAlmostEqual(result["terminal_state"]["time_s"], 17.0, places=7)
        self.assertEqual(result["policy_steps_after_injection"], 375)

    def test_condition_file_preserves_existing_result(self):
        """Catches overwriting the previous robustness study or completed parts."""
        audit = audit_module()
        with tempfile.TemporaryDirectory() as scratch:
            path = audit.condition_path(Scenario("mass", 0.99), Path(scratch))
            audit.write_condition(path, {"first": 1})
            with self.assertRaises(FileExistsError):
                audit.write_condition(path, {"second": 2})
            self.assertEqual(json.loads(path.read_text()), {"first": 1})

    def test_condition_write_is_atomic_on_publish_error(self):
        """A crash before publication must not leave an invalid .json part."""
        audit = audit_module()
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "sample.json"
            with patch.object(audit.os, "link", side_effect=OSError("publish failed")):
                with self.assertRaises(OSError):
                    audit.write_condition(path, {"new": 1})
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_hold_runner_resumes_valid_saved_condition(self):
        """An interrupted multi-condition hold audit should skip finished probes."""
        audit = audit_module()
        importlib.import_module("summarize_attribution_audit")
        nominal = Scenario("nominal", 0.0)
        disturbed = Scenario("mass", 1.05)
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            source = (Path(__file__).resolve().parents[1] / "reports" /
                      "ppo_baseline_v1_attribution_parts" / "hold" / "nominal.json")
            saved = json.loads(source.read_text())
            audit.write_condition(audit.hold_path(nominal, root), saved)
            new = copy.deepcopy(saved)
            new["scenario"] = {"kind": "mass", "value": 1.05}
            new["plant_mass_after_injection_kg"] = 7.35
            with patch.object(audit, "HOLD_SCENARIOS", (nominal, disturbed)), \
                    patch.object(audit, "evaluate_hold_condition", return_value=new) as run:
                audit.run_hold(root=root)
            run.assert_called_once_with(disturbed)
            self.assertEqual(json.loads(audit.hold_path(nominal, root).read_text()), saved)
            self.assertEqual(json.loads(audit.hold_path(disturbed, root).read_text()), new)

    def test_nominal_gate_rejects_changed_frozen_ppo_outcome(self):
        """Catches a changed evaluator passing through to factor sweeps."""
        audit = audit_module()
        old = json.loads((Path(__file__).resolve().parents[1] / "reports" /
                          "ppo_baseline_v1_robustness_parts" / "nominal.json").read_text())
        fake = {"evaluations": [
            {"policy": "scripted", "training_seed": None,
             "metrics": {"successes": 100}},
            *[{"policy": "ppo", "training_seed": item["training_seed"],
               "metrics": {key: item["metrics"][key] for key in (
                   "successes", "mean_final_distance_m",
                   "mean_successful_completion_time_s", "mean_episode_length",
                   "crossing_rate")}} for item in old["evaluations"]],
        ]}
        audit.validate_nominal_condition(fake)
        fake["evaluations"][1]["metrics"]["successes"] -= 1
        with self.assertRaises(RuntimeError):
            audit.validate_nominal_condition(fake)


if __name__ == "__main__":
    unittest.main()
