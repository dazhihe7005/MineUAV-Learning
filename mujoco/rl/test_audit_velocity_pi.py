"""Inference-only velocity-PI hold and frozen-policy comparison regressions."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_dynamics import RobustnessEnv, Scenario  # noqa: E402
from test_env_scripted_policy import scripted_action  # noqa: E402
from train_ppo_lowstd_multiseed import build_holdout_waypoints  # noqa: E402


def audit_module():
    return importlib.import_module("audit_velocity_pi")


class VelocityPIAuditTests(unittest.TestCase):
    def test_configure_pi_keeps_existing_kp_and_model(self):
        """A supposed PI audit must not silently change the Kp/plant/control chain."""
        audit = audit_module()
        env = RobustnessEnv(Scenario("nominal", 0.0))
        try:
            body_id = env.model.body("mine_uav").id
            before_mass = float(env.model.body_mass[body_id])
            before_kp = env.velocity_controller.kv.copy()
            attitude = env.velocity_controller.attitude_controller
            audit.configure_pi_env(env)
            np.testing.assert_array_equal(env.velocity_controller.kv, before_kp)
            self.assertIs(env.velocity_controller.attitude_controller, attitude)
            self.assertAlmostEqual(env.model.body_mass[body_id], before_mass)
            self.assertAlmostEqual(env.velocity_controller.ki_xy, 0.5)
            self.assertAlmostEqual(env.velocity_controller.ki_z, 0.8)
        finally:
            env.close()

    def test_nominal_zero_command_hold_records_complete_pi_diagnostics(self):
        """The hold probe must expose the state needed to separate drift/stopping."""
        audit = audit_module()
        row = audit.evaluate_pi_hold(Scenario("nominal", 0.0),
                                     warmup_s=0.2, observe_s=0.4)
        self.assertAlmostEqual(row["injection_time_s"], 0.2, places=8)
        self.assertAlmostEqual(row["actual_observation_s"], 0.4, places=8)
        self.assertEqual(row["terminal_reason"], "probe_complete")
        self.assertEqual(row["policy_steps_after_injection"], 10)
        self.assertEqual(len(row["steps"]), 10)
        self.assertLess(abs(row["terminal_state"]["position_m"][2] - 1), 1e-4)
        np.testing.assert_allclose(row["tail_5_s"]["mean_velocity_xyz_m_s"], 0,
                                   atol=1e-4)
        for step in row["steps"]:
            np.testing.assert_array_equal(step["velocity_command_m_s"], [0, 0, 0])
            self.assertEqual(len(step["integral_error_m"]), 3)
            self.assertEqual(len(step["integral_acceleration_m_s2"]), 3)
            self.assertEqual(len(step["desired_acceleration_m_s2"]), 3)
            self.assertGreater(step["max_motor_rpm"], 0)

    def test_force_hold_integral_opposes_external_force_and_is_bounded(self):
        """A +X force must induce a negative, bounded X integral correction."""
        audit = audit_module()
        row = audit.evaluate_pi_hold(Scenario("force_x_n", 5.0),
                                     warmup_s=0.2, observe_s=2.0)
        self.assertGreater(row["terminal_state"]["position_m"][0], 0)
        self.assertLess(row["steps"][-1]["integral_acceleration_m_s2"][0], 0)
        self.assertLessEqual(row["max_integral_accel_xy_m_s2"], 1.5 + 1e-9)
        self.assertLessEqual(row["max_integral_accel_z_m_s2"], 1.5 + 1e-9)
        self.assertEqual(row["controller_mass_kg"], 7.0)

    def test_new_part_writer_preserves_existing_output(self):
        audit = audit_module()
        with tempfile.TemporaryDirectory() as scratch:
            path = audit.hold_output_path(Scenario("mass", 1.05), Path(scratch))
            audit.write_new_part(path, {"first": 1})
            with self.assertRaises(FileExistsError):
                audit.write_new_part(path, {"second": 2})
            self.assertEqual(json.loads(path.read_text()), {"first": 1})

    def test_pi_waypoint_condition_rejects_wrong_targets_or_policy_set(self):
        """Scripted/PPO comparison cannot silently drop a seed or change targets."""
        audit = audit_module()
        waypoints = build_holdout_waypoints()
        with self.assertRaises(ValueError):
            audit.evaluate_pi_waypoint_condition(Scenario("mass", 1.05), [],
                                                 waypoints[:99])
        with self.assertRaises(ValueError):
            audit.evaluate_pi_waypoint_condition(Scenario("mass", 1.05), [],
                                                 waypoints)

    def test_pi_single_policy_uses_same_holdout_target_and_reward_v2(self):
        """The new evaluator should retain the old task and per-episode metrics."""
        audit = audit_module()
        env = RobustnessEnv(Scenario("nominal", 0.0))
        try:
            audit.configure_pi_env(env)
            waypoints = build_holdout_waypoints()[:1]
            result = audit.evaluate_pi_single_policy(env, scripted_action, waypoints)
            self.assertEqual(result["metrics"]["episodes"], 1)
            self.assertEqual(result["episodes"][0]["target_position_m"],
                             waypoints[0][1])
            self.assertEqual(env.reward_version, "v2")
            self.assertAlmostEqual(env.velocity_controller.ki_xy, 0.5)
        finally:
            env.close()

    def test_waypoint_output_path_is_independent_of_old_p_audit(self):
        audit = audit_module()
        with tempfile.TemporaryDirectory() as scratch:
            path = audit.waypoint_output_path(Scenario("force_x_n", 5), Path(scratch))
            self.assertEqual(path.parent.name, "force_x_n")
            self.assertTrue(str(path).startswith(scratch))
            self.assertNotIn("ppo_baseline_v1_attribution_parts", str(path))

    def test_hold_runner_validates_existing_part_and_resumes_remaining(self):
        """Interrupted seven-probe audit must not overwrite a finished probe."""
        audit = audit_module()
        summary = importlib.import_module("summarize_velocity_pi")
        nominal = Scenario("nominal", 0.0)
        mass = Scenario("mass", 1.05)
        old_row = json.loads(audit.hold_output_path(nominal).read_text())
        new_row = json.loads(audit.hold_output_path(mass).read_text())
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            audit.write_new_part(audit.hold_output_path(nominal, root), old_row)
            with patch.object(audit, "HOLD_SCENARIOS", (nominal, mass)), \
                    patch.object(audit, "evaluate_pi_hold", return_value=new_row):
                paths = audit.run_holds(root=root)
            self.assertEqual(len(paths), 2)
            self.assertEqual(json.loads(audit.hold_output_path(nominal, root).read_text()),
                             old_row)
            self.assertEqual(json.loads(audit.hold_output_path(mass, root).read_text()),
                             new_row)
            summary.validate_pi_hold(new_row, mass)


if __name__ == "__main__":
    unittest.main()
