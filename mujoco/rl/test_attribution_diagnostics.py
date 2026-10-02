"""Terminal attribution statistics must be paired and physically interpretable."""

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_dynamics import RobustnessEnv, Scenario  # noqa: E402


def diagnostics_module():
    return importlib.import_module("attribution_diagnostics")


class AttributionDiagnosticsTests(unittest.TestCase):
    def test_runtime_mass_change_preserves_state_and_nominal_controller(self):
        """Catches resetting the UAV or retuning its controller at the disturbance."""
        env = RobustnessEnv(Scenario("nominal", 0.0))
        try:
            env.reset(seed=20271001, options={"target_position": [2, 2, 1]})
            for _ in range(5):
                env.step(np.zeros(4, dtype=np.float32))
            before_qpos = env.data.qpos.copy()
            before_qvel = env.data.qvel.copy()
            before_time = float(env.data.time)
            before_inertia = env.model.body_inertia[env.uav_body_id].copy()
            before_allocator = env.allocator.B.copy()
            env.activate_scenario(Scenario("mass", 1.05))
            self.assertAlmostEqual(env.model.body_mass[env.uav_body_id], 7.35)
            np.testing.assert_array_equal(env.model.body_inertia[env.uav_body_id], before_inertia)
            np.testing.assert_array_equal(env.allocator.B, before_allocator)
            self.assertEqual(env.velocity_controller.attitude_controller.mass_kg, 7.0)
            np.testing.assert_array_equal(env.data.qpos, before_qpos)
            np.testing.assert_array_equal(env.data.qvel, before_qvel)
            self.assertEqual(env.data.time, before_time)
            with self.assertRaises(ValueError):
                env.activate_scenario(Scenario("mass", 1.05))
        finally:
            env.close()

    def test_runtime_force_acts_at_first_subsequent_step(self):
        """Catches force injection only at reset, not after the settled hover."""
        env = RobustnessEnv(Scenario("nominal", 0.0))
        try:
            env.reset(seed=20271001, options={"target_position": [2, 2, 1]})
            env.activate_scenario(Scenario("force_x_n", -5.0))
            np.testing.assert_array_equal(env.data.xfrc_applied[env.uav_body_id],
                                          [-5, 0, 0, 0, 0, 0])
            env.step(np.zeros(4, dtype=np.float32))
            np.testing.assert_array_equal(env.data.xfrc_applied[env.uav_body_id],
                                          [-5, 0, 0, 0, 0, 0])
        finally:
            env.close()

    def test_tail_window_uses_available_span_and_vector_signs(self):
        """Catches zero-padding a short success or reversing target-minus-position."""
        diagnostics = diagnostics_module()
        steps = [
            {"time_s": 0.0, "post_time_s": 0.04,
             "position_m": [0.0, 0.0, 1.0], "target_m": [1.0, 0.0, 1.0],
             "position_error_m": [1.0, 0.0, 0.0], "velocity_m_s": [0.2, 0, 0],
             "velocity_command_m_s": [0.4, 0, 0]},
            {"time_s": 0.04, "post_time_s": 0.08,
             "position_m": [0.2, 0.0, 1.0], "target_m": [1.0, 0.0, 1.0],
             "position_error_m": [0.8, 0.0, 0.0], "velocity_m_s": [0.1, 0, 0],
             "velocity_command_m_s": [-0.2, 0, 0]},
        ]
        tail = diagnostics.terminal_window_stats(steps, [1, 0, 1], 5.0)
        self.assertEqual(tail["policy_steps"], 2)
        self.assertAlmostEqual(tail["observed_span_s"], 0.08)
        np.testing.assert_allclose(tail["mean_position_error_xyz_m"], [0.9, 0, 0])
        np.testing.assert_allclose(tail["mean_velocity_xyz_m_s"], [0.15, 0, 0])
        np.testing.assert_allclose(tail["mean_command_velocity_xyz_m_s"], [0.1, 0, 0])

    def test_tail_window_excludes_early_approach_samples(self):
        """Catches accidentally averaging the whole episode as steady state."""
        diagnostics = diagnostics_module()
        steps = [
            {"time_s": t, "post_time_s": t + 1,
             "position_m": [x, 0, 1], "target_m": [1, 0, 1],
             "position_error_m": [1 - x, 0, 0], "velocity_m_s": [v, 0, 0],
             "velocity_command_m_s": [c, 0, 0]}
            for t, x, v, c in [(0, 0, 1, 1), (5, 0.5, 0.5, 0.5),
                               (9, 0.8, 0.1, -0.2)]
        ]
        tail5 = diagnostics.terminal_window_stats(steps, [1, 0, 1], 5.0)
        tail1 = diagnostics.terminal_window_stats(steps, [1, 0, 1], 1.0)
        self.assertEqual(tail5["policy_steps"], 2)
        self.assertEqual(tail1["policy_steps"], 1)
        self.assertAlmostEqual(tail5["mean_position_error_xyz_m"][0], 0.35)
        self.assertAlmostEqual(tail1["mean_command_velocity_xyz_m_s"][0], -0.2)

    def test_episode_summary_keeps_terminal_bias_and_timeout_distinct(self):
        """Catches converting a timeout into success or dropping its tail command."""
        diagnostics = diagnostics_module()
        step = {"time_s": 0.0, "post_time_s": 0.04,
                "position_m": [0.0, 0.0, 1.0], "target_m": [1.0, 0.0, 1.0],
                "position_error_m": [1.0, 0.0, 0.0], "distance_m": 1.0,
                "velocity_m_s": [0.1, 0, 0],
                "velocity_command_m_s": [-0.2, 0, 0]}
        trace = {"seed": 5, "target_m": [1.0, 0.0, 1.0], "steps": [step],
                 "episode_length": 1, "final_distance_m": 0.996,
                 "termination_reason": "time_limit"}
        result = diagnostics.episode_attribution_summary(trace)
        self.assertFalse(result["success"])
        self.assertIsNone(result["completion_time_s"])
        self.assertFalse(result["crossed_after_first_0_2_m"])
        self.assertEqual(result["tail_5_s"]["mean_command_velocity_xyz_m_s"],
                         [-0.2, 0, 0])


if __name__ == "__main__":
    unittest.main()
