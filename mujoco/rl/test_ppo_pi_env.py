"""Fixed-PI experiment wrappers must preserve the Baseline v1 task boundary."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppo_pi_env import (MineUAVPIEnv, PIRobustnessEnv,
                        make_pi_training_envs)  # noqa: E402
from robustness_dynamics import Scenario  # noqa: E402


class PPOPIEnvTests(unittest.TestCase):
    def test_pi_env_preserves_task_and_reset_clears_memory(self):
        env = MineUAVPIEnv(render_mode=None, reward_version="v2",
                           target_distribution="full")
        try:
            self.assertEqual(env.observation_space.shape, (7,))
            self.assertEqual(env.action_space.shape, (4,))
            self.assertAlmostEqual(env.physics_dt, 0.002)
            self.assertAlmostEqual(env.control_dt, 0.01)
            self.assertAlmostEqual(env.policy_dt, 0.04)
            controller = env.velocity_controller
            self.assertAlmostEqual(controller.ki_xy, 0.5)
            self.assertAlmostEqual(controller.ki_z, 0.8)
            self.assertAlmostEqual(controller.integral_accel_limit_xy, 1.5)
            self.assertAlmostEqual(controller.integral_accel_limit_z, 1.5)
            observation, _ = env.reset(seed=17)
            self.assertEqual(observation.shape, (7,))
            for _ in range(4):
                _, _, done, truncated, _ = env.step(
                    np.array([0.5, 0, 0, 0], dtype=np.float32))
                self.assertFalse(done or truncated)
            self.assertGreater(controller.integral_error[0], 0.0)
            observation, _ = env.reset(seed=18)
            np.testing.assert_array_equal(controller.integral_error, [0, 0, 0])
            np.testing.assert_array_equal(controller.integral_acceleration_world,
                                          [0, 0, 0])
            self.assertEqual(observation.shape, (7,))
        finally:
            env.close()

    def test_evaluation_capture_aligns_policy_steps_and_dynamics(self):
        env = PIRobustnessEnv(Scenario("force_x_n", 2.0))
        try:
            self.assertAlmostEqual(env.velocity_controller.ki_xy, 0.5)
            self.assertAlmostEqual(env.data.xfrc_applied[env.uav_body_id, 0], 2.0)
            env.begin_integral_capture()
            _, info = env.reset(seed=20271001)
            target = info["target_position_m"].tolist()
            for _ in range(5):
                _, _, done, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
                self.assertFalse(done or truncated)
            captured = env.take_integral_capture()
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["seed"], 20271001)
            self.assertEqual(captured[0]["target_m"], target)
            np.testing.assert_array_equal(captured[0]["initial_integral_error_m"],
                                          [0, 0, 0])
            self.assertEqual(len(captured[0]["steps"]), 5)
            for index, row in enumerate(captured[0]["steps"]):
                self.assertEqual(row["policy_step"], index)
                self.assertEqual(len(row["integral_error_m"]), 3)
                np.testing.assert_allclose(
                    row["integral_acceleration_m_s2"],
                    np.asarray(row["integral_error_m"]) * [0.5, 0.5, 0.8],
                    atol=1e-10)
                self.assertTrue(np.isfinite(row["integral_error_m"]).all())
            self.assertAlmostEqual(env.data.xfrc_applied[env.uav_body_id, 0], 2.0)
        finally:
            env.close()

    def test_all_eight_monitored_training_envs_use_pi(self):
        with tempfile.TemporaryDirectory() as scratch:
            vec = make_pi_training_envs(Path(scratch) / "monitor", n_envs=8)
            try:
                self.assertEqual(vec.num_envs, 8)
                self.assertEqual(vec.seed(40), list(range(40, 48)))
                for wrapped in vec.envs:
                    env = wrapped.unwrapped
                    self.assertIsInstance(env, MineUAVPIEnv)
                    self.assertEqual(env.reward_version, "v2")
                    self.assertEqual(env.target_distribution, "full")
                    self.assertIsNone(env.render_mode)
                    self.assertAlmostEqual(env.velocity_controller.ki_xy, 0.5)
                    self.assertAlmostEqual(env.velocity_controller.ki_z, 0.8)
            finally:
                vec.close()


if __name__ == "__main__":
    unittest.main()
