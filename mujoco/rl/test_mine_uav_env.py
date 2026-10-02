"""API and control-boundary regressions for the MineUAV Gymnasium baseline."""

import math
import sys
import unittest
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "control"))


class MineUAVEnvTests(unittest.TestCase):
    def setUp(self):
        from mine_uav_env import MineUAVEnv

        self.env = MineUAVEnv()

    def tearDown(self):
        self.env.close()

    def test_spaces_seeded_reset_and_clear_controller_state(self):
        obs_a, _ = self.env.reset(seed=42)
        target_a = self.env.target_position.copy()
        self.assertIsInstance(self.env, gym.Env)
        self.assertEqual(obs_a.shape, (7,))
        self.assertEqual(obs_a.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(obs_a))
        self.assertEqual(self.env.action_space.shape, (4,))
        self.assertEqual(self.env.action_space.dtype, np.float32)
        np.testing.assert_allclose(self.env.data.qpos[:3], [0, 0, 1])
        np.testing.assert_allclose(self.env.data.qvel, 0)
        self.assertTrue((-2 <= target_a[:2]).all() and (target_a[:2] <= 2).all())
        self.assertTrue(0.7 <= target_a[2] <= 1.5)
        self.env.step(np.array([0, 0, 0, 1], dtype=np.float32))
        self.env.reset(seed=42)
        np.testing.assert_allclose(self.env.target_position, target_a)
        np.testing.assert_allclose(self.env.data.qpos[:3], [0, 0, 1])
        np.testing.assert_allclose(self.env.data.qvel, 0)
        np.testing.assert_allclose(self.env.data.ctrl, 0)
        self.assertEqual(self.env.velocity_controller.yaw_target, 0.0)
        self.assertEqual(self.env.success_streak, 0)

    def test_env_reset_clears_enabled_velocity_pi_after_episode(self):
        """Reusing one env for multiple waypoints must not leak integral state."""
        from velocity_command_controller import VelocityCommandController

        old = self.env.velocity_controller
        self.env.velocity_controller = VelocityCommandController(
            old.attitude_controller, self.env.model.opt.gravity,
            ki_xy=0.5, ki_z=0.8)
        self.env.reset(seed=91)
        self.env.step(np.array([0.5, 0, 0.5, 0], dtype=np.float32))
        self.assertGreater(np.linalg.norm(self.env.velocity_controller.integral_error), 0)
        self.env.reset(seed=92)
        np.testing.assert_array_equal(self.env.velocity_controller.integral_error,
                                      [0, 0, 0])

    def test_action_mapping_timing_and_reward_breakdown(self):
        self.env.reset(seed=3)
        before = self.env.data.time
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([1, -1, 1, -1], dtype=np.float32))
        np.testing.assert_allclose(info["velocity_command_m_s"], [1.5, -1.5, 1.0])
        self.assertEqual(info["yaw_rate_command_rad_s"], -1.0)
        self.assertAlmostEqual(self.env.data.time - before, 0.04, places=10)
        self.assertEqual(info["physics_steps"], 20)
        self.assertEqual(info["controller_updates"], 4)
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertTrue(np.isfinite(reward))
        self.assertAlmostEqual(reward, sum(info["reward_breakdown"].values()))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(np.all(self.env.data.ctrl >= 0))
        self.assertTrue(np.all(self.env.data.ctrl <= self.env.allocator.u_max))

    def test_yaw_error_uses_shortest_wrap(self):
        self.env.reset(seed=1)
        self.env.target_yaw = math.radians(-179)
        self.env.data.qpos[3:7] = [math.cos(math.radians(179) / 2), 0, 0,
                                   math.sin(math.radians(179) / 2)]
        mujoco.mj_forward(self.env.model, self.env.data)
        obs = self.env._get_obs()
        self.assertAlmostEqual(float(obs[6]), math.radians(2), places=5)

    def test_success_requires_five_consecutive_steps(self):
        self.env.reset(seed=1, options={"target_position": [0, 0, 1]})
        for _ in range(4):
            _, _, terminated, truncated, _ = self.env.step(np.zeros(4, dtype=np.float32))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        _, reward, terminated, truncated, info = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "success")
        self.assertGreater(info["reward_breakdown"]["success_bonus"], 0)
        self.assertTrue(np.isfinite(reward))

    def test_failure_and_time_limit_are_distinct(self):
        from mine_uav_env import MineUAVEnv

        self.env.reset(seed=2)
        self.env.data.qpos[2] = -0.01
        _, reward, terminated, truncated, info = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "z_below_zero")
        self.assertLess(info["reward_breakdown"]["failure_penalty"], 0)
        self.assertTrue(np.isfinite(reward))

        short_env = MineUAVEnv(max_episode_seconds=0.08)
        try:
            short_env.reset(seed=2)
            short_env.step(np.zeros(4, dtype=np.float32))
            _, _, terminated, truncated, info = short_env.step(np.zeros(4, dtype=np.float32))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["termination_reason"], "time_limit")
        finally:
            short_env.close()

    def test_nonfinite_state_terminates_without_nonfinite_reward(self):
        self.env.reset(seed=1)
        self.env.data.qpos[0] = np.nan
        obs, reward, terminated, truncated, info = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "nonfinite_state")
        self.assertTrue(np.isfinite(reward))
        self.assertTrue(np.isfinite(obs).all())

    def test_flight_area_and_tilt_failures_terminate(self):
        self.env.reset(seed=1)
        self.env.data.qpos[0] = 6.1
        _, _, terminated, truncated, info = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "outside_flight_area")

        self.env.reset(seed=1)
        half = math.radians(61) / 2
        self.env.data.qpos[3:7] = [math.cos(half), math.sin(half), 0, 0]
        _, _, terminated, truncated, info = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "excessive_tilt")

    def test_scripted_velocity_policy_reaches_waypoint_without_training(self):
        from test_env_scripted_policy import scripted_action

        observation, _ = self.env.reset(seed=20260929)
        for _ in range(self.env.max_episode_steps):
            observation, _, terminated, truncated, info = self.env.step(
                scripted_action(observation))
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "success")


if __name__ == "__main__":
    unittest.main()
