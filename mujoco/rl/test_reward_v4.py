"""Reward V4 changes only V2's near-target tangential-speed cost."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_uav_env import MineUAVEnv  # noqa: E402


class RewardV4Tests(unittest.TestCase):
    def test_only_tangent_term_is_added_to_v2_at_intermediate_distance(self):
        v2 = MineUAVEnv(reward_version="v2")
        v4 = MineUAVEnv(reward_version="v4")
        try:
            v2.reset(seed=1)
            v4.reset(seed=1)
            for env in (v2, v4):
                env.target_position[:] = [0.3, 0.0, 1.0]
                env.data.qvel[:3] = [2.0, 3.0, 0.0]
            args = (0.4, 0.3, np.ones(4), False, False)
            reward_v2, parts_v2 = v2._compute_reward(*args)
            reward_v4, parts_v4 = v4._compute_reward(*args)
            self.assertEqual(set(parts_v4), {"progress", "action", "brake", "tangent",
                                             "success", "failure", "total"})
            for key in ("progress", "action", "brake", "success", "failure"):
                self.assertEqual(parts_v4[key], parts_v2[key])
            self.assertAlmostEqual(parts_v4["tangent"], -2.25)
            self.assertAlmostEqual(reward_v4 - reward_v2, -2.25)
            self.assertAlmostEqual(reward_v4, -4.52)
        finally:
            v2.close()
            v4.close()

    def test_tangent_penalty_is_zero_beyond_half_meter_and_full_below_tenth(self):
        env = MineUAVEnv(reward_version="v4")
        try:
            env.reset(seed=1)
            env.data.qvel[:3] = [0, 3, 0]
            for distance, expected in [(0.6, 0.0), (0.5, 0.0), (0.1, -4.5),
                                       (0.05, -4.5)]:
                with self.subTest(distance=distance):
                    env.target_position[:] = [distance, 0, 1]
                    _, parts = env._compute_reward(
                        distance, distance, np.zeros(4), False, False)
                    self.assertAlmostEqual(parts["tangent"], expected)
        finally:
            env.close()

    def test_exact_target_has_finite_tangent_penalty(self):
        env = MineUAVEnv(reward_version="v4")
        try:
            env.reset(seed=1)
            env.target_position[:] = env.data.qpos[:3]
            env.data.qvel[:3] = [1, 2, 0]
            reward, parts = env._compute_reward(0, 0, np.zeros(4), False, False)
            self.assertTrue(math.isfinite(reward))
            self.assertAlmostEqual(parts["tangent"], -2.5)
        finally:
            env.close()

    def test_nonfinite_failure_keeps_v4_reward_finite(self):
        env = MineUAVEnv(reward_version="v4")
        try:
            env.reset(seed=1)
            env.data.qvel[0] = np.nan
            _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
            self.assertTrue(terminated)
            self.assertTrue(math.isfinite(reward))
            self.assertEqual(info["termination_reason"], "nonfinite_state")
            self.assertTrue(math.isfinite(info["reward_breakdown"]["total"]))
        finally:
            env.close()

    def test_v2_and_v4_have_identical_dynamics_spaces_and_task_conditions(self):
        v2 = MineUAVEnv(reward_version="v2")
        v4 = MineUAVEnv(reward_version="v4")
        try:
            obs2, info2 = v2.reset(seed=20261001)
            obs4, info4 = v4.reset(seed=20261001)
            np.testing.assert_array_equal(obs2, obs4)
            np.testing.assert_array_equal(info2["target_position_m"], info4["target_position_m"])
            self.assertEqual(v2.observation_space, v4.observation_space)
            self.assertEqual(v2.action_space, v4.action_space)
            self.assertEqual(v2.max_episode_steps, v4.max_episode_steps)
            self.assertEqual((v2.physics_dt, v2.control_dt, v2.policy_dt),
                             (v4.physics_dt, v4.control_dt, v4.policy_dt))
            action = np.array([0.3, -0.4, 0.2, 0.1], dtype=np.float32)
            for _ in range(3):
                obs2, _, terminated2, truncated2, _ = v2.step(action)
                obs4, _, terminated4, truncated4, _ = v4.step(action)
                np.testing.assert_array_equal(obs2, obs4)
                np.testing.assert_array_equal(v2.data.qpos, v4.data.qpos)
                np.testing.assert_array_equal(v2.data.qvel, v4.data.qvel)
                np.testing.assert_array_equal(v2.data.ctrl, v4.data.ctrl)
                self.assertEqual((terminated2, truncated2), (terminated4, truncated4))
        finally:
            v2.close()
            v4.close()


if __name__ == "__main__":
    unittest.main()
