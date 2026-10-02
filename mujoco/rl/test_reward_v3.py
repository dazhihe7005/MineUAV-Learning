"""Reward V3 adds exactly one continuous distance cost to V2."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_uav_env import MineUAVEnv  # noqa: E402


class RewardV3Tests(unittest.TestCase):
    def test_v2_and_v3_have_identical_state_transition_for_same_action(self):
        v2 = MineUAVEnv(reward_version="v2")
        v3 = MineUAVEnv(reward_version="v3")
        try:
            obs_v2, info_v2 = v2.reset(seed=20261001)
            obs_v3, info_v3 = v3.reset(seed=20261001)
            np.testing.assert_array_equal(obs_v2, obs_v3)
            np.testing.assert_array_equal(info_v2["target_position_m"],
                                          info_v3["target_position_m"])
            action = np.array([0.3, -0.4, 0.2, 0.1], dtype=np.float32)
            for _ in range(3):
                obs_v2, reward_v2, terminated_v2, truncated_v2, info_v2 = v2.step(action)
                obs_v3, reward_v3, terminated_v3, truncated_v3, info_v3 = v3.step(action)
                np.testing.assert_array_equal(obs_v2, obs_v3)
                np.testing.assert_array_equal(v2.data.qpos, v3.data.qpos)
                np.testing.assert_array_equal(v2.data.qvel, v3.data.qvel)
                np.testing.assert_array_equal(v2.data.ctrl, v3.data.ctrl)
                self.assertEqual((terminated_v2, truncated_v2),
                                 (terminated_v3, truncated_v3))
                self.assertAlmostEqual(reward_v3 - reward_v2,
                                       -0.02 * info_v3["distance_m"])
        finally:
            v2.close()
            v3.close()

    def test_v3_adds_only_current_distance_cost_to_v2(self):
        v2 = MineUAVEnv(reward_version="v2")
        v3 = MineUAVEnv(reward_version="v3")
        try:
            v2.reset(seed=11)
            v3.reset(seed=11)
            v2.data.qvel[:3] = [3, 4, 0]
            v3.data.qvel[:3] = [3, 4, 0]
            args = (0.4, 0.3, np.ones(4), True, False)
            reward_v2, parts_v2 = v2._compute_reward(*args)
            reward_v3, parts_v3 = v3._compute_reward(*args)
            self.assertEqual(set(parts_v3), {"progress", "distance", "action", "brake",
                                              "success", "failure", "total"})
            self.assertAlmostEqual(parts_v3["distance"], -0.006)
            for key in ("progress", "action", "brake", "success", "failure"):
                self.assertEqual(parts_v3[key], parts_v2[key])
            self.assertAlmostEqual(reward_v3 - reward_v2, -0.006)
            self.assertAlmostEqual(reward_v3, sum(parts_v3[key] for key in parts_v3
                                                   if key != "total"))
            self.assertAlmostEqual(parts_v3["total"], reward_v3)
        finally:
            v2.close()
            v3.close()

    def test_distance_cost_is_active_outside_braking_zone(self):
        env = MineUAVEnv(reward_version="v3")
        try:
            env.reset(seed=3)
            env.data.qvel[:3] = [3, 4, 0]
            reward, parts = env._compute_reward(0.6, 0.6, np.zeros(4), False, False)
            self.assertEqual(parts["brake"], 0)
            self.assertAlmostEqual(parts["distance"], -0.012)
            self.assertAlmostEqual(reward, -0.012)
        finally:
            env.close()

    def test_nonfinite_failure_keeps_finite_v3_reward(self):
        env = MineUAVEnv(reward_version="v3")
        try:
            env.reset(seed=3)
            env.data.qvel[0] = np.nan
            _, reward, terminated, _, info = env.step(np.zeros(4, np.float32))
            self.assertTrue(terminated)
            self.assertEqual(info["termination_reason"], "nonfinite_state")
            self.assertTrue(math.isfinite(reward))
            self.assertAlmostEqual(reward, info["reward_breakdown"]["total"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
