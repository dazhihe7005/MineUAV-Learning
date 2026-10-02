"""Reward V2 is opt-in and adds only a distance-gated speed penalty."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_uav_env import MineUAVEnv  # noqa: E402


class RewardV2Tests(unittest.TestCase):
    def test_default_reward_retains_v1_values_and_keys(self):
        env = MineUAVEnv()
        try:
            env.reset(seed=1)
            env.data.qvel[:3] = [3, 4, 0]
            reward, parts = env._compute_reward(0.4, 0.3, np.ones(4), False, False)
            self.assertEqual(set(parts), {"progress", "action_penalty", "success_bonus", "failure_penalty"})
            self.assertAlmostEqual(reward, 0.98)
        finally:
            env.close()

    def test_v2_distance_weight_uses_only_world_linear_speed(self):
        env = MineUAVEnv(reward_version="v2")
        try:
            env.reset(seed=1)
            env.data.qvel[:] = [3, 4, 0, 100, 100, 100]
            for distance, expected_brake in [(0.6, 0), (0.5, 0), (0.3, -6.25),
                                             (0.1, -12.5), (0.05, -12.5)]:
                with self.subTest(distance=distance):
                    reward, parts = env._compute_reward(
                        distance, distance, np.zeros(4), False, False)
                    self.assertEqual(set(parts), {"progress", "action", "brake", "success", "failure", "total"})
                    self.assertAlmostEqual(parts["brake"], expected_brake)
                    self.assertAlmostEqual(parts["total"], reward)
                    self.assertAlmostEqual(reward, expected_brake)
        finally:
            env.close()

    def test_v2_keeps_original_reward_terms(self):
        env = MineUAVEnv(reward_version="v2")
        try:
            env.reset(seed=1)
            env.data.qvel[:3] = [0, 0, 0]
            reward, parts = env._compute_reward(0.4, 0.3, np.ones(4), True, False)
            self.assertAlmostEqual(parts["progress"], 1.0)
            self.assertAlmostEqual(parts["action"], -0.02)
            self.assertEqual(parts["success"], 10.0)
            self.assertEqual(parts["failure"], 0.0)
            self.assertAlmostEqual(reward, 10.98)
        finally:
            env.close()

    def test_invalid_reward_version_is_rejected(self):
        with self.assertRaises(ValueError):
            MineUAVEnv(reward_version="v5")

    def test_nonfinite_velocity_failure_has_finite_reward(self):
        env = MineUAVEnv(reward_version="v2")
        try:
            env.reset(seed=1)
            env.data.qvel[0] = np.nan
            _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
            self.assertTrue(terminated)
            self.assertEqual(info["termination_reason"], "nonfinite_state")
            self.assertTrue(math.isfinite(reward))
            self.assertTrue(math.isfinite(info["reward_breakdown"]["total"]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
