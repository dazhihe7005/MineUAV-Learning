"""Curriculum evaluations use fixed targets and unchanged V2 diagnostics."""

import unittest

import numpy as np

from curriculum_diagnostics import evaluate_curriculum
from mine_uav_env import MineUAVEnv


class ZeroPolicy:
    def predict(self, observation, deterministic=False):
        if not deterministic:
            raise AssertionError("Evaluation must be deterministic")
        return np.zeros(4, dtype=np.float32), None


class CurriculumDiagnosticsTests(unittest.TestCase):
    def test_local_evaluation_is_reproducible_and_uses_v2_reward(self):
        env = MineUAVEnv(max_episode_seconds=0.08, reward_version="v2",
                         target_distribution="local_a")
        try:
            first = evaluate_curriculum(ZeroPolicy(), env, episodes=3, seed=8123)
            second = evaluate_curriculum(ZeroPolicy(), env, episodes=3, seed=8123)
            self.assertEqual(first["seeds"], [8123, 8124, 8125])
            self.assertEqual(first["reward_version"], "v2")
            self.assertEqual(first["target_distribution"], "local_a")
            self.assertEqual(first["episode_summaries"], second["episode_summaries"])
            self.assertEqual(first["crossing_after_first_0_2_m"],
                             second["crossing_after_first_0_2_m"])
            for episode in first["episode_summaries"]:
                radius = np.linalg.norm(np.asarray(episode["target_position_m"]) - [0, 0, 1])
                self.assertGreaterEqual(radius, 0.25)
                self.assertLessEqual(radius, 0.50)
            self.assertIn("mean_command_norm_m_s",
                          first["braking_diagnostic"]["near_target"]["lt_0_1"])
        finally:
            env.close()

    def test_full_evaluation_keeps_original_seed_to_waypoint_mapping(self):
        env = MineUAVEnv(max_episode_seconds=0.04, reward_version="v2",
                         target_distribution="full")
        try:
            result = evaluate_curriculum(ZeroPolicy(), env, episodes=1, seed=20261001)
            np.testing.assert_allclose(result["episode_summaries"][0]["target_position_m"],
                                       [0.5636109068725292, 1.1732819373145285,
                                        0.7371474882999531], rtol=0, atol=1e-14)
        finally:
            env.close()

    def test_non_v2_or_human_environment_is_rejected(self):
        env = MineUAVEnv(reward_version="v1")
        try:
            with self.assertRaises(ValueError):
                evaluate_curriculum(ZeroPolicy(), env, episodes=1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
