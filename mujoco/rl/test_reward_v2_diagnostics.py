"""Hand-checked metric expectations for the common checkpoint evaluator."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_v2_diagnostics import evaluate_diagnostics, summarize_episodes  # noqa: E402


class DiagnosticsTests(unittest.TestCase):
    def test_real_environment_uses_requested_fixed_seeds(self):
        from mine_uav_env import MineUAVEnv
        from test_env_scripted_policy import scripted_action

        class ScriptedModel:
            def predict(self, observation, deterministic=False):
                if not deterministic:
                    raise AssertionError("Diagnostics must be deterministic")
                return scripted_action(observation), None

        env = MineUAVEnv(reward_version="v2")
        try:
            result = evaluate_diagnostics(ScriptedModel(), env, episodes=2, seed=20261001)
            self.assertEqual(result["seeds"], [20261001, 20261002])
            self.assertEqual(result["reward_version"], "v2")
            self.assertEqual(result["successes"], 2)
            self.assertEqual(len(result["episode_summaries"]), 2)
        finally:
            env.close()

    def test_proximity_speed_hold_and_action_fractions(self):
        episodes = [
            {"seed": 10, "final_distance_m": 0.09, "episode_length": 2,
             "episode_reward": 3.0, "termination_reason": "success",
             "steps": [
                 {"distance_m": 0.4, "speed_m_s": 1.0, "success_streak": 0,
                  "action": [1.0, 0, 0, 0]},
                 {"distance_m": 0.09, "speed_m_s": 0.1, "success_streak": 5,
                  "action": [0, -0.96, 0, 0]}]},
            {"seed": 11, "final_distance_m": 0.3, "episode_length": 2,
             "episode_reward": 1.0, "termination_reason": "time_limit",
             "steps": [
                 {"distance_m": 0.15, "speed_m_s": 0.5, "success_streak": 0,
                  "action": [0, 0, 0.95, 0]},
                 {"distance_m": 0.3, "speed_m_s": 0.05, "success_streak": 0,
                  "action": [0, 0, 0, 0]}]},
        ]
        result = summarize_episodes(episodes, policy_dt=0.04)
        self.assertEqual(result["seeds"], [10, 11])
        self.assertEqual(result["success_rate"], 0.5)
        self.assertAlmostEqual(result["mean_final_distance_m"], 0.195)
        self.assertEqual(result["mean_episode_length"], 2)
        self.assertEqual(result["mean_reward"], 2)
        self.assertAlmostEqual(result["mean_successful_completion_time_s"], 0.08)
        self.assertEqual(result["ever_within_0_5_m_fraction"], 1.0)
        self.assertEqual(result["ever_within_0_2_m_fraction"], 1.0)
        self.assertEqual(result["ever_within_0_1_m_fraction"], 0.5)
        self.assertEqual(result["ever_speed_below_0_15_m_s_fraction"], 1.0)
        self.assertEqual(result["ever_distance_and_speed_fraction"], 0.5)
        self.assertEqual(result["max_success_streak_policy_steps"], 5)
        self.assertAlmostEqual(result["mean_speed_within_0_5_m_s"], 0.4125)
        self.assertAlmostEqual(result["mean_speed_within_0_2_m_s"], 0.3)
        self.assertAlmostEqual(result["mean_speed_within_0_1_m_s"], 0.1)
        self.assertEqual(result["action_near_boundary_fraction_per_axis"],
                         [0.25, 0.25, 0.25, 0.0])

    def test_empty_near_target_speed_bucket_is_null(self):
        episodes = [{"seed": 7, "final_distance_m": 1.0, "episode_length": 1,
                     "episode_reward": -1.0, "termination_reason": "time_limit",
                     "steps": [{"distance_m": 1.0, "speed_m_s": 0.2,
                                "success_streak": 0, "action": [0, 0, 0, 0]}]}]
        result = summarize_episodes(episodes, policy_dt=0.04)
        self.assertIsNone(result["mean_speed_within_0_5_m_s"])
        self.assertIsNone(result["mean_successful_completion_time_s"])
        self.assertEqual(result["ever_distance_and_speed_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
