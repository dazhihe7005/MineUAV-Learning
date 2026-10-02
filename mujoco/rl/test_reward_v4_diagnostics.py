"""V4 evaluation shares fixed waypoints and V2's crossing definition."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_ppo_braking import make_step_record, summarize_policy  # noqa: E402
from mine_uav_env import MineUAVEnv  # noqa: E402
from reward_v4_diagnostics import (crossing_counts, evaluate_reward_v4,
                                   validate_fixed_waypoints)  # noqa: E402


class ConstantPolicy:
    def predict(self, observation, deterministic):
        if not deterministic:
            raise AssertionError("Evaluation must be deterministic")
        return np.zeros(4, dtype=np.float32), None


class RewardV4DiagnosticsTests(unittest.TestCase):
    def test_real_environment_evaluation_records_both_sampling_conventions(self):
        env = MineUAVEnv(max_episode_seconds=0.12, reward_version="v4")
        try:
            result = evaluate_reward_v4(ConstantPolicy(), env, episodes=2, seed=101)
        finally:
            env.close()
        self.assertEqual(result["seeds"], [101, 102])
        self.assertEqual(result["reward_version"], "v4")
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["braking_diagnostic"]["episodes"], 2)
        self.assertEqual(result["crossing_after_first_0_2_m"]["within_0_5_s_count"], 0)
        self.assertEqual(len(result["action_near_boundary_fraction_per_axis"]), 4)
        self.assertEqual(sum(zone["steps"] for zone in
                             result["braking_diagnostic"]["distance_bins"].values()),
                         sum(ep["episode_length"] for ep in result["episode_summaries"]))

    def test_crossing_counts_use_signed_target_plane_not_distance_minimum(self):
        def trace(seed, last_x):
            steps = [make_step_record(t, [x, 0, 0], [0, 0, 0], [0, 0, 0],
                                      [0, 0, 0, 0], [0, 0, 0], 0)
                     for t, x in [(0.0, 0.3), (0.04, 0.15), (0.08, last_x)]]
            return {"seed": seed, "target_m": [0, 0, 0], "steps": steps,
                    "episode_length": 3, "episode_reward": 0,
                    "final_distance_m": abs(last_x), "termination_reason": "time_limit"}

        summary = summarize_policy([trace(1, -0.03), trace(2, 0.04)])
        result = crossing_counts(summary)
        self.assertEqual(result["entered_0_2_m_count"], 2)
        self.assertEqual(result["within_0_5_s_count"], 1)
        self.assertEqual(result["before_episode_end_count"], 1)
        self.assertAlmostEqual(result["within_0_5_s_fraction_all_episodes"], 0.5)

    def test_waypoint_validation_rejects_a_single_changed_target(self):
        seeds = list(range(20261001, 20261101))
        v2 = {"seeds": seeds, "episode_summaries": [
            {"seed": seed, "target_position_m": [0, 0, 1]} for seed in seeds]}
        braking = {"episode_summaries": [
            {"seed": seed, "target_m": [0, 0, 1]} for seed in seeds]}
        v4 = {"seeds": seeds, "episode_summaries": [
            {"seed": seed, "target_position_m": [0, 0, 1]} for seed in seeds]}
        self.assertTrue(validate_fixed_waypoints(v2, braking, v4))
        v4["episode_summaries"][17]["target_position_m"] = [0.01, 0, 1]
        with self.assertRaisesRegex(ValueError, "waypoint"):
            validate_fixed_waypoints(v2, braking, v4)


if __name__ == "__main__":
    unittest.main()
