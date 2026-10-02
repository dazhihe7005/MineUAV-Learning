"""Return audit must preserve V2's actual per-step reward accounting."""

import unittest

import numpy as np

from mine_uav_env import MineUAVEnv
from reward_v2_alignment_audit import (accumulate_reward_terms,
                                       evaluate_episode_return, fixed_waypoints)


class RewardV2AlignmentAuditTests(unittest.TestCase):
    def test_component_sums_use_gamma_power_at_each_policy_step(self):
        rows = [
            {"progress": 1.0, "action": -0.2, "brake": -0.3,
             "success": 0.0, "failure": 0.0, "total": 0.5},
            {"progress": 2.0, "action": -0.1, "brake": -0.4,
             "success": 10.0, "failure": 0.0, "total": 11.5},
        ]
        result = accumulate_reward_terms(rows, gamma=0.99)
        self.assertAlmostEqual(result["undiscounted"]["total"], 12.0)
        self.assertAlmostEqual(result["discounted"]["total"], 11.885)
        self.assertAlmostEqual(result["discounted"]["progress"], 2.98)
        self.assertAlmostEqual(result["discounted"]["action"], -0.299)
        self.assertAlmostEqual(result["discounted"]["brake"], -0.696)
        self.assertAlmostEqual(result["discounted"]["success"], 9.9)

    def test_real_episode_total_matches_environment_reward_and_fixed_target(self):
        env = MineUAVEnv(max_episode_seconds=0.08, reward_version="v2")
        target = np.array([0.2, 0.0, 1.0])
        try:
            result = evaluate_episode_return(
                env, lambda obs: np.zeros(4, dtype=np.float32),
                target=target, seed=21, gamma=0.99)
            self.assertEqual(result["episode_length"], 2)
            self.assertEqual(result["target_position_m"], target.tolist())
            self.assertAlmostEqual(result["undiscounted"]["total"],
                                   sum(result["undiscounted"][key] for key in
                                       ("progress", "action", "brake", "success", "failure")))
            self.assertAlmostEqual(result["discounted"]["total"],
                                   sum(result["discounted"][key] for key in
                                       ("progress", "action", "brake", "success", "failure")))
        finally:
            env.close()

    def test_fixed_sets_preserve_curriculum_targets(self):
        local = fixed_waypoints("local")
        full = fixed_waypoints("full")
        self.assertEqual(len(local), 100)
        self.assertEqual(len(full), 100)
        self.assertEqual(local[0][0], 20261101)
        self.assertEqual(full[0][0], 20261001)
        np.testing.assert_allclose(full[0][1],
                                   [0.5636109068725292, 1.1732819373145285,
                                    0.7371474882999531], rtol=0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
