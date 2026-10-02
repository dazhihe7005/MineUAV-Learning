"""The exploration audit must measure the saved policy and matched tasks."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_uav_env import MineUAVEnv  # noqa: E402
from ppo_exploration_audit import (  # noqa: E402
    CHECKPOINT, evaluate_exploration, inspect_policy_distribution, raw_gaussian_mean,
)
from reward_v2_alignment_audit import fixed_waypoints  # noqa: E402


class ExplorationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = PPO.load(CHECKPOINT, device="cpu")

    def test_saved_gaussian_std_maps_to_physical_command_scale(self):
        audit = inspect_policy_distribution(self.model)
        self.assertEqual(audit["distribution"], "DiagGaussianDistribution")
        self.assertTrue(audit["state_independent_trainable_log_std"])
        self.assertEqual(audit["axes"], ["vx", "vy", "vz", "yaw_rate"])
        self.assertEqual(audit["num_timesteps"], 100352)
        for value, scale, physical in zip(audit["normalized_std"],
                                          [1.5, 1.5, 1.0, 1.0],
                                          audit["physical_command_std"]):
            self.assertAlmostEqual(physical, value * scale, places=6)
            self.assertTrue(math.isfinite(value) and value > 0)

    def test_real_rollout_separates_mean_and_sampled_actions_on_same_targets(self):
        waypoints = fixed_waypoints("full")[:2]
        env = MineUAVEnv(max_episode_seconds=0.08, reward_version="v2")
        try:
            deterministic = evaluate_exploration(
                self.model, env, waypoints, deterministic=True, repeats=1)
            stochastic = evaluate_exploration(
                self.model, env, waypoints, deterministic=False, repeats=2)
        finally:
            env.close()
        self.assertEqual(deterministic["episodes"], 2)
        self.assertEqual(stochastic["episodes"], 4)
        self.assertEqual(deterministic["targets"], [target for _, target in waypoints])
        self.assertEqual(stochastic["targets"], [target for _, target in waypoints] * 2)
        self.assertAlmostEqual(deterministic["mean_action_norm"],
                               deterministic["executed_action_norm"], places=6)
        self.assertGreater(abs(stochastic["mean_action_norm"] -
                               stochastic["executed_action_norm"]), 0.01)
        self.assertEqual(len(stochastic["executed_action_near_boundary_fraction_per_axis"]), 4)
        self.assertEqual(stochastic["braking_diagnostic"]["episodes"], 4)

    def test_raw_gaussian_mean_is_not_confused_with_clipped_deterministic_action(self):
        model = PPO.load(CHECKPOINT, device="cpu")
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.fill_(2.0)
        env = MineUAVEnv(max_episode_seconds=0.08, reward_version="v2")
        try:
            observation, _ = env.reset(seed=20261001)
            np.testing.assert_allclose(raw_gaussian_mean(model, observation), [2.0] * 4)
            clipped, _ = model.predict(observation, deterministic=True)
            np.testing.assert_allclose(clipped, [1.0] * 4)
            result = evaluate_exploration(model, env, fixed_waypoints("full")[:1],
                                          deterministic=True)
        finally:
            env.close()
        self.assertAlmostEqual(result["raw_gaussian_mean_action_norm"], 4.0)
        self.assertAlmostEqual(result["clipped_deterministic_action_norm"], 2.0)
        self.assertEqual(result["raw_mean_outside_action_box_fraction_per_axis"], [1.0] * 4)


if __name__ == "__main__":
    unittest.main()
