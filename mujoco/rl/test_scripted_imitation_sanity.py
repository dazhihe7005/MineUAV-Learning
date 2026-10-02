"""A fixed-size tanh actor must learn from whole scripted episodes only."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripted_imitation_sanity import (ActorMeanMLP, ImitationPolicy,
                                       collect_success_episodes,
                                       offline_action_metrics, train_imitation_model)
from test_env_scripted_policy import scripted_action


class ScriptedImitationSanityTests(unittest.TestCase):
    def test_actor_shape_and_hidden_activation_match_sb3_default(self):
        actor = ActorMeanMLP()
        self.assertEqual(tuple(actor(torch.zeros(3, 7)).shape), (3, 4))
        self.assertEqual([type(module) for module in actor.policy_net],
                         [torch.nn.Linear, torch.nn.Tanh,
                          torch.nn.Linear, torch.nn.Tanh])
        self.assertEqual((actor.policy_net[0].in_features,
                          actor.policy_net[0].out_features,
                          actor.policy_net[2].out_features,
                          actor.action_net.out_features), (7, 64, 64, 4))

    def test_successful_episodes_keep_step_pairs_and_episode_ids(self):
        data = collect_success_episodes("local", seed_start=20270001,
                                        required_successes=2, max_attempts=2)
        self.assertEqual(data["successes"], 2)
        self.assertEqual(len(data["episode_seeds"]), 2)
        self.assertEqual(data["observations"].shape[1], 7)
        self.assertEqual(data["actions"].shape[1], 4)
        self.assertEqual(data["observations"].dtype, np.float32)
        self.assertTrue(np.all(np.abs(data["actions"]) <= 1))
        self.assertEqual(set(data["episode_ids"].tolist()), set(data["episode_seeds"]))
        for obs, action in zip(data["observations"], data["actions"]):
            np.testing.assert_allclose(action, scripted_action(obs), atol=0, rtol=0)

    def test_imitation_policy_clips_linear_mean_before_environment(self):
        actor = ActorMeanMLP()
        with torch.no_grad():
            actor.action_net.weight.zero_()
            actor.action_net.bias[:] = torch.tensor([3.0, -2.0, 0.1, 0.0])
        action, _ = ImitationPolicy(actor).predict(np.zeros(7, dtype=np.float32),
                                                  deterministic=True)
        np.testing.assert_allclose(action, [1.0, -1.0, 0.1, 0.0], atol=1e-7)

    def test_supervised_fit_reduces_loss_without_step_level_split(self):
        rng = np.random.default_rng(33)
        train_x = rng.uniform(-1, 1, size=(128, 7)).astype(np.float32)
        val_x = rng.uniform(-1, 1, size=(32, 7)).astype(np.float32)
        train_y = np.stack([scripted_action(x) for x in train_x])
        val_y = np.stack([scripted_action(x) for x in val_x])
        actor, history, selection = train_imitation_model(
            train_x, train_y, val_x, val_y, max_epochs=8, batch_size=64)
        self.assertTrue(all(np.isfinite(row["train_mse"]) and
                            np.isfinite(row["validation_mse"]) for row in history))
        self.assertLess(history[-1]["train_mse"], history[0]["train_mse"])
        with torch.no_grad():
            saved_mse = float(torch.mean((actor(torch.from_numpy(val_x))
                                          - torch.from_numpy(val_y)) ** 2))
        self.assertAlmostEqual(saved_mse, selection["validation_mse"], places=9)
        self.assertIn(selection["epoch"], [row["epoch"] for row in history])
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "actor.pt"
            torch.save(actor.state_dict(), path)
            restored = ActorMeanMLP()
            restored.load_state_dict(torch.load(path, weights_only=True))
            self.assertEqual(tuple(restored(torch.zeros(1, 7)).shape), (1, 4))

    def test_offline_metrics_mark_constant_yaw_correlation_undefined(self):
        expected = np.array([[0.1, 0.2, 0.3, 0.0], [0.3, 0.4, 0.5, 0.0]],
                            dtype=np.float32)
        actual = expected.copy()
        result = offline_action_metrics(actual, expected)
        self.assertAlmostEqual(result["overall_mse"], 0.0)
        self.assertEqual(result["correlation_per_axis"][3], None)
        self.assertAlmostEqual(result["max_absolute_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
