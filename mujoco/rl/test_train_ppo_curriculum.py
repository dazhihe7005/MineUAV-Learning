"""Curriculum stages must preserve PPO configuration and model continuity."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mine_uav_env import MineUAVEnv
from train_ppo_curriculum import (require_fresh_curriculum_artifacts,
                                  stage_schedule, train_stage)
from train_ppo_waypoint import make_ppo, make_training_envs


class CurriculumTrainingTests(unittest.TestCase):
    def test_stage_budgets_are_complete_rollouts(self):
        schedule = stage_schedule()
        self.assertEqual([(item["label"], item["target_distribution"],
                           item["stage_timesteps"], item["cumulative_timesteps"])
                          for item in schedule], [
                              ("a", "local_a", 51_200, 51_200),
                              ("b", "local_b", 51_200, 102_400),
                              ("c", "full", 100_352, 202_752),
                          ])

    def test_eight_training_environments_are_headless_v2_with_local_targets(self):
        with tempfile.TemporaryDirectory() as scratch:
            envs = make_training_envs(Path(scratch) / "monitor", n_envs=8,
                                      reward_version="v2", target_distribution="local_a")
            try:
                self.assertEqual(envs.num_envs, 8)
                self.assertTrue(all(env.unwrapped.reward_version == "v2" and
                                    env.unwrapped.render_mode is None and
                                    env.unwrapped.target_distribution == "local_a"
                                    for env in envs.envs))
                model = make_ppo(envs, Path(scratch) / "tensorboard")
                self.assertEqual((model.n_steps, model.batch_size, model.n_epochs),
                                 (256, 256, 10))
                self.assertEqual(model.device.type, "cpu")
                self.assertEqual(model.policy.mlp_extractor.policy_net[0].out_features, 64)
                self.assertEqual(model.policy.mlp_extractor.value_net[0].out_features, 64)
            finally:
                envs.close()

    def test_stages_continue_same_policy_and_optimizer(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            env_a = make_training_envs(root / "a", n_envs=1, reward_version="v2",
                                       target_distribution="local_a")
            env_b = make_training_envs(root / "b", n_envs=1, reward_version="v2",
                                       target_distribution="local_b")
            try:
                model = make_ppo(env_a, root / "tensorboard")
                policy_id = id(model.policy)
                optimizer_id = id(model.policy.optimizer)
                model.verbose = 0
                train_stage(model, env_a, 256, first=True)
                first_weights = model.policy.action_net.weight.detach().cpu().numpy().copy()
                self.assertEqual((model.num_timesteps, model._n_updates), (256, 10))
                train_stage(model, env_b, 256, first=False)
                self.assertEqual((model.num_timesteps, model._n_updates), (512, 20))
                self.assertEqual(id(model.policy), policy_id)
                self.assertEqual(id(model.policy.optimizer), optimizer_id)
                self.assertFalse(np.array_equal(
                    first_weights, model.policy.action_net.weight.detach().cpu().numpy()))
                self.assertEqual(model.get_env().envs[0].unwrapped.target_distribution,
                                 "local_b")
            finally:
                env_a.close()
                env_b.close()

    def test_existing_artifacts_are_rejected_without_touching_v2(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            models = root / "models"
            models.mkdir()
            old = models / "ppo_waypoint_reward_v2_100k.zip"
            new = models / "ppo_waypoint_curriculum_stage_a.zip"
            old.write_bytes(b"preserve")
            new.write_bytes(b"preserve new")
            with self.assertRaises(FileExistsError):
                require_fresh_curriculum_artifacts(models, root / "report.json",
                                                   root / "logs", root / "tensorboard")
            self.assertEqual(old.read_bytes(), b"preserve")
            self.assertEqual(new.read_bytes(), b"preserve new")


if __name__ == "__main__":
    unittest.main()
