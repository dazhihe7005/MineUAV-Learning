"""PPO integration contracts; no training or desktop viewer in this suite."""

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mine_uav_env import MineUAVEnv
from test_env_scripted_policy import scripted_action


def training_module():
    try:
        return importlib.import_module("train_ppo_waypoint")
    except ImportError as error:
        raise AssertionError("PPO training entry point is missing") from error


class ScriptedModel:
    def predict(self, observation, deterministic=False):
        if not deterministic:
            raise AssertionError("Evaluation must be deterministic")
        return scripted_action(observation), None


class PpoWaypointTests(unittest.TestCase):
    def test_eight_distinct_headless_monitored_environments(self):
        module = training_module()
        with tempfile.TemporaryDirectory() as directory:
            env = module.make_training_envs(Path(directory), n_envs=8)
            try:
                self.assertIsInstance(env, DummyVecEnv)
                self.assertEqual(env.num_envs, 8)
                self.assertEqual(len({id(wrapper.unwrapped) for wrapper in env.envs}), 8)
                self.assertTrue(all(isinstance(wrapper, Monitor) for wrapper in env.envs))
                self.assertTrue(all(wrapper.unwrapped.render_mode is None for wrapper in env.envs))
                self.assertTrue(all(wrapper.unwrapped._viewer is None for wrapper in env.envs))
                self.assertEqual(env.reset().shape, (8, 7))
                observation, reward, done, info = env.step(np.zeros((8, 4), np.float32))
                self.assertEqual(observation.shape, (8, 7))
                self.assertTrue(np.isfinite(reward).all())
            finally:
                env.close()

    def test_ppo_cpu_network_and_rollout_configuration(self):
        module = training_module()
        with tempfile.TemporaryDirectory() as directory:
            env = module.make_training_envs(Path(directory) / "monitor", n_envs=8)
            try:
                model = module.make_ppo(env, Path(directory) / "tensorboard")
                self.assertIsInstance(model, PPO)
                self.assertEqual(model.device.type, "cpu")
                self.assertEqual(model.n_steps, 256)
                self.assertEqual(model.batch_size, 256)
                self.assertEqual(model.n_epochs, 10)
                self.assertEqual(model.n_steps * env.num_envs, 2048)
                self.assertEqual(model.learning_rate, 3e-4)
                self.assertEqual(model.gamma, 0.99)
                self.assertEqual(model.gae_lambda, 0.95)
                self.assertEqual(model.clip_range(1), 0.2)
                self.assertEqual(model.ent_coef, 0.0)
                self.assertEqual(model.vf_coef, 0.5)
                self.assertEqual(model.max_grad_norm, 0.5)
                extractor = model.policy.mlp_extractor
                self.assertEqual([layer.out_features for layer in extractor.policy_net
                                  if isinstance(layer, torch.nn.Linear)], [64, 64])
                self.assertEqual([layer.out_features for layer in extractor.value_net
                                  if isinstance(layer, torch.nn.Linear)], [64, 64])
                self.assertIsNot(extractor.policy_net[0], extractor.value_net[0])
            finally:
                env.close()

    def test_monitor_records_final_distance_and_reason(self):
        module = training_module()
        with tempfile.TemporaryDirectory() as directory:
            env = module.make_training_envs(Path(directory), n_envs=1)
            try:
                env.envs[0].reset(seed=7, options={"target_position": [0, 0, 1]})
                for _ in range(5):
                    _, _, done, info = env.step(np.zeros((1, 4), np.float32))
                self.assertTrue(done[0])
                self.assertEqual(info[0]["episode"]["termination_reason"], "success")
                self.assertLess(info[0]["episode"]["distance_m"], 0.1)
            finally:
                env.close()
            self.assertEqual(len(list(Path(directory).glob("*.monitor.csv"))), 1)

    def test_deterministic_evaluation_reports_waypoint_metrics(self):
        module = training_module()
        env = MineUAVEnv(render_mode=None)
        try:
            result = module.evaluate_waypoints(ScriptedModel(), env, episodes=2,
                                               seed=20260929)
            self.assertEqual(result["episodes"], 2)
            self.assertEqual(result["successes"], 2)
            self.assertLess(result["mean_final_distance_m"], 0.1)
            self.assertGreater(result["mean_episode_length"], 0)
            self.assertGreater(result["mean_completion_time_s"], 0)
        finally:
            env.close()

    def test_deterministic_evaluation_accepts_monitored_environment(self):
        module = training_module()
        env = Monitor(MineUAVEnv(render_mode=None),
                      info_keywords=("distance_m", "termination_reason"))
        try:
            result = module.evaluate_waypoints(ScriptedModel(), env, episodes=1,
                                               seed=20260929)
            self.assertEqual(result["successes"], 1)
        finally:
            env.close()

    def test_saved_policy_can_be_evaluated_headless(self):
        try:
            cli = importlib.import_module("eval_ppo_waypoint")
        except ImportError as error:
            self.fail(f"PPO checkpoint evaluation entry point is missing: {error}")
        module = training_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_env = module.make_training_envs(root / "monitor", n_envs=1)
            try:
                model = module.make_ppo(train_env, root / "tensorboard")
                checkpoint = root / "candidate.zip"
                model.save(checkpoint)
            finally:
                train_env.close()
            result = cli.run_ppo_evaluation(checkpoint, episodes=1,
                                            render_mode=None, seed=99)
            self.assertEqual(result["episodes"], 1)
            self.assertTrue(np.isfinite(result["mean_final_distance_m"]))


if __name__ == "__main__":
    unittest.main()
