"""Reward V2 training must preserve PPO setup and original artifacts."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ppo_waypoint import make_training_envs  # noqa: E402
from train_ppo_reward_v2 import (RewardV2Callback, checkpoint_schedule,
                                 require_fresh_v2_artifacts)  # noqa: E402
from eval_ppo_waypoint import run_ppo_evaluation  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from train_ppo_waypoint import make_ppo  # noqa: E402


class RewardV2TrainingTests(unittest.TestCase):
    def test_undefined_near_target_speed_is_not_logged_as_zero(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            callback = RewardV2Callback(None, root / "models", root / "logs")
            callback.model = SimpleNamespace(logger=configure(str(root / "logger"), []))
            callback.record_evaluation_metrics({
                "success_rate": 0.0,
                "mean_final_distance_m": 0.4,
                "mean_speed_within_0_1_m_s": None,
                "speed_samples_within_0_1_m": 0,
            })
            self.assertNotIn("eval/mean_speed_within_0_1_m_s", callback.logger.name_to_value)
            self.assertEqual(callback.logger.name_to_value["eval/speed_samples_within_0_1_m"], 0)
            callback.record_evaluation_metrics({
                "success_rate": 0.1,
                "mean_final_distance_m": 0.2,
                "mean_speed_within_0_1_m_s": 0.79,
                "speed_samples_within_0_1_m": 123,
            })
            self.assertEqual(callback.logger.name_to_value["eval/mean_speed_within_0_1_m_s"], 0.79)

    def test_milestones_use_complete_rollouts(self):
        schedule = checkpoint_schedule()
        self.assertEqual([entry["label"] for entry in schedule], ["20k", "50k", "100k"])
        self.assertEqual([entry["actual_timesteps"] for entry in schedule],
                         [20_480, 51_200, 100_352])

    def test_training_env_selection_is_explicit_and_defaults_v1(self):
        with tempfile.TemporaryDirectory() as scratch:
            v1 = make_training_envs(Path(scratch) / "v1", n_envs=1)
            v2 = make_training_envs(Path(scratch) / "v2", n_envs=1, reward_version="v2")
            try:
                self.assertEqual(v1.envs[0].unwrapped.reward_version, "v1")
                self.assertEqual(v2.envs[0].unwrapped.reward_version, "v2")
                self.assertEqual(v1.observation_space, v2.observation_space)
                self.assertEqual(v1.action_space, v2.action_space)
            finally:
                v1.close()
                v2.close()

    def test_existing_v2_checkpoint_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            model_dir = root / "models"
            model_dir.mkdir()
            (model_dir / "ppo_waypoint_reward_v2_20k.zip").touch()
            with self.assertRaises(FileExistsError):
                require_fresh_v2_artifacts(model_dir, root / "report.json",
                                           root / "logs", root / "tensorboard")

    def test_viewer_evaluator_selects_v2_reward_without_changing_default(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            env = make_training_envs(root / "monitor", n_envs=1)
            try:
                model = make_ppo(env, root / "tensorboard")
                checkpoint = root / "candidate.zip"
                model.save(checkpoint)
            finally:
                env.close()
            self.assertIsInstance(PPO.load(checkpoint, device="cpu"), PPO)
            result = run_ppo_evaluation(checkpoint, episodes=1, seed=17,
                                        reward_version="v2")
            self.assertEqual(result["reward_version"], "v2")


if __name__ == "__main__":
    unittest.main()
