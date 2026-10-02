"""Reward V3 training and Viewer contracts without a long PPO run."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure


RL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RL_DIR))
from mine_uav_env import MineUAVEnv  # noqa: E402
from train_ppo_waypoint import make_ppo, make_training_envs  # noqa: E402
from train_ppo_reward_v3 import (RewardV3Callback, checkpoint_schedule,
                                 require_fresh_v3_artifacts,
                                 validated_prior_comparison)  # noqa: E402
from train_ppo_waypoint import TRAIN_METRICS  # noqa: E402


class RewardV3TrainingTests(unittest.TestCase):
    def test_checkpoint_refuses_pre_update_policy_then_saves_updated_policy(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            train_env = make_training_envs(root / "monitor", n_envs=8, reward_version="v3")
            eval_env = MineUAVEnv(max_episode_seconds=0.04, reward_version="v3")
            (root / "logs").mkdir()
            (root / "models").mkdir()
            try:
                model = make_ppo(train_env, root / "tensorboard")
                model.set_logger(configure(str(root / "logger"), []))
                model.num_timesteps = 20_480
                for metric in TRAIN_METRICS:
                    model.logger.record(f"train/{metric}", 0.1)
                callback = RewardV3Callback(eval_env, root / "models", root / "logs")
                callback.init_callback(model)
                with self.assertRaisesRegex(RuntimeError, "precedes PPO update"):
                    callback.record_updated_state()
                self.assertFalse((root / "models" / "ppo_waypoint_reward_v3_20k.zip").exists())
                model._n_updates = 100
                callback.record_updated_state()
                checkpoint = root / "models" / "ppo_waypoint_reward_v3_20k.zip"
                self.assertTrue(checkpoint.is_file())
                self.assertEqual(PPO.load(checkpoint, device="cpu").num_timesteps, 20_480)
                self.assertEqual(callback.milestones["20k"]["ppo_n_updates"], 100)
            finally:
                eval_env.close()
                train_env.close()

    def test_v3_schedule_uses_only_three_complete_updated_rollouts(self):
        self.assertEqual([entry["actual_timesteps"] for entry in checkpoint_schedule()],
                         [20_480, 51_200, 100_352])

    def test_v3_envs_are_independent_headless_instances(self):
        with tempfile.TemporaryDirectory() as scratch:
            envs = make_training_envs(Path(scratch), n_envs=8, reward_version="v3")
            try:
                self.assertEqual(envs.num_envs, 8)
                self.assertEqual(len({id(wrapper.unwrapped) for wrapper in envs.envs}), 8)
                self.assertTrue(all(wrapper.unwrapped.reward_version == "v3"
                                    and wrapper.unwrapped.render_mode is None
                                    for wrapper in envs.envs))
                model = make_ppo(envs, Path(scratch) / "tensorboard")
                self.assertEqual(model.n_steps * envs.num_envs, 2048)
                self.assertEqual(model.device.type, "cpu")
            finally:
                envs.close()

    def test_v3_existing_artifact_is_protected_without_touching_v2(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            model_dir = root / "models"
            model_dir.mkdir()
            v2 = model_dir / "ppo_waypoint_reward_v2_100k.zip"
            v2.write_bytes(b"keep-v2")
            v3 = model_dir / "ppo_waypoint_reward_v3_20k.zip"
            v3.write_bytes(b"keep-v3")
            with self.assertRaises(FileExistsError):
                require_fresh_v3_artifacts(model_dir, root / "report.json",
                                           root / "logs", root / "tensorboard")
            self.assertEqual(v2.read_bytes(), b"keep-v2")
            self.assertEqual(v3.read_bytes(), b"keep-v3")

    def test_comparison_rejects_different_targets_even_with_same_seed_count(self):
        with tempfile.TemporaryDirectory() as scratch:
            prior = Path(scratch) / "prior.json"
            seeds = list(range(20261001, 20261101))
            base = {"seeds": seeds, "episode_summaries": [
                {"seed": seed, "target_position_m": [0.0, 0.0, 1.0]} for seed in seeds]}
            v2 = {"seeds": seeds, "episode_summaries": [
                {"seed": seed, "target_position_m": [0.0, 0.0, 1.0]} for seed in seeds]}
            prior.write_text(json.dumps({"baseline_100k_reevaluation": base,
                                         "milestones": {"100k": {"evaluation": v2}}}),
                             encoding="utf-8")
            v3 = {"seeds": seeds, "episode_summaries": [
                {"seed": seed, "target_position_m": [0.1, 0.0, 1.0]} for seed in seeds]}
            with self.assertRaisesRegex(ValueError, "waypoint"):
                validated_prior_comparison(prior, v3)

    def test_comparison_rejects_episode_seed_mismatch(self):
        with tempfile.TemporaryDirectory() as scratch:
            prior = Path(scratch) / "prior.json"
            seeds = list(range(20261001, 20261101))
            correct = {"seeds": seeds, "episode_summaries": [
                {"seed": seed, "target_position_m": [0.0, 0.0, 1.0]}
                for seed in seeds]}
            prior.write_text(json.dumps({"baseline_100k_reevaluation": correct,
                                         "milestones": {"100k": {"evaluation": correct}}}),
                             encoding="utf-8")
            mislabeled = {"seeds": seeds, "episode_summaries": [
                {"seed": seed, "target_position_m": [0.0, 0.0, 1.0]}
                for seed in seeds]}
            mislabeled["episode_summaries"][0]["seed"] = seeds[1]
            with self.assertRaisesRegex(ValueError, "episode seed"):
                validated_prior_comparison(prior, mislabeled)

    def test_viewer_cli_accepts_reward_v3(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            env = make_training_envs(root / "monitor", n_envs=1, reward_version="v3")
            try:
                checkpoint = root / "candidate.zip"
                make_ppo(env, root / "tensorboard").save(checkpoint)
            finally:
                env.close()
            self.assertIsInstance(PPO.load(checkpoint, device="cpu"), PPO)
            command = [sys.executable, str(RL_DIR / "eval_ppo_waypoint.py"),
                       "--model", str(checkpoint), "--reward-version", "v3",
                       "--episodes", "1", "--seed", "17"]
            result = subprocess.run(command, cwd=RL_DIR.parents[1], text=True,
                                    capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["reward_version"], "v3")


if __name__ == "__main__":
    unittest.main()
