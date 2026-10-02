"""V4-specific training isolation and checkpoint contracts without training."""

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
from train_ppo_waypoint import N_ENVS, N_STEPS, TRAIN_METRICS, make_ppo, make_training_envs  # noqa: E402
from train_ppo_reward_v4 import (RewardV4Callback, checkpoint_schedule,
                                 require_fresh_v4_artifacts)  # noqa: E402


class RewardV4TrainingTests(unittest.TestCase):
    def test_schedule_ends_on_complete_100k_rollout(self):
        self.assertEqual([entry["actual_timesteps"] for entry in checkpoint_schedule()],
                         [20_480, 51_200, 100_352])

    def test_eight_independent_headless_envs_keep_fixed_ppo_configuration(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            envs = make_training_envs(root / "monitor", n_envs=N_ENVS,
                                      reward_version="v4")
            try:
                self.assertEqual(N_ENVS, 8)
                self.assertEqual(len({id(env.unwrapped) for env in envs.envs}), 8)
                self.assertTrue(all(env.unwrapped.reward_version == "v4" and
                                    env.unwrapped.render_mode is None for env in envs.envs))
                model = make_ppo(envs, root / "tensorboard")
                self.assertEqual(model.n_steps, N_STEPS)
                self.assertEqual(model.n_steps * envs.num_envs, 2048)
                self.assertEqual(model.device.type, "cpu")
                self.assertEqual(model.batch_size, 256)
                self.assertEqual(model.n_epochs, 10)
                self.assertEqual(model.policy.mlp_extractor.policy_net[0].out_features, 64)
                self.assertEqual(model.policy.mlp_extractor.value_net[0].out_features, 64)
            finally:
                envs.close()

    def test_existing_v4_artifact_is_rejected_without_touching_v2_or_v3(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            models = root / "models"
            models.mkdir()
            v2 = models / "ppo_waypoint_reward_v2_100k.zip"
            v3 = models / "ppo_waypoint_reward_v3_100k.zip"
            v4 = models / "ppo_waypoint_reward_v4_20k.zip"
            v2.write_bytes(b"v2-keep")
            v3.write_bytes(b"v3-keep")
            v4.write_bytes(b"v4-keep")
            with self.assertRaises(FileExistsError):
                require_fresh_v4_artifacts(models, root / "report.json",
                                           root / "logs", root / "tensorboard")
            self.assertEqual(v2.read_bytes(), b"v2-keep")
            self.assertEqual(v3.read_bytes(), b"v3-keep")
            self.assertEqual(v4.read_bytes(), b"v4-keep")

    def test_checkpoint_refuses_preupdate_then_saves_updated_policy(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            train_env = make_training_envs(root / "monitor", n_envs=8,
                                           reward_version="v4")
            eval_env = MineUAVEnv(max_episode_seconds=0.04, reward_version="v4")
            (root / "logs").mkdir()
            (root / "models").mkdir()
            try:
                model = make_ppo(train_env, root / "tensorboard")
                model.set_logger(configure(str(root / "logger"), []))
                model.num_timesteps = 20_480
                for metric in TRAIN_METRICS:
                    model.logger.record(f"train/{metric}", 0.1)
                callback = RewardV4Callback(eval_env, root / "models", root / "logs")
                callback.init_callback(model)
                with self.assertRaisesRegex(RuntimeError, "precedes PPO update"):
                    callback.record_updated_state()
                self.assertFalse((root / "models" / "ppo_waypoint_reward_v4_20k.zip").exists())
                model._n_updates = 100
                callback.record_updated_state()
                checkpoint = root / "models" / "ppo_waypoint_reward_v4_20k.zip"
                self.assertTrue(checkpoint.is_file())
                self.assertEqual(PPO.load(checkpoint, device="cpu").num_timesteps, 20_480)
                self.assertEqual(callback.milestones["20k"]["ppo_n_updates"], 100)
                self.assertEqual(callback.milestones["20k"]["evaluation"]["episodes"], 100)
            finally:
                eval_env.close()
                train_env.close()

    def test_evaluator_cli_accepts_v4_checkpoint(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            env = make_training_envs(root / "monitor", n_envs=1, reward_version="v4")
            try:
                checkpoint = root / "candidate.zip"
                make_ppo(env, root / "tensorboard").save(checkpoint)
            finally:
                env.close()
            command = [sys.executable, str(RL_DIR / "eval_ppo_waypoint.py"),
                       "--model", str(checkpoint), "--reward-version", "v4",
                       "--episodes", "1", "--seed", "17"]
            result = subprocess.run(command, cwd=RL_DIR.parents[1], text=True,
                                    capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["reward_version"], "v4")


if __name__ == "__main__":
    unittest.main()
