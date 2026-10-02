"""Long-baseline contracts without performing PPO gradient updates."""

import importlib
import json
import math
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from mine_uav_env import MineUAVEnv
from test_env_scripted_policy import scripted_action
from train_ppo_waypoint import TRAIN_METRICS, make_ppo, make_training_envs


def baseline_module():
    try:
        return importlib.import_module("train_ppo_baseline")
    except ImportError as error:
        raise AssertionError("Long PPO baseline entry point is missing") from error


class ScriptedModel:
    def predict(self, observation, deterministic=False):
        if not deterministic:
            raise AssertionError("Evaluation must use deterministic actions")
        return scripted_action(observation), None


class LongBaselineTests(unittest.TestCase):
    def test_milestone_steps_use_complete_rollouts_without_exceeding_200k(self):
        schedule = baseline_module().checkpoint_schedule()
        self.assertEqual(schedule, [
            {"label": "20k", "requested_timesteps": 20000, "actual_timesteps": 20480},
            {"label": "50k", "requested_timesteps": 50000, "actual_timesteps": 51200},
            {"label": "100k", "requested_timesteps": 100000, "actual_timesteps": 100352},
            {"label": "200k", "requested_timesteps": 200000, "actual_timesteps": 198656},
        ])

    def test_fixed_seed_evaluation_reports_success_reward_and_action_behavior(self):
        module = baseline_module()
        env = MineUAVEnv(render_mode=None)
        try:
            result = module.evaluate_baseline(ScriptedModel(), env,
                                              episodes=2, seed=20261001)
            self.assertEqual(result["episodes"], 2)
            self.assertEqual(result["seeds"], [20261001, 20261002])
            self.assertEqual(result["successes"], 2)
            self.assertEqual(result["success_rate"], 1.0)
            self.assertLess(result["max_final_distance_m"], 0.1)
            self.assertTrue(math.isfinite(result["mean_reward"]))
            self.assertGreater(result["mean_completion_time_s"], 0)
            self.assertEqual(len(result["episode_summaries"]), 2)
            self.assertTrue(all(item["max_success_streak"] >= 5
                                for item in result["episode_summaries"]))
            self.assertEqual(len(result["actions"]["max_abs_per_axis"]), 4)
            self.assertTrue(all(0 <= x <= 1 for x in result["actions"]["near_boundary_fraction_per_axis"]))
            self.assertTrue(all(np.isfinite(x) for x in result["actions"]["max_abs_per_axis"]))
        finally:
            env.close()

    def test_milestone_checkpoint_waits_for_ppo_update(self):
        module = baseline_module()
        self.assertTrue(hasattr(module, "BaselineCallback"), "missing milestone callback")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_env = make_training_envs(root / "monitors", n_envs=8)
            eval_env = MineUAVEnv(render_mode=None)
            try:
                model = make_ppo(train_env, root / "tensorboard")
                model.set_logger(configure(str(root / "logger"), ["stdout"]))
                model.num_timesteps = 20480
                model.ep_info_buffer = deque([{"r": 1.5, "l": 123}], maxlen=100)
                for metric in TRAIN_METRICS:
                    model.logger.record(f"train/{metric}", 0.1)
                callback = module.BaselineCallback(eval_env, root / "models", root / "logs",
                                                   eval_episodes=1)
                callback.init_callback(model)
                with self.assertRaisesRegex(RuntimeError, "before PPO update"):
                    callback._on_rollout_start()
                self.assertFalse((root / "models" / "ppo_waypoint_20k.zip").exists())
                model._n_updates = 100
                callback._on_rollout_start()
                checkpoint = root / "models" / "ppo_waypoint_20k.zip"
                self.assertTrue(checkpoint.is_file())
                self.assertEqual(PPO.load(checkpoint, device="cpu").num_timesteps, 20480)
                self.assertEqual(callback.milestone_results["20k"]["evaluation"]["episodes"], 1)
            finally:
                eval_env.close()
                train_env.close()

    def test_existing_baseline_checkpoint_is_preserved(self):
        module = baseline_module()
        self.assertTrue(hasattr(module, "require_fresh_baseline_artifacts"),
                        "missing artifact protection")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            existing = root / "ppo_waypoint_20k.zip"
            existing.write_bytes(b"original checkpoint")
            with self.assertRaises(FileExistsError):
                module.require_fresh_baseline_artifacts(root)
            self.assertEqual(existing.read_bytes(), b"original checkpoint")

    def test_training_entry_point_refuses_existing_checkpoint_before_start(self):
        module = baseline_module()
        self.assertTrue(hasattr(module, "run_baseline_training"),
                        "missing from-scratch baseline entry point")
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            checkpoint = model_dir / "ppo_waypoint_20k.zip"
            checkpoint.write_bytes(b"keep me")
            with patch.object(module, "MODEL_DIR", model_dir):
                with self.assertRaises(FileExistsError):
                    module.run_baseline_training()
            self.assertEqual(checkpoint.read_bytes(), b"keep me")

    def test_four_training_and_evaluation_curves_are_generated(self):
        try:
            plots = importlib.import_module("plot_ppo_baseline")
        except ImportError as error:
            self.fail(f"PPO baseline plotting entry point is missing: {error}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            report.write_text(json.dumps({
                "training_history": [
                    {"timesteps": 2048, "rollout_ep_rew_mean": -10,
                     "rollout_ep_len_mean": 375},
                    {"timesteps": 4096, "rollout_ep_rew_mean": 2,
                     "rollout_ep_len_mean": 300},
                ],
                "milestones": {
                    label: {"actual_timesteps": step,
                            "evaluation": {"success_rate": rate,
                                           "mean_final_distance_m": distance}}
                    for label, step, rate, distance in (
                        ("20k", 20480, 0.0, 1.0), ("50k", 51200, 0.1, 0.8),
                        ("100k", 100352, 0.2, 0.5), ("200k", 198656, 0.4, 0.2))
                },
            }), encoding="utf-8")
            created = plots.plot_report(report, root)
            self.assertEqual({path.name for path in created}, {
                "ppo_training_reward.png", "ppo_training_episode_length.png",
                "ppo_eval_success_rate.png", "ppo_eval_final_distance.png",
            })
            self.assertTrue(all(path.stat().st_size > 1000 for path in created))


if __name__ == "__main__":
    unittest.main()
