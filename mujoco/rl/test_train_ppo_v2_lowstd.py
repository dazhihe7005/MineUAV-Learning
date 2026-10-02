"""Low-std PPO must be a single-variable, fresh Reward V2 experiment."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ppo_waypoint import make_ppo, make_training_envs  # noqa: E402
from train_ppo_v2_lowstd import (  # noqa: E402
    checkpoint_schedule, read_policy_std, require_fresh_outputs,
)


class LowStdPPOTests(unittest.TestCase):
    def test_only_log_std_initialization_differs_from_v2_factory(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            envs = make_training_envs(root / "monitor", n_envs=1, reward_version="v2")
            try:
                default = make_ppo(envs, root / "default_tensorboard")
                lowstd = make_ppo(envs, root / "lowstd_tensorboard", log_std_init=-2.0)
                self.assertEqual(default.policy_kwargs,
                                 {"net_arch": {"pi": [64, 64], "vf": [64, 64]}})
                self.assertEqual(lowstd.policy_kwargs,
                                 {"net_arch": {"pi": [64, 64], "vf": [64, 64]},
                                  "log_std_init": -2.0})
                self.assertEqual(default.policy.action_dist.__class__,
                                 lowstd.policy.action_dist.__class__)
                self.assertEqual(default.policy.log_std.detach().tolist(), [0.0] * 4)
                reading = read_policy_std(lowstd)
                self.assertEqual(reading["log_std"], [-2.0] * 4)
                np.testing.assert_allclose(reading["std"], [math.exp(-2)] * 4,
                                           rtol=1e-6)
                for key, value in default.policy.state_dict().items():
                    if key != "log_std":
                        self.assertTrue(torch.equal(value, lowstd.policy.state_dict()[key]), key)
                for key in ("learning_rate", "gamma", "gae_lambda", "ent_coef", "vf_coef",
                            "max_grad_norm", "n_steps", "batch_size", "n_epochs", "seed"):
                    self.assertEqual(getattr(default, key), getattr(lowstd, key), key)
                self.assertEqual(default.clip_range(1.0), lowstd.clip_range(1.0))
            finally:
                envs.close()

    def test_schedule_uses_original_full_rollout_boundaries(self):
        self.assertEqual([(x["label"], x["actual_timesteps"])
                          for x in checkpoint_schedule()],
                         [("20k", 20480), ("50k", 51200), ("100k", 100352)])

    def test_existing_lowstd_artifact_blocks_overwrite(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "models").mkdir()
            (root / "models" / "ppo_waypoint_v2_lowstd_20k.zip").touch()
            with self.assertRaises(FileExistsError):
                require_fresh_outputs(root / "models", root / "report.json",
                                      root / "logs", root / "tensorboard")


if __name__ == "__main__":
    unittest.main()
