"""The 10D observability run must change only observation and output names."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppo_pi_observable_env import MineUAVPIObservableEnv  # noqa: E402
from ppo_pi_env import make_pi_training_envs  # noqa: E402
from train_ppo_pi_lowstd import checkpoint_schedule, output_paths as old_paths  # noqa: E402
from train_ppo_pi_observable import (MODEL_PREFIX, output_paths,
                                     require_fresh_observable_outputs)  # noqa: E402
from train_ppo_waypoint import make_ppo  # noqa: E402


class ObservablePITrainingTests(unittest.TestCase):
    def test_paths_and_complete_rollout_checkpoints_are_separate(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            old = old_paths(root)
            new = output_paths(root)
            self.assertNotEqual(new["report"], old["report"])
            self.assertNotEqual(new["log_dir"], old["log_dir"])
            self.assertNotEqual(new["tensorboard_dir"], old["tensorboard_dir"])
            self.assertNotEqual(new["trace_dir"], old["trace_dir"])
            self.assertEqual(MODEL_PREFIX, "ppo_waypoint_pi_observable_lowstd")
            self.assertEqual([row["actual_timesteps"]
                              for row in checkpoint_schedule()],
                             [20_480, 51_200, 100_352])
            require_fresh_observable_outputs(root)
            new["model_dir"].mkdir(parents=True)
            (new["model_dir"] / f"{MODEL_PREFIX}_20k.zip").write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                require_fresh_observable_outputs(root)

    def test_real_eight_env_ppo_has_exact_fixed_settings_and_ten_inputs(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            vec = make_pi_training_envs(root / "monitor", n_envs=8,
                                        env_type=MineUAVPIObservableEnv)
            try:
                vec.seed(0)
                observation = vec.reset()
                self.assertEqual(observation.shape, (8, 10))
                model = make_ppo(vec, root / "tensorboard", log_std_init=-2.0,
                                 seed=0)
                self.assertEqual(model.observation_space.shape, (10,))
                self.assertEqual(model.action_space.shape, (4,))
                self.assertEqual(model.policy.net_arch,
                                 {"pi": [64, 64], "vf": [64, 64]})
                self.assertEqual(model.n_envs, 8)
                self.assertEqual(model.n_steps, 256)
                self.assertEqual(model.batch_size, 256)
                self.assertEqual(model.n_epochs, 10)
                self.assertEqual(model.learning_rate, 3e-4)
                self.assertEqual(model.gamma, 0.99)
                self.assertEqual(model.gae_lambda, 0.95)
                self.assertEqual(model.clip_range(1.0), 0.2)
                self.assertEqual(model.ent_coef, 0)
                self.assertEqual(model.vf_coef, 0.5)
                self.assertEqual(model.max_grad_norm, 0.5)
                self.assertEqual(model.policy.log_std_init, -2.0)
                np.testing.assert_allclose(
                    model.policy.log_std.detach().cpu().numpy(), [-2] * 4)
                self.assertEqual(model.device.type, "cpu")
            finally:
                vec.close()


if __name__ == "__main__":
    unittest.main()
