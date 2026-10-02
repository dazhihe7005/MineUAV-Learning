"""Fresh PPO/PI run must preserve fixed settings and existing experiments."""

import tempfile
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ppo_pi_lowstd import (LOG_STD_INIT, SEED, checkpoint_schedule,
                                 output_paths, require_fresh_outputs,
                                 run_training)  # noqa: E402
from ppo_pi_env import make_pi_training_envs  # noqa: E402
from train_ppo_waypoint import make_ppo  # noqa: E402


class PPOPITrainingTests(unittest.TestCase):
    def test_complete_rollout_checkpoint_schedule_is_exact(self):
        self.assertEqual(SEED, 0)
        self.assertEqual(LOG_STD_INIT, -2.0)
        self.assertEqual(checkpoint_schedule(), [
            {"label": "20k", "requested_timesteps": 20_000,
             "actual_timesteps": 20_480},
            {"label": "50k", "requested_timesteps": 50_000,
             "actual_timesteps": 51_200},
            {"label": "100k", "requested_timesteps": 100_000,
             "actual_timesteps": 100_352},
        ])

    def test_new_output_guard_blocks_checkpoint_and_report_collisions(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            model_dir = root / "models"
            log_dir = root / "logs"
            tb_dir = root / "tensorboard"
            traces = root / "traces"
            report = root / "report.json"
            require_fresh_outputs(model_dir, log_dir, tb_dir, traces, report)
            model_dir.mkdir()
            checkpoint = model_dir / "ppo_waypoint_pi_lowstd_20k.zip"
            checkpoint.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                require_fresh_outputs(model_dir, log_dir, tb_dir, traces, report)
            checkpoint.unlink()
            report.write_text("{}")
            with self.assertRaises(FileExistsError):
                require_fresh_outputs(model_dir, log_dir, tb_dir, traces, report)

    def test_real_model_construction_is_cpu_lowstd_and_64x64_per_head(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            vec = make_pi_training_envs(root / "monitor", n_envs=8)
            try:
                model = make_ppo(vec, root / "tensorboard", log_std_init=-2.0,
                                 seed=0)
                self.assertEqual(model.n_envs, 8)
                self.assertEqual(model.n_steps, 256)
                self.assertEqual(model.batch_size, 256)
                self.assertEqual(model.n_epochs, 10)
                self.assertEqual(model.device.type, "cpu")
                self.assertEqual(model.policy.log_std_init, -2.0)
                self.assertEqual(model.policy.net_arch,
                                 {"pi": [64, 64], "vf": [64, 64]})
                self.assertEqual(model.num_timesteps, 0)
            finally:
                vec.close()

    def test_training_entrypoint_rejects_existing_output_before_training(self):
        with tempfile.TemporaryDirectory() as scratch:
            report = output_paths(Path(scratch))["report"]
            report.parent.mkdir(parents=True)
            report.write_text("{}")
            with self.assertRaises(FileExistsError):
                run_training(output_root=Path(scratch))


if __name__ == "__main__":
    unittest.main()
