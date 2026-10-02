"""Five-seed robustness must preserve PPO settings and target isolation."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_uav_env import MineUAVEnv  # noqa: E402
from reward_v2_alignment_audit import fixed_waypoints  # noqa: E402
from train_ppo_waypoint import SEED, make_ppo, make_training_envs  # noqa: E402
from train_ppo_lowstd_multiseed import (  # noqa: E402
    ALL_TRAIN_SEEDS, aggregate_numeric, build_holdout_waypoints, compact_evaluation,
    expected_training_targets, first_observed_thresholds,
    require_fresh_seed_outputs, seed_paths,
    seed_training_envs,
)
from summarize_ppo_lowstd_multiseed import aggregate_seed_rows  # noqa: E402


class MultiSeedTests(unittest.TestCase):
    def test_seed_zero_and_one_change_initial_weights_but_same_seed_reproduces(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            env = make_training_envs(root / "monitor", n_envs=1, reward_version="v2")
            try:
                models = [make_ppo(env, root / "tb", log_std_init=-2.0, seed=s)
                          for s in (0, 1, 0)]
                weights = [m.policy.action_net.weight.detach().clone() for m in models]
                self.assertTrue(torch.equal(weights[0], weights[2]))
                self.assertFalse(torch.equal(weights[0], weights[1]))
                self.assertEqual([m.seed for m in models], [0, 1, 0])
                self.assertEqual([m.policy.log_std_init for m in models], [-2.0] * 3)
                default = make_ppo(env, root / "tb")
                self.assertEqual(default.seed, SEED)
                self.assertEqual(default.policy.log_std_init, 0.0)
                for key in ("learning_rate", "gamma", "gae_lambda", "ent_coef",
                            "vf_coef", "max_grad_norm", "n_steps", "batch_size", "n_epochs"):
                    self.assertEqual(getattr(models[0], key), getattr(default, key), key)
            finally:
                env.close()

    def test_vector_env_rank_seeds_reproduce_first_full_targets(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            targets = []
            for run in range(2):
                env = make_training_envs(root / f"monitor_{run}", n_envs=2,
                                         reward_version="v2", target_distribution="full")
                try:
                    self.assertEqual(seed_training_envs(env, 0), [0, 1])
                    env.reset()
                    targets.append([wrapped.unwrapped.target_position.copy()
                                    for wrapped in env.envs])
                finally:
                    env.close()
            np.testing.assert_array_equal(targets[0], targets[1])
            self.assertFalse(np.array_equal(targets[0][0], targets[0][1]))

    def test_training_seed_blocks_do_not_overlap(self):
        self.assertEqual(ALL_TRAIN_SEEDS, (20260929, 0, 8, 16, 24))
        blocks = [set(range(seed, seed + 8)) for seed in ALL_TRAIN_SEEDS]
        for index, block in enumerate(blocks):
            self.assertTrue(all(block.isdisjoint(other) for other in blocks[index + 1:]))
        first_targets = [set(map(tuple, expected_training_targets(seed)))
                         for seed in ALL_TRAIN_SEEDS]
        for index, targets in enumerate(first_targets):
            self.assertTrue(all(targets.isdisjoint(other)
                                for other in first_targets[index + 1:]))

    def test_holdout_uses_independent_fixed_full_sampler_seeds(self):
        holdout = build_holdout_waypoints()
        benchmark = fixed_waypoints("full")
        self.assertEqual(len(holdout), 100)
        self.assertEqual([seed for seed, _ in holdout], list(range(20271001, 20271101)))
        self.assertFalse(set(seed for seed, _ in holdout) &
                         set(seed for seed, _ in benchmark))
        env = MineUAVEnv(reward_version="v2", target_distribution="full")
        try:
            _, info = env.reset(seed=20271001)
            self.assertEqual(holdout[0][1], info["target_position_m"].tolist())
        finally:
            env.close()
        self.assertEqual(holdout, build_holdout_waypoints())
        self.assertTrue(all(-2 <= target[0] <= 2 and -2 <= target[1] <= 2
                            and 0.7 <= target[2] <= 1.5 for _, target in holdout))

    def test_per_seed_paths_are_isolated_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            zero = seed_paths(0, root)
            one = seed_paths(1, root)
            self.assertNotEqual(zero["model_dir"], one["model_dir"])
            self.assertNotEqual(zero["report"], one["report"])
            zero["model_dir"].mkdir(parents=True)
            (zero["model_dir"] / "ppo_waypoint_v2_lowstd_20k.zip").touch()
            with self.assertRaises(FileExistsError):
                require_fresh_seed_outputs(zero)
            require_fresh_seed_outputs(one)

    def test_thresholds_are_first_observed_full_checkpoint_not_interpolated(self):
        milestones = {
            "20k": {"actual_timesteps": 20480,
                    "evaluation": {"full": {"task_metrics": {"success_rate": 0.20}}}},
            "50k": {"actual_timesteps": 51200,
                    "evaluation": {"full": {"task_metrics": {"success_rate": 0.60}}}},
            "100k": {"actual_timesteps": 100352,
                     "evaluation": {"full": {"task_metrics": {"success_rate": 1.0}}}},
        }
        self.assertEqual(first_observed_thresholds(milestones),
                         {"50pct": 51200, "90pct": 100352, "100pct": 100352})
        milestones["100k"]["evaluation"]["full"]["task_metrics"]["success_rate"] = 0.89
        self.assertEqual(first_observed_thresholds(milestones),
                         {"50pct": 51200, "90pct": None, "100pct": None})

    def test_aggregation_uses_sample_std_and_preserves_undefined_completion(self):
        aggregate = aggregate_numeric([0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(aggregate["count"], 5)
        self.assertAlmostEqual(aggregate["mean"], 0.5)
        self.assertAlmostEqual(aggregate["std"], math.sqrt(0.15625))
        self.assertEqual((aggregate["min"], aggregate["max"]), (0.0, 1.0))
        completion = aggregate_numeric([None, 4.0, 6.0, None, 8.0])
        self.assertEqual(completion["count"], 3)
        self.assertEqual(completion["missing_count"], 2)
        self.assertAlmostEqual(completion["mean"], 6.0)
        self.assertIsNone(aggregate_numeric([None] * 5)["mean"])
        self.assertIsNone(aggregate_numeric([None, 4.0, None, None, None])["std"])

    def test_compact_evaluation_keeps_success_and_near_target_diagnostics(self):
        evaluation = {
            "task_metrics": {
                "episodes": 100, "successes": 87, "success_rate": 0.87,
                "mean_final_distance_m": 0.12, "mean_episode_length": 190,
                "mean_successful_completion_time_s": 7.0,
                "ever_within_0_1_m_fraction": 0.9,
                "ever_distance_and_speed_fraction": 0.88,
            },
            "braking_diagnostic": {"near_target": {"lt_0_1": {
                "mean_actual_speed_m_s": 0.11,
                "mean_command_norm_m_s": 0.05,
                "mean_v_tangent_m_s": 0.02,
            }}},
            "crossing_after_first_0_2_m": {
                "before_episode_end_fraction_all_episodes": 0.04,
            },
            "executed_action_near_boundary_fraction_overall": 0.01,
        }
        summary = compact_evaluation(evaluation)
        self.assertEqual(summary["success_rate"], 0.87)
        self.assertEqual(summary["near_target_actual_speed_m_s"], 0.11)
        self.assertEqual(summary["near_target_command_speed_m_s"], 0.05)
        self.assertEqual(summary["near_target_tangential_speed_m_s"], 0.02)
        self.assertEqual(summary["crossing_rate"], 0.04)
        self.assertEqual(summary["action_saturation_fraction"], 0.01)

    def test_five_seed_group_statistics_keep_missing_success_times(self):
        def metrics(rate, distance, completion_time):
            return {
                "success_rate": rate,
                "mean_final_distance_m": distance,
                "mean_successful_completion_time_s": completion_time,
                "mean_episode_length": 190.0,
                "ever_within_0_1_m_fraction": rate,
                "ever_distance_and_speed_fraction": rate,
                "near_target_actual_speed_m_s": 0.1,
                "near_target_command_speed_m_s": 0.05,
                "near_target_tangential_speed_m_s": 0.02,
                "crossing_rate": 0.01,
                "action_saturation_fraction": 0.0,
            }

        rows = [{"fixed": metrics(rate, distance, completion_time),
                 "holdout": metrics(rate, distance, completion_time),
                 "first_observed_full_success_threshold_timesteps": {
                     "50pct": 20480 if rate >= 0.5 else None,
                     "90pct": 51200 if rate >= 0.9 else None,
                     "100pct": 100352 if rate >= 1.0 else None}}
                for rate, distance, completion_time in (
                    (1.0, 0.08, 7.0), (0.0, 0.5, None), (0.9, 0.1, 8.0),
                    (1.0, 0.09, 6.0), (0.5, 0.2, 10.0))]
        result = aggregate_seed_rows(rows)
        self.assertAlmostEqual(result["fixed"]["success_rate"]["mean"], 0.68)
        self.assertEqual(result["holdout"]["mean_successful_completion_time_s"]
                         ["missing_count"], 1)
        self.assertEqual(result["first_observed_full_success_threshold_timesteps"]
                         ["100pct"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
