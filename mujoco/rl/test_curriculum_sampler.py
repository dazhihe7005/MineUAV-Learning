"""Reset-only curriculum targets must not alter the original flight task."""

import unittest

import numpy as np

from mine_uav_env import MineUAVEnv


class CurriculumSamplerTests(unittest.TestCase):
    def test_original_full_seed_target_is_unchanged(self):
        env = MineUAVEnv(reward_version="v2")
        try:
            observation, _ = env.reset(seed=20261001)
            np.testing.assert_allclose(env.target_position, [
                0.5636109068725292, 1.1732819373145285,
                0.7371474882999531], rtol=0, atol=1e-14)
            np.testing.assert_allclose(observation[:3], env.target_position - [0, 0, 1])
        finally:
            env.close()

    def test_local_a_and_b_sample_3d_shells_with_safe_altitude(self):
        for distribution, low, high in (("local_a", 0.25, 0.50),
                                        ("local_b", 0.50, 1.00)):
            with self.subTest(distribution=distribution):
                env = MineUAVEnv(reward_version="v2", target_distribution=distribution)
                try:
                    offsets = []
                    for seed in range(300):
                        env.reset(seed=seed)
                        offsets.append(env.target_position - [0, 0, 1])
                        self.assertGreaterEqual(env.target_position[2], 0.7)
                        self.assertLessEqual(env.target_position[2], 1.5)
                        np.testing.assert_allclose(env.data.qpos[:3], [0, 0, 1])
                        np.testing.assert_allclose(env.data.qpos[3:7], [1, 0, 0, 0])
                        np.testing.assert_allclose(env.data.qvel, 0)
                        self.assertEqual(env.target_yaw, 0.0)
                    distances = np.linalg.norm(offsets, axis=1)
                    self.assertTrue(np.all(distances >= low - 1e-12))
                    self.assertTrue(np.all(distances <= high + 1e-12))
                    self.assertGreater(np.std(np.asarray(offsets)[:, 2]), 0.05)
                finally:
                    env.close()

    def test_explicit_target_overrides_local_sampler_and_reward_is_v2(self):
        target = [0.9, -0.1, 1.2]
        envs = [MineUAVEnv(reward_version="v2", target_distribution=value)
                for value in ("full", "local_a")]
        try:
            for env in envs:
                env.reset(seed=42, options={"target_position": target})
                np.testing.assert_allclose(env.target_position, target)
            result_full = envs[0].step(np.zeros(4, dtype=np.float32))
            result_local = envs[1].step(np.zeros(4, dtype=np.float32))
            np.testing.assert_allclose(result_full[0], result_local[0])
            self.assertAlmostEqual(result_full[1], result_local[1])
            self.assertEqual(set(result_local[4]["reward_breakdown"]),
                             {"progress", "action", "brake", "success", "failure", "total"})
        finally:
            for env in envs:
                env.close()

    def test_invalid_distribution_is_rejected(self):
        with self.assertRaises(ValueError):
            MineUAVEnv(target_distribution="not-a-distribution")


if __name__ == "__main__":
    unittest.main()
