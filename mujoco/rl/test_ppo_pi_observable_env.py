"""Only the live PI acceleration contribution may extend the PPO observation."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env


sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppo_pi_env import MineUAVPIEnv, make_pi_training_envs  # noqa: E402
from ppo_pi_observable_env import MineUAVPIObservableEnv  # noqa: E402


class ObservablePIEnvTests(unittest.TestCase):
    def test_only_integral_acceleration_is_appended_at_policy_boundary(self):
        old = MineUAVPIEnv(render_mode=None, reward_version="v2")
        new = MineUAVPIObservableEnv(render_mode=None, reward_version="v2")
        try:
            old_obs, _ = old.reset(seed=11, options={"target_position": [1, 0, 1]})
            new_obs, _ = new.reset(seed=11, options={"target_position": [1, 0, 1]})
            self.assertEqual(old.observation_space.shape, (7,))
            self.assertEqual(new.observation_space.shape, (10,))
            self.assertEqual(new_obs.dtype, np.float32)
            np.testing.assert_array_equal(new_obs[:7], old_obs)
            np.testing.assert_array_equal(new_obs[7:], [0, 0, 0])
            self.assertEqual(old.action_space, new.action_space)
            actions = ([0.5, -0.25, 0.3, 0.1], [-0.1, 0.2, 0.0, -0.2],
                       [0.2, 0.0, -0.1, 0.0])
            for raw in actions:
                action = np.asarray(raw, dtype=np.float32)
                old_result = old.step(action)
                new_result = new.step(action)
                np.testing.assert_array_equal(new_result[0][:7], old_result[0])
                np.testing.assert_allclose(
                    new_result[0][7:], new.velocity_controller.integral_acceleration_world,
                    rtol=0, atol=1e-7)
                self.assertEqual(new_result[1:4], old_result[1:4])
                self.assertEqual(new_result[4]["reward_breakdown"],
                                 old_result[4]["reward_breakdown"])
                np.testing.assert_array_equal(new.data.qpos, old.data.qpos)
                np.testing.assert_array_equal(new.data.qvel, old.data.qvel)
                np.testing.assert_array_equal(new.data.ctrl, old.data.ctrl)
            self.assertGreater(float(np.linalg.norm(new_obs := new_result[0][7:])), 0)
            reset_obs, _ = new.reset(seed=12)
            np.testing.assert_array_equal(reset_obs[7:], [0, 0, 0])
            np.testing.assert_array_equal(new.velocity_controller.integral_error,
                                          [0, 0, 0])
        finally:
            old.close()
            new.close()

    def test_nonfinite_plant_state_returns_finite_ten_vector(self):
        env = MineUAVPIObservableEnv(render_mode=None, reward_version="v2")
        try:
            env.reset(seed=21)
            env.data.qpos[0] = np.nan
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(4, dtype=np.float32))
            self.assertEqual(obs.shape, (10,))
            np.testing.assert_array_equal(obs, np.zeros(10, dtype=np.float32))
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["termination_reason"], "nonfinite_state")
        finally:
            env.close()

    def test_gymnasium_checker_and_eight_monitored_training_envs(self):
        env = MineUAVPIObservableEnv(render_mode=None, reward_version="v2")
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()
        with tempfile.TemporaryDirectory() as scratch:
            vec = make_pi_training_envs(Path(scratch) / "monitor", n_envs=8,
                                        env_type=MineUAVPIObservableEnv)
            try:
                obs = vec.reset()
                self.assertEqual(obs.shape, (8, 10))
                for wrapped in vec.envs:
                    self.assertIsInstance(wrapped.unwrapped, MineUAVPIObservableEnv)
                    self.assertEqual(wrapped.unwrapped.reward_version, "v2")
            finally:
                vec.close()


if __name__ == "__main__":
    unittest.main()
