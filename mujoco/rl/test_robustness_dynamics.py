"""Runtime-only model mismatch must preserve the frozen nominal controller."""

import sys
import unittest
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_uav_env import MineUAVEnv  # noqa: E402


def robustness_types():
    try:
        from robustness_dynamics import RobustnessEnv, Scenario
    except ImportError as error:
        raise AssertionError("Robustness dynamics implementation is missing") from error
    return RobustnessEnv, Scenario


class EnvironmentPhysicsHookTests(unittest.TestCase):
    def test_rotor_hook_can_replace_physical_u_without_changing_allocator(self):
        """Catches bypassing the hook and writing allocated u directly to ctrl."""
        class ZeroRotorEnv(MineUAVEnv):
            last_command_u = None

            def _apply_rotor_command(self, command_u):
                self.last_command_u = np.asarray(command_u, dtype=float).copy()
                self.data.ctrl[:] = 0.0

        env = ZeroRotorEnv(reward_version="v2")
        try:
            env.reset(seed=20271001)
            env.step(np.zeros(4, dtype=np.float32))
            self.assertIsNotNone(env.last_command_u)
            self.assertGreater(float(np.linalg.norm(env.last_command_u)), 0.0)
            np.testing.assert_array_equal(env.data.ctrl, np.zeros(4))
        finally:
            env.close()


class RobustnessDynamicsTests(unittest.TestCase):
    def test_nominal_scenario_matches_frozen_environment_step_for_step(self):
        """Catches altered controller/physics behavior in the nominal audit."""
        RobustnessEnv, Scenario = robustness_types()
        baseline = MineUAVEnv(reward_version="v2")
        audited = RobustnessEnv(Scenario("nominal", 0.0))
        try:
            left, _ = baseline.reset(seed=20271001)
            right, _ = audited.reset(seed=20271001)
            np.testing.assert_array_equal(left, right)
            for action in (np.zeros(4), np.array([0.2, -0.1, 0.1, 0]),
                           np.array([-0.3, 0.2, 0, 0.1])):
                left, reward_left, term_left, trunc_left, _ = baseline.step(action)
                right, reward_right, term_right, trunc_right, _ = audited.step(action)
                np.testing.assert_array_equal(left, right)
                np.testing.assert_array_equal(baseline.data.qpos, audited.data.qpos)
                np.testing.assert_array_equal(baseline.data.qvel, audited.data.qvel)
                np.testing.assert_array_equal(baseline.data.ctrl, audited.data.ctrl)
                self.assertEqual((reward_left, term_left, trunc_left),
                                 (reward_right, term_right, trunc_right))
        finally:
            baseline.close()
            audited.close()

    def test_mass_override_changes_only_mass_and_known_force_acceleration(self):
        """Catches inadvertently scaling inertia or retuning nominal control."""
        RobustnessEnv, Scenario = robustness_types()
        env = RobustnessEnv(Scenario("mass", 0.8))
        try:
            body = env.model.body("mine_uav").id
            self.assertAlmostEqual(env.model.body_mass[body], 5.6)
            nominal = MineUAVEnv(reward_version="v2")
            try:
                np.testing.assert_array_equal(env.model.body_inertia[body],
                                              nominal.model.body_inertia[body])
                np.testing.assert_array_equal(env.allocator.B, nominal.allocator.B)
            finally:
                nominal.close()
            self.assertEqual(env.velocity_controller.attitude_controller.mass_kg, 7.0)
            env.reset(seed=20271001)
            env.data.xfrc_applied[body, 0] = 7.0
            mujoco.mj_forward(env.model, env.data)
            self.assertAlmostEqual(env.data.qacc[0], 1.25, places=8)
        finally:
            env.close()

    def test_inertia_override_scales_full_tensor_not_mass(self):
        """Catches scaling just diagonal body-frame terms incorrectly."""
        RobustnessEnv, Scenario = robustness_types()
        nominal = MineUAVEnv(reward_version="v2")
        env = RobustnessEnv(Scenario("inertia", 1.2))
        try:
            body = env.model.body("mine_uav").id

            def full_tensor(model):
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
                frame = rotation.reshape(3, 3)
                return frame @ np.diag(model.body_inertia[body]) @ frame.T

            np.testing.assert_allclose(full_tensor(env.model),
                                       full_tensor(nominal.model) * 1.2,
                                       rtol=1e-12, atol=1e-12)
            self.assertEqual(env.model.body_mass[body], nominal.model.body_mass[body])
            np.testing.assert_array_equal(env.model.actuator_gear, nominal.model.actuator_gear)
            np.testing.assert_array_equal(
                env.velocity_controller.attitude_controller.inertia_kg_m2,
                nominal.velocity_controller.attitude_controller.inertia_kg_m2)
        finally:
            nominal.close()
            env.close()

    def test_thrust_override_scales_only_four_force_gears(self):
        """Catches changing yaw coefficient or allocating with perturbed B."""
        RobustnessEnv, Scenario = robustness_types()
        nominal = MineUAVEnv(reward_version="v2")
        env = RobustnessEnv(Scenario("thrust", 0.8))
        try:
            np.testing.assert_allclose(env.model.actuator_gear[:, 2],
                                       nominal.model.actuator_gear[:, 2] * 0.8,
                                       rtol=0, atol=1e-15)
            np.testing.assert_array_equal(env.model.actuator_gear[:, [0, 1, 3, 4, 5]],
                                          nominal.model.actuator_gear[:, [0, 1, 3, 4, 5]])
            np.testing.assert_array_equal(env.allocator.B, nominal.allocator.B)
            env.reset(seed=20271001)
            nominal.reset(seed=20271001)
            env.data.ctrl[:] = [10000, 0, 0, 0]
            nominal.data.ctrl[:] = env.data.ctrl
            mujoco.mj_forward(env.model, env.data)
            mujoco.mj_forward(nominal.model, nominal.data)
            self.assertAlmostEqual(env.data.qfrc_actuator[2] /
                                   nominal.data.qfrc_actuator[2], 0.8, places=9)
            self.assertAlmostEqual(env.data.qfrc_actuator[5],
                                   nominal.data.qfrc_actuator[5], places=9)
        finally:
            nominal.close()
            env.close()

    def test_motor_lag_integrates_omega_not_omega_squared_and_resets(self):
        """Catches filtering u directly or retaining hidden motor state."""
        RobustnessEnv, Scenario = robustness_types()
        env = RobustnessEnv(Scenario("motor_lag_s", 0.020))
        try:
            env.reset(seed=20271001)
            env._apply_rotor_command(np.array([10000.0, 40000.0, 0.0, 0.0]))
            np.testing.assert_array_equal(env.omega_command_rad_s, [100, 200, 0, 0])
            np.testing.assert_array_equal(env.omega_actual_rad_s, [0, 0, 0, 0])
            env._before_physics_step()
            fraction = 1 - np.exp(-0.002 / 0.020)
            np.testing.assert_allclose(env.omega_actual_rad_s,
                                       [100 * fraction, 200 * fraction, 0, 0], rtol=1e-14)
            np.testing.assert_allclose(env.data.ctrl, env.omega_actual_rad_s ** 2,
                                       rtol=1e-14)
            env.reset(seed=20271001)
            np.testing.assert_array_equal(env.omega_command_rad_s, np.zeros(4))
            np.testing.assert_array_equal(env.omega_actual_rad_s, np.zeros(4))
            np.testing.assert_array_equal(env.data.ctrl, np.zeros(4))
        finally:
            env.close()

    def test_world_x_force_is_at_com_and_stays_world_x_at_yaw_90(self):
        """Catches injecting body-X force or a spurious torque."""
        RobustnessEnv, Scenario = robustness_types()
        env = RobustnessEnv(Scenario("force_x_n", 10.0))
        try:
            body = env.model.body("mine_uav").id
            for quaternion in ([1, 0, 0, 0],
                               [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]):
                env.reset(seed=20271001)
                env.data.qpos[3:7] = quaternion
                env._before_physics_step()
                np.testing.assert_array_equal(env.data.xfrc_applied[body],
                                              [10, 0, 0, 0, 0, 0])
                mujoco.mj_forward(env.model, env.data)
                self.assertAlmostEqual(env.data.qacc[0], 10 / 7, places=8)
                self.assertAlmostEqual(env.data.qacc[1], 0, places=8)
            env.reset(seed=20271001)
            np.testing.assert_array_equal(env.data.xfrc_applied[body],
                                          [10, 0, 0, 0, 0, 0])
        finally:
            env.close()

    def test_invalid_scenarios_are_rejected_before_model_mutation(self):
        """Catches nonphysical factor/tau and NaN entering an audit."""
        _, Scenario = robustness_types()
        for kind, value in (("mass", 0), ("inertia", -1), ("thrust", np.nan),
                            ("motor_lag_s", -0.001), ("force_x_n", np.inf),
                            ("unknown", 1.0)):
            with self.subTest(kind=kind, value=value), self.assertRaises(ValueError):
                Scenario(kind, value)

    def test_physics_hook_runs_once_for_each_500hz_step(self):
        """Catches a lag/force hook accidentally running only at controller rate."""
        class CountingEnv(MineUAVEnv):
            calls = 0

            def _before_physics_step(self):
                self.calls += 1

        env = CountingEnv(reward_version="v2")
        try:
            env.reset(seed=20271001)
            _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
            self.assertEqual(info["physics_steps"], 20)
            self.assertEqual(env.calls, 20)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
