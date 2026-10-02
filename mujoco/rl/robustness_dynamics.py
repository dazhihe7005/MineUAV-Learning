"""Runtime-only, single-factor dynamics mismatches for frozen PPO evaluation.

The source MJCF, controller, allocator and policy remain nominal. In the
optional motor-lag case, controller output u_cmd = omega_cmd**2 is held for
10 ms; omega_actual is integrated at each 2 ms physics step and the original
rotor actuator receives u_actual = omega_actual**2. Thus the same actual
input produces thrust and yaw reaction torque through the original site gear.
"""

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from mine_uav_env import MineUAVEnv


@dataclass(frozen=True)
class Scenario:
    kind: str
    value: float

    def __post_init__(self) -> None:
        if self.kind not in {"nominal", "mass", "inertia", "thrust",
                             "motor_lag_s", "force_x_n"}:
            raise ValueError(f"Unknown robustness factor: {self.kind}")
        if not math.isfinite(self.value):
            raise ValueError("Robustness value must be finite")
        if self.kind in {"mass", "inertia", "thrust"} and self.value <= 0:
            raise ValueError(f"{self.kind} factor must be positive")
        if self.kind == "motor_lag_s" and self.value < 0:
            raise ValueError("Motor lag must be nonnegative")
        if self.kind == "nominal" and self.value != 0:
            raise ValueError("Nominal scenario value must be zero")


class RobustnessEnv(MineUAVEnv):
    """A private compiled model with one disturbance; never edits XML files."""

    def __init__(self, scenario: Scenario, *, max_episode_seconds: float = 15.0):
        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        self.scenario = Scenario("nominal", 0.0)
        super().__init__(max_episode_seconds=max_episode_seconds,
                         render_mode=None, reward_version="v2",
                         target_distribution="full")
        self.uav_body_id = self.model.body("mine_uav").id
        self.omega_command_rad_s = np.zeros(4, dtype=float)
        self.omega_actual_rad_s = np.zeros(4, dtype=float)
        self.activate_scenario(scenario)

    def activate_scenario(self, scenario: Scenario) -> None:
        """Apply one plant-only mismatch, including after a nominal hover warm-up."""
        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        if self.scenario.kind != "nominal" or self.scenario.value != 0:
            raise ValueError("A plant mismatch is already active")
        if scenario.kind == "mass":
            self.model.body_mass[self.uav_body_id] *= scenario.value
        elif scenario.kind == "inertia":
            # MuJoCo stores principal moments in body_inertia; inertial-frame
            # orientation stays fixed, so this scales the full 3x3 tensor.
            self.model.body_inertia[self.uav_body_id] *= scenario.value
        elif scenario.kind == "thrust":
            # Only the site gear's translational +Z entry represents k_f.
            # Rotational +Z entries (s_i*k_m) are left untouched.
            self.model.actuator_gear[:, 2] *= scenario.value
        if scenario.kind in {"mass", "inertia", "thrust"}:
            # mj_setConst evaluates the reference qpos0 and overwrites data.qpos.
            # Preserve integration state when injecting after a settled hover.
            state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(self.model, state_spec))
            mujoco.mj_getState(self.model, self.data, state, state_spec)
            mujoco.mj_setConst(self.model, self.data)
            mujoco.mj_setState(self.model, self.data, state, state_spec)
            mujoco.mj_forward(self.model, self.data)
        elif scenario.kind == "force_x_n":
            self.data.xfrc_applied[self.uav_body_id, :3] = [scenario.value, 0, 0]
            self.data.xfrc_applied[self.uav_body_id, 3:] = 0.0
        self.scenario = scenario

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self.omega_command_rad_s.fill(0.0)
        self.omega_actual_rad_s.fill(0.0)
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        if self.scenario.kind == "force_x_n":
            self.data.xfrc_applied[self.uav_body_id, 0] = self.scenario.value
        return observation, info

    def _apply_rotor_command(self, command_u: np.ndarray) -> None:
        if self.scenario.kind != "motor_lag_s" or self.scenario.value == 0:
            super()._apply_rotor_command(command_u)
            return
        bounded_u = np.clip(np.asarray(command_u, dtype=float), 0, self.allocator.u_max)
        if bounded_u.shape != (4,) or not np.isfinite(bounded_u).all():
            raise ValueError("Motor command must be four finite squared speeds")
        self.omega_command_rad_s[:] = np.sqrt(bounded_u)

    def _before_physics_step(self) -> None:
        if self.scenario.kind == "motor_lag_s" and self.scenario.value > 0:
            decay = math.exp(-self.physics_dt / self.scenario.value)
            self.omega_actual_rad_s[:] = (self.omega_command_rad_s
                                          + (self.omega_actual_rad_s
                                             - self.omega_command_rad_s) * decay)
            self.data.ctrl[:] = self.omega_actual_rad_s ** 2
        if self.scenario.kind == "force_x_n":
            # xfrc_applied is a world-frame Cartesian wrench at body COM.
            self.data.xfrc_applied[self.uav_body_id, :3] = [self.scenario.value, 0, 0]
            self.data.xfrc_applied[self.uav_body_id, 3:] = 0.0
