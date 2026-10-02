"""Minimal waypoint Gymnasium environment over the MineUAV v2 controller stack.

Policy action: normalized world-frame [vx, vy, vz] and yaw-rate command.
Physics 500 Hz, low-level controller 100 Hz, policy 25 Hz. No motor
commands, wrench commands, sensors, noise, random dynamics or RL trainer.
"""

import math
import sys
import time
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "control"))
from control_allocator import ControlAllocator, MODEL  # noqa: E402
from hover_controller import HoverController  # noqa: E402
from velocity_command_controller import (VelocityCommandController,
                                         map_normalized_action)  # noqa: E402


PHYSICS_STEPS_PER_CONTROL = 5
CONTROL_UPDATES_PER_POLICY = 4
PHYSICS_STEPS_PER_POLICY = PHYSICS_STEPS_PER_CONTROL * CONTROL_UPDATES_PER_POLICY
SUCCESS_HOLD_STEPS = 5
PROGRESS_GAIN = 10.0
ACTION_PENALTY_GAIN = 0.005
SUCCESS_BONUS = 10.0
FAILURE_PENALTY = -10.0
BRAKE_PENALTY_GAIN = 0.5
DISTANCE_PENALTY_GAIN = 0.02
TANGENT_PENALTY_GAIN = 0.5
MAX_TILT_RAD = math.radians(60.0)
MAX_HORIZONTAL_RADIUS_M = 6.0
MAX_ALTITUDE_M = 4.0


def _wrapped_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class MineUAVEnv(gym.Env):
    """Velocity-command waypoint task with only seven observable scalars."""

    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, max_episode_seconds: float = 15.0,
                 render_mode: str | None = None,
                 reward_version: str = "v1",
                 target_distribution: str = "full"):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if reward_version not in ("v1", "v2", "v3", "v4"):
            raise ValueError("reward_version must be 'v1', 'v2', 'v3' or 'v4'")
        if target_distribution not in ("full", "local_a", "local_b"):
            raise ValueError("target_distribution must be 'full', 'local_a' or 'local_b'")
        self.render_mode = render_mode
        self.reward_version = reward_version
        self.target_distribution = target_distribution
        self._viewer = None
        self._viewer_closed_by_user = False
        self._wall_time_origin = None
        self._camera_needs_update = True
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        if not np.allclose(self.model.opt.gravity, [0, 0, -9.81], atol=1e-12):
            raise ValueError("MineUAV v2 must use world +Z up, gravity -9.81")
        if not math.isclose(float(self.model.opt.timestep), 0.002, abs_tol=1e-12):
            raise ValueError("MineUAV v2 physics timestep must be 0.002 s")
        self.physics_dt = float(self.model.opt.timestep)
        self.control_dt = self.physics_dt * PHYSICS_STEPS_PER_CONTROL
        self.policy_dt = self.physics_dt * PHYSICS_STEPS_PER_POLICY
        if (not math.isfinite(max_episode_seconds) or max_episode_seconds <= 0
                or max_episode_seconds < self.policy_dt):
            raise ValueError("max_episode_seconds must be finite and >= policy_dt")
        self.max_episode_steps = max(1, round(max_episode_seconds / self.policy_dt))
        report_path = ROOT / "reports" / "dynamics_v2_report.json"
        import json
        report = json.loads(report_path.read_text(encoding="utf-8"))
        inertia = np.asarray(report["estimated_inertia_v2_kg_m2"], dtype=float)
        if not np.allclose(self.model.body("mine_uav").ipos,
                           report["estimated_com_v2_m"], atol=1e-10):
            raise ValueError("Model COM and inertia report disagree")
        self.allocator = ControlAllocator.from_v2_model()
        attitude_controller = HoverController(
            mass_kg=mujoco.mj_getTotalmass(self.model), inertia_kg_m2=inertia,
            max_total_thrust_n=float(np.sum(self.allocator.B[0]) * self.allocator.u_max),
            gravity_m_s2=abs(float(self.model.opt.gravity[2])))
        self.velocity_controller = VelocityCommandController(
            attitude_controller, self.model.opt.gravity)
        self.data = mujoco.MjData(self.model)
        self.observation_space = spaces.Box(
            low=np.array([-100] * 6 + [-math.pi], dtype=np.float32),
            high=np.array([100] * 6 + [math.pi], dtype=np.float32),
            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.target_position = np.array([0.0, 0.0, 1.0], dtype=float)
        self.target_yaw = 0.0
        self.previous_distance = 0.0
        self.episode_steps = 0
        self.success_streak = 0
        self._done = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary or None")
        if "target_position" in options:
            target = np.asarray(options["target_position"], dtype=float)
            if (target.shape != (3,) or not np.isfinite(target).all()
                    or np.any(target[:2] < -2) or np.any(target[:2] > 2)
                    or not 0.7 <= target[2] <= 1.5):
                raise ValueError("target_position must be within x/y ±2 m, z 0.7–1.5 m")
            self.target_position = target.copy()
        elif self.target_distribution == "full":
            self.target_position = np.array([
                self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2),
                self.np_random.uniform(0.7, 1.5)], dtype=float)
        else:
            low, high = ((0.25, 0.50) if self.target_distribution == "local_a"
                         else (0.50, 1.00))
            radius = self.np_random.uniform(low, high)
            # A random 3D direction conditioned on the original safe target
            # altitude. Keep the original full-task RNG branch untouched.
            while True:
                direction = self.np_random.normal(size=3)
                direction /= np.linalg.norm(direction)
                target = np.array([0.0, 0.0, 1.0]) + radius * direction
                if 0.7 <= target[2] <= 1.5:
                    self.target_position = target
                    break
        self.target_yaw = 0.0
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, 1.0]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.velocity_controller.reset(yaw_target=0.0)
        self.previous_distance = float(np.linalg.norm(self.target_position - self.data.qpos[:3]))
        self.episode_steps = 0
        self.success_streak = 0
        self._done = False
        if self.render_mode == "human":
            self._wall_time_origin = None  # New episode begins at simulated t=0.
            self._camera_needs_update = True
            self.render()
        return self._get_obs(), {"target_position_m": self.target_position.copy(),
                                  "target_yaw_rad": self.target_yaw}

    def _get_obs(self) -> np.ndarray:
        if not self._state_is_finite():
            # A numerical failure is terminal. Do not leak NaN into a replay buffer.
            return np.zeros(7, dtype=np.float32)
        error = self.target_position - self.data.qpos[:3]
        yaw_error = _wrapped_angle(self.target_yaw - _yaw_from_wxyz(self.data.qpos[3:7]))
        observation = np.r_[error, self.data.qvel[:3], yaw_error].astype(np.float32)
        return observation

    def _state_is_finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
                    and np.isfinite(self.data.ctrl).all())

    def _failure_reason(self) -> str | None:
        if not self._state_is_finite():
            return "nonfinite_state"
        position = self.data.qpos[:3]
        if position[2] < 0:
            return "z_below_zero"
        if float(np.linalg.norm(position[:2])) > MAX_HORIZONTAL_RADIUS_M or position[2] > MAX_ALTITUDE_M:
            return "outside_flight_area"
        q = self.data.qpos[3:7]
        tilt_cosine = float(1 - 2 * (q[1] * q[1] + q[2] * q[2]))
        if tilt_cosine < math.cos(MAX_TILT_RAD):
            return "excessive_tilt"
        return None

    def _compute_reward(self, previous_distance: float, current_distance: float,
                        action: np.ndarray, success: bool, failed: bool) -> tuple[float, dict]:
        if self.reward_version in ("v2", "v3", "v4"):
            # qvel[:3] is world-frame translation for the freejoint. A
            # numerical failure must still produce a finite terminal reward.
            velocity = self.data.qvel[:3]
            speed_squared = (float(np.dot(velocity, velocity))
                             if np.isfinite(velocity).all() else 0.0)
            braking_weight = float(np.clip((0.5 - current_distance) / 0.4, 0.0, 1.0))
            breakdown = {"progress": PROGRESS_GAIN * (previous_distance - current_distance)}
            if self.reward_version == "v3":
                breakdown["distance"] = -DISTANCE_PENALTY_GAIN * current_distance
            breakdown.update({
                "action": -ACTION_PENALTY_GAIN * float(np.dot(action, action)),
                "brake": -BRAKE_PENALTY_GAIN * braking_weight * speed_squared,
            })
            if self.reward_version == "v4":
                error = self.target_position - self.data.qpos[:3]
                distance = float(np.linalg.norm(error))
                if np.isfinite(velocity).all() and math.isfinite(distance):
                    if distance > 1e-9:
                        direction = error / distance
                        tangent_velocity = velocity - float(np.dot(velocity, direction)) * direction
                        tangent_speed_squared = float(np.dot(tangent_velocity, tangent_velocity))
                    else:
                        # No well-defined radial axis at the exact waypoint.
                        tangent_speed_squared = speed_squared
                else:
                    tangent_speed_squared = 0.0
                breakdown["tangent"] = -TANGENT_PENALTY_GAIN * braking_weight * tangent_speed_squared
            breakdown.update({
                "success": SUCCESS_BONUS if success else 0.0,
                "failure": FAILURE_PENALTY if failed else 0.0,
            })
            breakdown["total"] = float(sum(breakdown.values()))
            return breakdown["total"], breakdown
        breakdown = {
            "progress": PROGRESS_GAIN * (previous_distance - current_distance),
            "action_penalty": -ACTION_PENALTY_GAIN * float(np.dot(action, action)),
            "success_bonus": SUCCESS_BONUS if success else 0.0,
            "failure_penalty": FAILURE_PENALTY if failed else 0.0,
        }
        return float(sum(breakdown.values())), breakdown

    def _apply_rotor_command(self, command_u: np.ndarray) -> None:
        """Nominal instantaneous rotor response; audit subclasses may override."""
        self.data.ctrl[:] = command_u

    def _before_physics_step(self) -> None:
        """No nominal disturbance; audit subclasses may override per 500 Hz step."""

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode ended or not initialized; call reset() before step()")
        normalized_action = np.asarray(action, dtype=float)
        velocity_command, yaw_rate_command = map_normalized_action(normalized_action)
        failure_reason = self._failure_reason()
        physics_steps = 0
        controller_updates = 0
        saturation_count = 0
        max_motor_rpm = 0.0
        if failure_reason is None:
            for physics_index in range(PHYSICS_STEPS_PER_POLICY):
                if physics_index % PHYSICS_STEPS_PER_CONTROL == 0:
                    command = self.velocity_controller.compute(
                        velocity=self.data.qvel[:3], quaternion=self.data.qpos[3:7],
                        angular_velocity=self.data.qvel[3:6],
                        velocity_command=velocity_command,
                        yaw_rate_command=yaw_rate_command, control_dt=self.control_dt)
                    allocation = self.allocator.allocate(command.desired_wrench_body)
                    self._apply_rotor_command(allocation.u)
                    saturation_count += int(allocation.saturated)
                    max_motor_rpm = max(max_motor_rpm, float(np.max(allocation.rpm)))
                    controller_updates += 1
                self._before_physics_step()
                mujoco.mj_step(self.model, self.data)
                physics_steps += 1
                failure_reason = self._failure_reason()
                if failure_reason is not None:
                    break
        self.episode_steps += 1
        distance = (float(np.linalg.norm(self.target_position - self.data.qpos[:3]))
                    if failure_reason != "nonfinite_state" else self.previous_distance)
        speed = (float(np.linalg.norm(self.data.qvel[:3]))
                 if failure_reason != "nonfinite_state" else math.inf)
        if failure_reason is None and distance < 0.10 and speed < 0.15:
            self.success_streak += 1
        else:
            self.success_streak = 0
        success = failure_reason is None and self.success_streak >= SUCCESS_HOLD_STEPS
        terminated = bool(success or failure_reason is not None)
        truncated = bool(not terminated and self.episode_steps >= self.max_episode_steps)
        reward, breakdown = self._compute_reward(
            self.previous_distance, distance, normalized_action, success,
            failure_reason is not None)
        self.previous_distance = distance
        self._done = terminated or truncated
        reason = "success" if success else failure_reason
        if truncated:
            reason = "time_limit"
        info = {
            "reward_breakdown": breakdown,
            "distance_m": distance, "speed_m_s": speed,
            "target_position_m": self.target_position.copy(),
            "termination_reason": reason,
            "velocity_command_m_s": velocity_command.copy(),
            "yaw_rate_command_rad_s": yaw_rate_command,
            "physics_steps": physics_steps,
            "controller_updates": controller_updates,
            "allocator_saturation_count": saturation_count,
            "max_motor_rpm": max_motor_rpm,
            "success_streak": self.success_streak,
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    @property
    def viewer_is_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def render(self):
        if self.render_mode is None or self._viewer_closed_by_user:
            return None
        if self._viewer is None:
            # Importing the desktop viewer is optional for headless training.
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if not self._viewer.is_running():
            self._viewer_closed_by_user = True
            return None
        if self._camera_needs_update:
            with self._viewer.lock():
                # Frame both start and waypoint, so the full flight stays visible.
                self._viewer.cam.lookat[:] = (self.target_position + [0.0, 0.0, 1.0]) / 2
                self._viewer.cam.distance = 4.5
                self._viewer.cam.azimuth = 135.0
                self._viewer.cam.elevation = -25.0
            self._camera_needs_update = False
        self._viewer.sync()
        if self._wall_time_origin is None:
            self._wall_time_origin = time.monotonic() - float(self.data.time)
        remaining = self._wall_time_origin + float(self.data.time) - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            # MuJoCo's passive close requests the daemon UI thread to exit;
            # allow X11/GLX teardown even after a manual window close.
            time.sleep(0.5)
        self._viewer_closed_by_user = True
