"""Real-physics LiDAR waypoint environment using the historical 100 Hz PI stack.

This is a simulation-ground-truth preflight, not a PX4 or trained LiDAR policy.
The central-body collision geom remains a TEMPORARY_PLACEHOLDER.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from mine_uav_env import MineUAVEnv
from velocity_command_controller import VelocityCommandController

from .async_observation import AsynchronousObservationBuffer, ObservationRead
from .navigation_scene import NavigationScene, build_scene, sample_start_target
from .observation import AngularRangeEncoder
from .scanner import Mid360Scanner


REWARD_KEYS = ("progress", "action", "brake", "success", "collision",
               "timeout_or_time_cost", "physical_failure", "sensor_failure", "total")
_FAILURE_OUTCOMES = {"collision", "physical_failure", "sensor_failure"}


def navigation_reward(previous_distance_m: float, distance_m: float,
                      speed_m_s: float, normalized_action: object,
                      outcome: str, executed_policy_steps: int) -> tuple[float, dict]:
    """Frozen Reward-V2-derived candidate, without post-failure progress credit.

    R = 10 Δdistance - .005||a||² - .5 w(d)||v||² + 10 success
        - 30 collision/physical failure - 10 sensor failure
        - .01 executed step - 2 timeout.
    w(d) = clip((.5-d)/.4, 0, 1). Exactly one terminal class is active.
    """
    action = np.asarray(normalized_action, dtype=float)
    if (action.shape != (4,) or not np.isfinite(action).all()
            or np.any(abs(action) > 1) or outcome not in
            {"running", "success", "collision", "physical_failure", "sensor_failure", "time_limit"}
            or executed_policy_steps not in (0, 1)
            or not all(math.isfinite(value) and value >= 0
                       for value in (previous_distance_m, distance_m, speed_m_s))):
        raise ValueError("reward inputs must be finite, bounded and classified")
    failed = outcome in _FAILURE_OUTCOMES
    weight = float(np.clip((.5 - distance_m) / .4, 0., 1.))
    parts = {
        "progress": (0. if failed else 10. * (previous_distance_m - distance_m)),
        "action": (-.005 * float(np.dot(action, action)) if executed_policy_steps else 0.),
        "brake": (-.5 * weight * speed_m_s ** 2 if executed_policy_steps else 0.),
        "success": 10. if outcome == "success" else 0.,
        "collision": -30. if outcome == "collision" else 0.,
        "timeout_or_time_cost": -.01 * executed_policy_steps - (2. if outcome == "time_limit" else 0.),
        "physical_failure": -30. if outcome == "physical_failure" else 0.,
        "sensor_failure": -10. if outcome == "sensor_failure" else 0.,
    }
    parts["total"] = float(sum(parts.values()))
    return parts["total"], parts


class LidarNavigationEnv(MineUAVEnv):
    """Five isolated obstacle layouts; same v2 PI, allocation and mj_step loop."""

    metadata = {"render_modes": [], "render_fps": 25}

    def __init__(self, layout: str, scene_seed: int, upstream_root: str | Path,
                 max_episode_seconds: float = 15.,
                 scene: NavigationScene | None = None) -> None:
        super().__init__(max_episode_seconds=max_episode_seconds,
                         render_mode=None, reward_version="v2", target_distribution="full")
        old = self.velocity_controller
        # Exact fixed gains from audit_velocity_pi.configure_pi_env, applied to
        # the same VelocityCommandController without importing the PPO trainer.
        self.velocity_controller = VelocityCommandController(
            old.attitude_controller, self.model.opt.gravity, kv=old.kv,
            horizontal_accel_limit=old.horizontal_accel_limit,
            vertical_accel_limit=old.vertical_accel_limit, ki_xy=.5, ki_z=.8)
        if scene is not None and (scene.layout != layout or scene.scene_seed != scene_seed):
            raise ValueError("injected scene identity does not match requested layout/seed")
        self.scene: NavigationScene = scene if scene is not None else build_scene(layout, scene_seed)
        if (not math.isclose(mujoco.mj_getTotalmass(self.scene.model),
                             mujoco.mj_getTotalmass(self.model), abs_tol=1e-10)
                or not np.array_equal(self.scene.model.actuator_gear,
                                      self.model.actuator_gear)):
            raise RuntimeError("isolated scene changed the inherited plant")
        self.model, self.data = self.scene.model, self.scene.data
        self.upstream_root = Path(upstream_root)
        self.encoder = AngularRangeEncoder()
        self.scanner = Mid360Scanner(self.model, self.upstream_root)
        self.lidar_buffer = AsynchronousObservationBuffer(experiment_id="uninitialized")
        self._episode_serial = 0
        self._experiment_id = "uninitialized"
        self._last_scan_tick = -1
        self.lidar_frame_count = 0
        self.last_frame = None
        self.last_scan_ms = 0.
        self.last_encode_ms = 0.
        self._last_contact_pairs: list[tuple[str, str]] = []
        self._last_collision_kind: str | None = None
        self._sensor_error: str | None = None
        self._lidar_active = False
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(0., 1., shape=(3, 12, 24), dtype=np.float32),
            "state": spaces.Box(low=np.array([-100] * 6 + [-math.pi], dtype=np.float32),
                                high=np.array([100] * 6 + [math.pi], dtype=np.float32),
                                dtype=np.float32),
            "lidar_age": spaces.Box(0., np.finfo(np.float32).max,
                                    shape=(1,), dtype=np.float32),
        })

    def _contact_kind(self) -> str | None:
        body_id = self.model.geom("coarse_collision").id
        observed = []
        result = None
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            if body_id not in (a, b):
                continue
            other = b if a == body_id else a
            name_a = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, a)
            name_b = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, b)
            observed.append((name_a, name_b))
            if other in self.scene.obstacle_geom_ids:
                result = "obstacle"
            elif other in self.scene.boundary_geom_ids and result != "obstacle":
                result = "boundary"
            elif self.model.geom("floor").id == other and result is None:
                result = "floor"
        self._last_contact_pairs = observed
        self._last_collision_kind = result
        return result

    def _scan_due(self) -> None:
        tick = round(float(self.data.time) / self.physics_dt)
        if (tick < 0 or abs(self.data.time - tick * self.physics_dt) > 1e-8):
            raise RuntimeError("LiDAR sampling requires the actual MuJoCo physics clock")
        if tick % 50 or tick == self._last_scan_tick:
            return
        started = time.perf_counter()
        frame = self.scanner.scan_next(self.data, downsample=1)
        self.last_frame = frame
        self.last_scan_ms = (time.perf_counter() - started) * 1000.
        started = time.perf_counter()
        observation = self.encoder.encode_frame(frame)
        self.last_encode_ms = (time.perf_counter() - started) * 1000.
        self.lidar_buffer.push(observation, arrival_time_s=frame.frame_timestamp,
                               experiment_id=self._experiment_id)
        self._last_scan_tick = tick
        self.lidar_frame_count += 1

    def _failure_reason(self) -> str | None:
        inherited = super()._failure_reason()
        if inherited == "nonfinite_state":
            return inherited
        if self._contact_kind() is not None:
            return "collision"
        if inherited is not None:
            return inherited
        if self._sensor_error is not None:
            return "sensor_failure"
        if self._lidar_active:
            try:
                self._scan_due()
            except (ValueError, RuntimeError) as exc:
                self._sensor_error = f"{type(exc).__name__}: {exc}"
                return "sensor_failure"
        return None

    def _read_lidar(self) -> ObservationRead:
        return self.lidar_buffer.read(float(self.data.time),
                                      experiment_id=self._experiment_id)

    @staticmethod
    def _observation(state: np.ndarray, read: ObservationRead) -> dict[str, np.ndarray]:
        if read.status == "FRESH" and read.observation is not None:
            lidar = read.observation.network_tensor.copy()
        else:
            lidar = np.zeros((3, 12, 24), dtype=np.float32)
            lidar[0] = 1.  # no obstacle claim; masks explicitly say no data
        age = float(read.age_s) if read.age_s is not None else 0.
        return {"lidar": lidar, "state": np.asarray(state, dtype=np.float32).copy(),
                "lidar_age": np.array([age], dtype=np.float32)}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        opts = {} if options is None else dict(options)
        start_seed = opts.pop("start_seed", 1000 if seed is None else 1000 + seed)
        target_seed = opts.pop("target_seed", 2000 if seed is None else 2000 + seed)
        explicit_start = opts.pop("start_position", None)
        explicit_yaw = opts.pop("start_yaw_rad", None)
        explicit_target = opts.pop("target_position", None)
        endpoints = (None if all(item is not None for item in
                                 (explicit_start, explicit_yaw, explicit_target))
                     else sample_start_target(self.scene, start_seed, target_seed))
        start = np.asarray(explicit_start if explicit_start is not None else endpoints.start_xyz,
                           dtype=float)
        yaw = float(explicit_yaw if explicit_yaw is not None else endpoints.start_yaw_rad)
        target = explicit_target if explicit_target is not None else endpoints.target_xyz
        if opts:
            raise ValueError(f"unsupported navigation reset options: {sorted(opts)}")
        target = np.asarray(target, dtype=float)
        if (start.shape != (3,) or not np.isfinite(start).all()
                or self.scene.overlaps_obstacle(start, clearance_m=.20)
                or np.any(np.abs(start[:2]) >= self.scene.boundary_half_extent_m - .20)
                or not .7 < start[2] < 2.95 or not math.isfinite(yaw)):
            raise ValueError("start pose must be finite, clear and inside bounds")
        if (target.shape != (3,) or not np.isfinite(target).all()
                or self.scene.overlaps_obstacle(target, clearance_m=.12)
                or np.any(np.abs(target[:2]) >= self.scene.boundary_half_extent_m - .12)
                or not .12 < target[2] < 2.95):
            raise ValueError("target must be finite, inside bounds and clear of obstacle/floor")
        self._lidar_active = False
        state, info = super().reset(seed=seed, options={"target_position": target})
        self.data.qpos[:3] = start
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        self.velocity_controller.reset(yaw_target=yaw)
        self.previous_distance = float(np.linalg.norm(self.target_position - self.data.qpos[:3]))
        self._episode_serial += 1
        self._experiment_id = f"{self.scene.geometry_hash}:{self._episode_serial}"
        self.lidar_buffer.reset(self._experiment_id)
        self.scanner = Mid360Scanner(self.model, self.upstream_root)
        self._last_scan_tick = -1
        self.lidar_frame_count = 0
        self.last_frame = None
        self.last_scan_ms = self.last_encode_ms = 0.
        self._sensor_error = None
        self._last_contact_pairs = []
        self._last_collision_kind = None
        if self._failure_reason() is not None:
            raise RuntimeError("initial physical state invalid or overlapping")
        self._lidar_active = True
        self._scan_due()
        read = self._read_lidar()
        if read.status != "FRESH":
            raise RuntimeError("initial LiDAR frame missing")
        info.update({"state_source": "SIM_GROUND_TRUTH_STATE",
                     "scene_seed": self.scene.scene_seed,
                     "start_seed": start_seed, "target_seed": target_seed,
                     "layout": self.scene.layout,
                     "obstacle_configuration_hash": self.scene.geometry_hash,
                     "lidar_status": read.status,
                     "lidar_source_time_s": read.source_frame_timestamp,
                     "lidar_age_s": read.age_s})
        return self._observation(self._get_obs(), read), info

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode ended or not initialized; call reset() before step()")
        before = self._read_lidar()
        if before.status != "FRESH":
            self._done = True
            distance = self.previous_distance
            speed = (float(np.linalg.norm(self.data.qvel[:3]))
                     if np.isfinite(self.data.qvel[:3]).all() else 0.)
            reward, parts = navigation_reward(distance, distance, speed, np.zeros(4),
                                              "sensor_failure", 0)
            info = {"termination_reason": "sensor_failure", "sensor_failure": True,
                    "collision": False, "physical_failure": False, "timeout": False,
                    "success": False, "reward_breakdown": parts,
                    "physics_steps": 0, "controller_updates": 0,
                    "distance_m": distance, "speed_m_s": speed,
                    "lidar_status": before.status,
                    "lidar_source_time_s": before.source_frame_timestamp,
                    "lidar_age_s": before.age_s,
                    "state_source": "SIM_GROUND_TRUTH_STATE"}
            return self._observation(self._get_obs(), before), reward, True, False, info
        previous_distance = self.previous_distance
        state, _, terminated, truncated, info = super().step(action)
        after = self._read_lidar()
        reason = info["termination_reason"]
        if after.status != "FRESH" and reason is None:
            # Preserve any terminal outcome already established by real
            # physics or the time limit. Only a continuing policy step needs
            # a fresh observation to authorize the next action.
            reason = "sensor_failure"
            terminated, truncated, self._done = True, False, True
            info["termination_reason"] = reason
        outcome = (reason if reason in {"success", "collision", "sensor_failure", "time_limit"}
                   else "physical_failure" if reason else "running")
        distance = float(info["distance_m"])
        speed = float(info["speed_m_s"])
        if not math.isfinite(speed):
            speed = 0.  # Numerical failure gets finite terminal reward.
        reward, parts = navigation_reward(previous_distance, distance, speed, action,
                                          outcome, 1)
        info.update({"reward_breakdown": parts,
                     "state_source": "SIM_GROUND_TRUTH_STATE",
                     "scene_seed": self.scene.scene_seed,
                     "layout": self.scene.layout,
                     "obstacle_configuration_hash": self.scene.geometry_hash,
                     "collision": outcome == "collision", "success": outcome == "success",
                     "timeout": outcome == "time_limit",
                     "physical_failure": outcome == "physical_failure",
                     "sensor_failure": outcome == "sensor_failure",
                     "collision_kind": self._last_collision_kind if outcome == "collision" else None,
                     "contact_pairs": list(self._last_contact_pairs) if outcome == "collision" else [],
                     "lidar_status": after.status,
                     "lidar_source_time_s": after.source_frame_timestamp,
                     "lidar_age_s": after.age_s,
                     "lidar_frame_count": self.lidar_frame_count,
                     "physics_tick": round(float(self.data.time) / self.physics_dt),
                     "sensor_error": self._sensor_error})
        return self._observation(state, after), reward, terminated, truncated, info
