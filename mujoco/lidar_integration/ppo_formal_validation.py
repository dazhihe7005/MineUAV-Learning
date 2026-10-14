"""Independent, deterministic Validation-only interface for future PPO runs."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

from .navigation_env import LidarNavigationEnv
from .ppo_formal_scheduler import (L6_MANIFEST_SHA256, require_memory,
                                   verify_l6_runtime_sources)
from .ppo_preflight import HeadingObservation
from .ppo_scenarios import build_ppo_scene
from .ppo_training_runner import _stats, _weights_sha, protocol_hash


TERMINAL_OUTCOMES = ("success", "collision", "time_limit", "physical_failure",
                     "sensor_failure")


def _optimizer_states_equal(a, b) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return bool(torch.equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_optimizer_states_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, type(a)):
        return len(a) == len(b) and all(_optimizer_states_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)


class LidarPPOValidationEvaluator:
    """Only six frozen Validation scenes; never changes the training environment."""

    def __init__(self, prereg: dict, upstream_root: Path):
        verify_l6_runtime_sources(prereg)
        if protocol_hash(prereg["scenario_manifest"]) != L6_MANIFEST_SHA256:
            raise ValueError("Validation requires the unchanged L6 scenario manifest")
        rows = prereg["scenario_manifest"]["validation"]
        if len(rows) != 6 or any(row["split"] != "validation" for row in rows):
            raise ValueError("only the six frozen Validation identities are supported")
        self.rows = tuple(copy.deepcopy(rows))
        self.upstream_root = Path(upstream_root)

    def evaluate(self, model, *, max_policy_steps: int = 375) -> dict:
        if (isinstance(max_policy_steps, bool) or not isinstance(max_policy_steps, int)
                or not 1 <= max_policy_steps <= 375):
            raise ValueError("Validation probe budget must be 1..375 policy steps")
        require_memory()
        policy_before = _weights_sha(model.policy)
        optimizer_before = copy.deepcopy(model.policy.optimizer.state_dict())
        training_steps_before = model.num_timesteps
        rows = []
        whole_started = time.perf_counter()
        for row in self.rows:
            require_memory()
            scene = build_ppo_scene(row)
            if scene.geometry_hash != row["obstacle_configuration_hash"]:
                raise ValueError("Validation compiled geometry identity mismatch")
            env = HeadingObservation(LidarNavigationEnv(
                row["layout"], row["scene_seed"], self.upstream_root, scene=scene))
            try:
                obs, info = env.reset(seed=row["scene_seed"], options={
                    "start_seed": row["start_seed"], "target_seed": row["target_seed"],
                    "start_position": row["start_xyz"], "target_position": row["target_xyz"],
                    "start_yaw_rad": row["start_yaw_rad"]})
                if info["obstacle_configuration_hash"] != row["obstacle_configuration_hash"]:
                    raise RuntimeError("Validation reset scene identity changed")
                previous_position = env.unwrapped.data.qpos[:3].copy()
                path_length = 0.
                rewards = 0.
                minimum_range = None
                ages, inference_ms = [], []
                final_info = None
                outcome = "probe_cutoff"
                for policy_step in range(1, max_policy_steps + 1):
                    require_memory()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        action, _ = model.predict(obs, deterministic=True)
                    inference_ms.append((time.perf_counter() - started) * 1000.)
                    if np.asarray(action).shape != (4,) or not np.isfinite(action).all():
                        raise RuntimeError("Validation model emitted invalid action")
                    obs, reward, terminated, truncated, final_info = env.step(action)
                    rewards += float(reward)
                    position = env.unwrapped.data.qpos[:3].copy()
                    path_length += float(np.linalg.norm(position - previous_position))
                    previous_position = position
                    if final_info["lidar_age_s"] is not None:
                        ages.append(float(final_info["lidar_age_s"]))
                    frame = env.unwrapped.last_frame
                    if frame is not None and np.any(frame.valid_mask):
                        sample = float(np.min(frame.ranges[frame.valid_mask]))
                        minimum_range = sample if minimum_range is None else min(minimum_range, sample)
                    if terminated or truncated:
                        reason = final_info["termination_reason"]
                        outcome = reason if reason in TERMINAL_OUTCOMES else "physical_failure"
                        break
                if final_info is None:
                    raise RuntimeError("Validation did not execute a policy step")
                if outcome != "probe_cutoff":
                    flags = [bool(final_info.get(key)) for key in
                             ("success", "collision", "timeout", "physical_failure", "sensor_failure")]
                    if sum(flags) != 1:
                        raise RuntimeError("Validation terminal class is not mutually exclusive")
                rows.append({
                    "scenario_identity_sha256": row["identity_sha256"],
                    "layout": row["layout"], "outcome": outcome,
                    "success": outcome == "success", "collision": outcome == "collision",
                    "timeout": outcome == "time_limit",
                    "physical_failure": outcome == "physical_failure",
                    "sensor_failure": outcome == "sensor_failure",
                    "policy_steps": policy_step, "episode_return": rewards,
                    "trajectory_length_m": path_length,
                    "completion_time_s": float(env.unwrapped.data.time) if outcome == "success" else None,
                    "final_target_error_m": float(np.linalg.norm(
                        env.unwrapped.target_position - position)),
                    "minimum_valid_lidar_range_m": minimum_range,
                    "mean_lidar_age_s": float(np.mean(ages)) if ages else None,
                    "inference_ms": _stats(inference_ms),
                    "collision_contact_pairs": final_info.get("contact_pairs", []),
                    "last_sim_time_s": float(env.unwrapped.data.time),
                })
            finally:
                env.close()
        if _weights_sha(model.policy) != policy_before or model.num_timesteps != training_steps_before:
            raise RuntimeError("Validation mutated training model")
        # The optimizer is not invoked; compare serialized state-dict values,
        # including any tensor-valued moments, without consuming its RNG.
        optimizer_after = model.policy.optimizer.state_dict()
        if not _optimizer_states_equal(optimizer_before, optimizer_after):
            raise RuntimeError("Validation mutated PPO optimizer state")
        counts = {key: sum(row["outcome"] == key for row in rows)
                  for key in (*TERMINAL_OUTCOMES, "probe_cutoff")}
        complete = counts["probe_cutoff"] == 0
        rates = {
            "navigation_success_rate": counts["success"] / len(rows) if complete else None,
            "collision_rate": counts["collision"] / len(rows) if complete else None,
            "timeout_rate": counts["time_limit"] / len(rows) if complete else None,
            "physical_failure_rate": counts["physical_failure"] / len(rows) if complete else None,
            "sensor_failure_rate": counts["sensor_failure"] / len(rows) if complete else None,
        }
        times_to_goal = [row["completion_time_s"] for row in rows if row["success"]]
        return {
            "split": "validation", "model_status": "UNTRAINED_RANDOM_INITIALIZATION"
            if training_steps_before == 0 else "FROZEN_MODEL_DIAGNOSTIC",
            "probe_only": not complete, "episodes": rows,
            "overall": {**counts, **rates, "total": len(rows),
                        "mean_return": float(np.mean([row["episode_return"] for row in rows])),
                        "mean_trajectory_length_m": float(np.mean(
                            [row["trajectory_length_m"] for row in rows])),
                        "mean_time_to_goal_s": float(np.mean(times_to_goal)) if times_to_goal else None},
            "per_scene": {row["scenario_identity_sha256"]: dict(row) for row in rows},
            "policy_sha256_before_after": policy_before,
            "optimizer_unchanged": True,
            "training_timesteps_before_after": training_steps_before,
            "wall_seconds": time.perf_counter() - whole_started,
        }
