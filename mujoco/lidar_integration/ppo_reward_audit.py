"""Read-only reward audit using immutable Phase 3E-L5 real-physics traces."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np


# Chosen before this bounded comparison is run. These are privileged analysis
# waypoints, not an obstacle-blind or LiDAR-learned controller.
FIXED_FRONT_BOX_WAYPOINTS = ((.45, .85, 1.), (1.45, .85, 1.))


def _audit_action(condition: str, state: np.ndarray, position: np.ndarray,
                  stage: int) -> tuple[np.ndarray, int]:
    from test_env_scripted_policy import scripted_action

    if condition == "A_zero_command":
        return np.zeros(4, dtype=np.float32), stage
    if condition == "B_obstacle_blind_direct":
        return scripted_action(state), stage
    if condition != "C_privileged_fixed_waypoint_detour":
        raise ValueError("unknown fixed Reward audit condition")
    if stage == 0 and position[0] > .30 and position[1] > .68:
        stage = 1
    if stage == 1 and position[0] > 1.37 and position[1] > .53:
        stage = 2
    if stage == 2:
        return scripted_action(state), stage
    proxy = state.copy()
    proxy[:3] = np.asarray(FIXED_FRONT_BOX_WAYPOINTS[stage]) - position
    return scripted_action(proxy), stage


def run_same_scene_reward_audit(upstream_root: Path, raw_dir: Path) -> dict:
    """Exactly three fixed Train-scene episodes; no optimizer, no retries.

    The privileged path is only a reward-semantics probe. Its success or
    failure is reported literally and never used for training/model choice.
    """
    from .navigation_env import LidarNavigationEnv

    conditions = ("A_zero_command", "B_obstacle_blind_direct",
                  "C_privileged_fixed_waypoint_detour")
    if raw_dir.exists() and any(raw_dir.iterdir()):
        raise FileExistsError("reward audit raw directory is not empty")
    raw_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    initial_identity = None
    for condition in conditions:
        env = LidarNavigationEnv("front_box", 1002, upstream_root)
        try:
            obs, reset_info = env.reset(seed=0, options={"start_seed": 1102,
                                                        "target_seed": 1202})
            identity = hashlib.sha256(np.asarray(env.data.qpos, dtype="<f8").tobytes()
                                      + np.asarray(env.data.qvel, dtype="<f8").tobytes()
                                      + np.asarray(env.target_position, dtype="<f8").tobytes()).hexdigest()
            if initial_identity is None:
                initial_identity = identity
            elif identity != initial_identity:
                raise RuntimeError("Reward audit initial physical states differ")
            rewards, actions, positions, distances = [], [], [env.data.qpos[:3].copy()], []
            stage = 0
            started = time.perf_counter()
            for _ in range(env.max_episode_steps):
                action, stage = _audit_action(condition, obs["state"],
                                              env.data.qpos[:3].copy(), stage)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                actions.append(np.asarray(action, dtype=np.float32))
                positions.append(env.data.qpos[:3].copy())
                distances.append(float(info["distance_m"]))
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("fixed audit episode did not terminate")
            path = raw_dir / f"{condition}.npz"
            with tempfile.NamedTemporaryFile(dir=raw_dir, suffix=".partial.npz",
                                             delete=False) as file:
                temporary = Path(file.name)
                np.savez_compressed(file, reward=np.array(rewards),
                                    action=np.array(actions), position_enu_m=np.array(positions),
                                    distance_m=np.array(distances))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
            results[condition] = {
                "outcome": info["termination_reason"], "return": float(sum(rewards)),
                "policy_steps": len(rewards), "physics_time_s": float(env.data.time),
                "final_distance_m": float(info["distance_m"]),
                "waypoint_stage_reached": stage if condition.startswith("C_") else None,
                "scene_hash": reset_info["obstacle_configuration_hash"],
                "initial_state_sha256": identity,
                "raw_path_local_only": str(path),
                "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "wall_seconds": time.perf_counter() - started,
            }
        finally:
            env.close()
    return {"conditions": results, "actual_episode_count": 3,
            "same_initial_state_sha256": initial_identity,
            "privileged_waypoints": FIXED_FRONT_BOX_WAYPOINTS,
            "training_or_optimizer_steps": 0}


def audit_reward(l5_report_path: Path) -> dict:
    report = json.loads(l5_report_path.read_text())
    if report["reward"]["version"] != "NAV_V2_COLLISION_PREFLIGHT_V1":
        raise ValueError("historical reward version differs")
    observed = {}
    for row in report["raw_local_only"]:
        if row["local_only_path"].endswith("evidence_cloud.npz"):
            continue
        path = l5_report_path.parent / row["local_only_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("historical trajectory identity mismatch")
        layout = path.stem
        with np.load(path, allow_pickle=False) as data:
            rewards = data["reward"]
            parts = [data[f"reward_component_{name}"] for name in
                     ("progress", "action", "brake", "success", "collision",
                      "timeout_or_time_cost", "physical_failure", "sensor_failure")]
            if not np.isfinite(rewards).all() or not np.allclose(rewards, sum(parts), atol=1e-10):
                raise ValueError("stored rewards disagree with stored components")
            observed[layout] = {"return": float(rewards.sum()),
                                "steps": len(rewards), "raw_sha256": row["sha256"]}
    episodes = {row["layout"]: row for row in report["episodes"]}
    for layout, item in observed.items():
        if not math.isclose(item["return"], episodes[layout]["return"], abs_tol=1e-9):
            raise ValueError("historical report and raw return disagree")
    collision = observed["front_box"]
    channel = observed["channel"]
    if not episodes["front_box"]["collision"] or not episodes["channel"]["success"]:
        raise ValueError("historical outcome classification differs")
    # An exactly stationary, zero-command, zero-speed policy has zero progress,
    # zero brake/action cost, 375 time-cost steps and a -2 timeout penalty.
    hover_ideal_return = -.01 * 375 - 2.
    return {
        "reward_version": report["reward"]["version"],
        "A_ideal_hover_math_not_executed": {"return": hover_ideal_return,
                                              "termination": "timeout",
                                              "assumptions": "zero speed, zero command, zero progress for 375 policy steps"},
        "B_front_box_actual_obstacle_blind_collision": {**collision,
                                                         "termination": "collision"},
        "C_channel_actual_clearance_and_stop": {**channel,
                                                "termination": "success"},
        "same_front_box_collision_return_upper_bound":
            10. * episodes["front_box"]["initial_distance_m"] - 30.,
        "same_front_box_success_return_upper_bound_not_a_trajectory":
            10. * episodes["front_box"]["initial_distance_m"] + 10.,
        "same_scene_three_policy_comparison": "NOT TESTED",
        "C_interpretation": "physical corridor clearance; not a front-box detour planner",
        "fixed_small_scene_ordering": hover_ideal_return > collision["return"]
                                      and channel["return"] > hover_ideal_return,
        "limitation": "A is mathematical ideal; B/C are different Phase L5 Train layouts. "
                      "This does not prove obstacle-aware PPO learning or general reward alignment.",
    }
