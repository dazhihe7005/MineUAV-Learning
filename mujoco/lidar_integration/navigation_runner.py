"""Three fixed, sequential real-MuJoCo navigation episodes; never train PPO."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

from test_env_scripted_policy import scripted_action

from .navigation_env import LidarNavigationEnv
from .navigation_plots import plot_navigation_preflight
from .navigation_scene import LAYOUTS, build_scene, fixed_split_manifest
from .runner import _atomic_json, _available_mib, _stats
from .scanner import PATTERN_SHA256, UPSTREAM_COMMIT
from .scene import BASE_MODEL, BASE_SHA256


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
        temporary = Path(file.name)
        np.savez_compressed(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("process RSS unavailable")


def classify_contact_failure(episodes: list[dict]) -> str:
    """Observed front-box contact, not mere inclusion of a collidable scene."""
    front = next((row for row in episodes if row["layout"] == "front_box"), None)
    if front is None:
        return "NOT TESTED"
    return ("PASS" if front["collision"] and front["collision_kind"] == "obstacle"
            and any("coarse_collision" in pair for pair in front["contact_pairs"])
            else "FAIL")


def run_preflight(output_report: Path, figures_dir: Path, upstream_root: Path,
                  *, layouts: tuple[str, ...] = ("empty", "front_box", "channel")) -> dict:
    """Bounded interface proof; obstacle-blind Scripted is not a LiDAR policy."""
    output, figures = Path(output_report), Path(figures_dir)
    parts = output.parent / "mujoco_lidar_navigation_env_parts" / "preflight_run_01"
    if output.exists():
        raise ValueError("report exists; refusing to overwrite evidence")
    if figures.exists() and any(figures.iterdir()):
        raise ValueError("figure directory is nonempty; refusing to overwrite evidence")
    if parts.exists() and any(parts.iterdir()):
        raise ValueError("local raw part directory is nonempty; refusing to overwrite evidence")
    if (not layouts or len(layouts) > 3 or len(set(layouts)) != len(layouts)
            or any(layout not in LAYOUTS for layout in layouts)):
        raise ValueError("preflight permits one to three distinct pinned layouts")
    if importlib.metadata.version("mujoco") != "3.14.0" or not hasattr(mujoco, "MjModel"):
        raise RuntimeError("real MuJoCo 3.14.0 Python package required")
    if hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() != BASE_SHA256:
        raise RuntimeError("historical physical model identity changed")
    available_start = _available_mib()
    if available_start < 1536:
        raise MemoryError("insufficient system memory for bounded MuJoCo preflight")
    rss_start = _rss_mib()
    manifest = fixed_split_manifest()
    split_identity = {
        split: [{**row, "obstacle_configuration_hash":
                 build_scene(row["layout"], row["scene_seed"]).geometry_hash}
                for row in rows]
        for split, rows in manifest.items()
    }
    rows_by_layout = {row["layout"]: row for row in manifest["train"]}
    traces: list[dict] = []
    episodes: list[dict] = []
    all_step_ms: list[float] = []
    all_scan_ms: list[float] = []
    all_encode_ms: list[float] = []
    min_available = available_start
    first_cloud: np.ndarray | None = None
    first_scene = None
    raw_files: list[dict] = []
    for layout in layouts:
        spec = rows_by_layout[layout]
        env = LidarNavigationEnv(layout, spec["scene_seed"], upstream_root)
        try:
            obs, reset_info = env.reset(seed=0, options={
                "start_seed": spec["start_seed"], "target_seed": spec["target_seed"]})
            if first_cloud is None or layout == "front_box":
                first_cloud = env.last_frame.world_points.copy()
                first_scene = env.scene
            initial_scan_ms = env.last_scan_ms
            initial_encode_ms = env.last_encode_ms
            scan_wall_ms = [initial_scan_ms]
            encode_wall_ms = [initial_encode_ms]
            all_scan_ms.append(initial_scan_ms)
            all_encode_ms.append(initial_encode_ms)
            positions = [env.data.qpos[:3].copy()]
            times = [float(env.data.time)]
            distance = [float(env.previous_distance)]
            source_times = [float(reset_info["lidar_source_time_s"])]
            scan_times = [float(reset_info["lidar_source_time_s"])]
            reward_parts = []
            step_ms = []
            rewards = []
            max_speed = 0.
            max_saturation = 0
            physics_ticks = 0
            controller_updates = 0
            last_frame_count = env.lidar_frame_count
            for _ in range(env.max_episode_steps):
                action = scripted_action(obs["state"])  # obstacle-blind state-only baseline
                began = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                step_duration = (time.perf_counter() - began) * 1000.
                if _available_mib() < 1536:
                    raise MemoryError("minimum system memory reserve reached")
                min_available = min(min_available, _available_mib())
                step_ms.append(step_duration)
                all_step_ms.append(step_duration)
                positions.append(env.data.qpos[:3].copy())
                times.append(float(env.data.time))
                distance.append(float(info["distance_m"]))
                source_times.append(info["lidar_source_time_s"])
                reward_parts.append(info["reward_breakdown"])
                rewards.append(float(reward))
                max_speed = max(max_speed, float(info["speed_m_s"]))
                max_saturation += int(info.get("allocator_saturation_count", 0))
                physics_ticks += int(info["physics_steps"])
                controller_updates += int(info["controller_updates"])
                if env.lidar_frame_count != last_frame_count:
                    assert env.lidar_frame_count == last_frame_count + 1
                    scan_times.append(float(info["lidar_source_time_s"]))
                    all_scan_ms.append(float(env.last_scan_ms))
                    all_encode_ms.append(float(env.last_encode_ms))
                    scan_wall_ms.append(float(env.last_scan_ms))
                    encode_wall_ms.append(float(env.last_encode_ms))
                    last_frame_count = env.lidar_frame_count
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("episode did not terminate by fixed time limit")
            if not math.isclose(float(env.data.time), physics_ticks * env.physics_dt,
                                abs_tol=1e-8):
                raise RuntimeError("physical clock and mj_step count disagree")
            positions_array = np.asarray(positions)
            times_array = np.asarray(times)
            reward_array = np.asarray(rewards)
            local_path = parts / f"{layout}.npz"
            _atomic_npz(local_path, position_enu_m=positions_array,
                        time_s=times_array, distance_m=np.asarray(distance),
                        reward=reward_array,
                        lidar_source_time_s=np.asarray(source_times, dtype=float),
                        scan_sample_time_s=np.asarray(scan_times, dtype=float),
                        scan_wall_ms=np.asarray(scan_wall_ms, dtype=float),
                        encode_wall_ms=np.asarray(encode_wall_ms, dtype=float),
                        step_wall_ms=np.asarray(step_ms, dtype=float),
                        **{f"reward_component_{key}": np.asarray(
                            [row[key] for row in reward_parts], dtype=float)
                           for key in reward_parts[0]})
            raw_files.append({"local_only_path": str(local_path.relative_to(output.parent)),
                              "sha256": hashlib.sha256(local_path.read_bytes()).hexdigest()})
            traces.append({"layout": layout, "scene": env.scene,
                           "target": env.target_position.copy(),
                           "position": positions_array, "time": times_array,
                           "distance": np.asarray(distance),
                           "source_times": np.asarray(source_times, dtype=float),
                           "reward_parts": reward_parts, "step_ms": step_ms,
                           "scan_times": scan_times})
            episodes.append({"layout": layout, "scene_seed": spec["scene_seed"],
                             "start_seed": spec["start_seed"],
                             "target_seed": spec["target_seed"],
                             "obstacle_configuration_hash": env.scene.geometry_hash,
                             "policy": "STATE_ONLY_SCRIPTED_NO_LIDAR_INPUT",
                             "steps": len(rewards), "physics_ticks": physics_ticks,
                             "controller_updates": controller_updates,
                             "lidar_frames": env.lidar_frame_count,
                             "ray_count_per_frame": env.last_frame.ray_count,
                             "termination_reason": info["termination_reason"],
                             "collision_kind": info.get("collision_kind"),
                             "contact_pairs": info.get("contact_pairs", []),
                             "success": bool(info["success"]),
                             "collision": bool(info["collision"]),
                             "timeout": bool(info["timeout"]),
                             "physical_failure": bool(info["physical_failure"]),
                             "sensor_failure": bool(info["sensor_failure"]),
                             "elapsed_sim_s": float(env.data.time),
                             "initial_distance_m": float(distance[0]),
                             "final_distance_m": float(distance[-1]),
                             "max_translational_speed_m_s": max_speed,
                             "allocator_saturation_updates": max_saturation,
                             "return": float(np.sum(reward_array)),
                             "reward_component_sums": {
                                 key: float(sum(row[key] for row in reward_parts))
                                 for key in reward_parts[0]},
                             "initial_scan_ms": initial_scan_ms,
                             "initial_encode_ms": initial_encode_ms,
                             "step_ms": _stats(step_ms),
                             "scan_sample_times_s": scan_times})
        finally:
            env.close()
    assert first_cloud is not None and first_scene is not None
    cloud_path = parts / "evidence_cloud.npz"
    _atomic_npz(cloud_path, world_points=first_cloud,
                scene_geometry_sha256=np.asarray(first_scene.geometry_hash))
    raw_files.append({"local_only_path": str(cloud_path.relative_to(output.parent)),
                      "sha256": hashlib.sha256(cloud_path.read_bytes()).hexdigest()})
    plot_navigation_preflight(traces, first_cloud, first_scene, figures)
    contact_status = classify_contact_failure(episodes)
    physics_ok = all(row["physics_ticks"] > 0 and row["controller_updates"] > 0
                     and math.isclose(row["elapsed_sim_s"], row["physics_ticks"] * .002,
                                      abs_tol=1e-8) for row in episodes)
    rays_ok = all(row["lidar_frames"] > 0 and row["ray_count_per_frame"] == 24000
                  and len(row["scan_sample_times_s"]) == row["lidar_frames"]
                  for row in episodes)
    causal_ok = all(
        not row["sensor_failure"] and np.all(trace["time"] + 1e-9 >=
                                             np.asarray(trace["source_times"]))
        for row, trace in zip(episodes, traces))
    reward_ok = all(
        math.isclose(row["return"], row["reward_component_sums"]["total"], abs_tol=1e-8)
        and sum(bool(row[key]) for key in
                ("success", "collision", "timeout", "physical_failure", "sensor_failure")) == 1
        and (not row["collision"] or trace["reward_parts"][-1]["progress"] == 0.)
        for row, trace in zip(episodes, traces))
    acceptance = {"gymnasium_reset_step": "PASS" if len(episodes) == len(layouts) else "FAIL",
                  "real_physics": "PASS" if physics_ok else "FAIL",
                  "actual_lidar_rays": "PASS" if rays_ok else "FAIL",
                  "causal_async_input": "PASS" if causal_ok else "FAIL",
                  "contact_failure": contact_status,
                  "reward_accounting": "PASS" if reward_ok else "FAIL",
                  "ppo_training": "NOT TESTED", "px4_execution": "NOT TESTED",
                  "real_mid360s_collision_safety": "NOT TESTED"}
    core_statuses = [acceptance[key] for key in
                     ("gymnasium_reset_step", "real_physics", "actual_lidar_rays",
                      "causal_async_input", "contact_failure", "reward_accounting")]
    status = ("FAIL" if "FAIL" in core_statuses else
              "NOT TESTED" if "NOT TESTED" in core_statuses else "PASS")
    result = {
        "status": status, "scope": "REAL_MUJOCO_NAVIGATION_ENV_PREFLIGHT_ONLY",
        "episode_count": len(episodes), "episodes": episodes,
        "physics_ticks_total": sum(row["physics_ticks"] for row in episodes),
        "controller_updates_total": sum(row["controller_updates"] for row in episodes),
        "lidar_frames_total": sum(row["lidar_frames"] for row in episodes),
        "frequencies_hz": {"physics": 500, "pi": 100, "policy": 25, "lidar": 10},
        "observation": {"lidar_shape": [3, 12, 24], "state_shape": [7],
                        "lidar_age_shape": [1], "state_source": "SIM_GROUND_TRUTH_STATE",
                        "stale_after_s": .15},
        "action": {"normalized_shape": [4], "bounds": [-1, 1],
                   "world_velocity_scale_m_s": [1.5, 1.5, 1.0],
                   "yaw_rate_scale_rad_s": 1.0},
        "reward": {"version": "NAV_V2_COLLISION_PREFLIGHT_V1",
                   "formula": "10*(d_prev-d_next) [0 on failure] - 0.005*||a||² - 0.5*clip((0.5-d)/0.4,0,1)*||v||² + 10*success - 30*collision - 30*physical_failure - 10*sensor_failure - 0.01*executed_policy_step - 2*timeout",
                   "units": "dimensionless per policy step", "collision_progress_credit": False},
        "upstream": {"commit": UPSTREAM_COMMIT, "pattern_sha256": PATTERN_SHA256,
                     "backend": "cpu", "ray_count_per_frame": 24000,
                     "lidar_extrinsic_provenance": "SIM_CONFIG"},
        "historical_model_sha256": BASE_SHA256,
        "scene_split_manifest": split_identity,
        "raw_local_only": raw_files,
        "performance": {"whole_step_ms": _stats(all_step_ms),
                        "lidar_scan_ms": _stats(all_scan_ms),
                        "lidar_encode_ms": _stats(all_encode_ms),
                        "frame_samples": len(all_scan_ms),
                        "step_samples": len(all_step_ms)},
        "resource": {"worker_count": 1,
                     "rss_start_mib": rss_start, "rss_end_mib": _rss_mib(),
                     "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                     "min_mem_available_mib": min_available,
                     "oom_observed": False, "px4_runs": 0,
                     "fast_lio2_runs": 0, "training_updates": 0},
        "acceptance": acceptance,
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    args = parser.parse_args()
    result = run_preflight(args.output_report, args.figures_dir, args.upstream_root)
    print(json.dumps({"status": result["status"], "episodes": result["episodes"],
                      "performance": result["performance"],
                      "resource": result["resource"]}, indent=2))


if __name__ == "__main__":
    main()
