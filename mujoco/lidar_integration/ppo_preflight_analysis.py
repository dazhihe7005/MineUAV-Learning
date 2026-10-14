"""Offline evidence aggregation after the sole PPO smoke attempt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .ppo_reward_audit import audit_reward
from .ppo_training_runner import _atomic_json, protocol_hash


def _verified_same_scene_audit(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    names = ("A_zero_command", "B_obstacle_blind_direct",
             "C_privileged_fixed_waypoint_detour")
    if summary.get("actual_episode_count") != 3 or summary.get("training_or_optimizer_steps") != 0:
        raise ValueError("same-scene reward audit episode/training count invalid")
    rows = [summary["conditions"][name] for name in names]
    if (len({row["scene_hash"] for row in rows}) != 1
            or len({row["initial_state_sha256"] for row in rows}) != 1
            or [row["outcome"] for row in rows] != ["time_limit", "collision", "success"]):
        raise ValueError("reward audit conditions are not paired or outcomes differ")
    for row in rows:
        path = Path(row["raw_path_local_only"])
        if not path.is_absolute():
            path = Path(__file__).parents[2] / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["raw_sha256"]:
            raise ValueError("same-scene reward raw identity mismatch")
        with np.load(path, allow_pickle=False) as raw:
            if len(raw["reward"]) != row["policy_steps"] or not np.isclose(
                    raw["reward"].sum(), row["return"], atol=1e-8):
                raise ValueError("same-scene reward sum mismatch")
    summary["observed_ordering_C_gt_A_gt_B"] = rows[2]["return"] > rows[0]["return"] > rows[1]["return"]
    summary["analysis_timing"] = "POST_SMOKE; not used for Reward or checkpoint selection"
    summary["C_controller_privilege"] = "fixed geometry-derived waypoints; not a LiDAR policy"
    summary["initial_identity_scope"] = (
        "SHA256 covers qpos, qvel and target only; PI/act/ctrl are reset by the "
        "same deterministic environment reset, but are not included in this hash"
    )
    return summary


def build_report(smoke_path: Path, prereg_path: Path, checkpoint_path: Path,
                 historical_l5_report: Path,
                 same_scene_audit_path: Path | None = None) -> dict:
    smoke = json.loads(smoke_path.read_text())
    prereg = json.loads(prereg_path.read_text())
    if smoke["status"] != "PASS" or smoke["timesteps"] != 512 or smoke["worker_count"] != 1:
        raise ValueError("smoke did not satisfy frozen training budget")
    if hashlib.sha256(prereg_path.read_bytes()).hexdigest() != smoke["prereg_sha256"]:
        raise ValueError("preregistration identity changed after smoke")
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != smoke["checkpoint_sha256"]:
        raise ValueError("smoke checkpoint identity mismatch")
    if smoke["scenario_manifest_sha256"] != protocol_hash(prereg["scenario_manifest"]):
        raise ValueError("scene manifest identity mismatch")
    if smoke["protocol_sha256"] != protocol_hash(prereg["smoke_protocol"]):
        raise ValueError("training protocol identity mismatch")
    if prereg["smoke_train_identity_sha256"] != smoke["scenario_identity_sha256"]:
        raise ValueError("smoke used an unregistered Train scenario")
    if not smoke["historical_checkpoints_unchanged"]:
        raise ValueError("historical model identity failed")
    source_root = Path(__file__).parents[1]
    for relative, expected in prereg["preflight_source_sha256"].items():
        if hashlib.sha256((source_root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"runtime source changed after smoke: {relative}")
    audit = audit_reward(historical_l5_report)
    audit["evidence_scope"] = (
        "Historical L5 trajectories from different layouts; its same-scene status "
        "does not describe the separate POST_SMOKE front_box audit"
    )
    same_scene = (_verified_same_scene_audit(same_scene_audit_path)
                  if same_scene_audit_path is not None else None)
    seen = set()
    for split, rows in prereg["scenario_manifest"].items():
        for row in rows:
            if row["split"] != split or row["identity_sha256"] in seen:
                raise ValueError("scenario split identity collision")
            seen.add(row["identity_sha256"])
    return {
        "task": "Phase 3E-L6 LiDAR-CNN-PPO training preflight",
        "status": {
            "observation_interface": "PASS", "cnn_actor_critic_shapes_and_gradients": "PASS",
            "scene_split_identity": "PASS", "geometric_feasibility_at_nominal_height": "PASS",
            "fixed_reward_formula_and_accounting": "PASS",
            "same_scene_ABC_reward_trajectory_comparison": "PASS" if same_scene else "NOT TESTED",
            "bounded_training_smoke": "PASS", "checkpoint_reload": "PASS",
            "formal_multiseed_training": "NOT TESTED",
            "obstacle_avoidance_learned": "NOT TESTED",
            "test_generalization": "NOT TESTED", "px4_fast_lio2_real_sensor": "NOT TESTED",
            "ready_to_run_formal_training_without_further_implementation": "BLOCKED",
        },
        "observation": {
            "lidar": "float32[3,12,24] normalized range/sampled mask/hit mask",
            "state": "float32[7] target_position−position ENU, velocity ENU, wrapped yaw error",
            "heading": "float32[2] sin/cos of current ENU body yaw",
            "lidar_age": "float32[1] MuJoCo simulation seconds",
            "state_source": "SIM_GROUND_TRUTH_STATE",
            "feature_dim": 74, "normalize_images": False,
        },
        "action": {"shape": [4], "normalized_bounds": [-1, 1],
                   "world_velocity_scale_m_s": [1.5, 1.5, 1.0],
                   "yaw_rate_scale_rad_s": 1.0,
                   "physical_scaling_location": "inherited MineUAVEnv.step/map_normalized_action"},
        "reward_audit": audit,
        "same_scene_reward_audit": same_scene,
        "scenario_counts": {split: len(rows) for split, rows in prereg["scenario_manifest"].items()},
        "scenario_manifest_sha256": smoke["scenario_manifest_sha256"],
        "smoke_scene": {"split": "train", "layout": "front_box",
                        "identity_sha256": smoke["scenario_identity_sha256"]},
        "formal_protocol_proposal_not_run": prereg["formal_proposal_not_run"],
        "formal_protocol_sha256": prereg["formal_proposal_sha256"],
        "complete_prereg_sha256": prereg["complete_protocol_sha256"],
        "software_versions": prereg["software_versions"],
        "runtime_source_sha256": prereg["preflight_source_sha256"],
        "historical_model_sha256": prereg["historical_model_sha256"],
        "upstream_lidar_commit": prereg["upstream_lidar_commit"],
        "upstream_pattern_sha256": prereg["lidar_pattern_sha256"],
        "historical_checkpoint_sha256": prereg["historical_checkpoint_sha256"],
        "smoke": smoke,
        "tests": {"inherited_bridge_and_lidar": 627, "old_controller_targeted": 25,
                  "new_preflight": 15, "total": 667, "failed": 0, "warnings": 2},
        "resource_note": "One CPU process, one MuJoCo environment; cv2/NumPy ABI import warning "
                         "printed by SB3 import chain but process completed. No PX4 process.",
        "offline_separate_process_init_memory_probe_ru_maxrss_mib": {
            "after_import": 339.23828125, "after_environment_init": 1274.875,
            "after_model_init": 1274.875, "training_in_this_probe": False,
            "note": "Separate process; peak-so-far ru_maxrss, not same-process incremental RSS."},
        "limitations": [
            "One 512-step smoke is a pipeline check, not obstacle-learning evidence.",
            "Only front_box Train[1] was sampled; uniform 12-scene formal scheduler is not implemented.",
            "The 12/6/6 layouts are small synthetic configurations, not mine generalization proof.",
            "Same-scene post-smoke Reward ordering uses a privileged geometry-waypoint "
            "controller for C; it does not prove LiDAR-policy reward alignment.",
            "The body contact geom and LiDAR pattern/extrinsic are simulation approximations.",
            "State and heading are MuJoCo truth, not PX4 EKF or SLAM outputs.",
            "No mid-scan timing, motion distortion, real MID360S calibration or PX4 joint realtime test.",
        ],
    }


def write_report(output_path: Path, smoke_path: Path, prereg_path: Path,
                 checkpoint_path: Path, historical_l5_report: Path,
                 same_scene_audit_path: Path | None = None) -> dict:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite analysis evidence")
    report = build_report(smoke_path, prereg_path, checkpoint_path,
                          historical_l5_report, same_scene_audit_path)
    _atomic_json(output_path, report)
    return report
