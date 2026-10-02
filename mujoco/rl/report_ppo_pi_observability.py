"""Paired nominal A/B/C/D waypoint comparison; no training or perturbations."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from stable_baselines3 import PPO

from attribution_audit import write_condition
from audit_ppo_robustness import MUJOCO_DIR, waypoint_sha256
from mine_uav_env import MineUAVEnv
from ppo_exploration_audit import evaluate_exploration
from ppo_pi_env import MineUAVPIEnv
from reward_v2_alignment_audit import fixed_waypoints
from train_ppo_lowstd_multiseed import (build_holdout_waypoints,
                                        compact_evaluation)
from train_ppo_pi_observable import output_paths as observable_paths
from train_ppo_pi_lowstd import output_paths as previous_pi_paths


REFERENCE_NAMES = ("A_7d_ppo_p", "B_7d_ppo_direct_pi")
COMMON_METRICS = (
    "episodes", "successes", "success_rate", "mean_final_distance_m",
    "mean_successful_completion_time_s", "mean_episode_length",
    "near_target_actual_speed_m_s", "near_target_command_speed_m_s",
    "crossing_rate", "action_saturation_fraction",
)


def report_dir(root: Path = MUJOCO_DIR) -> Path:
    return root / "reports" / "ppo_pi_observability"


def _publish_or_validate(path: Path, value: dict) -> dict:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(f"Existing report differs: {path}")
        return existing
    write_condition(path, value)
    return value


def validate_reference_replay(saved: dict, replayed: dict) -> None:
    """A cached near-target summary is usable only after a fresh identical run."""
    for key in COMMON_METRICS:
        left, right = saved[key], replayed[key]
        if left is None or right is None:
            matches = left is right
        elif isinstance(left, (float, int)) and isinstance(right, (float, int)):
            matches = math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-10)
        else:
            matches = left == right
        if not matches:
            raise ValueError(f"Cached reference replay differs for {key}")


def evaluate_old_policy_pair(root: Path = MUJOCO_DIR, *,
                             verify_cached_by_rerun: bool = False) -> dict:
    """Run the same frozen seed-0 actor with P and then direct PI on holdout."""
    source = json.loads((root / "reports" /
                         "ppo_waypoint_v2_lowstd_multiseed.json").read_text())
    rows = [row for row in source["rows"] if row["seed"] == 0]
    if len(rows) != 1:
        raise ValueError("Need exactly the seed-0 frozen 7D PPO checkpoint")
    checkpoint = Path(rows[0]["checkpoint"])
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = PPO.load(checkpoint, device="cpu")
    if (model.observation_space.shape != (7,)
            or model.action_space.shape != (4,)
            or model.num_timesteps != 100_352):
        raise ValueError("Frozen P-controller PPO checkpoint differs")
    waypoints = build_holdout_waypoints()
    digest = waypoint_sha256(waypoints)
    if len(waypoints) != 100:
        raise ValueError("Need the same fixed 100 holdout targets")
    results = {}
    for name, env_type in ((REFERENCE_NAMES[0], MineUAVEnv),
                           (REFERENCE_NAMES[1], MineUAVPIEnv)):
        path = report_dir(root) / f"{name}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved["waypoint_sha256"] != digest
                    or saved["checkpoint_sha256"] != checkpoint_sha
                    or saved["training_seed"] != 0
                    or saved["evaluation"]["episodes"] != 100):
                raise ValueError(f"Existing reference evaluation differs: {path}")
            results[name] = saved
            if not verify_cached_by_rerun:
                continue
        env = env_type(render_mode=None, reward_version="v2",
                       target_distribution="full")
        try:
            evaluation = evaluate_exploration(model, env, waypoints,
                                              deterministic=True)
        finally:
            env.close()
        if (evaluation["seeds"] != [seed for seed, _ in waypoints]
                or evaluation["targets"] != [target for _, target in waypoints]):
            raise ValueError(f"{name} waypoint pairing differs")
        replayed = {
            "experiment": name,
            "training_seed": 0,
            "waypoint_sha256": digest,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "deterministic": True,
            "evaluation": compact_evaluation(evaluation),
        }
        if path.exists():
            validate_reference_replay(results[name]["evaluation"],
                                      replayed["evaluation"])
        else:
            results[name] = _publish_or_validate(path, replayed)
    return results


def build_paired_comparison(references: dict, c_report: dict, d_report: dict,
                            *, expected_digest: str) -> dict:
    """Require all four to share seed/targets, then keep only common metrics."""
    if (c_report["seed"] != 0 or d_report["seed"] != 0
            or c_report["reward_version"] != "v2"
            or d_report["reward_version"] != "v2"
            or c_report["actual_timesteps"] != 100_352
            or d_report["actual_timesteps"] != 100_352):
        raise ValueError("Seed, Reward V2 or training budget differs")
    if (c_report["observation_shape"] != [7]
            or d_report["observation_shape"] != [10]):
        raise ValueError("7D/10D observation comparison differs")
    if (c_report["waypoint_sha256"]["holdout"] != expected_digest
            or d_report["waypoint_sha256"]["holdout"] != expected_digest
            or any(references[name]["waypoint_sha256"] != expected_digest
                   for name in REFERENCE_NAMES)):
        raise ValueError("The holdout waypoint sets differ")
    if any(references[name]["training_seed"] != 0 for name in REFERENCE_NAMES):
        raise ValueError("Reference training seed differs")
    evaluations = {
        **{name: references[name]["evaluation"] for name in REFERENCE_NAMES},
        "C_7d_ppo_trained_pi": c_report["milestones"]["100k"]
            ["evaluation"]["holdout"],
        "D_10d_ppo_trained_pi": d_report["milestones"]["100k"]
            ["evaluation"]["holdout"],
    }
    rows = {}
    for name, evaluation in evaluations.items():
        if (evaluation["episodes"] != 100
                or evaluation["success_rate"] != evaluation["successes"] / 100):
            raise ValueError(f"Incomplete fixed-100 evaluation: {name}")
        rows[name] = {key: evaluation[key] for key in COMMON_METRICS}
    return {
        "experiment": "PI integral-acceleration observation ablation; nominal only",
        "training_seed": 0,
        "holdout_digest": expected_digest,
        "holdout": rows,
        "d_minus_c_success_percentage_points":
            rows["D_10d_ppo_trained_pi"]["successes"]
            - rows["C_7d_ppo_trained_pi"]["successes"],
        "d_minus_c_crossing_percentage_points":
            100 * (rows["D_10d_ppo_trained_pi"]["crossing_rate"]
                   - rows["C_7d_ppo_trained_pi"]["crossing_rate"]),
    }


def verify_report_artifacts(report: dict, *, expected_shape: tuple[int, ...],
                            waypoints: dict, require_training_hash: bool) -> dict:
    """Read back every milestone model and PI trace before using its metrics."""
    if (report["seed"] != 0 or report["reward_version"] != "v2"
            or tuple(report["observation_shape"]) != expected_shape
            or report["action_shape"] != [4]
            or report["actual_timesteps"] != 100_352
            or (report["physics_hz"], report["controller_hz"],
                report["policy_hz"]) != (500, 100, 25)
            or report["n_envs"] != 8 or report["n_steps"] != 256):
        raise ValueError("Saved PPO/PI task configuration differs")
    controller = report["controller"]
    if (controller["mode"] != "velocity_pi"
            or controller["ki_xy_s2"] != 0.5
            or controller["ki_z_s2"] != 0.8
            or controller["integral_accel_limit_xy_m_s2"] != 1.5
            or controller["integral_accel_limit_z_m_s2"] != 1.5):
        raise ValueError("Saved PI controller configuration differs")
    if (set(waypoints) != {"benchmark", "holdout"}
            or any(len(waypoints[task]) != 100 or
                   report["waypoint_sha256"][task] != waypoint_sha256(waypoints[task])
                   for task in waypoints)):
        raise ValueError("Saved benchmark/holdout target digest differs")
    checkpoint_hashes = {}
    trace_count = 0
    for label, expected_steps in (("20k", 20_480), ("50k", 51_200),
                                  ("100k", 100_352)):
        milestone = report["milestones"][label]
        if milestone["actual_timesteps"] != expected_steps:
            raise ValueError("Checkpoint timestep differs")
        checkpoint = Path(milestone["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        recorded = milestone.get("checkpoint_sha256")
        if (recorded is not None and digest != recorded) or (
                require_training_hash and recorded is None):
            raise ValueError(f"{label} checkpoint hash differs or is absent")
        checkpoint_hashes[label] = digest
        model = PPO.load(checkpoint, device="cpu")
        if (model.num_timesteps != expected_steps
                or model.observation_space.shape != expected_shape
                or model.action_space.shape != (4,)
                or model.n_envs != 8 or model.n_steps != 256
                or model.batch_size != 256 or model.n_epochs != 10
                or model.policy.net_arch != {"pi": [64, 64], "vf": [64, 64]}
                or model.policy.log_std_init != -2.0
                or model.learning_rate != 3e-4 or model.gamma != 0.99
                or model.gae_lambda != 0.95 or model.clip_range(1.0) != 0.2
                or model.ent_coef != 0.0 or model.vf_coef != 0.5
                or model.max_grad_norm != 0.5 or model.device.type != "cpu"):
            raise ValueError(f"{label} checkpoint PPO metadata differs")
        for task, targets in waypoints.items():
            evaluation = milestone["evaluation"][task]
            trace_path = Path(evaluation["integral_trace_path"])
            if not trace_path.is_file():
                raise FileNotFoundError(trace_path)
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            rows = trace["episodes"]
            if (trace["waypoint_seeds"] != [seed for seed, _ in targets]
                    or trace["targets"] != [target for _, target in targets]
                    or len(rows) != evaluation["episodes"]
                    or evaluation["episodes"] != 100
                    or trace["policy_steps"] != sum(len(row["steps"]) for row in rows)
                    or trace["policy_steps"] != evaluation["integral"]["policy_step_count"]
                    or not evaluation["integral"]["all_resets_zero"]):
                raise ValueError(f"{label}/{task} PI trace summary differs")
            for row, (seed, target) in zip(rows, targets):
                if (row["seed"] != seed or row["target_m"] != target
                        or row["initial_integral_error_m"] != [0, 0, 0]
                        or not row["steps"]
                        or any(step["policy_step"] != index
                               for index, step in enumerate(row["steps"]))):
                    raise ValueError(f"{label}/{task} PI trace episode differs")
            trace_count += 1
    return {"checkpoint_sha256": checkpoint_hashes,
            "trace_count": trace_count}


def publish_comparison(root: Path = MUJOCO_DIR) -> dict:
    references = evaluate_old_policy_pair(root, verify_cached_by_rerun=True)
    c = json.loads(previous_pi_paths(root)["report"].read_text())
    d = json.loads(observable_paths(root)["report"].read_text())
    waypoints = {"benchmark": fixed_waypoints("full"),
                 "holdout": build_holdout_waypoints()}
    c_verified = verify_report_artifacts(c, expected_shape=(7,),
                                         waypoints=waypoints,
                                         require_training_hash=False)
    d_verified = verify_report_artifacts(d, expected_shape=(10,),
                                         waypoints=waypoints,
                                         require_training_hash=True)
    expected_digest = waypoint_sha256(waypoints["holdout"])
    result = build_paired_comparison(references, c, d,
                                     expected_digest=expected_digest)
    result["artifact_verification"] = {
        "C_7d": c_verified,
        "D_10d": d_verified,
        "A_B_cached_reference_replayed": True,
        "note": "The previous 7D report predates training-time checkpoint hashes; "
                "10D hashes are present in each milestone.",
    }
    result["benchmark_10d"] = {
        label: d["milestones"][label]["evaluation"]["benchmark"]
        for label in ("20k", "50k", "100k")
    }
    result["holdout_10d"] = {
        label: d["milestones"][label]["evaluation"]["holdout"]
        for label in ("20k", "50k", "100k")
    }
    return _publish_or_validate(report_dir(root) / "comparison_verified.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--references-only", action="store_true")
    args = parser.parse_args()
    if args.references_only:
        outcome = evaluate_old_policy_pair()
        print(json.dumps({name: row["evaluation"]["successes"]
                          for name, row in outcome.items()}, indent=2))
    else:
        outcome = publish_comparison()
        print(json.dumps({name: row["successes"]
                          for name, row in outcome["holdout"].items()}, indent=2))
