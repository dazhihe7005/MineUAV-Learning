"""Read-only integrity audit and figures for the one formal LiDAR PPO seed0 run."""

from __future__ import annotations

import hashlib
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .ppo_formal_scheduler import load_l6_prereg
from .ppo_training_orchestration import (_runtime_paths, _weights_sha, aligned_milestones,
                                         l8_execution_protocol, source_hashes)
from .ppo_training_runner import _atomic_json, _historical_model_hashes, protocol_hash
from .ppo_formal_seed0_health import L9_RUNTIME_PATHS
from .scene import BASE_MODEL, BASE_SHA256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines()]


def _roll(values: list[float], window: int = 100) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    numerator = np.cumsum(np.insert(source, 0, 0.))
    denominator = np.minimum(np.arange(1, len(source) + 1), window)
    left = np.maximum(np.arange(len(source)) + 1 - window, 0)
    return (numerator[1:] - numerator[left]) / denominator


def build_interrupted_seed0_report(prereg_path: Path, raw_dir: Path) -> tuple[dict, dict]:
    """Audit surviving evidence without inventing final counters or a completion marker."""
    raw = Path(raw_dir)
    if (raw / "formal.complete.json").exists() or (raw / "formal_summary.json").exists():
        raise ValueError("interrupted analysis rejects completion evidence")
    if not (raw / "formal.attempt").is_file():
        raise ValueError("interrupted run lacks its unique attempt marker")
    prereg = load_l6_prereg(prereg_path)
    preflight = json.loads((raw / "preflight_identity.json").read_text())
    failure = json.loads((raw / "failure_capture.json").read_text())
    protocol_sha = protocol_hash(l8_execution_protocol(prereg))
    l8_sources = source_hashes(_runtime_paths())
    l9_sources = source_hashes(L9_RUNTIME_PATHS)
    if (preflight["runtime_source_sha256"] != l8_sources
            or preflight["l8_protocol_sha256"] != protocol_sha
            or preflight["historical_checkpoint_sha256"] != _historical_model_hashes()
            or preflight["l6_manifest_sha256"] != prereg["scenario_manifest_sha256"]
            or _sha(BASE_MODEL) != BASE_SHA256
            or failure["exit_code"] != 1
            or failure["exception_message"] != "invalid or nonexclusive training episode outcome"):
        raise ValueError("interrupted run identity or captured failure mismatch")
    episodes = _read_jsonl(raw / "train_episodes.jsonl")
    resources = _read_jsonl(raw / "resource_samples.jsonl")
    train_ids = {r["identity_sha256"] for r in prereg["scenario_manifest"]["train"]}
    validation_ids = {r["identity_sha256"] for r in prereg["scenario_manifest"]["validation"]}
    test_ids = {r["identity_sha256"] for r in prereg["scenario_manifest"]["test"]}
    if (train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids
            or not resources or not episodes
            or [r["policy_step"] for r in resources] !=
            list(range(128, resources[-1]["policy_step"] + 1, 128))
            or any(a["policy_end_step"] >= b["policy_end_step"] for a, b in zip(episodes, episodes[1:]))
            or any(r["scenario_identity_sha256"] not in train_ids for r in episodes)
            or any(r["policy_end_step"] > resources[-1]["policy_step"] + 127 for r in episodes)):
        raise ValueError("interrupted Train/resource evidence invalid")
    terminal_types = ("success", "collision", "time_limit", "physical_failure", "sensor_failure")
    for row in episodes:
        if row["outcome"] not in terminal_types or any(not np.isfinite(row[key]) for key in
                ("episode_return", "trajectory_length_m", "net_displacement_m",
                 "mean_action_norm", "final_target_distance_m", "sim_end_time_s")):
            raise ValueError("invalid persisted episode telemetry")
    event_files = sorted(raw.glob("L8_SEED0_STEP*_event.json"))
    if len(event_files) != 1:
        raise ValueError("this interrupted attempt requires its one certified milestone")
    event = json.loads(event_files[0].read_text())
    if event["actual_execution_step"] != 100096 or event["nominal_step"] != 100000:
        raise ValueError("unexpected completed milestone")
    step = event["actual_execution_step"]
    archive = raw / f"L8_SEED0_STEP{step:07d}.zip"
    sidecar = json.loads((raw / f"L8_SEED0_STEP{step:07d}.json").read_text())
    signature = sidecar.pop("sidecar_payload_sha256")
    health = json.loads((raw / f"L9_HEALTH_STEP{step:07d}.json").read_text())
    validation = event["validation"]
    if (signature != protocol_hash(sidecar)
            or sidecar["checkpoint_sha256"] != _sha(archive)
            or event["checkpoint"]["checkpoint_sha256"] != _sha(archive)
            or sidecar["training_timesteps"] != step
            or sidecar["execution_protocol_sha256"] != protocol_sha
            or sidecar["historical_checkpoint_sha256"] != _historical_model_hashes()
            or sidecar["interrupted_formal_run_policy"] != "RESTART_REQUIRED"
            or health["l9_runtime_source_sha256"] != l9_sources
            or health["actual_execution_step"] != step
            or event["event_after_optimizer_updates"] != 1564
            or event["train_rng_unchanged_during_validation"] is not True
            or event["test_episode_count"] != 0
            or validation["optimizer_unchanged"] is not True
            or validation["probe_only"] is not False
            or validation["training_timesteps_before_after"] != step
            or validation["overall"]["total"] != 6
            or validation["overall"]["probe_cutoff"] != 0
            or {r["scenario_identity_sha256"] for r in validation["episodes"]} != validation_ids
            or sum(validation["overall"][key] for key in terminal_types) != 6):
        raise ValueError("surviving checkpoint/Validation identity incomplete")
    from stable_baselines3 import PPO
    import torch
    model = PPO.load(archive, device="cpu")
    optimizer_steps = {int(state["step"].item()) for state in model.policy.optimizer.state.values()}
    if model.num_timesteps != step or model._n_updates != 1564 or optimizer_steps != {3128}:
        raise ValueError("surviving checkpoint optimizer/timestep identity invalid")
    if any(not torch.isfinite(value).all() for value in model.policy.state_dict().values()):
        raise ValueError("surviving checkpoint policy nonfinite")
    if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
           for state in model.policy.optimizer.state.values() for value in state.values()):
        raise ValueError("surviving checkpoint optimizer nonfinite")
    del model
    counts = Counter(r["outcome"] for r in episodes)
    last_step = resources[-1]["policy_step"]
    visits = sidecar["scheduler_state"]["visit_counts"]
    selected = sidecar["scheduler_state"]["selected_identities"]
    if set(visits) != train_ids or dict(Counter(selected)) != visits:
        raise ValueError("surviving scheduler identity mismatch")
    milestone = {
        "nominal_step": 100000, "actual_execution_step": step,
        "checkpoint_sha256": _sha(archive),
        "validation_overall": validation["overall"],
        "validation_by_scenario": [{"scenario_identity_sha256": r["scenario_identity_sha256"],
                                    "outcome": r["outcome"]} for r in validation["episodes"]],
        "train_completed_episodes": health["completed_episodes"],
        "train_terminal_counts": health["terminal_counts"],
        "ppo_diagnostics": health["ppo_diagnostics"],
    }
    report = {
        "task": "Phase 3E-L9 Formal LiDAR-CNN-PPO Training — Seed 0",
        "overall_result": "FAIL", "formal_training_complete": False,
        "FORMAL_TRAINING_READY": "NO", "actual_timesteps": None,
        "requested_actual_timesteps": 1000064, "seed": 0, "workers": 1,
        "formal_attempt_count": 1, "automatic_restart_count": 0,
        "continuation_policy": "RESTART_REQUIRED", "test_episode_count": 0,
        "last_persisted_environment_step": last_step,
        "successful_returned_environment_step_bounds": [last_step, last_step + 127],
        "failing_environment_step_bounds": [last_step + 1, last_step + 128],
        "last_completed_episode_step": episodes[-1]["policy_end_step"],
        "exact_failure_step": "NOT RECORDED", "exact_failure_info": "NOT RECORDED",
        "final_model_sha256": None,
        "status": {
            "formal_seed0_training": "FAIL", "all_10_milestones": "FAIL",
            "surviving_checkpoint_identity": "PASS", "six_scenario_validation_at_100096": "PASS",
            "final_checkpoint": "NOT TESTED", "test_sealed": "PASS",
            "historical_models_unchanged": "PASS", "original_7kg_dynamics_unchanged": "PASS",
            "runtime_source_identity": "PASS", "recorded_resource_guard": "PASS",
            "complete_train_visit_history": "BLOCKED", "formal_navigation_performance": "NOT TESTED",
            "px4_or_real_mid360s": "NOT TESTED",
        },
        "failure": {
            "captured_exception": failure,
            "accounting_contract_defect": "Confirmed",
            "explanation": "navigation_env retains raw physical termination_reason while setting canonical physical_failure flag; frozen L8 TrainingMetricsEnv rejects that raw reason",
            "specific_physical_trigger": "Unresolved: failing info was not persisted",
            "source_evidence": ["navigation_env.py:284-300", "ppo_training_orchestration.py:223-226",
                                "mine_uav_env.py:177-189"],
            "offline_reproduction": "outside_flight_area with exactly one physical_failure flag raises the same accounting RuntimeError; zero optimizer updates",
        },
        "formal_protocol": l8_execution_protocol(prereg),
        "formal_protocol_sha256": protocol_sha,
        "scenario_manifest_sha256": prereg["scenario_manifest_sha256"],
        "software_versions": {name: __import__("importlib.metadata", fromlist=["version"]).version(name)
                              for name in ("mujoco", "stable-baselines3", "torch", "gymnasium", "numpy")},
        "recorded_train_episode_count": len(episodes),
        "recorded_train_terminal_counts": {key: counts[key] for key in terminal_types},
        "recorded_train_mean_return": float(np.mean([r["episode_return"] for r in episodes])),
        "recorded_motionless_episode_fraction": sum(r["motionless_episode"] for r in episodes) / len(episodes),
        "recorded_train_completed_episode_counts_by_identity": dict(Counter(
            r["scenario_identity_sha256"] for r in episodes)),
        "train_visit_counts_at_checkpoint_100096": visits,
        "milestones": [milestone],
        "resource": {
            "last_recorded_wall_seconds": resources[-1]["wall_elapsed_s"],
            "last_recorded_cpu_seconds": resources[-1]["cpu_elapsed_s"],
            "last_recorded_peak_rss_mib": max(r["rss_peak_mib"] for r in resources),
            "last_recorded_current_rss_mib": resources[-1]["rss_current_mib"],
            "recorded_mem_available_min_mib": min(r["mem_available_mib"] for r in resources),
            "recorded_disk_free_min_bytes": min(r["disk_free_bytes"] for r in resources),
            "sample_count": len(resources), "oom_observed": False,
            "termination_is_oom": False, "exact_final_wall_time": "NOT RECORDED",
            "exact_final_peak_rss": "NOT RECORDED",
            "environment_scan_encoding_update_timings": "NOT RECORDED: in-memory aggregates lost on exception",
        },
        "raw_evidence_sha256": {path.name: _sha(path) for path in sorted(raw.iterdir()) if path.is_file()},
        "runtime_source_sha256": {"l8": l8_sources, "l9": l9_sources},
        "historical_checkpoint_sha256": _historical_model_hashes(),
        "original_mjcf_sha256": BASE_SHA256,
        "verification": {"tests": "PENDING", "posttrain_independent_review": "PENDING"},
        "limitations": [
            "No final checkpoint or complete result: exact terminal counters and final train step are unavailable.",
            "Recorded 706 episode outcomes exclude the unrecorded failing transition.",
            "Complete scene reset history is certified through step100096 only; subsequent terminal episode identities are recorded.",
            "Only one post-update PPO diagnostic sample and one Validation event survived.",
            "The physical trigger is not recorded; it is not valid to conclude a specific tilt, altitude or nonfinite failure.",
            "Seven old models and frozen physics/protocol remain unchanged; future repair needs a separately audited protocol and new authorized attempt.",
        ],
    }
    data = {"episodes": episodes, "resources": resources, "event_summaries": [milestone],
            "summary": {"train_scenario_visit_counts": visits},
            "visits_label": "Train selections at surviving checkpoint step100096 (partial run)",
            "partial_run": True}
    return report, data


def build_formal_seed0_report(prereg_path: Path, raw_dir: Path, *,
                              tests: dict, pretrain_review: str,
                              posttrain_review: str) -> tuple[dict, dict]:
    """Reject incomplete/identity-mismatched evidence; do not run an evaluation episode."""
    raw = Path(raw_dir)
    prereg = load_l6_prereg(prereg_path)
    summary_path = raw / "formal_summary.json"
    summary = json.loads(summary_path.read_text())
    complete = json.loads((raw / "formal.complete.json").read_text())
    preflight = json.loads((raw / "preflight_identity.json").read_text())
    if (pretrain_review != "GO" or posttrain_review != "GO"
            or tests.get("failed") != 0 or tests.get("passed", 0) < 714):
        raise ValueError("formal report requires passing tests and both independent reviews")
    expected_events = aligned_milestones(1_000_000, 100_000, 128)
    expected_protocol_sha = protocol_hash(l8_execution_protocol(prereg))
    expected_l8_sources = source_hashes(_runtime_paths())
    expected_l9_sources = source_hashes(L9_RUNTIME_PATHS)
    if (complete != {"summary_sha256": _sha(summary_path),
                     "actual_timesteps": 1_000_064, "events": 10}
            or preflight["runtime_source_sha256"] != expected_l8_sources
            or summary["runtime_source_sha256"] != expected_l8_sources
            or preflight["l8_protocol_sha256"] != expected_protocol_sha
            or summary["l8_protocol_sha256"] != expected_protocol_sha
            or summary["l6_manifest_sha256"] != prereg["scenario_manifest_sha256"]
            or summary["historical_checkpoint_sha256"] != _historical_model_hashes()
            or _sha(BASE_MODEL) != BASE_SHA256):
        raise ValueError("formal completion, frozen protocol or historical identity mismatch")
    if (summary["mode"] != "formal" or summary["seed"] != 0 or summary["workers"] != 1
            or summary["training_actual_timesteps"] != 1_000_064
            or summary["rollout_count"] != 7813 or summary["ppo_n_updates"] != 15626
            or summary["optimizer_step_count"] != 31252
            or summary["environment_step_ms"]["count"] != 1_000_064
            or summary["event_count"] != 10 or len(summary["events"]) != 10
            or summary["test_episode_count"] != 0
            or summary["finite_policy_and_gradients"] is not True
            or summary["out_of_memory"] is not False):
        raise ValueError("formal training step, optimizer or event accounting incomplete")
    train_ids = {x["identity_sha256"] for x in prereg["scenario_manifest"]["train"]}
    validation_ids = {x["identity_sha256"] for x in prereg["scenario_manifest"]["validation"]}
    test_ids = {x["identity_sha256"] for x in prereg["scenario_manifest"]["test"]}
    selected = summary["train_scenario_selected_identities"]
    counts = summary["train_scenario_visit_counts"]
    if (set(selected) != train_ids or dict(Counter(selected)) != counts
            or set(counts) != train_ids or sum(counts.values()) != len(selected)
            or train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids):
        raise ValueError("Train/Validation/Test scenario identity separation failed")
    episodes = _read_jsonl(raw / "train_episodes.jsonl")
    resources = _read_jsonl(raw / "resource_samples.jsonl")
    terminal_types = ("success", "collision", "time_limit", "physical_failure", "sensor_failure")
    terminals = Counter(row["outcome"] for row in episodes)
    if (len(episodes) != summary["episode_count"]
            or any(row["scenario_identity_sha256"] not in train_ids for row in episodes)
            or any(row["outcome"] not in terminal_types for row in episodes)
            or {key: terminals[key] for key in terminal_types} != summary["terminal_counts"]
            or any(not 0 < row["policy_end_step"] <= 1_000_064 for row in episodes)
            or any(a["policy_end_step"] >= b["policy_end_step"] for a, b in zip(episodes, episodes[1:]))
            or len(resources) != 7813
            or [r["policy_step"] for r in resources] != list(range(128, 1_000_065, 128))):
        raise ValueError("raw Train episode or resource telemetry incomplete")
    for row in episodes:
        for key in ("episode_return", "trajectory_length_m", "net_displacement_m",
                    "mean_action_norm", "final_target_distance_m", "sim_end_time_s"):
            if not np.isfinite(row[key]):
                raise ValueError("nonfinite Train episode telemetry")
    event_summaries = []
    for scheduled, event in zip(expected_events, summary["events"]):
        step = scheduled.actual_step
        run_id = f"L8_SEED0_STEP{step:07d}"
        event_path = raw / f"{run_id}_event.json"
        sidecar_path = raw / f"{run_id}.json"
        archive = raw / f"{run_id}.zip"
        health_path = raw / f"L9_HEALTH_STEP{step:07d}.json"
        sidecar = json.loads(sidecar_path.read_text())
        signature = sidecar.pop("sidecar_payload_sha256")
        validation = event["validation"]
        health = json.loads(health_path.read_text())
        if (event != json.loads(event_path.read_text())
                or event["nominal_step"] != scheduled.nominal_step
                or event["actual_execution_step"] != step
                or event["event_after_optimizer_updates"] != 2 * (step // 128)
                or event["train_rng_unchanged_during_validation"] is not True
                or event["test_episode_count"] != 0
                or signature != protocol_hash(sidecar)
                or sidecar["checkpoint_sha256"] != _sha(archive)
                or event["checkpoint"]["checkpoint_sha256"] != _sha(archive)
                or sidecar["training_timesteps"] != step
                or sidecar["execution_protocol_sha256"] != expected_protocol_sha
                or sidecar["continuation_class"] != "WARM_START"
                or sidecar["interrupted_formal_run_policy"] != "RESTART_REQUIRED"
                or validation["optimizer_unchanged"] is not True
                or validation["training_timesteps_before_after"] != step
                or validation["probe_only"] is not False
                or validation["overall"]["total"] != 6
                or validation["overall"]["probe_cutoff"] != 0
                or {r["scenario_identity_sha256"] for r in validation["episodes"]} != validation_ids
                or sum(validation["overall"][name] for name in terminal_types) != 6
                or health["actual_execution_step"] != step
                or health["optimizer_updates"] != 2 * (step // 128)
                or health["policy_steps"] != step or health["test_episode_count"] != 0
                or health["l9_runtime_source_sha256"] != expected_l9_sources
                or health["completed_episodes"] != sum(health["terminal_counts"].values())
                or health["train_episodes_jsonl_rows"] != health["completed_episodes"]
                or health["resource_sample_count"] != step // 128):
            raise ValueError(f"milestone {step} checkpoint, health or Validation incomplete")
        event_summaries.append({
            "nominal_step": scheduled.nominal_step,
            "actual_execution_step": step,
            "checkpoint_sha256": _sha(archive),
            "checkpoint_sidecar_sha256": _sha(sidecar_path),
            "event_sha256": _sha(event_path),
            "health_sha256": _sha(health_path),
            "validation_overall": validation["overall"],
            "validation_by_scenario": [{"scenario_identity_sha256": row["scenario_identity_sha256"],
                                         "outcome": row["outcome"]}
                                        for row in validation["episodes"]],
            "train_completed_episodes": health["completed_episodes"],
            "train_terminal_counts": health["terminal_counts"],
            "ppo_diagnostics": health["ppo_diagnostics"],
            "mean_action_norm": health["mean_action_norm"],
            "mean_displacement_m": health["mean_displacement_m"],
            "motionless_episode_fraction": health["motionless_episode_fraction"],
            "rss_peak_mib": health["rss_peak_mib"],
            "mem_available_min_mib": health["mem_available_min_mib"],
        })
    previous_step = 0
    for milestone in event_summaries:
        current_step = milestone["actual_execution_step"]
        segment_rows = [row for row in episodes if previous_step < row["policy_end_step"] <= current_step]
        segment_counts = Counter(row["outcome"] for row in segment_rows)
        milestone["train_segment"] = {
            "completed_episodes": len(segment_rows),
            "terminal_counts": {name: segment_counts[name] for name in terminal_types},
            "success_rate": segment_counts["success"] / len(segment_rows) if segment_rows else None,
            "collision_rate": segment_counts["collision"] / len(segment_rows) if segment_rows else None,
            "timeout_rate": segment_counts["time_limit"] / len(segment_rows) if segment_rows else None,
            "mean_episode_return": float(np.mean([r["episode_return"] for r in segment_rows]))
            if segment_rows else None,
            "mean_action_norm": float(np.mean([r["mean_action_norm"] for r in segment_rows]))
            if segment_rows else None,
            "mean_net_displacement_m": float(np.mean([r["net_displacement_m"] for r in segment_rows]))
            if segment_rows else None,
            "motionless_episode_fraction": sum(r["motionless_episode"] for r in segment_rows) /
            len(segment_rows) if segment_rows else None,
        }
        previous_step = current_step
    if (len(resources) == 0 or min(r["mem_available_mib"] for r in resources) < 1536
            or summary["system_min_available_mib"] < 1536):
        raise ValueError("resource guard or telemetry evidence incomplete")
    # The final checkpoint, not the best Validation milestone, is the frozen selection.
    from stable_baselines3 import PPO
    import torch
    final_archive = raw / f"L8_SEED0_STEP{1_000_064:07d}.zip"
    final_model = PPO.load(final_archive, device="cpu")
    if (final_model.num_timesteps != 1_000_064 or final_model._n_updates != 15626
            or _weights_sha(final_model.policy.action_net) != summary["actor_sha256_after"]
            or _weights_sha(final_model.policy.value_net) != summary["critic_sha256_after"]
            or not final_model.policy.optimizer.state):
        raise ValueError("final checkpoint does not reload the selected trained model")
    for value in final_model.policy.state_dict().values():
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite final policy weights")
    for state in final_model.policy.optimizer.state.values():
        if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
               for value in state.values()):
            raise ValueError("nonfinite final optimizer state")
    del final_model
    report = {
        "task": "Phase 3E-L9 Formal LiDAR-CNN-PPO Training — Seed 0",
        "status": {
            "formal_seed0_training": "PASS", "all_10_milestones": "PASS",
            "final_checkpoint_identity": "PASS", "six_scenario_validation": "PASS",
            "test_sealed": "PASS", "historical_models_unchanged": "PASS",
            "original_7kg_dynamics_unchanged": "PASS", "no_oom": "PASS",
            "multi_seed_comparison": "NOT TESTED", "sealed_test_evaluation": "NOT TESTED",
            "px4_or_real_mid360s": "NOT TESTED",
        },
        "formal_protocol_sha256": expected_protocol_sha,
        "scenario_manifest_sha256": prereg["scenario_manifest_sha256"],
        "seed": 0, "workers": 1, "actual_timesteps": 1_000_064,
        "rollout_count": 7813, "ppo_n_updates": 15626,
        "optimizer_step_count": 31252, "test_episode_count": 0,
        "reward_version": l8_execution_protocol(prereg)["reward_version"],
        "train_scenario_visit_counts": counts,
        "train_episode_count": len(episodes),
        "actual_training_physics_ticks": summary["training_physics_ticks"],
        "train_terminal_counts": summary["terminal_counts"],
        "mean_train_episode_return": summary["mean_completed_episode_return"],
        "final_model_sha256": event_summaries[-1]["checkpoint_sha256"],
        "milestones": event_summaries,
        "resource": {
            "wall_seconds": summary["wall_seconds"],
            "rss_peak_mib": summary["rss_peak_mib"],
            "mem_available_min_mib": summary["system_min_available_mib"],
            "disk_free_min_bytes": min(r["disk_free_bytes"] for r in resources),
            "last_resource_cpu_elapsed_s": resources[-1]["cpu_elapsed_s"],
            "last_resource_wall_elapsed_s": resources[-1]["wall_elapsed_s"],
            "environment_step_ms": summary["environment_step_ms"],
            "lidar_scan_ms": summary["lidar_scan_ms"],
            "lidar_encoding_ms": summary["lidar_encoding_ms"],
            "ppo_train_update_ms": summary["ppo_train_update_ms"],
            "out_of_memory": False,
        },
        "raw_evidence": {
            "summary_sha256": _sha(summary_path),
            "completion_marker_sha256": _sha(raw / "formal.complete.json"),
            "preflight_identity_sha256": _sha(raw / "preflight_identity.json"),
            "episodes_jsonl_sha256": _sha(raw / "train_episodes.jsonl"),
            "resource_jsonl_sha256": _sha(raw / "resource_samples.jsonl"),
            "l8_runtime_source_sha256": expected_l8_sources,
            "l9_runtime_source_sha256": expected_l9_sources,
            "historical_checkpoint_sha256": _historical_model_hashes(),
            "original_mjcf_sha256": BASE_SHA256,
        },
        "verification": {"tests": tests, "pretrain_independent_review": pretrain_review,
                         "posttrain_independent_review": posttrain_review},
        "limitations": [
            "One seed and six Validation scenes cannot establish seed variance or held-out generalization.",
            "Validation is diagnostic; final checkpoint is the preregistered selection.",
            "LiDAR is an approximate frozen-pose MuJoCo raycast and state is simulation ground truth.",
            "An interrupted run is RESTART_REQUIRED, not an exact checkpoint resume.",
            "System OpenCV/NumPy ABI import warning persists in optional SB3 Atari path; no cv2 hot-path use.",
        ],
    }
    return report, {"episodes": episodes, "resources": resources,
                    "event_summaries": event_summaries, "summary": summary}


def draw_formal_seed0_figures(data: dict, output_dir: Path) -> list[str]:
    """All figures use only observed Train/Validation/monitor samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = data["episodes"]
    resources = data["resources"]
    milestones = data["event_summaries"]
    x = np.asarray([r["policy_end_step"] for r in episodes])
    returns = [r["episode_return"] for r in episodes]
    files = []

    def save(name: str):
        if data.get("partial_run"):
            plt.gcf().suptitle(f"INTERRUPTED: recorded data only (through resource step {resources[-1]['policy_step']})",
                              fontsize=10)
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=150)
        plt.close()
        files.append(name)

    plt.figure(figsize=(9, 4))
    plt.scatter(x, returns, s=3, alpha=.18, label="completed episode")
    plt.plot(x, _roll(returns), label="last 100 completed mean")
    plt.xlabel("Actual training timesteps"); plt.ylabel("Episode return"); plt.legend()
    save("episode_reward.png")

    for outcome, name in (("success", "training_success_rate.png"),
                          ("collision", "training_collision_rate.png"),
                          ("time_limit", "training_timeout_rate.png")):
        plt.figure(figsize=(9, 4))
        plt.plot(x, _roll([float(r["outcome"] == outcome) for r in episodes]))
        plt.xlabel("Actual training timesteps")
        plt.ylabel(f"{outcome} fraction, last 100 completed episodes")
        plt.ylim(0, 1)
        save(name)

    scene_ids = sorted({r["scenario_identity_sha256"] for m in milestones
                        for r in m["validation_by_scenario"]})
    matrix = np.asarray([[float(next(r["outcome"] == "success" for r in m["validation_by_scenario"]
                                    if r["scenario_identity_sha256"] == sid))
                          for m in milestones] for sid in scene_ids])
    plt.figure(figsize=(10, 4))
    plt.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    plt.yticks(range(6), [sid[:10] for sid in scene_ids])
    plt.xticks(range(len(milestones)), [str(m["actual_execution_step"]) for m in milestones], rotation=45)
    plt.xlabel("Actual checkpoint step"); plt.ylabel("Validation scene identity")
    plt.colorbar(label="Success (0/1)")
    save("validation_six_scene_success.png")

    visits = data["summary"]["train_scenario_visit_counts"]
    plt.figure(figsize=(9, 4))
    plt.bar(range(12), [visits[key] for key in sorted(visits)])
    plt.xticks(range(12), [key[:8] for key in sorted(visits)], rotation=60)
    plt.ylabel("Training episode selections")
    if data.get("visits_label"):
        plt.title("At surviving checkpoint step100096; later reset visits incomplete", fontsize=10)
    save("train_scene_visits.png")

    actual = [m["actual_execution_step"] for m in milestones]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for key in ("train/policy_gradient_loss", "train/value_loss", "train/entropy_loss"):
        values = [m["ppo_diagnostics"].get(key, float("nan")) for m in milestones]
        axes[0].plot(actual, values, marker="o", label=key)
    axes[0].legend(); axes[0].set_ylabel("PPO loss")
    axes[1].plot(actual, [m["ppo_diagnostics"].get("train/approx_kl", float("nan"))
                          for m in milestones], marker="o", label="approx KL")
    axes[1].set_xlabel("Actual checkpoint step"); axes[1].set_ylabel("Approx KL")
    save("ppo_loss_approx_kl.png")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(x, _roll([r["mean_action_norm"] for r in episodes]))
    axes[0].set_ylabel("Mean action norm, last 100")
    axes[1].plot(x, _roll([r["net_displacement_m"] for r in episodes]), label="net displacement")
    axes[1].plot(x, _roll([r["trajectory_length_m"] for r in episodes]), label="trajectory length")
    axes[1].legend(); axes[1].set_ylabel("Meters, last 100"); axes[1].set_xlabel("Actual training timesteps")
    save("action_and_progress.png")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    rx = [r["policy_step"] for r in resources]
    axes[0].plot(rx, [r["rss_current_mib"] for r in resources], label="current RSS")
    axes[0].plot(rx, [r["rss_peak_mib"] for r in resources], label="peak RSS")
    axes[0].legend(); axes[0].set_ylabel("MiB")
    axes[1].plot(rx, [r["wall_elapsed_s"] / 3600 for r in resources])
    axes[1].set_ylabel("Elapsed wall hours"); axes[1].set_xlabel("Actual training timesteps")
    save("rss_and_training_time.png")
    return files


def write_formal_seed0_report(output: Path, prereg_path: Path, raw_dir: Path,
                              figure_dir: Path, *, tests: dict,
                              pretrain_review: str, posttrain_review: str) -> dict:
    if output.exists():
        raise FileExistsError("formal seed0 report must not overwrite existing evidence")
    report, data = build_formal_seed0_report(prereg_path, raw_dir, tests=tests,
                                             pretrain_review=pretrain_review,
                                             posttrain_review=posttrain_review)
    report["figures"] = draw_formal_seed0_figures(data, figure_dir)
    _atomic_json(output, report)
    return report


def write_interrupted_seed0_report(output: Path, prereg_path: Path, raw_dir: Path,
                                  figure_dir: Path, *, tests: dict,
                                  posttrain_review: str) -> dict:
    if output.exists():
        raise FileExistsError("formal seed0 failure report must not overwrite evidence")
    if posttrain_review != "GO" or tests.get("failed") != 0:
        raise ValueError("failure report requires passing analysis tests and independent review")
    report, data = build_interrupted_seed0_report(prereg_path, raw_dir)
    report["verification"] = {"tests": tests, "pretrain_independent_review": "GO",
                              "posttrain_independent_review": "GO: failure evidence accepted; no retry",
                              "formal_training_acceptance": "FAIL"}
    report["figures"] = draw_formal_seed0_figures(data, figure_dir)
    _atomic_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit completed formal seed0 evidence; no rollout")
    root = Path(__file__).resolve().parents[1] / "reports"
    parser.add_argument("--raw", type=Path,
                        default=root / "mujoco_lidar_ppo_formal_seed0_parts/run_01")
    parser.add_argument("--report", type=Path,
                        default=root / "mujoco_lidar_ppo_formal_seed0.json")
    parser.add_argument("--figures", type=Path,
                        default=root / "mujoco_lidar_ppo_formal_seed0_figures")
    parser.add_argument("--passed", type=int, required=True)
    parser.add_argument("--failed", type=int, required=True)
    parser.add_argument("--warnings", type=int, required=True)
    parser.add_argument("--posttrain-review", choices=("GO", "NO_GO"), required=True)
    parser.add_argument("--interrupted", action="store_true",
                        help="audit a failed attempt without inventing completion evidence")
    arguments = parser.parse_args(argv)
    tests = {"inherited": 698, "new": arguments.passed - 698,
             "passed": arguments.passed, "failed": arguments.failed,
             "skipped": 0, "warnings": arguments.warnings}
    if arguments.interrupted:
        report = write_interrupted_seed0_report(
            arguments.report, root / "mujoco_lidar_ppo_training_prereg_seed0.json",
            arguments.raw, arguments.figures, tests=tests,
            posttrain_review=arguments.posttrain_review)
    else:
        report = write_formal_seed0_report(
            arguments.report, root / "mujoco_lidar_ppo_training_prereg_seed0.json",
            arguments.raw, arguments.figures, tests=tests,
            pretrain_review="GO", posttrain_review=arguments.posttrain_review)
    print(json.dumps({"actual_timesteps": report["actual_timesteps"],
                      "final_model_sha256": report["final_model_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
