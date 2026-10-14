"""Read-only integrity audit of the single L8 Pilot; never performs inference."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .ppo_formal_scheduler import load_l6_prereg
from .ppo_training_orchestration import (PILOT_ID, _runtime_paths, aligned_milestones,
                                         l8_execution_protocol, source_hashes)
from .ppo_training_runner import _atomic_json, _historical_model_hashes, protocol_hash


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_orchestration_report(prereg_path: Path, raw_dir: Path, *,
                               verification: dict | None = None) -> dict:
    prereg = load_l6_prereg(prereg_path)
    raw = Path(raw_dir)
    preflight = json.loads((raw / "preflight_identity.json").read_text())
    summary_path = raw / "pilot_summary.json"
    summary = json.loads(summary_path.read_text())
    complete = json.loads((raw / "pilot.complete.json").read_text())
    if complete.get("summary_sha256") != _sha(summary_path):
        raise ValueError("Pilot summary hash differs from completion marker")
    if complete.get("actual_timesteps") != 8192 or complete.get("events") != 1:
        raise ValueError("Pilot completion marker is incomplete")
    expected_protocol = protocol_hash(l8_execution_protocol(prereg))
    current_sources = source_hashes(_runtime_paths())
    if (preflight.get("runtime_source_sha256") != current_sources
            or summary.get("runtime_source_sha256") != current_sources
            or preflight.get("l8_protocol_sha256") != expected_protocol
            or summary.get("l8_protocol_sha256") != expected_protocol
            or preflight.get("l6_manifest_sha256") != prereg["scenario_manifest_sha256"]
            or summary.get("l6_manifest_sha256") != prereg["scenario_manifest_sha256"]
            or preflight.get("network_implementation_sha256") !=
            prereg["network_implementation_sha256"]
            or preflight.get("reward_version") != l8_execution_protocol(prereg)["reward_version"]
            or preflight.get("historical_checkpoint_sha256") != _historical_model_hashes()
            or summary.get("historical_checkpoint_sha256") != _historical_model_hashes()):
        raise ValueError("preflight/runtime source, protocol or historical identity mismatch")
    if (summary.get("experiment_id") != PILOT_ID or summary.get("mode") != "pilot"
            or summary.get("seed") != 0 or summary.get("workers") != 1
            or summary.get("training_actual_timesteps") != 8192
            or summary.get("rollout_count") != 64
            or summary.get("ppo_n_updates") != 128
            or summary.get("optimizer_step_count") != 256
            or summary.get("ppo_train_update_ms", {}).get("count") != 64
            or summary.get("environment_step_ms", {}).get("count") != 8192
            or summary.get("training_physics_ticks", 0) <= 0
            or summary.get("test_episode_count") != 0
            or summary.get("event_count") != 1
            or summary.get("finite_policy_and_gradients") is not True
            or summary.get("out_of_memory") is not False
            or summary.get("actor_sha256_before") == summary.get("actor_sha256_after")
            or summary.get("critic_sha256_before") == summary.get("critic_sha256_after")
            or summary.get("system_min_available_mib", 0) < 1536):
        raise ValueError("Pilot step, optimizer, identity or memory evidence incomplete")
    train_rows = prereg["scenario_manifest"]["train"]
    train_ids = {row["identity_sha256"] for row in train_rows}
    selected = summary["train_scenario_selected_identities"]
    counts = summary["train_scenario_visit_counts"]
    if (not selected or set(selected) != train_ids or set(counts) != train_ids
            or dict(Counter(selected)) != counts or sum(counts.values()) != len(selected)):
        raise ValueError("Train scene visit record does not match frozen identities")
    outcomes = summary["terminal_counts"]
    if (set(outcomes) != {"success", "collision", "time_limit", "physical_failure", "sensor_failure"}
            or sum(outcomes.values()) != summary["episode_count"]
            or summary["episode_count"] > len(selected)):
        raise ValueError("Train terminal accounting is incomplete")
    event = summary["events"][0]
    event_path = raw / "L8_SEED0_STEP0008192_event.json"
    if event != json.loads(event_path.read_text()):
        raise ValueError("Pilot milestone event file differs from summary")
    archive = raw / "L8_SEED0_STEP0008192.zip"
    sidecar = json.loads((raw / "L8_SEED0_STEP0008192.json").read_text())
    signature = sidecar.pop("sidecar_payload_sha256")
    if (signature != protocol_hash(sidecar) or event["checkpoint"]["checkpoint_sha256"] != _sha(archive)
            or sidecar["checkpoint_sha256"] != _sha(archive)
            or sidecar["execution_protocol_sha256"] != expected_protocol
            or sidecar["training_timesteps"] != 8192
            or sidecar["continuation_class"] != "WARM_START"
            or sidecar["interrupted_formal_run_policy"] != "RESTART_REQUIRED"):
        raise ValueError("Pilot checkpoint or continuation identity mismatch")
    validation = event["validation"]
    validation_ids = {row["identity_sha256"] for row in prereg["scenario_manifest"]["validation"]}
    if (event["nominal_step"] != 8192 or event["actual_execution_step"] != 8192
            or event["event_after_optimizer_updates"] != 128
            or event["train_rng_unchanged_during_validation"] is not True
            or event["test_episode_count"] != 0
            or validation["probe_only"] is not False
            or validation["optimizer_unchanged"] is not True
            or validation["training_timesteps_before_after"] != 8192
            or validation["overall"]["total"] != 6
            or validation["overall"]["probe_cutoff"] != 0
            or {row["scenario_identity_sha256"] for row in validation["episodes"]} != validation_ids):
        raise ValueError("Validation did not use six independent complete episodes")
    if (sum(validation["overall"][key] for key in
            ("success", "collision", "time_limit", "physical_failure", "sensor_failure")) != 6):
        raise ValueError("Validation terminal accounting is incomplete")
    evidence = verification or {"independent_review": "PENDING", "tests": "PENDING"}
    tests = evidence.get("tests")
    tests_ready = (isinstance(tests, dict) and
                   all(type(tests.get(key)) is int for key in
                       ("inherited", "new", "passed", "failed", "skipped")) and
                   tests["inherited"] >= 679 and tests["new"] >= 19 and
                   tests["passed"] == tests["inherited"] + tests["new"] and
                   tests["failed"] == 0 and tests["skipped"] == 0)
    ready = tests_ready and evidence.get("independent_review") == "GO"
    milestones = aligned_milestones(1_000_000, 100_000, 128)
    return {
        "task": "Phase 3E-L8 Multi-Scenario PPO Training Orchestration and Bounded Pilot",
        "TRAINING_DRIVER_READY": "YES" if ready else "NO",
        "FORMAL_SEED0_READY": "YES" if ready else "NO",
        "LONG_RUN_MEMORY_PROVEN": "NO",
        "formal_training_executed": False,
        "pilot_is_not_navigation_performance_claim": True,
        "pilot_checkpoint_local_only": True,
        "status": {
            "frozen_protocol_and_source_identity": "PASS",
            "multi_scenario_train_execution": "PASS",
            "rollout_aligned_post_update_milestones": "PASS",
            "pilot_checkpoint_save_reload": "PASS",
            "independent_validation_execution": "PASS",
            "sealed_test_untouched": "PASS",
            "one_shot_8192_step_pilot": "PASS",
            "long_run_memory_stability": "NOT TESTED",
            "formal_1m_training": "NOT TESTED",
            "trained_navigation_success": "NOT TESTED",
            "px4_or_real_sensor": "NOT TESTED",
        },
        "l6_formal_proposal_sha256": prereg["formal_proposal_sha256"],
        "l7_formal_protocol_sha256": summary["l7_protocol_sha256"],
        "l8_formal_execution_protocol_sha256": expected_protocol,
        "protocol_revision": {
            "nominal_final_step": 1_000_000,
            "actual_final_step": 1_000_064,
            "milestones": [{"nominal_step": x.nominal_step, "actual_step": x.actual_step}
                           for x in milestones],
            "event_phase": "after complete SB3 2.9.0 rollout and corresponding PPO optimizer update",
            "selection": "Final checkpoint; Validation diagnostic only; Test sealed",
            "continuation": "RESTART_REQUIRED for interrupted formal runs",
        },
        "scenario_split_counts": {split: len(rows) for split, rows in
                                  prereg["scenario_manifest"].items()},
        "scenario_visits_by_layout": {row["layout"] + ":" + row["identity_sha256"][:12]:
                                      counts[row["identity_sha256"]] for row in train_rows},
        "pilot": summary,
        "raw_evidence": {
            "summary_sha256": _sha(summary_path),
            "checkpoint_sha256": _sha(archive),
            "preflight_identity_sha256": _sha(raw / "preflight_identity.json"),
            "milestone_event_sha256": _sha(event_path),
            "runtime_source_sha256": current_sources,
        },
        "verification": evidence,
        "limitations": [
            "The 8192-step Pilot is training-mechanics evidence, not obstacle-navigation success.",
            "The Pilot ended 42 Train collisions and 4 timeouts; Validation ended 6 collisions and no successes.",
            "Finite Pilot RSS cannot prove 1M-step memory stability; MemAvailable was guarded at 1.5 GiB.",
            "ZIP and sidecar are individually atomic but not a cross-file transaction; incomplete pairs fail closed.",
            "WARM_START is not lossless; interrupted formal runs are RESTART_REQUIRED.",
            "System OpenCV/NumPy ABI warning remains nonfatal in tested code path, but needs isolated environment cleanup.",
            "MuJoCo-LiDAR uses approximate geometry and SIM ground truth, not FAST-LIO2 or real MID360S.",
        ],
    }


def write_orchestration_report(output: Path, prereg_path: Path, raw_dir: Path,
                               *, verification: dict) -> dict:
    if Path(output).exists():
        raise FileExistsError("refusing to overwrite L8 official report")
    report = build_orchestration_report(prereg_path, raw_dir, verification=verification)
    _atomic_json(Path(output), report)
    return report
