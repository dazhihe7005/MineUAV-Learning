"""Hash-checked offline L7 readiness decision; no model inference or training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .ppo_formal_checkpoint import formal_readiness_protocol
from .ppo_formal_scheduler import (L6_PREREG_FILE_SHA256, load_l6_prereg,
                                   verify_l6_runtime_sources)
from .ppo_training_runner import _atomic_json, _historical_model_hashes, protocol_hash


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_readiness_report(prereg_path: Path, raw_dir: Path,
                           *, verification: dict | None = None) -> dict:
    prereg = load_l6_prereg(prereg_path)
    verify_l6_runtime_sources(prereg)
    raw = json.loads((Path(raw_dir) / "probe_summary.json").read_text())
    archive = Path(raw_dir) / "L7_UNTRAINED_CONTRACT.zip"
    sidecar = Path(raw_dir) / "L7_UNTRAINED_CONTRACT.json"
    checkpoint = json.loads(sidecar.read_text())
    sidecar_hash = checkpoint.pop("sidecar_payload_sha256")
    if sidecar_hash != protocol_hash(checkpoint):
        raise ValueError("checkpoint sidecar identity mismatch")
    validation = raw["validation"]
    outcomes = validation["overall"]
    scene_rows = raw["train_coverage"]["rows"]
    complete_validation = (len(validation["episodes"]) == 6
                           and outcomes["total"] == 6
                           and outcomes["probe_cutoff"] == 0
                           and sum(outcomes[key] for key in
                                   ("success", "collision", "time_limit",
                                    "physical_failure", "sensor_failure")) == 6)
    if (raw["l6_manifest_sha256"] != prereg["scenario_manifest_sha256"]
            or raw["l6_formal_proposal_sha256"] != prereg["formal_proposal_sha256"]
            or raw["l6_prereg_file_sha256"] != L6_PREREG_FILE_SHA256
            or raw["l7_formal_protocol_sha256"] !=
            protocol_hash(formal_readiness_protocol(prereg))
            or checkpoint["checkpoint_sha256"] != _sha(archive)
            or raw["checkpoint_contract"]["checkpoint_sha256"] != _sha(archive)
            or checkpoint["historical_checkpoint_sha256"] != _historical_model_hashes()
            or raw["ppo_optimizer_updates"] != 0 or raw["test_episode_count"] != 0
            or raw["formal_training_executed"] is not False
            or raw["train_coverage"]["actual_scene_count"] != 12
            or len(scene_rows) != 12 * raw["train_coverage"]["cycles"]
            or not complete_validation):
        raise ValueError("bounded readiness evidence is incomplete or identity differs")
    validation_ids = {row["identity_sha256"] for row in
                      prereg["scenario_manifest"]["validation"]}
    train_ids = {row["identity_sha256"] for row in prereg["scenario_manifest"]["train"]}
    if ({row["scenario_identity_sha256"] for row in validation["episodes"]} != validation_ids
            or {row["scenario_identity_sha256"] for row in scene_rows} != train_ids):
        raise ValueError("Train/Validation episode identity mismatch")
    l7_sources = {name: _sha(Path(__file__).parent / name) for name in
                  ("ppo_formal_scheduler.py", "ppo_formal_validation.py",
                   "ppo_formal_checkpoint.py", "ppo_formal_probe.py")}
    return {
        "task": "Phase 3E-L7 Multi-Scenario PPO Training and Evaluation Readiness",
        "FORMAL_TRAINING_READY": "NO",
        "readiness_reason": (
            "The bounded scene/Validation/checkpoint interfaces pass, but periodic 100k-step "
            "training orchestration has not been implemented or verified, and uninterrupted "
            "1M-step resource stability is untested. A warm start is not lossless resume."
        ),
        "status": {
            "frozen_l6_protocol_and_source_identity": "PASS",
            "train_scene_scheduler": "PASS",
            "train_12_scene_physics_coverage": "PASS",
            "independent_validation_interface": "PASS",
            "validation_six_complete_episodes": "PASS",
            "checkpoint_save_reload_identity": "PASS",
            "bitwise_exact_resume": "NOT TESTED",
            "safe_interrupted_run_policy": "PASS",
            "periodic_100k_training_orchestration": "BLOCKED",
            "long_duration_resource_stability": "NOT TESTED",
            "trained_navigation_success": "NOT TESTED",
            "sealed_test_generalization": "NOT TESTED",
            "formal_ppo_training": "NOT TESTED",
            "px4_fast_lio2_real_sensor": "NOT TESTED",
        },
        "l6_formal_proposal_sha256": prereg["formal_proposal_sha256"],
        "l6_scene_manifest_sha256": prereg["scenario_manifest_sha256"],
        "l7_formal_candidate_protocol": formal_readiness_protocol(prereg),
        "l7_formal_candidate_sha256": raw["l7_formal_protocol_sha256"],
        "explicit_l7_protocol_diff": {
            "train_scene_scheduler": [prereg["formal_proposal_not_run"]["train_scene_scheduler"],
                                      formal_readiness_protocol(prereg)["train_scene_scheduler"]],
            "added_continuation_policy": formal_readiness_protocol(prereg)["continuation_policy"],
        },
        "scenario_counts": {key: len(rows) for key, rows in
                            prereg["scenario_manifest"].items()},
        "evaluation_selection_rule": "Final checkpoint; Validation every 100k diagnostic only; Test sealed",
        "evidence": raw,
        "local_only_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "l7_runtime_source_sha256": l7_sources,
        "historical_checkpoint_sha256": prereg["historical_checkpoint_sha256"],
        "verification": verification or {"tests": "PENDING", "independent_review": "PENDING"},
        "limitations": [
            "Validation used an untrained random-init model; 0/6 success is not trained-policy performance.",
            "Six Validation and six sealed Test scenes are small synthetic distributions.",
            "SB3 ZIP contains optimizer state, not active MjData/PI/LiDAR/partial-rollout state.",
            "Checkpoint ZIP and sidecar are individually atomic; an interrupted pair is fail-closed, not a transaction.",
            "Finite scene-switch RSS observation cannot prove 1M-step memory stability.",
            "Legacy system OpenCV import prints a NumPy 1.x/2.x ABI warning; no system package was changed.",
            "LiDAR and state/heading are MuJoCo approximations/ground truth, not PX4/FAST-LIO2.",
        ],
    }


def write_readiness_report(output: Path, prereg_path: Path, raw_dir: Path,
                           *, verification: dict) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite L7 analysis report")
    report = build_readiness_report(prereg_path, raw_dir, verification=verification)
    _atomic_json(output, report)
    return report
