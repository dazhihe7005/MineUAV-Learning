"""L9R execution identity: old parameters retained, no training authorization."""

from __future__ import annotations

import copy
from pathlib import Path

from .failure_evidence import SCHEMA_VERSION, MAX_RECORD_BYTES
from .terminal_accounting import CONTRACT_VERSION, PHYSICAL_REASONS, TERMINAL_CLASSES
from .ppo_training_runner import protocol_hash


PARENT_L8_SHA256 = "fe07816f48bd4183165db482d8c93de50c637a32d13d86eccb47667d85e8cef6"


def _repair_source_paths() -> tuple[Path, ...]:
    root = Path(__file__).parent
    return tuple(root / name for name in (
        "terminal_accounting.py", "failure_evidence.py", "ppo_termination_repair_protocol.py",
        "ppo_training_orchestration.py", "ppo_formal_validation.py", "ppo_formal_seed0_health.py",
        "ppo_formal_seed0_entry.py", "ppo_formal_scheduler.py", "ppo_formal_checkpoint.py"))


def repair_execution_protocol(prereg: dict) -> dict:
    from .ppo_training_orchestration import l8_execution_protocol, source_hashes
    parent = l8_execution_protocol(prereg)
    if protocol_hash(parent) != PARENT_L8_SHA256:
        raise RuntimeError("L9R parent execution protocol identity mismatch")
    config = copy.deepcopy(parent)
    config.update({
        "parent_l8_execution_protocol_sha256": PARENT_L8_SHA256,
        "repair_revision": "L9R_TERMINAL_ACCOUNTING_AND_FAILURE_EVIDENCE_V1",
        "formal_retrain_authorized": False,
        "repair_source_sha256": source_hashes(_repair_source_paths()),
        "terminal_contract": {
            "version": CONTRACT_VERSION, "classes": list(TERMINAL_CLASSES),
            "physical_raw_reason_whitelist": list(PHYSICAL_REASONS),
            "fields": ["terminal_class", "raw_termination_reason"],
            "unknown_or_contradictory_transition": "FAIL_CLOSED_AND_LATCH",
            "time_limit": "truncated=True, terminated=False",
            "other_terminal_classes": "terminated=True, truncated=False",
            "reward_observation_physics_termination_changes": False,
        },
        "failure_capture": {
            "schema_version": SCHEMA_VERSION, "maximum_record_bytes": MAX_RECORD_BYTES,
            "returned_transition": "RETURNED_TRANSITION",
            "unreturned_step": "LAST_KNOWN_CONTEXT; unavailable fields NOT_AVAILABLE",
            "nonfinite_values": "NONFINITE_NAN/POSITIVE_INFINITY/NEGATIVE_INFINITY",
            "publication": "fsync + atomic hard link, exclusive unique event, never replace",
            "original_l9_run": "STRICTLY_READ_ONLY",
        },
    })
    return config


def verify_repair_protocol(prereg: dict, *, expected_sha256: str) -> None:
    if protocol_hash(repair_execution_protocol(prereg)) != expected_sha256:
        raise RuntimeError("L9R execution/source identity mismatch")
