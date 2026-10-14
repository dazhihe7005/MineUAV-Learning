"""Bounded, atomic, non-overwriting exception evidence; never changes physics."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import traceback
import uuid
from collections.abc import Mapping
from itertools import islice
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "FAILURE_EVIDENCE_V1"
NOT_AVAILABLE = "NOT_AVAILABLE"
MAX_RECORD_BYTES = 262144
MAX_CONTEXT_NODES = 512
MAX_CONTEXT_TEXT_CHARACTERS = 8192
PROTECTED_L9_RUN = Path(__file__).parents[1] / "reports/mujoco_lidar_ppo_formal_seed0_parts/run_01"
REQUIRED_CONTEXT_FIELDS = (
    "context_kind", "policy_step_attempt", "last_completed_policy_step",
    "scenario_identity_sha256", "simulation_time", "terminal_class", "raw_termination_reason",
    "terminated", "truncated", "terminal_flags", "last_action", "qpos", "qvel",
    "relevant_ctrl", "lidar_status", "physics_steps", "last_known_reward", "returned_reward",
)


def _event_id() -> str:
    return uuid.uuid4().hex


def _bounded_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "\n[TRUNCATED]"


class _SerializationBudget:
    def __init__(self):
        self.nodes = MAX_CONTEXT_NODES
        self.text = MAX_CONTEXT_TEXT_CHARACTERS

    def bounded_text(self, value: str, limit: int = 2048) -> str:
        count = min(len(value), limit, self.text)
        self.text -= count
        return value[:count] + ("[TRUNCATED_TEXT]" if count < len(value) else "")


def json_safe(value, *, depth: int = 0, _budget=None):
    """Explicit nonfinite markers; unavailable values never become invented zeros."""
    budget = _budget if _budget is not None else _SerializationBudget()
    if budget.nodes <= 0:
        return "[TRUNCATED_BUDGET]"
    budget.nodes -= 1
    if depth > 6:
        return "[TRUNCATED_DEPTH]"
    recurse = lambda item: json_safe(item, depth=depth + 1, _budget=budget)
    if isinstance(value, np.ndarray):
        # flat iteration avoids a full reshape copy of non-contiguous arrays.
        if value.size > 128:
            return {"shape": list(value.shape), "first_values": recurse(list(islice(value.flat, 128))),
                    "truncated": True}
        return recurse(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        number = int(value)
        return number if number.bit_length() <= 256 else "[TRUNCATED_INTEGER]"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "NONFINITE_NAN"
        if math.isinf(number):
            return "POSITIVE_INFINITY" if number > 0 else "NEGATIVE_INFINITY"
        return number
    if value is None:
        return None
    if isinstance(value, str):
        return budget.bounded_text(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in islice(value.items(), 64):
            if budget.nodes <= 0:
                break
            name = budget.bounded_text(key if isinstance(key, str) else str(key), 128)
            result[name] = recurse(item)
        if len(value) > len(result):
            result["truncated_mapping"] = True
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in islice(value, 128):
            if budget.nodes <= 0:
                break
            result.append(recurse(item))
        if len(value) > len(result):
            result.append("[TRUNCATED_ITEMS]")
        return result
    return {"unavailable_type": type(value).__name__}


def _resource_value(path: str, prefix: str):
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(prefix):
                return float(line.split()[1]) / 1024.
    except (OSError, ValueError, IndexError):
        pass
    return NOT_AVAILABLE


def simulator_context(scheduler, *, scenario_identity=None) -> dict:
    """Read only MjData and scheduler identity; do not call sensor read/step methods."""
    context = {key: NOT_AVAILABLE for key in
               ("scenario_identity_sha256", "simulation_time", "qpos", "qvel", "relevant_ctrl")}
    context["clock_domain"] = "MUJOCO_SIM_TIME"
    identity = scenario_identity if scenario_identity is not None else getattr(scheduler, "current_identity", None)
    if identity is not None:
        context["scenario_identity_sha256"] = identity
    try:
        plant = (scheduler.active if hasattr(scheduler, "active") else scheduler).unwrapped
        context.update({"simulation_time": float(plant.data.time),
                        "qpos": plant.data.qpos.copy(), "qvel": plant.data.qvel.copy(),
                        "relevant_ctrl": plant.data.ctrl.copy()})
    except (AttributeError, RuntimeError, ValueError):
        pass
    return context


def transition_failure_context(env, *, completed_before, action, known_reward=NOT_AVAILABLE,
                               returned=None, outcome=None, scenario_identity=None,
                               phase="TRAIN") -> dict:
    context = simulator_context(env, scenario_identity=scenario_identity)
    context.update({"phase": phase, "policy_step_attempt": completed_before + 1,
                    "last_completed_policy_step": completed_before, "last_action": action,
                    "last_known_reward": known_reward, "context_kind": "LAST_KNOWN_CONTEXT"})
    if isinstance(returned, tuple) and len(returned) == 5:
        _, reward, terminated, truncated, info = returned
        context.update({"context_kind": "RETURNED_TRANSITION", "returned_reward": reward,
                        "terminated": terminated, "truncated": truncated})
        if isinstance(info, Mapping):
            context.update({"raw_termination_reason": info.get("termination_reason", NOT_AVAILABLE),
                            "terminal_class": outcome.terminal_class if outcome else NOT_AVAILABLE,
                            "terminal_flags": {key: info.get(key, NOT_AVAILABLE) for key in
                                ("success", "collision", "timeout", "physical_failure", "sensor_failure")},
                            "physics_steps": info.get("physics_steps", NOT_AVAILABLE),
                            "lidar_status": info.get("lidar_status", NOT_AVAILABLE),
                            "returned_scenario_identity_sha256": info.get("scenario_identity_sha256", NOT_AVAILABLE)})
    return context


class FailureEvidenceRecorder:
    """Publish each unique event via hard link: atomic visibility, no replace race."""

    def __init__(self, directory: Path, *, run_id: str, attempt_id: str, origin: str):
        self.directory = Path(directory).resolve()
        if self.directory.is_relative_to(PROTECTED_L9_RUN.resolve()):
            raise ValueError("historical L9 failure evidence is strictly read-only")
        for value in (run_id, attempt_id, origin):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
                raise ValueError("failure evidence identities must be explicit safe names")
        self.run_id, self.attempt_id, self.origin = run_id, attempt_id, origin
        self.last_captured_exception: BaseException | None = None
        self.last_path: Path | None = None

    def capture(self, context: Mapping, error: BaseException) -> Path:
        if self.last_captured_exception is error and self.last_path is not None:
            return self.last_path
        event_id = _event_id()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", event_id):
            raise ValueError("invalid failure event identity")
        # Keep authoritative exception metadata outside the untrusted context
        # traversal. Only known fields are read; extra keys cannot displace it.
        budget = _SerializationBudget()
        record = {key: json_safe(context.get(key, NOT_AVAILABLE), _budget=budget)
                  for key in (*REQUIRED_CONTEXT_FIELDS, "clock_domain",
                              "returned_scenario_identity_sha256", "phase")}
        record.update({"schema_version": SCHEMA_VERSION, "event_id": event_id,
                       "run_id": self.run_id, "attempt_id": self.attempt_id, "origin": self.origin,
                       "exception_type": type(error).__name__,
                       "exception_message": _bounded_text(str(error), 8192),
                       "traceback": _bounded_text("".join(traceback.format_tb(
                           error.__traceback__, limit=32)), 16384)
                           + "\n" + type(error).__name__ + ": " + _bounded_text(str(error), 8192),
                       "current_rss_mib": _resource_value("/proc/self/status", "VmRSS:"),
                       "mem_available_mib": _resource_value("/proc/meminfo", "MemAvailable:")})
        encoded = json.dumps(record, sort_keys=True, allow_nan=False).encode()
        if len(encoded) > MAX_RECORD_BYTES:
            raise RuntimeError("failure evidence exceeds bounded record size")
        self.directory.mkdir(parents=True, exist_ok=True)
        final = self.directory / f"failure_{self.attempt_id}_{event_id}.json"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".failure_", suffix=".tmp", dir=self.directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, final)  # EEXIST is an error; never overwrite evidence.
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        self.last_captured_exception = error
        self.last_path = final
        return final
