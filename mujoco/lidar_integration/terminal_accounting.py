"""Strict shared terminal classification; does not change environment outcomes."""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np


CONTRACT_VERSION = "TERMINAL_CLASS_RAW_REASON_V1"
TERMINAL_CLASSES = ("success", "collision", "time_limit", "physical_failure", "sensor_failure")
TERMINAL_FLAG_KEYS = ("success", "collision", "timeout", "physical_failure", "sensor_failure")
PHYSICAL_REASONS = ("z_below_zero", "outside_flight_area", "excessive_tilt", "nonfinite_state")


@dataclass(frozen=True)
class TerminalOutcome:
    terminal_class: str | None
    raw_termination_reason: str | None


def classify_terminal(info: Mapping, terminated: bool, truncated: bool) -> TerminalOutcome:
    """Validate running or terminal transitions; unknown causes are never guessed."""
    if (not isinstance(info, Mapping)
            or not isinstance(terminated, (bool, np.bool_))
            or not isinstance(truncated, (bool, np.bool_))
            or any(key not in info or not isinstance(info[key], (bool, np.bool_))
                   for key in TERMINAL_FLAG_KEYS)
            or "termination_reason" not in info):
        raise RuntimeError("terminal contract requires explicit reason and boolean flags")
    raw = info["termination_reason"]
    if raw is None:
        terminal_class = None
    elif isinstance(raw, str) and raw in TERMINAL_CLASSES:
        terminal_class = raw
    elif isinstance(raw, str) and raw in PHYSICAL_REASONS:
        terminal_class = "physical_failure"
    else:
        raise RuntimeError("unknown raw termination reason; fail closed")
    expected_flag = "timeout" if terminal_class == "time_limit" else terminal_class
    if any(bool(info[key]) != (key == expected_flag) for key in TERMINAL_FLAG_KEYS):
        raise RuntimeError("terminal reason and mutually exclusive flags disagree")
    expected_terminated = terminal_class is not None and terminal_class != "time_limit"
    expected_truncated = terminal_class == "time_limit"
    if bool(terminated) != expected_terminated or bool(truncated) != expected_truncated:
        raise RuntimeError("terminal class and terminated/truncated disagree")
    if ("terminal_class" in info and info["terminal_class"] != terminal_class
            or "raw_termination_reason" in info and info["raw_termination_reason"] != raw):
        raise RuntimeError("preexisting terminal metadata disagrees with raw transition")
    return TerminalOutcome(terminal_class, raw)
