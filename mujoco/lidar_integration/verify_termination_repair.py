"""Single-threaded offline regressions, with learning/optimizer tripwires.

Run from the repository root via ``python -m lidar_integration.verify_termination_repair``.
Prints JSON evidence; does not write historical files or start a simulator process.
"""

from __future__ import annotations

import json
import resource
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from stable_baselines3 import PPO

PATHS = ("mujoco/px4_bridge", "mujoco/lidar_integration",
         "mujoco/tests/test_mid360s_interface_audit.py", "mujoco/rl/test_px4_policy_shadow.py",
         "mujoco/rl/test_mine_uav_env.py", "mujoco/rl/test_velocity_command_controller.py")
EXCLUDED = {
    "test_actor_critic_optimizer_and_checkpoint_roundtrip": "would execute optimizer.step",
    "test_real_sb3_rollouts_update_weights_and_switch_train_scenes": "would execute PPO.learn",
    "test_completed_real_rollout_milestone_saves_after_update_and_validates_without_mutation": "would execute PPO updates",
    "test_passive_ppo_train_timer_counts_actual_updates_and_is_removed_before_save": "would execute PPO updates",
    "test_pilot_refuses_reused_attempt_and_formal_mode_requires_separate_authorization": "old Pilot authorization contract superseded by L9R no-training gate",
    "test_pilot_analysis_rejects_tampered_completion_and_preserves_negative_outcome": "old recorded L8 runtime hashes; audited as history not current code",
    "test_report_readiness_fails_closed_on_missing_failed_or_incomplete_tests": "old recorded L8 runtime hashes; no new readiness claim",
}
ENTRY_KEEP = {
    "test_passive_health_wrapper_preserves_real_transitions_and_records_episode",
    "test_health_monitor_stops_on_disk_pressure_without_false_completion",
    "test_formal_health_fails_closed_if_l9_runtime_source_changes",
    "test_formal_analysis_cannot_claim_completion_without_raw_marker",
    "test_formal_rolling_metrics_use_past_episodes_only",
    "test_interrupted_report_rejects_a_completion_marker",
    "test_interrupted_figures_disclose_incomplete_status_in_visible_title",
}


def exclusion_reason(nodeid):
    name = nodeid.rsplit("::", 1)[-1].split("[")[0]
    if "test_ppo_formal_seed0_entry.py" in nodeid and name not in ENTRY_KEEP:
        return ("historical L9 entry/source identity or old-defect expectation, or optimizer "
                "execution; superseded by L9R targeted tests")
    return EXCLUDED.get(name)


@contextmanager
def no_training_updates():
    calls, original = [], []
    def forbidden(*args, **kwargs):
        calls.append("blocked training/update invocation")
        raise AssertionError("L9R forbids PPO learning and all optimizer updates")
    targets = [(PPO, "learn"), (PPO, "train")]
    targets.extend((cls, "step") for name in dir(torch.optim)
                   if isinstance((cls := getattr(torch.optim, name)), type)
                   and issubclass(cls, torch.optim.Optimizer) and "step" in cls.__dict__)
    try:
        for cls, method in targets:
            original.append((cls, method, getattr(cls, method)))
            setattr(cls, method, forbidden)
        yield calls
    finally:
        for cls, method, value in reversed(original):
            setattr(cls, method, value)


class OfflineScope:
    def __init__(self):
        self.removed, self.results, self.mem_samples = [], [], []

    def pytest_collection_modifyitems(self, config, items):
        keep, remove = [], []
        for item in items:
            reason = exclusion_reason(item.nodeid)
            if reason:
                remove.append(item)
                self.removed.append({"nodeid": item.nodeid, "reason": reason})
            else:
                keep.append(item)
        items[:] = keep
        config.hook.pytest_deselected(items=remove)

    def _resource_sample(self):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) / 1024.
                self.mem_samples.append(available)
                if available < 1536.:
                    pytest.exit("below 1.5 GiB MemAvailable; offline verification stopped", returncode=2)

    def pytest_runtest_setup(self, item):
        self._resource_sample()

    def pytest_runtest_teardown(self, item):
        self._resource_sample()

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.failed):
            self.results.append({"nodeid": report.nodeid, "outcome": report.outcome})


def main():
    scope, started = OfflineScope(), time.perf_counter()
    with no_training_updates() as calls:
        code = pytest.main(["-q", "--tb=short", *PATHS], plugins=[scope])
    print("L9R_VERIFICATION_JSON=" + json.dumps({
        "exit_code": int(code), "complete": int(code) in (0, 1),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
        "minimum_mem_available_mib": min(scope.mem_samples) if scope.mem_samples else None,
        "memory_sampling": "single-threaded test setup/teardown; not continuous sampling",
        "forbidden_calls": calls, "deselected": scope.removed, "results": scope.results,
    }, allow_nan=False))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
