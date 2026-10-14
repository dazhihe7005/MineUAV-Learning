"""Bounded failure evidence persistence; offline and zero optimizer updates."""

import json
from pathlib import Path

import numpy as np
import pytest

from .ppo_formal_scheduler import MultiScenarioTrainingEnv, load_l6_prereg
from .ppo_training_orchestration import TrainingMetricsEnv


UPSTREAM = Path("/tmp/mineuav-mujoco-lidar-audit")
PREREG = Path(__file__).resolve().parents[1] / "reports/mujoco_lidar_ppo_training_prereg_seed0.json"


def _recorder(tmp_path):
    from .failure_evidence import FailureEvidenceRecorder
    return FailureEvidenceRecorder(tmp_path, run_id="L9R_FORCED_TEST_STATE",
                                   attempt_id="TEST_ATTEMPT_01", origin="FORCED_TEST_STATE")


def test_failure_writer_preserves_nonfinite_values_without_invalid_json(tmp_path):
    recorder = _recorder(tmp_path)
    try:
        raise RuntimeError("test incident")
    except RuntimeError as error:
        path = recorder.capture({"context_kind": "LAST_KNOWN_CONTEXT",
                                 "policy_step_attempt": 7,
                                 "last_completed_policy_step": 6,
                                 "qpos": np.array([np.nan, np.inf, -np.inf]),
                                 "qvel": np.zeros(6)}, error)
    record = json.loads(path.read_text(), parse_constant=lambda token:
                        (_ for _ in ()).throw(AssertionError(token)))
    assert record["qpos"] == ["NONFINITE_NAN", "POSITIVE_INFINITY", "NEGATIVE_INFINITY"]
    assert record["last_known_reward"] == "NOT_AVAILABLE"
    assert record["exception_type"] == "RuntimeError"
    assert "test incident" in record["traceback"]
    assert record["run_id"] == "L9R_FORCED_TEST_STATE"
    assert record["attempt_id"] == "TEST_ATTEMPT_01"
    assert record["origin"] == "FORCED_TEST_STATE"
    assert path.stat().st_size < 262144
    assert not list(tmp_path.glob("*.tmp"))


def test_failure_writer_unique_identity_and_atomic_no_overwrite(tmp_path, monkeypatch):
    from . import failure_evidence as module
    recorder = _recorder(tmp_path)
    fixed = "fixed_event"
    monkeypatch.setattr(module, "_event_id", lambda: fixed)
    path = recorder.capture({"policy_step_attempt": 1}, ValueError("first"))
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        recorder.capture({"policy_step_attempt": 1}, ValueError("second"))
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_unknown_context_cannot_displace_authoritative_exception_metadata(tmp_path):
    context = {f"unknown_{index}": "ignored" for index in range(2000)}
    context.update(exception_type="spoofed", run_id="spoofed", policy_step_attempt=3)
    record = json.loads(_recorder(tmp_path).capture(context, ValueError("real failure")).read_text())
    assert record["exception_type"] == "ValueError"
    assert record["run_id"] == "L9R_FORCED_TEST_STATE"
    assert record["policy_step_attempt"] == 3
    assert not any(key.startswith("unknown_") for key in record)


def test_nested_large_context_has_global_serialization_budget(tmp_path):
    from .failure_evidence import json_safe
    # Repeated references test traversal cost, without allocating a huge source.
    payload = [["x" * 65536] * 128] * 128
    safe = json_safe(payload)
    assert len(json.dumps(safe).encode()) < 100000
    record = json.loads(_recorder(tmp_path).capture(
        {"terminal_flags": payload, "qpos": payload}, ValueError("bounded failure")).read_text())
    assert record["exception_type"] == "ValueError"
    assert record["raw_termination_reason"] == "NOT_AVAILABLE"


def test_mapping_serializer_never_materializes_entire_input():
    from collections.abc import Mapping
    from .failure_evidence import json_safe
    class LargeMapping(Mapping):
        def __len__(self):
            return 1000000
        def __iter__(self):
            for index in range(1000000):
                if index > 65:
                    raise AssertionError("unbounded mapping traversal")
                yield str(index)
        def __getitem__(self, key):
            return int(key)
    safe = json_safe(LargeMapping())
    assert safe["truncated_mapping"] is True


def test_failure_recorder_refuses_historical_l9_directory():
    from .failure_evidence import FailureEvidenceRecorder
    protected = PREREG.parent / "mujoco_lidar_ppo_formal_seed0_parts/run_01"
    with pytest.raises(ValueError, match="historical"):
        FailureEvidenceRecorder(protected, run_id="TEST", attempt_id="TEST", origin="FORCED_TEST_STATE")


def test_returned_invalid_terminal_records_complete_transition_before_raising(tmp_path):
    recorder = _recorder(tmp_path)
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = TrainingMetricsEnv(scheduler, failure_recorder=recorder)
    try:
        env.reset()
        original_step = scheduler.step
        def inconsistent_step(action):
            obs, reward, _, _, info = original_step(action)
            # An explicit unknown returned reason must fail closed, not be guessed.
            info = {**info, "termination_reason": "unknown_physical_issue", "physical_failure": True}
            return obs, reward, True, False, info
        scheduler.step = inconsistent_step
        with pytest.raises(RuntimeError, match="unknown"):
            env.step(np.zeros(4, dtype=np.float32))
        records = list(tmp_path.glob("failure_*.json"))
        assert len(records) == 1
        row = json.loads(records[0].read_text())
        assert row["context_kind"] == "RETURNED_TRANSITION"
        assert row["policy_step_attempt"] == 1 and row["last_completed_policy_step"] == 0
        assert row["raw_termination_reason"] == "unknown_physical_issue"
        assert row["terminal_class"] == "NOT_AVAILABLE"
        assert row["terminated"] is True and row["truncated"] is False
        assert row["terminal_flags"]["physical_failure"] is True
        assert row["returned_reward"] != "NOT_AVAILABLE"
        assert row["last_known_reward"] == "NOT_AVAILABLE"
        assert row["physics_steps"] == 20
        assert len(row["qpos"]) == 7 and len(row["qvel"]) == 6
        assert len(row["relevant_ctrl"]) == 4
        assert row["scenario_identity_sha256"] == scheduler.current_identity
        assert row["lidar_status"] == "FRESH"
        assert env.policy_steps == 0 and not env.episode_returns
    finally:
        env.close()


def test_lower_step_exception_records_last_known_context_without_fake_info(tmp_path):
    recorder = _recorder(tmp_path)
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = TrainingMetricsEnv(scheduler, failure_recorder=recorder)
    try:
        env.reset()
        env.step(np.zeros(4, dtype=np.float32))
        def broken_step(action):
            raise ValueError("lower step has no result")
        scheduler.step = broken_step
        with pytest.raises(ValueError, match="no result"):
            env.step(np.ones(4, dtype=np.float32))
        row = json.loads(next(tmp_path.glob("failure_*.json")).read_text())
        assert row["context_kind"] == "LAST_KNOWN_CONTEXT"
        assert row["policy_step_attempt"] == 2 and row["last_completed_policy_step"] == 1
        assert row["returned_reward"] == "NOT_AVAILABLE"
        assert row["raw_termination_reason"] == "NOT_AVAILABLE"
        assert row["terminated"] == "NOT_AVAILABLE" and row["truncated"] == "NOT_AVAILABLE"
        assert row["last_known_reward"] != "NOT_AVAILABLE"
        assert row["last_action"] == [1., 1., 1., 1.]
        assert row["current_rss_mib"] > 0 and row["mem_available_mib"] > 0
        assert env.policy_steps == 1
    finally:
        env.close()


def test_nonfinite_quaternion_exception_is_captured_as_unreturned_context(tmp_path):
    recorder = _recorder(tmp_path)
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = TrainingMetricsEnv(scheduler, failure_recorder=recorder)
    try:
        env.reset()
        scheduler.active.unwrapped.data.qpos[3] = np.nan
        with pytest.raises(ValueError):
            env.step(np.zeros(4, dtype=np.float32))
        row = json.loads(next(tmp_path.glob("failure_*.json")).read_text())
        assert row["context_kind"] == "LAST_KNOWN_CONTEXT"
        assert row["qpos"][3] == "NONFINITE_NAN"
        assert row["raw_termination_reason"] == "NOT_AVAILABLE"
        assert row["returned_reward"] == "NOT_AVAILABLE"
    finally:
        env.close()


def test_capture_write_failure_cannot_allow_step_to_continue(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path)
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = TrainingMetricsEnv(scheduler, failure_recorder=recorder)
    try:
        env.reset()
        def broken_step(action):
            raise ValueError("original lower failure")
        def broken_capture(context, error):
            raise OSError("forced read-only evidence disk")
        scheduler.step = broken_step
        monkeypatch.setattr(recorder, "capture", broken_capture)
        with pytest.raises(RuntimeError, match="failure evidence") as failure:
            env.step(np.zeros(4, dtype=np.float32))
        assert isinstance(failure.value.__cause__, ValueError)
        assert env.policy_steps == 0
    finally:
        env.close()


def test_failure_latch_cannot_be_cleared_by_reset(tmp_path):
    recorder = _recorder(tmp_path)
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = TrainingMetricsEnv(scheduler, failure_recorder=recorder)
    try:
        env.reset()
        scheduler.step = lambda action: (_ for _ in ()).throw(ValueError("test failure"))
        with pytest.raises(ValueError):
            env.step(np.zeros(4, dtype=np.float32))
        with pytest.raises(RuntimeError, match="latched"):
            env.step(np.zeros(4, dtype=np.float32))
        with pytest.raises(RuntimeError, match="cannot be reset"):
            env.reset()
        assert len(list(tmp_path.glob("failure_*.json"))) == 1
    finally:
        env.close()


def test_health_wrapper_logs_canonical_physical_terminal_in_new_directory(tmp_path):
    from .ppo_formal_seed0_health import FormalHealthMetricsEnv
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = FormalHealthMetricsEnv(scheduler, failure_recorder=_recorder(tmp_path / "failures"),
                                 diagnostics_dir=tmp_path)
    try:
        env.reset()
        scheduler.active.unwrapped.data.ctrl[0] = np.nan
        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        assert terminated and not truncated and info["terminal_class"] == "physical_failure"
        row = json.loads((tmp_path / "train_episodes.jsonl").read_text())
        assert row["outcome"] == row["terminal_class"] == "physical_failure"
        assert row["raw_termination_reason"] == "nonfinite_state"
        assert not list((tmp_path / "failures").glob("failure_*.json"))
    finally:
        env.close()


def test_health_nonfinite_position_is_recorded_as_unavailable_not_guessed_or_raised(tmp_path):
    from .ppo_formal_seed0_health import FormalHealthMetricsEnv
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = FormalHealthMetricsEnv(scheduler, failure_recorder=_recorder(tmp_path / "failures"),
                                 diagnostics_dir=tmp_path)
    try:
        env.reset()
        scheduler.active.unwrapped.data.qpos[0] = np.nan
        _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert terminated and info["terminal_class"] == "physical_failure"
        row = json.loads((tmp_path / "train_episodes.jsonl").read_text())
        assert row["trajectory_length_m"] == row["net_displacement_m"] == "NOT_AVAILABLE"
        assert row["motionless_episode"] == "NOT_AVAILABLE"
        assert row["final_target_distance_m"] == "NOT_AVAILABLE"
        assert row["raw_termination_reason"] == "nonfinite_state"
    finally:
        env.close()


def test_health_append_failure_captures_returned_result_and_latches(tmp_path, monkeypatch):
    from . import ppo_formal_seed0_health as health
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = health.FormalHealthMetricsEnv(scheduler, failure_recorder=_recorder(tmp_path / "failures"),
                                        diagnostics_dir=tmp_path)
    try:
        env.reset()
        scheduler.active.unwrapped.data.ctrl[0] = np.nan
        monkeypatch.setattr(health, "_append_jsonl", lambda *args:
                            (_ for _ in ()).throw(OSError("forced diagnostic append failure")))
        with pytest.raises(OSError, match="append"):
            env.step(np.zeros(4, dtype=np.float32))
        row = json.loads(next((tmp_path / "failures").glob("failure_*.json")).read_text())
        assert row["context_kind"] == "RETURNED_TRANSITION"
        assert row["last_completed_policy_step"] == 0 and row["policy_step_attempt"] == 1
        assert row["terminal_class"] == "physical_failure"
        assert row["raw_termination_reason"] == "nonfinite_state"
        assert row["relevant_ctrl"][0] == "NONFINITE_NAN"
        with pytest.raises(RuntimeError, match="latched"):
            env.step(np.zeros(4, dtype=np.float32))
    finally:
        env.close()
