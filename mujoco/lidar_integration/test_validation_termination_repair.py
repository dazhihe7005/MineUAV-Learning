"""FORCED_TEST_STATE Validation diagnostics, never PPO learning or Test episodes."""

from research_regression.upstream import resolve_upstream
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from .navigation_env import LidarNavigationEnv
from .ppo_formal_scheduler import MultiScenarioTrainingEnv, load_l6_prereg
from .ppo_formal_validation import LidarPPOValidationEvaluator
from .ppo_preflight import make_ppo

UPSTREAM = resolve_upstream()
PREREG = Path(__file__).resolve().parents[1] / "reports/mujoco_lidar_ppo_training_prereg_seed0.json"


def _untrained_model(prereg):
    train = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0)
    try:
        model = make_ppo(train, seed=0)
    finally:
        train.close()
    return model


@pytest.mark.parametrize("raw", ["outside_flight_area", "nonfinite_state"])
def test_validation_preserves_canonical_physical_reason_and_never_serializes_nan(raw, monkeypatch):
    prereg = load_l6_prereg(PREREG)
    model = _untrained_model(prereg)
    original_reset = LidarNavigationEnv.reset
    def forced_reset(plant, **kwargs):
        result = original_reset(plant, **kwargs)
        if raw == "nonfinite_state":
            plant.data.qpos[0] = np.nan
        else:
            plant.data.qpos[2] = 100.
            mujoco.mj_forward(plant.model, plant.data)
        return result
    monkeypatch.setattr(LidarNavigationEnv, "reset", forced_reset)
    report = LidarPPOValidationEvaluator(prereg, UPSTREAM).evaluate(model, max_policy_steps=1)
    assert report["overall"]["physical_failure"] == 6
    assert all(row["terminal_class"] == "physical_failure"
               and row["raw_termination_reason"] == raw for row in report["episodes"])
    assert model._n_updates == 0 and not model.policy.optimizer.state
    json.dumps(report, allow_nan=False)
    if raw == "nonfinite_state":
        assert all(row["trajectory_length_m"] == row["final_target_error_m"] == "NOT_AVAILABLE"
                   for row in report["episodes"])
        assert report["overall"]["trajectory_length_available_count"] == 0
        assert report["overall"]["mean_trajectory_length_m"] == "NOT_AVAILABLE"


def test_validation_inconsistent_return_is_persisted_before_environment_close(tmp_path, monkeypatch):
    from .failure_evidence import FailureEvidenceRecorder
    prereg = load_l6_prereg(PREREG)
    model = _untrained_model(prereg)
    original_step = LidarNavigationEnv.step
    def invalid_step(plant, action):
        obs, reward, _, _, info = original_step(plant, action)
        return obs, reward, True, False, {**info, "termination_reason": "unknown_issue",
                                       "physical_failure": True}
    monkeypatch.setattr(LidarNavigationEnv, "step", invalid_step)
    recorder = FailureEvidenceRecorder(tmp_path, run_id="L9R_VALIDATION_TEST",
                                      attempt_id="TEST", origin="FORCED_TEST_STATE")
    with pytest.raises(RuntimeError, match="unknown"):
        LidarPPOValidationEvaluator(prereg, UPSTREAM, failure_recorder=recorder).evaluate(
            model, max_policy_steps=1)
    record = json.loads(next(tmp_path.glob("failure_*.json")).read_text())
    assert record["phase"] == "VALIDATION"
    assert record["context_kind"] == "RETURNED_TRANSITION"
    assert record["raw_termination_reason"] == "unknown_issue"
    assert record["returned_reward"] != "NOT_AVAILABLE"
    assert record["physics_steps"] == 20
    assert len(record["qpos"]) == 7 and len(record["relevant_ctrl"]) == 4
    assert record["scenario_identity_sha256"] in {
        row["identity_sha256"] for row in prereg["scenario_manifest"]["validation"]}
    assert model._n_updates == 0 and not model.policy.optimizer.state


def test_validation_reset_exception_cannot_invent_a_policy_step_attempt(tmp_path, monkeypatch):
    from .failure_evidence import FailureEvidenceRecorder
    prereg = load_l6_prereg(PREREG)
    model = _untrained_model(prereg)
    def failed_reset(*args, **kwargs):
        raise ValueError("forced reset exception before any policy step")
    monkeypatch.setattr(LidarNavigationEnv, "reset", failed_reset)
    recorder = FailureEvidenceRecorder(tmp_path, run_id="L9R_VALIDATION_TEST",
                                      attempt_id="RESET_TEST", origin="FORCED_TEST_STATE")
    with pytest.raises(ValueError, match="reset"):
        LidarPPOValidationEvaluator(prereg, UPSTREAM, failure_recorder=recorder).evaluate(
            model, max_policy_steps=1)
    record = json.loads(next(tmp_path.glob("failure_*.json")).read_text())
    assert record["context_kind"] == "LAST_KNOWN_CONTEXT"
    assert record["policy_step_attempt"] == "NOT_AVAILABLE"
    assert record["last_completed_policy_step"] == 0
    assert record["last_action"] == record["returned_reward"] == "NOT_AVAILABLE"
