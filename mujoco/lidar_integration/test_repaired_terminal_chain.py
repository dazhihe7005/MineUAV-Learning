"""L9S FORCED_TEST_STATE integration evidence; never a training/Pilot result.

Real MuJoCo instances run sequentially. Checkpoints contain an untrained model;
the associated statistics describe forced diagnostics, not optimizer milestones.
"""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from .failure_evidence import FailureEvidenceRecorder, NOT_AVAILABLE
from .navigation_env import LidarNavigationEnv
from .ppo_formal_checkpoint import CheckpointManager
from .ppo_formal_scheduler import MultiScenarioTrainingEnv, load_l6_prereg
from .ppo_formal_seed0_health import FormalHealthMetricsEnv
from .ppo_formal_validation import LidarPPOValidationEvaluator
from .ppo_preflight import make_ppo
from .ppo_training_runner import _atomic_json, _weights_sha
from .test_termination_repair import PREREG, UPSTREAM, _force_state
from .test_validation_termination_repair import _untrained_model
from .verify_termination_repair import no_training_updates


CASES = (
    ("success", "success", True, False),
    ("collision", "collision", True, False),
    ("time_limit", "time_limit", False, True),
    ("sensor_failure", "sensor_failure", True, False),
    ("z_below_zero", "physical_failure", True, False),
    ("outside_flight_area", "physical_failure", True, False),
    ("excessive_tilt", "physical_failure", True, False),
    ("nonfinite_state", "physical_failure", True, False),
)
FLAGS = ("success", "collision", "timeout", "physical_failure", "sensor_failure")
EXPECTED_COUNTS = {"success": 1, "collision": 1, "time_limit": 1,
                   "physical_failure": 4, "sensor_failure": 1}


@pytest.fixture(scope="module", autouse=True)
def _forbid_updates():
    """Keep direct pytest invocation as constrained as the offline runner."""
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with no_training_updates() as calls:
            yield
        assert calls == [], "forced diagnostics attempted learning or optimization"
    finally:
        torch.set_num_threads(prior_threads)


def _recorder(directory):
    return FailureEvidenceRecorder(directory, run_id="L9S_TERMINAL_CHAIN",
                                   attempt_id="FORCED_TEST_STATE", origin="FORCED_TEST_STATE")


def _forced_episode(env, scheduler, raw):
    env.reset(seed=0)
    _force_state(scheduler.active.unwrapped, raw)
    transitions = []
    for _ in range(5 if raw == "success" else 1):
        result = env.step(np.zeros(4, dtype=np.float32))
        transitions.append(copy.deepcopy(result))
        if result[2] or result[3]:
            break
    return transitions


@pytest.fixture(scope="module")
def full_chain_evidence(tmp_path_factory, _forbid_updates):
    directory = tmp_path_factory.mktemp("l9s_forced_terminal_chain")
    prereg = load_l6_prereg(PREREG)
    scheduler = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0)
    try:
        original = {raw: _forced_episode(scheduler, scheduler, raw)
                    for raw, _, _, _ in CASES}
    finally:
        scheduler.close()

    # The raw reference has closed before constructing the measured simulator.
    scheduler = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0)
    env = FormalHealthMetricsEnv(scheduler, diagnostics_dir=directory,
                                 failure_recorder=_recorder(directory / "failures"))
    try:
        measured, counts = {}, {}
        for raw, _, _, _ in CASES:
            measured[raw] = _forced_episode(env, scheduler, raw)
            counts[raw] = dict(env.terminal_counts)

        model = make_ppo(env, seed=0)
        weights = _weights_sha(model.policy)
        manager = CheckpointManager(directory / "checkpoint", prereg)
        run_id = "L9S_FORCED_UNTRAINED_TERMINALS"
        checkpoint = manager.save(model, scheduler, run_id=run_id, seed=0)
        restored = manager.load_verified(env, run_id=run_id)
        assert _weights_sha(restored.policy) == weights
        for policy in (model, restored):
            assert policy.num_timesteps == policy._n_updates == 0
            assert not policy.policy.optimizer.state

        statistics = {
            "origin": "FORCED_TEST_STATE", "model_status": "UNTRAINED_RANDOM_INITIALIZATION",
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "training_timesteps": model.num_timesteps, "optimizer_updates": model._n_updates,
            "test_episode_count": 0, "diagnostic_policy_steps": env.policy_steps,
            "terminal_counts": dict(env.terminal_counts),
            "episode_returns": list(env.episode_returns),
            "episode_diagnostics": copy.deepcopy(env.episode_diagnostics),
        }
        statistics_path = directory / "checkpoint" / f"{run_id}_forced_statistics.json"
        _atomic_json(statistics_path, statistics)
        return {"directory": directory, "original": original, "measured": measured,
                "counts": counts, "statistics": statistics,
                "statistics_path": statistics_path, "run_id": run_id}
    finally:
        env.close()


@pytest.mark.parametrize("raw,want,terminated,truncated", CASES)
def test_real_terminal_chain_preserves_transitions_and_counts_once(
        full_chain_evidence, raw, want, terminated, truncated):
    """Catch Health dropping a class, altering a transition, or counting it twice."""
    evidence = full_chain_evidence
    original, measured = evidence["original"][raw], evidence["measured"][raw]
    assert len(measured) == len(original) == (5 if raw == "success" else 1)
    for before, after in zip(original, measured):
        assert after[1:4] == before[1:4]
        assert after[0].keys() == before[0].keys()
        for key in before[0]:
            np.testing.assert_allclose(after[0][key], before[0][key],
                                       rtol=0, atol=0, equal_nan=True)
        for key in (*FLAGS, "termination_reason", "reward_breakdown", "physics_steps",
                    "scenario_identity_sha256", "scenario_split"):
            assert after[4][key] == before[4][key], (raw, key)

    final = measured[-1]
    assert final[2:4] == (terminated, truncated)
    assert final[4]["terminal_class"] == want
    assert final[4]["termination_reason"] == final[4]["raw_termination_reason"] == raw
    assert sum(final[4][flag] for flag in FLAGS) == 1
    index = [case[0] for case in CASES].index(raw)
    prior = evidence["counts"][CASES[index - 1][0]] if index else dict.fromkeys(EXPECTED_COUNTS, 0)
    assert evidence["counts"][raw] == {key: count + int(key == want)
                                        for key, count in prior.items()}
    row = evidence["statistics"]["episode_diagnostics"][index]
    assert row["terminal_class"] == row["outcome"] == want
    assert row["raw_termination_reason"] == raw
    assert row["episode_return"] == sum(step[1] for step in measured)
    assert row["episode_policy_steps"] == len(measured)
    assert row["scenario_identity_sha256"] == final[4]["scenario_identity_sha256"]


def test_forced_terminal_statistics_survive_untrained_checkpoint_association(full_chain_evidence):
    """Catch missing persisted rows/counts or a checkpoint association with other weights."""
    evidence = full_chain_evidence
    persisted = json.loads(evidence["statistics_path"].read_text())
    assert persisted == evidence["statistics"]
    assert persisted["terminal_counts"] == EXPECTED_COUNTS
    assert persisted["diagnostic_policy_steps"] == 12
    assert persisted["training_timesteps"] == persisted["optimizer_updates"] == 0
    assert persisted["test_episode_count"] == 0
    assert persisted["origin"] == "FORCED_TEST_STATE"
    rows = [json.loads(line) for line in
            (evidence["directory"] / "train_episodes.jsonl").read_text().splitlines()]
    assert rows == persisted["episode_diagnostics"]
    assert len(rows) == len(persisted["episode_returns"]) == 8
    for terminal, count in EXPECTED_COUNTS.items():
        assert sum(row["terminal_class"] == terminal for row in rows) == count
    checkpoint = evidence["directory"] / "checkpoint" / evidence["run_id"]
    assert hashlib.sha256(checkpoint.with_suffix(".zip").read_bytes()).hexdigest() == persisted["checkpoint_sha256"]
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    assert sidecar["checkpoint_sha256"] == persisted["checkpoint_sha256"]
    assert sidecar["training_timesteps"] == 0
    assert not list((evidence["directory"] / "failures").glob("failure_*.json"))


@pytest.mark.parametrize("raw,want", [(raw, want) for raw, want, _, _ in CASES
                                     if raw not in {"outside_flight_area", "nonfinite_state"}])
def test_validation_missing_terminal_cases_share_classifier_without_updates(raw, want, tmp_path, monkeypatch):
    """Complete existing Validation physical-case coverage for the other six raw reasons."""
    prereg = load_l6_prereg(PREREG)
    model = _untrained_model(prereg)
    weights = _weights_sha(model.policy)
    original_reset = LidarNavigationEnv.reset

    def forced_reset(plant, **kwargs):
        result = original_reset(plant, **kwargs)
        _force_state(plant, raw)
        return result

    monkeypatch.setattr(LidarNavigationEnv, "reset", forced_reset)
    report = LidarPPOValidationEvaluator(prereg, UPSTREAM).evaluate(
        model, max_policy_steps=5 if raw == "success" else 1)
    assert report["overall"][want] == report["overall"]["total"] == 6
    assert report["overall"]["probe_cutoff"] == 0
    assert report["probe_only"] is False
    assert {row["scenario_identity_sha256"] for row in report["episodes"]} == {
        row["identity_sha256"] for row in prereg["scenario_manifest"]["validation"]}
    for row in report["episodes"]:
        assert row["terminal_class"] == row["outcome"] == want
        assert row["raw_termination_reason"] == raw
        assert sum(row[key] for key in FLAGS) == 1
        assert row["policy_steps"] == (5 if raw == "success" else 1)
    assert model.num_timesteps == model._n_updates == 0
    assert not model.policy.optimizer.state
    assert _weights_sha(model.policy) == weights
    path = tmp_path / "forced_validation.json"
    _atomic_json(path, {"origin": "FORCED_TEST_STATE", "test_episode_count": 0, "report": report})
    # Contact pairs are tuples in memory and JSON arrays on disk.
    assert json.loads(path.read_text())["report"] == json.loads(json.dumps(report, allow_nan=False))


@pytest.mark.parametrize("returned", [True, False], ids=["invalid_return", "bottom_step_raises"])
def test_health_nested_failure_capture_preserves_first_evidence_and_latches(tmp_path, monkeypatch, returned):
    """Catch the outer Health handler overwriting richer inner transition evidence."""
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    recorder = _recorder(tmp_path / "failures")
    env = FormalHealthMetricsEnv(scheduler, diagnostics_dir=tmp_path, failure_recorder=recorder)
    try:
        env.reset()
        prior = env.step(np.zeros(4, dtype=np.float32))
        plant = scheduler.active.unwrapped
        original_step = plant.step
        attempts = []

        def invalid_step(action):
            attempts.append(action.copy())
            if not returned:
                raise ValueError("FORCED_TEST_STATE bottom step returned nothing")
            obs, reward, _, _, info = original_step(action)
            return obs, reward, True, False, {**info, "termination_reason": "unknown_issue",
                                              "physical_failure": True}

        monkeypatch.setattr(plant, "step", invalid_step)
        with pytest.raises(RuntimeError if returned else ValueError):
            env.step(np.ones(4, dtype=np.float32))
        paths = list((tmp_path / "failures").glob("failure_*.json"))
        assert len(paths) == 1
        content = paths[0].read_bytes()
        row = json.loads(content)
        assert row["origin"] == "FORCED_TEST_STATE"
        assert row["context_kind"] == ("RETURNED_TRANSITION" if returned else "LAST_KNOWN_CONTEXT")
        assert row["policy_step_attempt"] == 2 and row["last_completed_policy_step"] == 1
        assert row["last_known_reward"] == prior[1]
        assert row["last_action"] == [1., 1., 1., 1.]
        assert row["scenario_identity_sha256"] == scheduler.current_identity
        assert len(row["qpos"]) == 7 and len(row["qvel"]) == 6 and len(row["relevant_ctrl"]) == 4
        assert row["terminal_class"] == NOT_AVAILABLE
        if returned:
            assert row["raw_termination_reason"] == "unknown_issue"
            assert row["terminated"] is True and row["truncated"] is False
            assert row["returned_reward"] != NOT_AVAILABLE and row["physics_steps"] == 20
            assert row["terminal_flags"]["physical_failure"] is True
        else:
            for key in ("raw_termination_reason", "returned_reward", "terminated", "truncated",
                        "terminal_flags", "physics_steps", "lidar_status"):
                assert row[key] == NOT_AVAILABLE
        with pytest.raises(RuntimeError, match="latched"):
            env.step(np.zeros(4, dtype=np.float32))
        with pytest.raises(RuntimeError, match="cannot be reset"):
            env.reset()
        assert len(attempts) == 1
        assert env.policy_steps == 1 and not env.episode_returns and not env.episode_diagnostics
        assert not any(env.terminal_counts.values())
        assert paths[0].read_bytes() == content
        assert list((tmp_path / "failures").glob("failure_*.json")) == paths
        assert not (tmp_path / "train_episodes.jsonl").exists()
    finally:
        env.close()


@pytest.mark.parametrize("persistence_error", ["write_failure", "existing_event"])
def test_health_evidence_persistence_failure_cannot_resume_or_overwrite(tmp_path, monkeypatch, persistence_error):
    """Catch nested capture swallowing storage failure or replacing an earlier incident."""
    from . import failure_evidence

    recorder = _recorder(tmp_path / "failures")
    original_path, original_bytes = None, None
    if persistence_error == "existing_event":
        monkeypatch.setattr(failure_evidence, "_event_id", lambda: "fixed_forced_event")
        original_path = recorder.capture({}, ValueError("prior independent incident"))
        original_bytes = original_path.read_bytes()
    else:
        def failed_publish(*args):
            raise OSError("FORCED_TEST_STATE evidence publication failed")
        monkeypatch.setattr(failure_evidence.os, "link", failed_publish)

    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = FormalHealthMetricsEnv(scheduler, diagnostics_dir=tmp_path, failure_recorder=recorder)
    try:
        env.reset()

        def broken_step(action):
            raise ValueError("FORCED_TEST_STATE original bottom failure")

        monkeypatch.setattr(scheduler.active.unwrapped, "step", broken_step)
        with pytest.raises(RuntimeError, match="failure evidence persistence failed") as failure:
            env.step(np.zeros(4, dtype=np.float32))
        cause = failure.value
        while cause.__cause__ is not None:
            cause = cause.__cause__
        assert isinstance(cause, ValueError) and "original bottom failure" in str(cause)
        with pytest.raises(RuntimeError, match="latched"):
            env.step(np.zeros(4, dtype=np.float32))
        with pytest.raises(RuntimeError, match="cannot be reset"):
            env.reset()
        assert env.policy_steps == 0 and not env.episode_returns and not env.episode_diagnostics
        assert not any(env.terminal_counts.values())
        assert env.last_failure_path is None
        if original_path is not None:
            assert original_path.read_bytes() == original_bytes
        records = list((tmp_path / "failures").glob("failure_*.json"))
        assert records == ([original_path] if original_path is not None else [])
        assert not list((tmp_path / "failures").glob("*.tmp"))
        assert not (tmp_path / "train_episodes.jsonl").exists()
    finally:
        env.close()
