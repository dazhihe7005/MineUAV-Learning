"""L8 rollout-aligned training orchestration contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import pytest
import torch

from .ppo_formal_scheduler import MultiScenarioTrainingEnv, load_l6_prereg
from .ppo_formal_checkpoint import CheckpointManager
from .ppo_preflight import make_ppo
from .ppo_training_orchestration import (
    MilestoneLedger, aligned_milestones, l8_execution_protocol,
    require_pilot_budget, execute_segments, TrainingMetricsEnv,
    FormalPPOTrainingDriver,
    source_hashes, verify_source_hashes,
    require_canonical_pilot_output,
    main,
)
from .ppo_training_orchestration_analysis import build_orchestration_report
from .ppo_training_runner import protocol_hash


PREREG = Path(__file__).resolve().parents[1] / "reports/mujoco_lidar_ppo_training_prereg_seed0.json"
UPSTREAM = Path("/tmp/mineuav-mujoco-lidar-audit")


def test_frozen_milestones_align_up_without_misreporting_nominal_budget():
    events = aligned_milestones(1_000_000, 100_000, 128)
    assert len(events) == 10
    assert [(x.nominal_step, x.actual_step) for x in events[:2]] == [
        (100_000, 100_096), (200_000, 200_064)]
    assert (events[-1].nominal_step, events[-1].actual_step) == (1_000_000, 1_000_064)
    assert all(x.actual_step % 128 == 0 for x in events)
    assert all(x.actual_step >= x.nominal_step for x in events)


def test_milestone_ledger_does_not_replay_or_skip_a_rollout_boundary():
    ledger = MilestoneLedger(aligned_milestones(200_000, 100_000, 128))
    assert ledger.acknowledge(99_968) is None
    assert ledger.acknowledge(100_000) is None
    assert ledger.acknowledge(100_096).nominal_step == 100_000
    assert ledger.acknowledge(100_096) is None
    with pytest.raises(ValueError, match="boundary"):
        ledger.acknowledge(200_192)
    assert ledger.acknowledge(200_064).nominal_step == 200_000
    assert ledger.acknowledge(200_064) is None


def test_l8_protocol_preserves_l6_budget_and_records_explicit_alignment_revision():
    prereg = load_l6_prereg(PREREG)
    protocol = l8_execution_protocol(prereg)
    assert protocol["nominal_total_timesteps_per_seed"] == 1_000_000
    assert protocol["actual_total_timesteps_per_seed"] == 1_000_064
    assert protocol["checkpoint_every_nominal_steps"] == 100_000
    assert protocol["n_steps"] == 128
    assert protocol["reward_version"] == "NAV_V2_COLLISION_PREFLIGHT_V1"
    assert protocol["parent_l7_protocol_sha256"] != protocol_hash(protocol)
    assert protocol["selection_rule"] == "final checkpoint, not test-selected"


def test_only_exact_pilot_budget_and_identity_can_run_in_this_phase():
    require_pilot_budget(seed=0, workers=1, total_timesteps=8192,
                         experiment_id="LIDAR_PPO_PILOT_ONLY_L8_SEED0")
    for kwargs in ({"seed": 1}, {"workers": 2}, {"total_timesteps": 8193},
                   {"experiment_id": "SMOKE_ONLY"}):
        request = {"seed": 0, "workers": 1, "total_timesteps": 8192,
                   "experiment_id": "LIDAR_PPO_PILOT_ONLY_L8_SEED0"}
        request.update(kwargs)
        with pytest.raises(ValueError):
            require_pilot_budget(**request)


def test_segment_calls_use_delta_and_process_events_after_updates():
    class CountingModel:
        def __init__(self):
            self.num_timesteps = 0
            self._n_updates = 0
            self.learn_calls = []

        def learn(self, total_timesteps, *, reset_num_timesteps):
            assert reset_num_timesteps is False
            assert total_timesteps > 0 and total_timesteps % 128 == 0
            self.learn_calls.append(total_timesteps)
            self.num_timesteps += total_timesteps
            self._n_updates += total_timesteps // 128 * 2
            return self

    model = CountingModel()
    observed = []
    execute_segments(model, aligned_milestones(200_000, 100_000, 128),
                     on_boundary=lambda event: observed.append(
                         (event.nominal_step, model.num_timesteps, model._n_updates)))
    assert model.learn_calls == [100_096, 99_968]
    assert observed == [(100_000, 100_096, 1564), (200_000, 200_064, 3126)]


def test_milestone_rejects_last_rollout_collected_without_its_optimizer_update():
    class InterruptedModel:
        n_steps = 128
        num_timesteps = 0
        _n_updates = 0

        def learn(self, total_timesteps, *, reset_num_timesteps):
            self.num_timesteps += total_timesteps
            self._n_updates += 2 * (total_timesteps // 128 - 1)
            return self

    with pytest.raises(RuntimeError, match="optimizer"):
        execute_segments(InterruptedModel(), aligned_milestones(256, 256, 128),
                         on_boundary=lambda _: pytest.fail("unoptimized checkpoint"))


def test_checkpoint_sidecar_binds_explicit_l8_execution_protocol(tmp_path):
    prereg = load_l6_prereg(PREREG)
    env = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0)
    try:
        env.reset()
        model = make_ppo(env, seed=0)
        new_hash = protocol_hash(l8_execution_protocol(prereg))
        manager = CheckpointManager(tmp_path, prereg, execution_protocol_sha256=new_hash)
        record = manager.save(model, env, run_id="L8_TEST_CONTRACT", seed=0)
        assert record["execution_protocol_sha256"] == new_hash
        assert manager.load_verified(env, run_id="L8_TEST_CONTRACT").num_timesteps == 0
        with pytest.raises(ValueError, match="execution protocol"):
            CheckpointManager(tmp_path, prereg, execution_protocol_sha256="0" * 64).load_verified(
                env, run_id="L8_TEST_CONTRACT")
    finally:
        env.close()


def test_real_sb3_rollouts_update_weights_and_switch_train_scenes():
    torch.set_num_threads(1)
    prereg = load_l6_prereg(PREREG)
    scheduler = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0)
    measured = TrainingMetricsEnv(scheduler)
    try:
        model = make_ppo(measured, seed=0)
        initial_actor = [p.detach().clone() for p in model.policy.action_net.parameters()]
        initial_critic = [p.detach().clone() for p in model.policy.value_net.parameters()]
        model.learn(512, reset_num_timesteps=False)
        assert model.num_timesteps == 512
        assert model._n_updates == 8
        assert measured.physics_ticks == 512 * 20
        assert sum(scheduler.visit_counts.values()) >= 2
        assert set(scheduler.selected_identities) <= set(scheduler.by_identity)
        assert measured.terminal_counts["time_limit"] >= 1
        assert any(not torch.equal(a, b) for a, b in
                   zip(initial_actor, model.policy.action_net.parameters()))
        assert any(not torch.equal(a, b) for a, b in
                   zip(initial_critic, model.policy.value_net.parameters()))
    finally:
        measured.close()


def test_pilot_refuses_reused_attempt_and_formal_mode_requires_separate_authorization(tmp_path,
                                                                                     monkeypatch):
    from . import ppo_training_orchestration as orchestration
    monkeypatch.setattr(orchestration, "PILOT_OUTPUT_DIR", tmp_path)
    driver = FormalPPOTrainingDriver(PREREG, UPSTREAM, tmp_path)
    with pytest.raises(PermissionError):
        driver.run_formal()
    (tmp_path / "pilot.attempt").touch()
    with pytest.raises(FileExistsError):
        driver.run_pilot()


def test_pilot_rejects_alternate_directory_even_if_it_is_empty(tmp_path):
    with pytest.raises(ValueError, match="canonical"):
        require_canonical_pilot_output(tmp_path)


def test_completed_real_rollout_milestone_saves_after_update_and_validates_without_mutation(tmp_path):
    torch.set_num_threads(1)
    prereg = load_l6_prereg(PREREG)
    scheduler = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0)
    measured = TrainingMetricsEnv(scheduler)
    try:
        model = make_ppo(measured, seed=0)
        driver = FormalPPOTrainingDriver(PREREG, UPSTREAM, tmp_path)
        event = aligned_milestones(128, 128, 128)[0]
        execute_segments(model, (event,), on_boundary=lambda item:
                         driver.persist_milestone(model, scheduler, item))
        assert model.num_timesteps == 128 and model._n_updates == 2
        assert len(driver.completed_events) == 1
        record = driver.completed_events[0]
        assert record["checkpoint"]["training_timesteps"] == 128
        assert record["validation"]["overall"]["total"] == 6
        assert record["validation"]["optimizer_unchanged"] is True
        assert record["train_rng_unchanged_during_validation"] is True
        with pytest.raises(FileExistsError):
            driver.persist_milestone(model, scheduler, event)
    finally:
        measured.close()


def test_passive_ppo_train_timer_counts_actual_updates_and_is_removed_before_save(tmp_path):
    torch.set_num_threads(1)
    prereg = load_l6_prereg(PREREG)
    scheduler = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0)
    measured = TrainingMetricsEnv(scheduler)
    try:
        model = make_ppo(measured, seed=0)
        timings = []
        execute_segments(model, aligned_milestones(256, 256, 128),
                         on_boundary=lambda _: None, update_times_ms=timings)
        assert model.num_timesteps == 256 and model._n_updates == 4
        assert len(timings) == 2 and all(value > 0 for value in timings)
        assert "train" not in vars(model)
        model.save(tmp_path / "timer_can_save.zip")
        assert (tmp_path / "timer_can_save.zip").is_file()
    finally:
        measured.close()


def test_pre_flight_runtime_source_fingerprint_detects_mid_run_change(tmp_path):
    first = tmp_path / "driver.py"
    second = tmp_path / "scheduler.py"
    first.write_text("initial driver")
    second.write_text("initial scheduler")
    frozen = source_hashes((first, second))
    verify_source_hashes((first, second), frozen)
    second.write_text("changed scheduler")
    with pytest.raises(RuntimeError, match="source identity"):
        verify_source_hashes((first, second), frozen)


def test_checkpoint_write_failure_cannot_create_completed_event(tmp_path, monkeypatch):
    from . import ppo_training_orchestration as orchestration
    driver = FormalPPOTrainingDriver(PREREG, UPSTREAM, tmp_path)
    def fail_save(*args, **kwargs):
        raise OSError("injected checkpoint failure")
    monkeypatch.setattr(orchestration.CheckpointManager, "save", fail_save)
    with pytest.raises(OSError, match="checkpoint failure"):
        driver.persist_milestone(SimpleNamespace(num_timesteps=128, _n_updates=2),
                                 SimpleNamespace(), aligned_milestones(128, 128, 128)[0])
    assert driver.completed_events == []
    assert not list(tmp_path.glob("*event.json"))
    assert not list(tmp_path.glob("*complete.json"))


def test_validation_rng_mutation_is_detected_before_event_completion(tmp_path, monkeypatch):
    from . import ppo_training_orchestration as orchestration
    prereg = load_l6_prereg(PREREG)
    scheduler = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0)
    try:
        scheduler.reset()
        model = make_ppo(scheduler, seed=0)
        model.num_timesteps = 128
        model._n_updates = 2
        driver = FormalPPOTrainingDriver(PREREG, UPSTREAM, tmp_path)
        monkeypatch.setattr(orchestration.CheckpointManager, "save", lambda *a, **k: {})
        monkeypatch.setattr(orchestration.CheckpointManager, "load_verified", lambda *a, **k: model)
        def tamper_validation(self, model):
            scheduler.reset()
            return {"probe_only": False}
        monkeypatch.setattr(orchestration.LidarPPOValidationEvaluator,
                            "evaluate", tamper_validation)
        with pytest.raises(RuntimeError, match="Validation changed Train scheduler"):
            driver.persist_milestone(model, scheduler, aligned_milestones(128, 128, 128)[0])
        assert driver.completed_events == []
        assert not list(tmp_path.glob("*event.json"))
    finally:
        scheduler.close()


def test_memory_guard_prevents_training_metrics_start(monkeypatch):
    from . import ppo_training_orchestration as orchestration
    def low_memory():
        raise MemoryError("below 1.5 GiB")
    monkeypatch.setattr(orchestration, "require_memory", low_memory)
    with pytest.raises(MemoryError, match="1.5 GiB"):
        TrainingMetricsEnv(SimpleNamespace())


def test_cli_help_does_not_start_training_or_create_pilot_attempt(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
    assert "pilot" in capsys.readouterr().out.lower()


def test_pilot_analysis_rejects_tampered_completion_and_preserves_negative_outcome(tmp_path):
    raw = Path(__file__).resolve().parents[1] / "reports/mujoco_lidar_ppo_orchestration_parts/run_01"
    if not (raw / "pilot_summary.json").exists():
        pytest.skip("local-only unique Pilot evidence absent in a clean clone")
    report = build_orchestration_report(PREREG, raw)
    assert report["TRAINING_DRIVER_READY"] == "NO"  # no test/review evidence supplied
    assert report["LONG_RUN_MEMORY_PROVEN"] == "NO"
    assert report["pilot"]["training_actual_timesteps"] == 8192
    assert report["pilot"]["terminal_counts"] == {
        "collision": 42, "physical_failure": 0, "sensor_failure": 0,
        "success": 0, "time_limit": 4}
    assert report["pilot"]["events"][0]["validation"]["overall"]["collision"] == 6
    for source in raw.iterdir():
        if source.is_file():
            shutil.copy2(source, tmp_path / source.name)
    complete = tmp_path / "pilot.complete.json"
    payload = json.loads(complete.read_text())
    payload["summary_sha256"] = "0" * 64
    complete.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="summary hash"):
        build_orchestration_report(PREREG, tmp_path)


def test_report_readiness_fails_closed_on_missing_failed_or_incomplete_tests():
    raw = Path(__file__).resolve().parents[1] / "reports/mujoco_lidar_ppo_orchestration_parts/run_01"
    if not (raw / "pilot_summary.json").exists():
        pytest.skip("local-only unique Pilot evidence absent in a clean clone")
    passing = {"independent_review": "GO", "tests": {
        "inherited": 679, "new": 19, "passed": 698, "failed": 0, "skipped": 0,
    }}
    assert build_orchestration_report(PREREG, raw, verification=passing)["FORMAL_SEED0_READY"] == "YES"
    for evidence in (
        None,
        {**passing, "independent_review": "PENDING"},
        {**passing, "tests": {**passing["tests"], "failed": 1}},
        {**passing, "tests": {**passing["tests"], "new": 18, "passed": 697}},
        {**passing, "tests": {**passing["tests"], "passed": 697}},
        {**passing, "tests": {**passing["tests"], "skipped": 1}},
    ):
        report = build_orchestration_report(PREREG, raw, verification=evidence)
        assert report["TRAINING_DRIVER_READY"] == "NO"
        assert report["FORMAL_SEED0_READY"] == "NO"
