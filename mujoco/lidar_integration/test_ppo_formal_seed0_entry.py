"""L9 formal seed0 launch fails closed before any PPO optimizer work."""

from __future__ import annotations
from research_regression.upstream import resolve_upstream

from pathlib import Path
from types import SimpleNamespace

import pytest

from .ppo_formal_seed0_entry import (
    FORMAL_ACTUAL_TIMESTEPS, FROZEN_L8_PROTOCOL_SHA256,
    preflight_formal_seed0, main,
)
from .ppo_formal_seed0_health import FormalHealthMetricsEnv, L9FormalPPOTrainingDriver
from .ppo_formal_scheduler import MultiScenarioTrainingEnv, load_l6_prereg
from .ppo_preflight import make_ppo
from .ppo_training_orchestration import Milestone, TrainingMetricsEnv, execute_segments
import numpy as np


UPSTREAM = resolve_upstream()
PREREG = Path(__file__).resolve().parents[1] / "reports/mujoco_lidar_ppo_training_prereg_seed0.json"


def _preflight(tmp_path, **changes):
    args = {
        "upstream_root": UPSTREAM, "output_dir": tmp_path / "formal_run",
        "expected_protocol_sha256": FROZEN_L8_PROTOCOL_SHA256,
        "seed": 0, "actual_timesteps": FORMAL_ACTUAL_TIMESTEPS,
        "authorized": True,
    }
    args.update(changes)
    return preflight_formal_seed0(**args)


def test_frozen_formal_preflight_accepts_only_current_l8_identity(tmp_path):
    result = _preflight(tmp_path)
    assert result["l8_protocol_sha256"] == "fe07816f48bd4183165db482d8c93de50c637a32d13d86eccb47667d85e8cef6"
    assert result["actual_timesteps"] == 1_000_064
    assert result["seed"] == 0
    assert result["train_scenes"] == 12 and result["validation_scenes"] == 6
    assert result["test_episode_count"] == 0
    assert not (tmp_path / "formal_run").exists()


def test_formal_preflight_checks_disk_on_existing_ancestor_without_creating_output(tmp_path):
    target = tmp_path / "not_created_yet" / "run_01"
    result = _preflight(tmp_path, output_dir=target)
    assert result["disk_free_bytes"] > 5 * 1024**3
    assert not target.exists() and not target.parent.exists()


@pytest.mark.parametrize("changes, error", [
    ({"authorized": False}, PermissionError),
    ({"seed": 1}, ValueError),
    ({"actual_timesteps": 1_000_000}, ValueError),
    ({"actual_timesteps": 1_000_192}, ValueError),
    ({"expected_protocol_sha256": "0" * 64}, ValueError),
])
def test_formal_preflight_rejects_wrong_authorization_seed_budget_or_protocol(tmp_path,
                                                                               changes, error):
    with pytest.raises(error):
        _preflight(tmp_path, **changes)
    assert not (tmp_path / "formal_run").exists()


def test_formal_preflight_rejects_existing_output_without_deleting_it(tmp_path):
    output = tmp_path / "formal_run"
    output.mkdir()
    marker = output / "formal.attempt"
    marker.write_text("prior run")
    with pytest.raises(FileExistsError):
        _preflight(tmp_path)
    assert marker.read_text() == "prior run"


def test_formal_preflight_rejects_low_disk_and_low_memory(tmp_path, monkeypatch):
    from . import ppo_formal_seed0_entry as entry
    monkeypatch.setattr(entry.shutil, "disk_usage", lambda path: SimpleNamespace(free=2 * 1024**3))
    with pytest.raises(OSError, match="disk"):
        _preflight(tmp_path)
    monkeypatch.undo()
    monkeypatch.setattr(entry, "require_memory", lambda: (_ for _ in ()).throw(MemoryError("guard")))
    with pytest.raises(MemoryError, match="guard"):
        _preflight(tmp_path)


def test_cli_requires_explicit_formal_mode_authorization_and_rejects_initializer(tmp_path,
                                                                                  monkeypatch):
    from . import ppo_formal_seed0_entry as entry
    monkeypatch.setattr(entry, "FORMAL_OUTPUT_DIR", tmp_path / "formal_run")
    argv = ["--formal-seed0", "--seed", "0", "--actual-timesteps", "1000064",
            "--expected-protocol-sha256", FROZEN_L8_PROTOCOL_SHA256,
            "--upstream", str(UPSTREAM)]
    for candidate in (argv, argv + ["--init-checkpoint", "SMOKE_ONLY.zip"]):
        with pytest.raises(SystemExit):
            main(candidate)
    called = []
    def capture_run(self, *, allow_formal_training=False):
        from . import ppo_training_orchestration as orchestration
        assert orchestration.TrainingMetricsEnv is FormalHealthMetricsEnv
        called.append((self.output_dir, allow_formal_training))
        return {"training_actual_timesteps": 1_000_064, "event_count": 10}
    monkeypatch.setattr(entry.FormalPPOTrainingDriver, "run_formal", capture_run)
    assert main(argv + ["--authorize-formal-training"]) == 0
    assert called == [(tmp_path / "formal_run", True)]
    assert not (tmp_path / "formal_run").exists()


def test_passive_health_wrapper_preserves_real_transitions_and_records_episode(tmp_path,
                                                                                monkeypatch):
    from . import ppo_formal_seed0_health as health
    monkeypatch.setattr(health, "FORMAL_HEALTH_DIR", tmp_path)
    prereg = load_l6_prereg(PREREG)
    base = TrainingMetricsEnv(MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0))
    enhanced = FormalHealthMetricsEnv(MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0))
    try:
        first_obs, first_info = base.reset()
        second_obs, second_info = enhanced.reset()
        assert first_info["scenario_identity_sha256"] == second_info["scenario_identity_sha256"]
        for key in first_obs:
            np.testing.assert_array_equal(first_obs[key], second_obs[key])
        zero = np.zeros(4, dtype=np.float32)
        for _ in range(375):
            left = base.step(zero)
            right = enhanced.step(zero)
            for key in left[0]:
                np.testing.assert_array_equal(left[0][key], right[0][key])
            assert left[1:4] == right[1:4]
            assert left[4]["termination_reason"] == right[4]["termination_reason"]
            if left[2] or left[3]:
                break
        assert len(enhanced.episode_diagnostics) == 1
        row = enhanced.episode_diagnostics[0]
        assert row["scenario_identity_sha256"] == first_info["scenario_identity_sha256"]
        assert row["outcome"] == right[4]["termination_reason"]
        assert row["policy_end_step"] == enhanced.policy_steps
        assert row["trajectory_length_m"] >= row["net_displacement_m"] >= 0
        assert np.isfinite(row["mean_action_norm"])
        assert row["final_target_distance_m"] == pytest.approx(right[4]["distance_m"])
        assert (tmp_path / "train_episodes.jsonl").is_file()
    finally:
        base.close()
        enhanced.close()


def test_health_monitor_stops_on_disk_pressure_without_false_completion(tmp_path, monkeypatch):
    from . import ppo_formal_seed0_health as health
    monkeypatch.setattr(health, "FORMAL_HEALTH_DIR", tmp_path)
    prereg = load_l6_prereg(PREREG)
    env = FormalHealthMetricsEnv(MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0))
    try:
        env.reset()
        monkeypatch.setattr(health.shutil, "disk_usage",
                            lambda path: SimpleNamespace(free=1024))
        with pytest.raises(OSError, match="disk"):
            env.step(np.zeros(4, dtype=np.float32))
        assert env.policy_steps == 0
        assert not list(tmp_path.glob("*.complete.json"))
    finally:
        env.close()


def test_formal_health_fails_closed_if_l9_runtime_source_changes(tmp_path, monkeypatch):
    from . import ppo_formal_seed0_health as health
    monkeypatch.setattr(health, "FORMAL_HEALTH_DIR", tmp_path)
    first = tmp_path / "entry.py"
    second = tmp_path / "health.py"
    first.write_text("frozen entry")
    second.write_text("frozen health")
    monkeypatch.setattr(health, "L9_RUNTIME_PATHS", (first, second))
    prereg = load_l6_prereg(PREREG)
    env = FormalHealthMetricsEnv(MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0))
    try:
        env.reset()
        second.write_text("changed mid-run")
        with pytest.raises(RuntimeError, match="source identity"):
            env.step(np.zeros(4, dtype=np.float32))
        assert env.policy_steps == 0
        assert not list(tmp_path.glob("*.complete.json"))
    finally:
        env.close()


def test_l9_driver_writes_real_post_optimizer_health_snapshot(tmp_path, monkeypatch):
    from . import ppo_formal_seed0_health as health
    monkeypatch.setattr(health, "FORMAL_HEALTH_DIR", tmp_path)
    prereg = load_l6_prereg(PREREG)
    env = FormalHealthMetricsEnv(MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0))
    try:
        model = make_ppo(env, seed=0)
        driver = L9FormalPPOTrainingDriver(PREREG, UPSTREAM, tmp_path)
        execute_segments(model, (Milestone(128, 128),), on_boundary=lambda item:
                         driver.persist_milestone(model, env.env, item))
        snapshot = tmp_path / "L9_HEALTH_STEP0000128.json"
        assert snapshot.is_file()
        payload = __import__("json").loads(snapshot.read_text())
        assert payload["actual_execution_step"] == 128
        assert payload["optimizer_updates"] == 2
        assert payload["policy_steps"] == 128
        assert payload["test_episode_count"] == 0
        assert payload["disk_free_bytes"] > 0
        resource_log = tmp_path / "resource_samples.jsonl"
        assert resource_log.is_file()
        assert len(resource_log.read_text().splitlines()) == 1
    finally:
        env.close()


@pytest.mark.parametrize("corruption", ["weight", "logger"])
def test_bad_post_update_model_cannot_be_checkpointed_or_validated(tmp_path,
                                                                   monkeypatch, corruption):
    from . import ppo_formal_seed0_health as health
    monkeypatch.setattr(health, "FORMAL_HEALTH_DIR", tmp_path)
    env = FormalHealthMetricsEnv(MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0))
    try:
        model = make_ppo(env, seed=0)
        model.learn(128, reset_num_timesteps=False)
        driver = L9FormalPPOTrainingDriver(PREREG, UPSTREAM, tmp_path)
        if corruption == "weight":
            with __import__("torch").no_grad():
                next(model.policy.parameters()).flatten()[0] = float("nan")
        else:
            model.logger.record("train/forced_nonfinite", float("nan"))
        with pytest.raises(RuntimeError, match="nonfinite"):
            driver.persist_milestone(model, env.env, Milestone(128, 128))
        assert not list(tmp_path.glob("L8_SEED0_STEP*_event.json"))
        assert not list(tmp_path.glob("L8_SEED0_STEP*.zip"))
    finally:
        env.close()


def test_formal_analysis_cannot_claim_completion_without_raw_marker(tmp_path):
    from .ppo_formal_seed0_analysis import build_formal_seed0_report
    with pytest.raises(FileNotFoundError):
        build_formal_seed0_report(
            PREREG, tmp_path, tests={"passed": 0},
            pretrain_review="GO", posttrain_review="PENDING")


def test_formal_rolling_metrics_use_past_episodes_only():
    from .ppo_formal_seed0_analysis import _roll
    np.testing.assert_array_equal(_roll([1., 0., 0., 1.], window=2),
                                  np.asarray([1., .5, 0., .5]))


def test_frozen_l8_physical_reason_contract_reproduces_formal_stop_without_training():
    """Keep the failing frozen contract visible; no optimizer or restart is performed."""
    prereg = load_l6_prereg(PREREG)
    scheduler = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0)
    measured = TrainingMetricsEnv(scheduler)
    try:
        measured.reset()
        plant = scheduler.active.unwrapped
        plant.data.qpos[2] = 100.  # controlled test-only physical boundary violation
        captured = []
        original_step = scheduler.step
        def capture_step(action):
            result = original_step(action)
            captured.append(result[4])
            return result
        scheduler.step = capture_step
        with pytest.raises(RuntimeError, match="invalid or nonexclusive training episode outcome"):
            measured.step(np.zeros(4, dtype=np.float32))
        assert captured[0]["termination_reason"] == "outside_flight_area"
        assert captured[0]["physical_failure"] is True
        assert sum(bool(captured[0].get(key)) for key in
                   ("success", "collision", "timeout", "physical_failure", "sensor_failure")) == 1
        assert measured.policy_steps == 1
        assert measured.physics_ticks == 0
        assert not measured.episode_returns
        assert sum(measured.terminal_counts.values()) == 0
    finally:
        measured.close()


def test_interrupted_formal_report_never_claims_exact_completion_or_final_model():
    from .ppo_formal_seed0_analysis import build_interrupted_seed0_report
    raw = PREREG.parent / "mujoco_lidar_ppo_formal_seed0_parts/run_01"
    report, data = build_interrupted_seed0_report(PREREG, raw)
    assert report["status"]["formal_seed0_training"] == "FAIL"
    assert report["actual_timesteps"] is None
    assert report["last_persisted_environment_step"] == 119552
    assert report["final_model_sha256"] is None
    assert report["automatic_restart_count"] == 0
    assert report["test_episode_count"] == 0
    assert len(data["episodes"]) == 706
    assert len(data["resources"]) == 934


def test_interrupted_report_rejects_a_completion_marker(tmp_path):
    from .ppo_formal_seed0_analysis import build_interrupted_seed0_report
    (tmp_path / "formal.complete.json").write_text("{}")
    with pytest.raises(ValueError, match="completion"):
        build_interrupted_seed0_report(PREREG, tmp_path)


def test_interrupted_figures_disclose_incomplete_status_in_visible_title():
    from PIL import Image
    directory = PREREG.parent / "mujoco_lidar_ppo_formal_seed0_figures"
    files = sorted(directory.glob("*.png"))
    assert len(files) == 9
    for path in files:
        with Image.open(path) as picture:
            pixels = np.asarray(picture.convert("RGB"))
        # The explicit header occupies the top 45px; axes begin below it.
        header = pixels[:45, 150:-150]
        assert np.count_nonzero(np.max(header, axis=2) < 100) > 200, path.name
