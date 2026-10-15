"""L7 contracts; real MuJoCo paths, no PPO learning call."""

from __future__ import annotations
from research_regression.upstream import resolve_upstream

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from .ppo_formal_scheduler import (MultiScenarioTrainingEnv, load_l6_prereg,
                                   sample_train_sequence, verify_l6_runtime_sources)
from .ppo_formal_validation import LidarPPOValidationEvaluator
from .ppo_preflight import make_ppo
from .ppo_formal_checkpoint import CheckpointManager
from .ppo_formal_probe import run_readiness_probe
from .ppo_formal_analysis import build_readiness_report


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "reports/mujoco_lidar_ppo_training_prereg_seed0.json"
UPSTREAM = resolve_upstream()


def test_frozen_manifest_rejects_recomputed_but_unregistered_scene(tmp_path):
    registered = load_l6_prereg(PREREG)
    assert [len(registered["scenario_manifest"][key])
            for key in ("train", "validation", "test")] == [12, 6, 6]
    altered = copy.deepcopy(registered)
    altered["scenario_manifest"]["train"][0]["start_xyz"][0] += .01
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError):
        load_l6_prereg(path)


def test_frozen_runtime_source_bytes_are_verified(tmp_path):
    prereg = load_l6_prereg(PREREG)
    for relative in prereg["preflight_source_sha256"]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    verify_l6_runtime_sources(prereg, source_root=tmp_path)
    target = tmp_path / "lidar_integration/ppo_preflight.py"
    target.write_bytes(target.read_bytes() + b"\n# tampered\n")
    with pytest.raises(ValueError, match="runtime source"):
        verify_l6_runtime_sources(prereg, source_root=tmp_path)


def test_train_scheduler_draws_only_train_and_is_reproducible():
    prereg = load_l6_prereg(PREREG)
    first = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=18)
    second = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=18)
    try:
        drawn_a, drawn_b = [], []
        for _ in range(24):
            _, a = first.reset()
            _, b = second.reset()
            drawn_a.append(a["scenario_identity_sha256"])
            drawn_b.append(b["scenario_identity_sha256"])
            assert a["scenario_identity_sha256"] == b["scenario_identity_sha256"]
            assert a["scenario_split"] == "train"
        assert len(set(drawn_a)) > 1
        assert set(drawn_a).isdisjoint(
            {row["identity_sha256"] for split in ("validation", "test")
             for row in prereg["scenario_manifest"][split]})
        assert sum(first.visit_counts.values()) == 24
        assert first.active_environment_count == 1
    finally:
        first.close()
        second.close()


def test_all_train_rows_initialize_exactly_and_switch_resets_state():
    prereg = load_l6_prereg(PREREG)
    env = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=0)
    try:
        for row in prereg["scenario_manifest"]["train"]:
            obs, info = env.reset(options={"scenario_identity_sha256": row["identity_sha256"]})
            assert info["obstacle_configuration_hash"] == row["obstacle_configuration_hash"]
            assert info["scenario_identity_sha256"] == row["identity_sha256"]
            plant = env.active.unwrapped
            np.testing.assert_allclose(plant.data.qpos[:3], row["start_xyz"])
            np.testing.assert_allclose(
                plant.data.qpos[3:7],
                [np.cos(row["start_yaw_rad"] / 2), 0., 0.,
                 np.sin(row["start_yaw_rad"] / 2)], atol=1e-7)
            np.testing.assert_allclose(plant.data.qvel, 0.)
            np.testing.assert_allclose(plant.data.ctrl, 0.)
            np.testing.assert_allclose(plant.data.act, 0.)
            np.testing.assert_allclose(plant.target_position, row["target_xyz"])
            np.testing.assert_allclose(plant.velocity_controller.integral_error, 0.)
            assert plant.data.time == 0
            assert plant.success_streak == 0
            assert plant.lidar_buffer.frame_count == 1
            assert info["lidar_source_time_s"] == 0
            assert plant._last_collision_kind is None
            assert plant._last_contact_pairs == []
            assert obs["lidar_age"][0] == 0
            env.step(np.array([.3, 0., 0., 0.], dtype=np.float32))
            assert plant.data.time > 0
            plant.velocity_controller._integral_error[:] = 1.
            plant.success_streak = 3
            plant._last_contact_pairs = [("coarse_collision", "nav_obstacle_0")]
        with pytest.raises(ValueError):
            env.reset(options={"scenario_identity_sha256":
                               prereg["scenario_manifest"]["validation"][0]["identity_sha256"]})
    finally:
        env.close()


def test_sampler_state_does_not_change_on_explicit_coverage_reset():
    prereg = load_l6_prereg(PREREG)
    env = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=7)
    try:
        before = env.scheduler_state()["rng_state"]
        env.reset(options={"scenario_identity_sha256":
                           prereg["scenario_manifest"]["train"][4]["identity_sha256"]})
        assert env.scheduler_state()["rng_state"] == before
    finally:
        env.close()


def test_validation_is_independent_and_reports_six_untrained_probe_outcomes():
    prereg = load_l6_prereg(PREREG)
    train = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=3)
    try:
        train.reset()
        scheduler_before = train.scheduler_state()
        model = make_ppo(train, seed=8)
        before = [parameter.detach().clone() for parameter in model.policy.parameters()]
        evaluator = LidarPPOValidationEvaluator(prereg, UPSTREAM)
        report = evaluator.evaluate(model, max_policy_steps=2)
        assert report["split"] == "validation"
        assert report["model_status"] == "UNTRAINED_RANDOM_INITIALIZATION"
        assert report["probe_only"] is True
        assert len(report["episodes"]) == 6
        assert {row["scenario_identity_sha256"] for row in report["episodes"]} == {
            row["identity_sha256"] for row in prereg["scenario_manifest"]["validation"]}
        assert all(row["outcome"] == "probe_cutoff" and row["policy_steps"] == 2
                   for row in report["episodes"])
        assert report["overall"]["probe_cutoff"] == 6
        assert report["overall"]["navigation_success_rate"] is None
        assert report["overall"]["timeout_rate"] is None
        assert report["overall"]["physical_failure_rate"] is None
        assert report["overall"]["sensor_failure_rate"] is None
        assert report["overall"]["mean_trajectory_length_m"] >= 0
        assert report["optimizer_unchanged"] is True
        assert train.scheduler_state() == scheduler_before
        assert all(np.array_equal(a.detach().numpy(), b.numpy())
                   for a, b in zip(model.policy.parameters(), before))
    finally:
        train.close()


def test_validation_rejects_tampered_split_and_invalid_budget():
    prereg = load_l6_prereg(PREREG)
    altered = copy.deepcopy(prereg)
    altered["scenario_manifest"]["validation"][0] = copy.deepcopy(
        altered["scenario_manifest"]["test"][0])
    with pytest.raises(ValueError):
        LidarPPOValidationEvaluator(altered, UPSTREAM)
    evaluator = LidarPPOValidationEvaluator(prereg, UPSTREAM)
    with pytest.raises(ValueError):
        evaluator.evaluate(object(), max_policy_steps=0)


def test_checkpoint_is_atomic_identity_verified_and_warm_start_only(tmp_path):
    prereg = load_l6_prereg(PREREG)
    env = MultiScenarioTrainingEnv(prereg, UPSTREAM, seed=11)
    try:
        obs, _ = env.reset()
        model = make_ppo(env, seed=11)
        manager = CheckpointManager(tmp_path, prereg)
        record = manager.save(model, env, run_id="L7_UNTRAINED_CONTRACT", seed=11)
        assert record["training_timesteps"] == 0
        assert record["continuation_class"] == "WARM_START"
        assert record["interrupted_formal_run_policy"] == "RESTART_REQUIRED"
        assert record["optimizer_state_in_sb3_zip"] is True
        restored = manager.load_verified(env, run_id="L7_UNTRAINED_CONTRACT")
        np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                      restored.predict(obs, deterministic=True)[0])
        with pytest.raises(ValueError):
            manager.load_verified(env, run_id="L7_UNTRAINED_CONTRACT", mode="EXACT_RESUME")
        with pytest.raises(FileExistsError):
            manager.save(model, env, run_id="L7_UNTRAINED_CONTRACT", seed=11)
        sidecar = tmp_path / "L7_UNTRAINED_CONTRACT.json"
        payload = json.loads(sidecar.read_text())
        payload["checkpoint_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            manager.load_verified(env, run_id="L7_UNTRAINED_CONTRACT")
    finally:
        env.close()


def test_checkpoint_rejects_prereg_tamper(tmp_path):
    prereg = load_l6_prereg(PREREG)
    prereg["formal_proposal_not_run"]["learning_rate"] = 1e-2
    with pytest.raises(ValueError):
        CheckpointManager(tmp_path, prereg)


def test_uniform_sampler_has_reproducible_bounded_empirical_counts():
    prereg = load_l6_prereg(PREREG)
    a = sample_train_sequence(prereg, seed=29, draws=1200)
    b = sample_train_sequence(prereg, seed=29, draws=1200)
    c = sample_train_sequence(prereg, seed=30, draws=1200)
    assert a == b and a != c
    assert len(a) == 1200
    registered = {row["identity_sha256"] for row in prereg["scenario_manifest"]["train"]}
    assert set(a) == registered
    counts = [a.count(identity) for identity in registered]
    assert min(counts) >= 60 and max(counts) <= 140


def test_bounded_probe_never_trains_or_runs_test_episode(tmp_path):
    report = run_readiness_probe(PREREG, UPSTREAM, tmp_path, validation_step_cap=2,
                                 train_cycles=1)
    assert report["train_coverage"]["registered_scene_count"] == 12
    assert report["train_coverage"]["actual_scene_count"] == 12
    assert report["validation"]["overall"]["total"] == 6
    assert report["validation"]["probe_only"] is True
    assert report["ppo_optimizer_updates"] == 0
    assert report["test_episode_count"] == 0
    assert report["formal_training_executed"] is False
    assert report["checkpoint_contract"]["continuation_class"] == "WARM_START"
    assert all(row["scenario_split"] == "train" for row in report["train_coverage"]["rows"])
    assert report["system_min_available_mib"] > 1536
    with pytest.raises(FileExistsError):
        run_readiness_probe(PREREG, UPSTREAM, tmp_path, validation_step_cap=2,
                            train_cycles=1)


def test_full_probe_evidence_cannot_be_promoted_to_formal_training_ready():
    raw_dir = ROOT / "reports/mujoco_lidar_ppo_formal_readiness_parts/run_01"
    if not (raw_dir / "probe_summary.json").exists():
        pytest.skip("local-only bounded probe evidence absent in clean clone")
    report = build_readiness_report(PREREG, raw_dir)
    assert report["FORMAL_TRAINING_READY"] == "NO"
    assert report["status"]["train_scene_scheduler"] == "PASS"
    assert report["status"]["validation_six_complete_episodes"] == "PASS"
    assert report["status"]["formal_ppo_training"] == "NOT TESTED"
    assert report["status"]["periodic_100k_training_orchestration"] == "BLOCKED"
    assert report["evidence"]["ppo_optimizer_updates"] == 0
    assert report["evidence"]["test_episode_count"] == 0
