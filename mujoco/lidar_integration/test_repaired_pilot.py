"""L9S authorization and evidence regression: never invoke PPO.learn here."""

import importlib
import importlib.util
import json
from pathlib import Path

import pytest


def api():
    name = "lidar_integration.ppo_repaired_pilot"
    assert importlib.util.find_spec(name) is not None, "independent L9S authorization entry missing"
    return importlib.import_module(name)


def validation_environment(factory):
    from .ppo_formal_scheduler import load_l6_prereg
    from .ppo_scenarios import build_ppo_scene
    from .test_termination_repair import PREREG, UPSTREAM
    row = load_l6_prereg(PREREG)["scenario_manifest"]["validation"][0]
    env = factory(row["layout"], row["scene_seed"], UPSTREAM, scene=build_ppo_scene(row))
    return env, row


def test_independent_protocol_preserves_l9r_denial_and_algorithm():
    m = api()
    protocol = m.execution_protocol()
    assert protocol["parent_l9r_sha256"] == "e267406ec4203353b4ee0c7f8349b57759ef64b31bd076c3db57b99b3682678f"
    assert protocol["formal_retrain_authorized"] is False
    assert protocol["actual_timesteps"] == 4096
    assert protocol["algorithm"]["ray_count"] == 24000
    assert protocol["algorithm"]["n_steps"] == 128
    assert protocol["algorithm"]["reward_version"] == "NAV_V2_COLLISION_PREFLIGHT_V1"
    assert protocol["algorithm"]["ent_coef"] == 0.0
    m.verify_parent()


def test_published_protocol_roundtrip_keeps_float_identity_and_denial():
    m = api()
    frozen = json.loads((m.REPORTS / "mujoco_lidar_ppo_repaired_integration_protocol.json").read_text())
    assert m.protocol_hash(frozen["protocol"]) == frozen["protocol_sha256"]
    assert m.protocol_hash(m.execution_protocol()) == frozen["protocol_sha256"]
    assert type(frozen["protocol"]["algorithm"]["ent_coef"]) is float
    assert frozen["protocol"]["formal_retrain_authorized"] is False


@pytest.mark.parametrize("change, error", [
    ({"authorized": False}, PermissionError), ({"seed": 1}, ValueError),
    ({"seed": True}, ValueError), ({"workers": 2}, ValueError),
    ({"timesteps": 4097}, ValueError), ({"timesteps": 128}, ValueError),
    ({"initializer": "SMOKE_ONLY.zip"}, ValueError),
    ({"expected_sha256": "0" * 64}, ValueError),
    ({"experiment_id": "FORMAL_SEED0"}, ValueError),
])
def test_rejects_unauthorized_or_changed_execution_before_output(change, error, tmp_path):
    m = api()
    values = dict(authorized=True, seed=0, workers=1, timesteps=4096,
                  initializer=None, expected_sha256=m.protocol_hash(m.execution_protocol()),
                  experiment_id="L9S_REPAIRED_PILOT_ONLY_SEED0")
    values.update(change)
    with pytest.raises(error):
        m.validate_request(**values)
    assert list(tmp_path.iterdir()) == []


def test_resources_stop_at_frozen_thresholds(tmp_path, monkeypatch):
    m = api()
    monkeypatch.setattr(m, "_available_mib", lambda: 1535.9)
    with pytest.raises(MemoryError):
        m.check_resources(tmp_path)
    monkeypatch.setattr(m, "_available_mib", lambda: 4096.)
    from collections import namedtuple
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(m.shutil, "disk_usage", lambda path: Usage(9, 8, 2 * 1024**3 - 1))
    with pytest.raises(OSError):
        m.check_resources(tmp_path)


def test_attempt_reservation_is_exclusive_and_never_overwrites(tmp_path):
    m = api()
    target = tmp_path / "new_run"
    first = m.reserve_attempt(target, {"experiment_id": "FORCED_TEST_STATE"})
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        m.reserve_attempt(target, {"experiment_id": "retry"})
    assert first.read_bytes() == before
    with pytest.raises(FileExistsError):
        m.publish_json(first, {"replacement": True})
    assert first.read_bytes() == before


def test_post_link_flush_failure_leaves_no_false_complete(tmp_path, monkeypatch):
    import stat
    m = api()
    real_fsync = m.os.fsync
    def fail_directory(fd):
        if stat.S_ISDIR(m.os.fstat(fd).st_mode):
            raise OSError("forced directory fsync failure")
        real_fsync(fd)
    monkeypatch.setattr(m.os, "fsync", fail_directory)
    target = tmp_path / "pilot.complete.json"
    with pytest.raises(OSError, match="fsync"):
        m.publish_json(target, {"status": "PASS"})
    assert not target.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_pilot_output_cannot_be_redirected_to_new_attempt_or_history(tmp_path):
    m = api()
    for path in (tmp_path / "retry", m.ROOT / "mujoco/reports/mujoco_lidar_ppo_formal_seed0_parts/run_01"):
        with pytest.raises(ValueError):
            m.require_output(path)


def test_raw_episode_audit_keeps_physical_reason_and_rejects_corruption():
    m = api()
    identity = "a" * 64
    row = dict(terminal_class="physical_failure", raw_termination_reason="outside_flight_area",
               outcome="physical_failure", scenario_identity_sha256=identity, episode_return=-1.,
               mean_action_norm=1., final_target_distance_m=5., trajectory_length_m=2.,
               policy_end_step=1, episode_policy_steps=1)
    result = m.audit_episodes([row], {identity}, {"success":0, "collision":0, "time_limit":0,
                                              "physical_failure":1, "sensor_failure":0})
    assert result["physical_raw_reason_counts"] == {"outside_flight_area":1}
    for change in ({"raw_termination_reason":"unknown"}, {"scenario_identity_sha256":"test"},
                   {"terminal_class":"collision"}, {"episode_return":float("nan")}):
        with pytest.raises(RuntimeError):
            m.audit_episodes([{**row, **change}], {identity}, result["terminal_counts"])


def test_cli_never_allows_formal_or_initializer():
    m = api()
    for args in ([], ["--formal-seed0"], ["--initializer", "PILOT_ONLY.zip"]):
        with pytest.raises(SystemExit):
            m.main(args)


def test_resource_probe_does_not_create_output(tmp_path):
    m = api()
    path = tmp_path / "nonexistent" / "run"
    result = m.check_resources(path)
    assert result["disk_free_bytes"] >= 2 * 1024**3
    assert not path.exists() and not path.parent.exists()


def test_real_pilot_observation_is_causal_and_budget_latches_before_physics(tmp_path):
    import numpy as np
    from .ppo_formal_scheduler import MultiScenarioTrainingEnv, load_l6_prereg
    from .failure_evidence import FailureEvidenceRecorder
    from .test_termination_repair import PREREG, UPSTREAM
    m = api()
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = m.PilotMetricsEnv(scheduler, protocol=m.execution_protocol(), diagnostics_dir=tmp_path,
        failure_recorder=FailureEvidenceRecorder(tmp_path / "failures", run_id="L9S_BUDGET_TEST",
            attempt_id="FORCED_TEST_STATE", origin="FORCED_TEST_STATE"))
    try:
        env.reset()
        env.step(np.zeros(4, dtype=np.float32))
        rows = [json.loads(line) for line in (tmp_path / "causal_observations.jsonl").read_text().splitlines()]
        assert [r["simulation_time"] for r in rows] == [0., pytest.approx(.04)]
        assert [r["lidar_sample_time"] for r in rows] == [0., 0.]
        assert env.lidar_scans == 1
        before = scheduler.active.unwrapped.data.time
        env.policy_steps = 4096  # FORCED_TEST_STATE counter, not performed training
        with pytest.raises(RuntimeError, match="budget"):
            env.step(np.zeros(4, dtype=np.float32))
        assert scheduler.active.unwrapped.data.time == before
        with pytest.raises(RuntimeError, match="latched"):
            env.step(np.zeros(4, dtype=np.float32))
    finally:
        env.close()


def test_future_lidar_and_nonfinite_state_rejected_before_policy(tmp_path):
    import numpy as np
    from .ppo_formal_scheduler import MultiScenarioTrainingEnv, load_l6_prereg
    from .test_termination_repair import PREREG, UPSTREAM
    m = api()
    env = m.PilotMetricsEnv(MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0),
                           protocol=m.execution_protocol(), diagnostics_dir=tmp_path)
    try:
        obs, info = env.reset()
        with pytest.raises(RuntimeError, match="future"):
            env._audit_observation(obs, {**info, "lidar_source_time_s": .1})
        bad = {**obs, "state": np.full(7, np.nan)}
        with pytest.raises(RuntimeError, match="nonfinite"):
            env._audit_observation(bad, info)
    finally:
        env.close()


def test_checkpoint_reload_rng_scope_restores_python_numpy_and_torch():
    import random
    import numpy as np
    import torch
    m = api()
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with m.preserve_rng():
        random.seed(90210)
        np.random.seed(90210)
        torch.manual_seed(90210)
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert np.random.get_state()[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


def test_validation_guard_checks_disk_before_step_and_restores_factory(tmp_path, monkeypatch):
    import numpy as np
    from . import ppo_formal_validation as validation
    from .test_termination_repair import UPSTREAM
    from .failure_evidence import FailureEvidenceRecorder
    m = api()
    assert hasattr(m, "guarded_validation"), "Validation lacks L9S disk/finite/causal guard"
    original = validation.LidarNavigationEnv
    recorder = FailureEvidenceRecorder(tmp_path / "failures", run_id="L9S_VAL_TEST",
        attempt_id="FORCED_TEST_STATE", origin="FORCED_TEST_STATE")
    with pytest.raises(OSError, match="disk"):
        with m.guarded_validation(tmp_path, recorder):
            env, _ = validation_environment(validation.LidarNavigationEnv)
            try:
                env.reset()
                before = env.data.time
                monkeypatch.setattr(m, "check_resources", lambda path:
                    (_ for _ in ()).throw(OSError("forced disk guard")))
                with pytest.raises(OSError, match="disk"):
                    env.step(np.zeros(4, dtype=np.float32))
                assert env.data.time == before
                raise OSError("exit disk fault")
            finally:
                env.close()
    assert validation.LidarNavigationEnv is original


@pytest.mark.parametrize("fault", ["nonfinite", "future"])
def test_validation_guard_preserves_returned_evidence_for_new_faults(tmp_path, monkeypatch, fault):
    import numpy as np
    from . import ppo_formal_validation as validation
    from .navigation_env import LidarNavigationEnv
    from .test_termination_repair import UPSTREAM
    from .failure_evidence import FailureEvidenceRecorder
    m = api()
    recorder = FailureEvidenceRecorder(tmp_path / "failures", run_id="L9S_VAL_GUARD",
        attempt_id="FORCED_TEST_STATE", origin="FORCED_TEST_STATE")
    original_step = LidarNavigationEnv.step
    if fault == "future":
        def future_result(plant, action):
            obs, reward, term, trunc, info = original_step(plant, action)
            if plant.data.time > .05:
                info = {**info, "lidar_source_time_s": plant.data.time + .1}
            return obs, reward, term, trunc, info
        monkeypatch.setattr(LidarNavigationEnv, "step", future_result)
    with m.guarded_validation(tmp_path, recorder):
        env, scenario = validation_environment(validation.LidarNavigationEnv)
        try:
            env.reset()
            prior = env.step(np.zeros(4, dtype=np.float32))
            if fault == "nonfinite":
                env.data.ctrl[0] = np.nan
            with pytest.raises(RuntimeError, match="nonfinite|causality"):
                env.step(np.zeros(4, dtype=np.float32))
            row = json.loads(next((tmp_path / "failures").glob("failure_*.json")).read_text())
            assert row["context_kind"] == "RETURNED_TRANSITION"
            assert row["policy_step_attempt"] == 2 and row["last_completed_policy_step"] == 1
            assert row["scenario_identity_sha256"] == scenario["identity_sha256"]
            assert row["last_known_reward"] == prior[1]
            assert row["returned_reward"] != "NOT_AVAILABLE"
            if fault == "nonfinite":
                assert row["terminal_class"] == "physical_failure"
                assert row["raw_termination_reason"] == "nonfinite_state"
        finally:
            env.close()
