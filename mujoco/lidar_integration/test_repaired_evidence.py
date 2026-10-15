"""Pure file/data checks for L9S evidence; no model or simulator imports."""

import hashlib
import importlib
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

import pytest


def api():
    name = "lidar_integration.analyze_repaired_integration"
    assert importlib.util.find_spec(name) is not None, "independent read-only L9S artifact validator missing"
    return importlib.import_module(name)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def required_fixture(tmp_path):
    names = (
        "pilot.attempt", "execution_protocol.json", "pilot_summary.json",
        "historical_identity_before.json", "train_episodes.jsonl",
        "resource_samples.jsonl", "causal_observations.jsonl", "formal_progress.json",
        "L9S_REPAIRED_PILOT_ONLY_SEED0.zip", "L9S_REPAIRED_PILOT_ONLY_SEED0.json",
    )
    for name in names:
        (tmp_path / name).write_bytes(b"{}")
    manifest = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in names}
    return {"actual_timesteps": 4096, "execution_protocol_sha256": "a" * 64,
            "evidence_sha256": manifest}


def counts(**updates):
    result = {"success": 0, "collision": 0, "time_limit": 0,
              "physical_failure": 0, "sensor_failure": 0}
    result.update(updates)
    return result


def episode(**updates):
    row = {"scenario_identity_sha256": "a" * 64, "outcome": "physical_failure",
           "terminal_class": "physical_failure", "raw_termination_reason": "outside_flight_area",
           "policy_end_step": 1, "episode_policy_steps": 1,
           "episode_return": -10., "mean_action_norm": 1., "trajectory_length_m": 0.,
           "net_displacement_m": 0., "final_target_distance_m": 2.,
           "sim_end_time_s": .04, "motionless_episode": True}
    row.update(updates)
    return row


def causal_fixture():
    rows, episodes = [], []
    local = 0
    for step in range(4097):
        if step == 0 or local == 375:
            rows.append({"policy_step": step, "scenario_identity_sha256": "a" * 64,
                         "simulation_time": 0., "lidar_sample_time": 0., "lidar_age_s": 0.,
                         "physics_steps": 0, "terminal_class": None,
                         "raw_termination_reason": None, "lidar_status": "FRESH"})
            local = 0
        if step == 4096:
            break
        local += 1
        now = local * .04
        sample = math.floor((now + 1e-10) * 10) / 10
        terminal = "time_limit" if local == 375 else None
        rows.append({"policy_step": step + 1, "scenario_identity_sha256": "a" * 64,
                     "simulation_time": now, "lidar_sample_time": sample,
                     "lidar_age_s": max(0., now - sample), "physics_steps": 20,
                     "terminal_class": terminal, "raw_termination_reason": terminal,
                     "lidar_status": "FRESH"})
        if terminal:
            episodes.append(episode(outcome=terminal, terminal_class=terminal,
                raw_termination_reason=terminal, policy_end_step=step + 1,
                episode_policy_steps=375, sim_end_time_s=15.))
    summary = {"actual_timesteps": 4096, "training_physics_ticks": 81920,
               "lidar_causal_rows": 4107, "max_lidar_age_s": max(r["lidar_age_s"] for r in rows),
               "reset_compile_ms": {"count": 11},
               "train_scheduler": {"selected_identities": ["a" * 64] * 11}}
    return rows, episodes, summary


def validation_fixture():
    identities = {str(i) * 64 for i in range(1, 7)}
    rows = []
    for identity in sorted(identities):
        rows.append({"scenario_identity_sha256": identity, "outcome": "collision",
                     "terminal_class": "collision", "raw_termination_reason": "collision",
                     "success": False, "collision": True, "timeout": False,
                     "physical_failure": False, "sensor_failure": False,
                     "policy_steps": 5, "episode_return": -30.})
    overall = {**counts(collision=6), "probe_cutoff": 0, "total": 6,
               "navigation_success_rate": 0., "collision_rate": 1., "timeout_rate": 0.,
               "physical_failure_rate": 0., "sensor_failure_rate": 0., "mean_return": -30.}
    result = {"split": "validation", "probe_only": False, "episodes": rows,
              "overall": overall, "per_scene": {r["scenario_identity_sha256"]: dict(r) for r in rows},
              "optimizer_unchanged": True, "training_timesteps_before_after": 4096,
              "policy_sha256_before_after": "f" * 64, "model_status": "FROZEN_MODEL_DIAGNOSTIC"}
    return result, identities


def test_artifact_analyzer_does_not_import_learning_or_simulator_modules():
    before = set(sys.modules)
    api()
    added = set(sys.modules) - before
    assert not any(name.split(".")[0] in {"torch", "stable_baselines3", "mujoco"}
                   or name == "lidar_integration.ppo_repaired_pilot" for name in added)


def test_missing_complete_marker_cannot_become_pass(tmp_path):
    write_json(tmp_path / "pilot_summary.json", {"status": "PASS"})
    with pytest.raises(ValueError, match="completion"):
        api().analyze_completed_pilot(tmp_path)


def test_complete_marker_with_sparse_manifest_cannot_become_pass(tmp_path):
    write_json(tmp_path / "pilot.complete.json", {"actual_timesteps": 4096,
        "execution_protocol_sha256": "a" * 64, "evidence_sha256": {}})
    with pytest.raises(ValueError, match="required"):
        api().analyze_completed_pilot(tmp_path)


@pytest.mark.parametrize("name", ["/tmp/outside.json", "../outside.json", "nested/file.json", "a\\b.json"])
def test_evidence_manifest_rejects_path_escape(tmp_path, name):
    complete = required_fixture(tmp_path)
    complete["evidence_sha256"][name] = "f" * 64
    with pytest.raises(ValueError, match="path"):
        api().validate_evidence_manifest(tmp_path, complete)


@pytest.mark.parametrize("fault", ["missing", "hash", "extra", "symlink", "failure"])
def test_evidence_manifest_rejects_missing_tampered_unsealed_or_failed_evidence(tmp_path, fault):
    complete = required_fixture(tmp_path)
    target = tmp_path / "pilot_summary.json"
    if fault == "missing":
        target.unlink()
    elif fault == "hash":
        target.write_text("changed")
    elif fault == "extra":
        (tmp_path / "unsealed.json").write_text("{}")
    elif fault == "symlink":
        target.unlink()
        target.symlink_to(tmp_path / "pilot.attempt")
    else:
        (tmp_path / "failures").mkdir()
        (tmp_path / "failures/failure_test.json").write_text("{}")
    with pytest.raises(ValueError):
        api().validate_evidence_manifest(tmp_path, complete)


def test_evidence_manifest_accepts_only_exact_hash_bound_file_set(tmp_path):
    complete = required_fixture(tmp_path)
    assert api().validate_evidence_manifest(tmp_path, complete) == complete["evidence_sha256"]


def test_episode_audit_recounts_physical_raw_reasons_independently():
    result = api().validate_episode_rows([episode()], {"a" * 64}, counts(physical_failure=1))
    assert result == {"episode_count": 1, "terminal_counts": counts(physical_failure=1),
                      "physical_raw_reason_counts": {"outside_flight_area": 1},
                      "natural_physical_failure": "OBSERVED"}


@pytest.mark.parametrize("changes", [
    {"raw_termination_reason": "unknown"}, {"terminal_class": "collision"},
    {"scenario_identity_sha256": "sealed_test"}, {"episode_return": float("nan")},
    {"policy_end_step": True}, {"episode_policy_steps": 0},
    {"policy_end_step": 4097, "episode_policy_steps": 4097},
])
def test_episode_audit_rejects_unknown_nontrain_or_impossible_rows(changes):
    with pytest.raises(ValueError):
        api().validate_episode_rows([episode(**changes)], {"a" * 64}, counts(physical_failure=1))


def test_episode_audit_rejects_summary_count_disagreement():
    with pytest.raises(ValueError, match="count"):
        api().validate_episode_rows([episode()], {"a" * 64}, counts())


def test_causal_audit_recounts_resets_physics_and_terminal_rows():
    rows, episodes, summary = causal_fixture()
    result = api().validate_causal_rows(rows, {"a" * 64}, episodes, summary)
    assert result["policy_steps"] == 4096
    assert result["reset_rows"] == 11
    assert result["physics_ticks"] == 81920
    assert result["terminal_rows"] == 10


@pytest.mark.parametrize("fault", ["future", "age", "ticks", "skip", "identity", "terminal", "reset"])
def test_causal_audit_rejects_inconsistent_timestamps_steps_and_terminal_chain(fault):
    rows, episodes, summary = causal_fixture()
    if fault == "future":
        rows[1]["lidar_sample_time"] = 100.
    elif fault == "age":
        rows[1]["lidar_age_s"] = 100.
    elif fault == "ticks":
        summary["training_physics_ticks"] += 1
    elif fault == "skip":
        rows[1]["policy_step"] = 2
    elif fault == "identity":
        rows[1]["scenario_identity_sha256"] = "b" * 64
    elif fault == "terminal":
        rows[375]["terminal_class"] = "collision"
        rows[375]["raw_termination_reason"] = "collision"
    else:
        rows[376]["simulation_time"] = .04
    with pytest.raises(ValueError):
        api().validate_causal_rows(rows, {"a" * 64}, episodes, summary)


def test_validation_audit_uses_exact_six_identities_and_recounts_outcomes():
    validation, identities = validation_fixture()
    assert api().validate_validation(validation, identities) == counts(collision=6)


@pytest.mark.parametrize("fault", ["identity", "duplicate", "raw", "flags", "probe", "count", "steps", "optimizer"])
def test_validation_audit_rejects_incomplete_or_inconsistent_diagnostics(fault):
    value, identities = validation_fixture()
    if fault == "identity":
        value["episodes"][0]["scenario_identity_sha256"] = "test"
    elif fault == "duplicate":
        value["episodes"][1] = dict(value["episodes"][0])
    elif fault == "raw":
        value["episodes"][0]["raw_termination_reason"] = "outside_flight_area"
    elif fault == "flags":
        value["episodes"][0]["success"] = True
    elif fault == "probe":
        value["probe_only"] = True
    elif fault == "count":
        value["overall"]["collision"] = 5
    elif fault == "steps":
        value["training_timesteps_before_after"] = 0
    else:
        value["optimizer_unchanged"] = False
    with pytest.raises(ValueError):
        api().validate_validation(value, identities)


def completed_fixture(directory):
    """Synthetic JSON/ZIP schema fixture; deliberately contains no learned tensors."""
    root = Path(__file__).resolve().parents[2]
    reports = root / "mujoco/reports"
    prereg = json.loads((reports / "mujoco_lidar_ppo_training_prereg_seed0.json").read_text())
    frozen = json.loads((reports / "mujoco_lidar_ppo_termination_repair.json").read_text())["frozen_repair_identity"]
    protocol = {"experiment_id": "L9S_REPAIRED_PILOT_ONLY_SEED0",
        "parent_l9r_sha256": "e267406ec4203353b4ee0c7f8349b57759ef64b31bd076c3db57b99b3682678f",
        "formal_retrain_authorized": False, "pilot_authorization_scope": "ONE_ATTEMPT_ONLY",
        "seed": 0, "workers": 1, "actual_timesteps": 4096,
        "initialization": "NEW_RANDOM_NETWORK_NO_CHECKPOINT", "test_episode_budget": 0,
        "continuation": "RESTART_REQUIRED; this task does not authorize a restart",
        "validation": "one final diagnostic, six frozen Validation scenes, deterministic",
        "scene_manifest_sha256": prereg["scenario_manifest_sha256"],
        "network_implementation_sha256": prereg["network_implementation_sha256"],
        "historical_checkpoint_sha256": prereg["historical_checkpoint_sha256"],
        "parent_source_sha256": frozen["protocol"]["repair_source_sha256"],
        "runtime_source_sha256": {"ppo_repaired_pilot.py": hashlib.sha256(
            Path(__file__).with_name("ppo_repaired_pilot.py").read_bytes()).hexdigest()},
        "minimum_mem_available_mib": 1536., "minimum_disk_free_bytes": 2 * 1024**3,
        "software_versions": frozen["software_versions"],
        "original_model_sha256": "cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520",
        "upstream_commit": "e6b879df4b96daaed30ef10fe02650d29048fa9b",
        "upstream_pattern_sha256": "9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3"}
    keys = ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda", "clip_range",
            "ent_coef", "vf_coef", "max_grad_norm", "policy", "normalize_images", "network",
            "ray_count", "physics_hz", "pi_hz", "policy_hz", "lidar_hz", "reward_version")
    protocol["algorithm"] = {k: prereg["formal_proposal_not_run"][k] for k in keys}
    digest = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    protocol_sha = digest(protocol)
    attempt = {"experiment_id": protocol["experiment_id"], "attempt_id": "L9S_" + "a" * 32,
               "execution_protocol_sha256": protocol_sha}
    train = prereg["scenario_manifest"]["train"]
    identity = train[0]["identity_sha256"]
    causal, episodes, summary = causal_fixture()
    for row in causal + episodes:
        row["scenario_identity_sha256"] = identity
    summary["train_scheduler"] = {"seed": 0, "current_identity": identity,
        "selected_identities": [identity] * 11,
        "visit_counts": {r["identity_sha256"]: 11 if r["identity_sha256"] == identity else 0 for r in train}}
    validation, _ = validation_fixture()
    for row, original in zip(validation["episodes"], prereg["scenario_manifest"]["validation"]):
        row["scenario_identity_sha256"] = original["identity_sha256"]
    validation["per_scene"] = {r["scenario_identity_sha256"]: dict(r) for r in validation["episodes"]}
    resources = [{"policy_step": i, "mem_available_mib": 8192., "disk_free_bytes": 10 * 1024**3}
                 for i in range(128, 4097, 128)]
    with zipfile.ZipFile(directory / (protocol["experiment_id"] + ".zip"), "w") as zipped:
        zipped.writestr("data", json.dumps({"num_timesteps": 4096, "_n_updates": 64}))
        zipped.writestr("policy.pth", b"SYNTHETIC_SCHEMA_FIXTURE_NOT_A_MODEL")
        zipped.writestr("policy.optimizer.pth", b"SYNTHETIC_SCHEMA_FIXTURE_NOT_AN_OPTIMIZER")
    checkpoint = {"run_id": protocol["experiment_id"], "training_timesteps": 4096, "training_seed": 0,
        "execution_protocol_sha256": protocol_sha, "checkpoint_sha256": hashlib.sha256(
            (directory / (protocol["experiment_id"] + ".zip")).read_bytes()).hexdigest(),
        "l6_manifest_sha256": prereg["scenario_manifest_sha256"],
        "l6_formal_proposal_sha256": prereg["formal_proposal_sha256"],
        "l7_formal_protocol_sha256": "7c0c032002a5ea04b3400a5d939215cb10996a40fc5ff2f008c076ec51f55e33",
        "reward_version": "NAV_V2_COLLISION_PREFLIGHT_V1",
        "network_implementation_sha256": prereg["network_implementation_sha256"],
        "historical_checkpoint_sha256": prereg["historical_checkpoint_sha256"],
        "optimizer_state_in_sb3_zip": True, "continuation_class": "WARM_START",
        "interrupted_formal_run_policy": "RESTART_REQUIRED", "scheduler_state": summary["train_scheduler"]}
    checkpoint["sidecar_payload_sha256"] = digest(checkpoint)
    summary.update(attempt, status="PASS", formal_retrain_authorized=False,
        rollouts=32, ppo_n_updates=64, optimizer_steps=128, test_episode_count=0,
        OOM=False, finite_gradients_and_optimizer=True, checkpoint_reload_equal=True,
        train_rng_unchanged_during_validation=True, global_rng_preserved=True, historical_identity_unchanged=True,
        runtime_source_sha256=protocol["runtime_source_sha256"],
        parameter_hash_before={"actor": "b" * 64, "critic": "c" * 64},
        parameter_hash_after={"actor": "d" * 64, "critic": "e" * 64},
        environment_step_ms={"count": 4096}, ppo_update_ms={"count": 32},
        min_mem_available_mib=8192., minimum_disk_free_bytes=10 * 1024**3, resource_guard_samples=5000,
        checkpoint=checkpoint, validation=validation,
        accounting={"terminal_counts": counts(time_limit=10), "physical_raw_reason_counts": {},
                    "natural_physical_failure": "NOT OBSERVED"},
        health={"terminal_counts": counts(time_limit=10), "actual_execution_step": 4096,
                "policy_steps": 4096, "optimizer_updates": 64, "test_episode_count": 0,
                "completed_episodes": 10, "train_episodes_jsonl_rows": 10,
                "resource_sample_count": 32, "ppo_diagnostics": {"train/loss": 1.}})
    from .analyze_termination_repair import audit_original_l9
    for name, value in (("pilot.attempt", attempt), ("execution_protocol.json", protocol),
                        ("pilot_summary.json", summary), ("historical_identity_before.json", audit_original_l9()),
                        (protocol["experiment_id"] + ".json", checkpoint),
                        ("formal_progress.json", {"status": "INCOMPLETE", "actual_timesteps": 4096,
                         "completed_episodes": 10, "latest_resource_sample": resources[-1]})):
        write_json(directory / name, value)
    for name, rows in (("train_episodes.jsonl", episodes), ("resource_samples.jsonl", resources),
                       ("causal_observations.jsonl", causal)):
        (directory / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    complete = {"actual_timesteps": 4096, "execution_protocol_sha256": protocol_sha,
                "evidence_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}}
    write_json(directory / "pilot.complete.json", complete)
    return protocol


def reseal(directory):
    complete = json.loads((directory / "pilot.complete.json").read_text())
    complete["evidence_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in directory.iterdir() if p.name != "pilot.complete.json"}
    write_json(directory / "pilot.complete.json", complete)


def test_complete_synthetic_schema_is_audited_without_loading_model_or_mutating_files(tmp_path):
    completed_fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = api().analyze_completed_pilot(tmp_path)
    assert result["status"] == "PASS"
    assert result["formal_retrain_authorized"] is False
    assert result["episode_audit"]["episode_count"] == 10
    assert result["validation_terminal_counts"] == counts(collision=6)
    assert "does not deserialize tensors" in result["limitations"][0]
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("field,value", [("status", "FAIL"), ("actual_timesteps", 4095),
    ("optimizer_steps", 127), ("formal_retrain_authorized", True), ("test_episode_count", 1)])
def test_resealed_failed_incomplete_or_unauthorized_summary_is_rejected(tmp_path, field, value):
    completed_fixture(tmp_path)
    path = tmp_path / "pilot_summary.json"
    summary = json.loads(path.read_text())
    summary[field] = value
    write_json(path, summary)
    reseal(tmp_path)
    with pytest.raises(ValueError):
        api().analyze_completed_pilot(tmp_path)


@pytest.mark.parametrize("field", ["continuation", "validation", "original_model_sha256",
                                  "upstream_commit", "upstream_pattern_sha256"])
def test_protocol_rejects_changed_frozen_execution_semantics(tmp_path, field):
    protocol = completed_fixture(tmp_path)
    protocol[field] = "tampered"
    with pytest.raises(ValueError, match="protocol|scope"):
        api()._verify_protocol(protocol)


def reseal_protocol(directory, protocol):
    """Model an attacker recomputing all untrusted artifact-level checksums."""
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    protocol_sha = digest(protocol)
    checkpoint_path = directory / "L9S_REPAIRED_PILOT_ONLY_SEED0.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["execution_protocol_sha256"] = protocol_sha
    checkpoint.pop("sidecar_payload_sha256")
    checkpoint["sidecar_payload_sha256"] = digest(checkpoint)
    write_json(checkpoint_path, checkpoint)
    for name in ("pilot.attempt", "pilot_summary.json", "pilot.complete.json"):
        path = directory / name
        value = json.loads(path.read_text())
        value["execution_protocol_sha256"] = protocol_sha
        if name == "pilot_summary.json":
            value["checkpoint"] = checkpoint
            value["runtime_source_sha256"] = protocol["runtime_source_sha256"]
        write_json(path, value)
    write_json(directory / "execution_protocol.json", protocol)
    reseal(directory)


@pytest.mark.parametrize("fault", ["extra_field", "replacement_runtime"])
def test_resealed_new_protocol_cannot_replace_the_authorized_identity(tmp_path, monkeypatch, fault):
    protocol = completed_fixture(tmp_path)
    module = api()
    if fault == "extra_field":
        protocol["unauthorized_scope_extension"] = True
    else:
        # Simulate changed bytes at the hashing boundary; the real running
        # Pilot source is never written or patched.
        runtime = Path(__file__).with_name("ppo_repaired_pilot.py").resolve()
        original_sha = module._sha
        monkeypatch.setattr(module, "_sha", lambda path:
                            "f" * 64 if Path(path).resolve() == runtime else original_sha(path))
        protocol["runtime_source_sha256"]["ppo_repaired_pilot.py"] = "f" * 64
    reseal_protocol(tmp_path, protocol)
    with pytest.raises(ValueError, match="authorized|frozen|protocol"):
        module.analyze_completed_pilot(tmp_path)


@pytest.mark.parametrize("fault", ["payload", "claimed_hash"])
def test_changed_published_freeze_is_rejected_without_mutating_frozen_file(tmp_path, monkeypatch, fault):
    completed_fixture(tmp_path)
    module = api()
    published_path = module.REPORTS / "mujoco_lidar_ppo_repaired_integration_protocol.json"
    published = json.loads(published_path.read_text())
    if fault == "payload":
        published["protocol"]["unauthorized_scope_extension"] = True
        published["protocol_sha256"] = hashlib.sha256(json.dumps(
            published["protocol"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    else:
        published["protocol_sha256"] = "f" * 64
    original_json = module._json
    monkeypatch.setattr(module, "_json", lambda path:
                        published if Path(path) == published_path else original_json(path))
    with pytest.raises(ValueError, match="authorized|frozen|protocol"):
        module.analyze_completed_pilot(tmp_path)


@pytest.mark.parametrize("text", ["1e309", "-1e309", '{"nested": [1e309]}'])
def test_json_float_overflow_cannot_enter_evidence_as_infinity(text):
    with pytest.raises(ValueError, match="nonfinite"):
        api()._parse(text)
