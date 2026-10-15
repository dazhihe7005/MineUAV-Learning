"""Read-only L9S artifact audit, using only the Python standard library.

No PPO deserialization, policy inference, training, simulator or entry imports.
Hashes establish consistency with the frozen producer; they do not independently
reexecute the producer's reported optimizer/reload checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "mujoco/reports"
OUTPUT = REPORTS / "mujoco_lidar_ppo_repaired_integration_parts/run_01"
EXPERIMENT_ID = "L9S_REPAIRED_PILOT_ONLY_SEED0"
PARENT_SHA256 = "e267406ec4203353b4ee0c7f8349b57759ef64b31bd076c3db57b99b3682678f"
AUTHORIZED_PROTOCOL_SHA256 = "09837c2104b12a1780cf81b6eb06c9b6c7f6e167d3378381def706e6af9f5cb3"
TERMINALS = ("success", "collision", "time_limit", "physical_failure", "sensor_failure")
PHYSICAL = ("z_below_zero", "outside_flight_area", "excessive_tilt", "nonfinite_state", "physical_failure")
REQUIRED_FILES = frozenset((
    "pilot.attempt", "execution_protocol.json", "pilot_summary.json",
    "historical_identity_before.json", "train_episodes.jsonl", "resource_samples.jsonl",
    "causal_observations.jsonl", "formal_progress.json",
    EXPERIMENT_ID + ".zip", EXPERIMENT_ID + ".json",
))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON evidence key")
        result[key] = value
    return result


def _parse(text):
    def reject(value):
        raise ValueError("nonfinite JSON evidence: " + value)
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            reject(value)
        return result
    return json.loads(text, parse_constant=reject, parse_float=finite_float,
                      object_pairs_hook=_pairs)


def _json(path):
    try:
        value = _parse(Path(path).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("missing or malformed JSON evidence: " + str(path)) from error
    _require(isinstance(value, dict), "JSON evidence must be an object")
    return value


def _jsonl(path):
    try:
        rows = [_parse(line) for line in Path(path).read_text().splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("missing or malformed JSONL evidence: " + str(path)) from error
    _require(all(isinstance(row, dict) for row in rows), "JSONL rows must be objects")
    return rows


def validate_evidence_manifest(directory, complete):
    """Require the complete file set, safe flat paths and matching SHA256 values."""
    directory = Path(directory)
    manifest = complete.get("evidence_sha256")
    _require(isinstance(manifest, dict), "completion evidence manifest required")
    for name, digest in manifest.items():
        _require(isinstance(name, str) and name not in ("", ".", "..")
                 and "/" not in name and "\\" not in name and not Path(name).is_absolute(),
                 "unsafe evidence manifest path")
        _require(_digest(digest), "invalid evidence SHA256")
    _require(REQUIRED_FILES <= set(manifest), "required completion evidence missing")
    _require(set(manifest) == REQUIRED_FILES, "unexpected completion evidence file")
    actual = set()
    for path in directory.iterdir():
        _require(not path.is_symlink(), "symlink evidence path is forbidden")
        if path.is_dir():
            _require(path.name == "failures" and not any(path.iterdir()),
                     "failure or unsealed nested evidence prevents completion")
            continue
        _require(path.is_file(), "invalid evidence file")
        if path.name != "pilot.complete.json":
            actual.add(path.name)
    _require(actual == set(manifest), "missing or unsealed evidence files")
    for name, digest in manifest.items():
        _require(_sha(directory / name) == digest, "evidence hash mismatch: " + name)
    return dict(manifest)


def _terminal(row, *, running=False):
    cls, raw = row.get("terminal_class"), row.get("raw_termination_reason")
    if running and cls is None and raw is None:
        return None
    _require(cls in TERMINALS and ((cls == "physical_failure" and raw in PHYSICAL)
                                  or (cls != "physical_failure" and raw == cls)),
             "unknown or inconsistent terminal class/raw reason")
    return cls


def _counts(value):
    _require(isinstance(value, dict) and set(value) == set(TERMINALS)
             and all(_integer(n) for n in value.values()), "invalid terminal counts")


def validate_episode_rows(rows, train_identities, expected_counts):
    """Independently recount completed episodes; never infer an absent failure."""
    _counts(expected_counts)
    counts = dict.fromkeys(TERMINALS, 0)
    physical = Counter()
    previous = 0
    for row in rows:
        cls = _terminal(row)
        end, length = row.get("policy_end_step"), row.get("episode_policy_steps")
        _require(row.get("outcome") == cls
                 and row.get("scenario_identity_sha256") in train_identities
                 and _integer(end, 1, 4096) and _integer(length, 1, 375)
                 and end - previous == length,
                 "invalid completed episode identity or step chain")
        _require(all(_finite(row.get(key)) for key in
                     ("episode_return", "mean_action_norm", "sim_end_time_s")),
                 "nonfinite episode metric")
        for key in ("trajectory_length_m", "net_displacement_m", "final_target_distance_m"):
            value = row.get(key)
            _require(value == "NOT_AVAILABLE" or (_finite(value) and value >= 0),
                     "invalid episode motion metric")
        counts[cls] += 1
        if cls == "physical_failure":
            physical[row["raw_termination_reason"]] += 1
        previous = end
    _require(counts == expected_counts, "raw episode count differs from summary")
    return {"episode_count": len(rows), "terminal_counts": counts,
            "physical_raw_reason_counts": dict(physical),
            "natural_physical_failure": "OBSERVED" if physical else "NOT OBSERVED"}


def validate_causal_rows(rows, train_identities, episode_rows, summary):
    """Audit reset/transition sequencing, sensor causality and actual physics ticks."""
    _require(bool(rows), "causal observations missing")
    completed, physics, resets, terminals = 0, 0, [], []
    previous_time = 0.
    identity = None
    must_reset = True
    maximum_age = 0.
    for index, row in enumerate(rows):
        step = row.get("policy_step")
        ticks = row.get("physics_steps")
        now, source, age = (row.get(k) for k in
                            ("simulation_time", "lidar_sample_time", "lidar_age_s"))
        _require(_integer(step, 0, 4096) and _integer(ticks, 0, 20)
                 and row.get("scenario_identity_sha256") in train_identities,
                 "invalid causal step/physics/Train identity")
        _require(all(_finite(v) for v in (now, source, age)) and now >= 0
                 and source >= -1e-9 and source <= now + 1e-9 and age >= -1e-9
                 and abs(now - source - age) <= 1e-8,
                 "future or inconsistent LiDAR timestamps")
        maximum_age = max(maximum_age, age)
        cls = _terminal(row, running=True)
        if step == completed:
            _require(must_reset and ticks == 0 and cls is None
                     and abs(now) <= 1e-9 and abs(source) <= 1e-9 and abs(age) <= 1e-9,
                     "invalid or duplicate reset causal row")
            resets.append(row["scenario_identity_sha256"])
            identity = row["scenario_identity_sha256"]
            previous_time = 0.
            must_reset = False
        else:
            _require(index > 0 and not must_reset and step == completed + 1
                     and row["scenario_identity_sha256"] == identity,
                     "causal transition sequence/identity mismatch")
            _require(abs(now - previous_time - ticks / 500.) <= 1e-7,
                     "causal simulation time differs from actual physics ticks")
            completed = step
            physics += ticks
            previous_time = now
            if cls is not None:
                terminals.append((step, identity, cls, row["raw_termination_reason"], now))
                must_reset = True
    expected_terminals = [(row["policy_end_step"], row["scenario_identity_sha256"],
                           row["terminal_class"], row["raw_termination_reason"], row["sim_end_time_s"])
                          for row in episode_rows]
    _require(len(terminals) == len(expected_terminals)
             and all(a[:4] == b[:4] and abs(a[4] - b[4]) <= 1e-7
                     for a, b in zip(terminals, expected_terminals)),
             "causal terminal rows disagree with completed episode rows")
    _require(completed == 4096 and summary.get("actual_timesteps") == completed
             and summary.get("training_physics_ticks") == physics
             and summary.get("lidar_causal_rows") == len(rows)
             and summary.get("reset_compile_ms", {}).get("count") == len(resets)
             and summary.get("train_scheduler", {}).get("selected_identities") == resets,
             "causal counters or reset selection history disagree with summary")
    _require(_finite(summary.get("max_lidar_age_s"))
             and abs(summary["max_lidar_age_s"] - maximum_age) <= 1e-8,
             "maximum causal LiDAR age differs from summary")
    return {"policy_steps": completed, "reset_rows": len(resets), "physics_ticks": physics,
            "terminal_rows": len(terminals), "observation_rows": len(rows),
            "maximum_lidar_age_s": maximum_age}


def validate_validation(validation, identities):
    """Validate the six frozen terminal diagnostics without executing episodes."""
    _require(len(identities) == 6 and validation.get("split") == "validation"
             and validation.get("probe_only") is False
             and validation.get("optimizer_unchanged") is True
             and type(validation.get("training_timesteps_before_after")) is int
             and validation["training_timesteps_before_after"] == 4096
             and validation.get("model_status") == "FROZEN_MODEL_DIAGNOSTIC"
             and _digest(validation.get("policy_sha256_before_after")),
             "incomplete Validation or changed training-state evidence")
    rows = validation.get("episodes")
    _require(isinstance(rows, list) and len(rows) == 6
             and {r.get("scenario_identity_sha256") for r in rows} == identities,
             "Validation must contain each frozen identity exactly once")
    counts = dict.fromkeys(TERMINALS, 0)
    for row in rows:
        cls = _terminal(row)
        _require(row.get("outcome") == cls and _integer(row.get("policy_steps"), 1, 375)
                 and _finite(row.get("episode_return")), "invalid Validation episode")
        for flag in ("success", "collision", "timeout", "physical_failure", "sensor_failure"):
            expected = cls == ("time_limit" if flag == "timeout" else flag)
            _require(row.get(flag) is expected, "inconsistent Validation terminal flags")
        counts[cls] += 1
    overall = validation.get("overall", {})
    _require(all(type(overall.get(k)) is int and overall[k] == n for k, n in counts.items())
             and type(overall.get("total")) is int and overall["total"] == 6
             and type(overall.get("probe_cutoff")) is int and overall["probe_cutoff"] == 0,
             "Validation aggregate terminal count mismatch")
    for key, cls in (("navigation_success_rate", "success"), ("collision_rate", "collision"),
                     ("timeout_rate", "time_limit"), ("physical_failure_rate", "physical_failure"),
                     ("sensor_failure_rate", "sensor_failure")):
        _require(_finite(overall.get(key)) and abs(overall[key] - counts[cls] / 6) < 1e-12,
                 "Validation terminal rate mismatch")
    _require(validation.get("per_scene") == {r["scenario_identity_sha256"]: r for r in rows},
             "Validation per-scene evidence mismatch")
    _require(_finite(overall.get("mean_return"))
             and abs(overall["mean_return"] - sum(r["episode_return"] for r in rows) / 6) < 1e-8,
             "Validation mean-return mismatch")
    return counts


def _verify_protocol(protocol):
    published = _json(REPORTS / "mujoco_lidar_ppo_repaired_integration_protocol.json")
    _require(set(published) == {"protocol", "protocol_sha256"}
             and published["protocol_sha256"] == AUTHORIZED_PROTOCOL_SHA256
             and _hash(published["protocol"]) == AUTHORIZED_PROTOCOL_SHA256,
             "published frozen L9S protocol differs from authorized identity")
    _require(_hash(protocol) == AUTHORIZED_PROTOCOL_SHA256
             and protocol == published["protocol"],
             "artifact protocol differs from exact authorized frozen L9S protocol")
    prereg_path = REPORTS / "mujoco_lidar_ppo_training_prereg_seed0.json"
    _require(_sha(prereg_path) == "c468490154627ba8f71df9c73bc8c80769f557054d9692313139188971e332c8",
             "frozen preregistration identity changed")
    prereg = _json(prereg_path)
    repair = _json(REPORTS / "mujoco_lidar_ppo_termination_repair.json")
    frozen = repair["frozen_repair_identity"]
    parent = frozen["protocol"]
    _require(frozen["protocol_sha256"] == PARENT_SHA256 and _hash(parent) == PARENT_SHA256
             and parent["formal_retrain_authorized"] is False
             and repair["FORMAL_RETRAIN_AUTHORIZED"] == "NO",
             "frozen L9R identity or formal denial changed")
    sources = parent["repair_source_sha256"]
    _require(len(sources) == 9 and protocol.get("parent_source_sha256") == sources,
             "L9R nine-source identity mismatch")
    for name, digest in sources.items():
        _require(Path(name).name == name and _sha(Path(__file__).with_name(name)) == digest,
                 "frozen L9R runtime source changed")
    runtime = {"ppo_repaired_pilot.py": _sha(Path(__file__).with_name("ppo_repaired_pilot.py"))}
    _require(protocol.get("runtime_source_sha256") == runtime, "L9S runtime source changed")
    expected = {"experiment_id": EXPERIMENT_ID, "parent_l9r_sha256": PARENT_SHA256,
                "formal_retrain_authorized": False, "pilot_authorization_scope": "ONE_ATTEMPT_ONLY",
                "seed": 0, "workers": 1, "actual_timesteps": 4096,
                "initialization": "NEW_RANDOM_NETWORK_NO_CHECKPOINT", "test_episode_budget": 0,
                "continuation": "RESTART_REQUIRED; this task does not authorize a restart",
                "validation": "one final diagnostic, six frozen Validation scenes, deterministic",
                "scene_manifest_sha256": prereg["scenario_manifest_sha256"],
                "network_implementation_sha256": prereg["network_implementation_sha256"],
                "historical_checkpoint_sha256": prereg["historical_checkpoint_sha256"],
                "minimum_mem_available_mib": 1536., "minimum_disk_free_bytes": 2 * 1024**3,
                "software_versions": frozen["software_versions"],
                "original_model_sha256": "cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520",
                "upstream_commit": "e6b879df4b96daaed30ef10fe02650d29048fa9b",
                "upstream_pattern_sha256": "9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3"}
    _require(all(protocol.get(k) == v and type(protocol.get(k)) is type(v) for k, v in expected.items()),
             "L9S execution scope/protocol mismatch")
    algorithm_keys = ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda",
                      "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "policy", "normalize_images",
                      "network", "ray_count", "physics_hz", "pi_hz", "policy_hz", "lidar_hz", "reward_version")
    algorithm = {k: prereg["formal_proposal_not_run"][k] for k in algorithm_keys}
    _require(_hash(protocol.get("algorithm")) == _hash(algorithm), "frozen PPO/plant algorithm changed")
    for relative, digest in prereg["preflight_source_sha256"].items():
        _require(_sha(ROOT / "mujoco" / relative) == digest, "frozen L6 runtime source changed")
    for name, digest in prereg["historical_checkpoint_sha256"].items():
        _require(_sha(ROOT / "mujoco/rl/models" / name) == digest, "historical checkpoint changed")
    return prereg


def _validate_summary(summary, protocol_sha, protocol):
    _require(summary.get("status") == "PASS" and summary.get("experiment_id") == EXPERIMENT_ID
             and summary.get("execution_protocol_sha256") == protocol_sha
             and summary.get("formal_retrain_authorized") is False
             and summary.get("OOM") is False, "Pilot summary is failed/incomplete or unauthorized")
    expected = {"actual_timesteps": 4096, "rollouts": 32, "ppo_n_updates": 64,
                "optimizer_steps": 128, "test_episode_count": 0}
    _require(all(type(summary.get(k)) is int and summary[k] == v for k, v in expected.items()),
             "Pilot step/rollout/optimizer/Test counters differ from authorized execution")
    for key in ("finite_gradients_and_optimizer", "checkpoint_reload_equal",
                "train_rng_unchanged_during_validation", "global_rng_preserved", "historical_identity_unchanged"):
        _require(summary.get(key) is True, "missing successful Pilot evidence: " + key)
    _require(summary.get("runtime_source_sha256") == protocol["runtime_source_sha256"],
             "summary runtime source identity mismatch")
    for component in ("actor", "critic"):
        before = summary.get("parameter_hash_before", {}).get(component)
        after = summary.get("parameter_hash_after", {}).get(component)
        _require(_digest(before) and _digest(after) and before != after, "actor/critic update evidence invalid")
    for key, count in (("environment_step_ms", 4096), ("ppo_update_ms", 32)):
        _require(summary.get(key, {}).get("count") == count, "Pilot timing sample count mismatch")
    _require(_finite(summary.get("min_mem_available_mib")) and summary["min_mem_available_mib"] >= 1536.
             and _integer(summary.get("minimum_disk_free_bytes"), 2 * 1024**3)
             and _integer(summary.get("resource_guard_samples"), 1), "resource guard evidence invalid")


def _validate_checkpoint(raw, summary, prereg, protocol_sha):
    record = _json(raw / (EXPERIMENT_ID + ".json"))
    payload = dict(record)
    signature = payload.pop("sidecar_payload_sha256", None)
    archive = raw / (EXPERIMENT_ID + ".zip")
    _require(signature == _hash(payload) and record == summary.get("checkpoint"),
             "checkpoint sidecar integrity/summary mismatch")
    _require(record.get("run_id") == EXPERIMENT_ID and record.get("training_timesteps") == 4096
             and record.get("training_seed") == 0 and record.get("execution_protocol_sha256") == protocol_sha
             and record.get("checkpoint_sha256") == _sha(archive)
             and record.get("l6_manifest_sha256") == prereg["scenario_manifest_sha256"]
             and record.get("network_implementation_sha256") == prereg["network_implementation_sha256"]
             and record.get("historical_checkpoint_sha256") == prereg["historical_checkpoint_sha256"]
             and record.get("optimizer_state_in_sb3_zip") is True
             and record.get("continuation_class") == "WARM_START"
             and record.get("interrupted_formal_run_policy") == "RESTART_REQUIRED"
             and record.get("scheduler_state") == summary.get("train_scheduler"),
             "checkpoint identity or continuation contract mismatch")
    try:
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
            _require(len(names) == len(set(names)) and
                     {"data", "policy.pth", "policy.optimizer.pth"} <= set(names),
                     "checkpoint ZIP lacks unique policy/optimizer payload")
            _require(zipped.testzip() is None, "corrupt checkpoint ZIP member")
            data = _parse(zipped.read("data").decode())
            _require(type(data.get("num_timesteps")) is int and data["num_timesteps"] == 4096
                     and type(data.get("_n_updates")) is int and data["_n_updates"] == 64,
                     "checkpoint ZIP training counters mismatch")
    except (OSError, zipfile.BadZipFile, UnicodeError) as error:
        raise ValueError("invalid checkpoint ZIP") from error
    return record["checkpoint_sha256"]


def analyze_completed_pilot(raw_directory=OUTPUT):
    """Return PASS only for complete, hash-bound, internally consistent evidence.

    A directory argument supports read-only examination of an evidence copy;
    experiment identity and current frozen source checks remain mandatory.
    """
    raw = Path(raw_directory)
    _require((raw / "pilot.complete.json").is_file(), "Pilot completion marker missing")
    try:
        complete = _json(raw / "pilot.complete.json")
        manifest = validate_evidence_manifest(raw, complete)
        protocol = _json(raw / "execution_protocol.json")
        prereg = _verify_protocol(protocol)
        protocol_sha = _hash(protocol)
        attempt = _json(raw / "pilot.attempt")
        summary = _json(raw / "pilot_summary.json")
        _require(type(complete.get("actual_timesteps")) is int and complete["actual_timesteps"] == 4096
                 and complete.get("execution_protocol_sha256") == protocol_sha,
                 "completion protocol or actual timestep mismatch")
        _require(attempt.get("experiment_id") == EXPERIMENT_ID
                 and attempt.get("execution_protocol_sha256") == protocol_sha
                 and isinstance(attempt.get("attempt_id"), str)
                 and re.fullmatch("L9S_[0-9a-f]{32}", attempt["attempt_id"]) is not None
                 and summary.get("attempt_id") == attempt["attempt_id"], "Pilot attempt identity mismatch")
        _validate_summary(summary, protocol_sha, protocol)
        train_ids = {r["identity_sha256"] for r in prereg["scenario_manifest"]["train"]}
        val_ids = {r["identity_sha256"] for r in prereg["scenario_manifest"]["validation"]}
        test_ids = {r["identity_sha256"] for r in prereg["scenario_manifest"]["test"]}
        _require(len(train_ids) == 12 and len(val_ids) == 6 and len(test_ids) == 6
                 and not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids),
                 "frozen scenario split identity mismatch")
        health = summary["health"]
        episodes = _jsonl(raw / "train_episodes.jsonl")
        accounting = validate_episode_rows(episodes, train_ids, health["terminal_counts"])
        _require(summary.get("accounting") == {k: v for k, v in accounting.items() if k != "episode_count"}
                 and health.get("actual_execution_step") == 4096 and health.get("policy_steps") == 4096
                 and health.get("optimizer_updates") == 64 and health.get("test_episode_count") == 0
                 and health.get("completed_episodes") == len(episodes)
                 and health.get("train_episodes_jsonl_rows") == len(episodes),
                 "health/episode accounting mismatch")
        diagnostics = health.get("ppo_diagnostics")
        _require(isinstance(diagnostics, dict) and diagnostics
                 and all(k.startswith("train/") and _finite(v) for k, v in diagnostics.items()),
                 "nonfinite or missing post-update PPO diagnostics")
        causal = validate_causal_rows(_jsonl(raw / "causal_observations.jsonl"), train_ids, episodes, summary)
        scheduler = summary["train_scheduler"]
        visits = scheduler["visit_counts"]
        _require(set(visits) == train_ids and all(_integer(n) for n in visits.values())
                 and dict(Counter(scheduler["selected_identities"])) == {k: v for k, v in visits.items() if v}
                 and scheduler.get("seed") == 0 and scheduler.get("current_identity") in train_ids,
                 "Train scheduler visit evidence mismatch")
        resources = _jsonl(raw / "resource_samples.jsonl")
        _require([r.get("policy_step") for r in resources] == list(range(128, 4097, 128))
                 and health.get("resource_sample_count") == 32,
                 "resource sample boundaries missing or duplicated")
        for row in resources:
            _require(_finite(row.get("mem_available_mib")) and row["mem_available_mib"] >= 1536.
                     and _integer(row.get("disk_free_bytes"), 2 * 1024**3), "resource sample guard violation")
        progress = _json(raw / "formal_progress.json")
        _require(progress.get("status") == "INCOMPLETE" and progress.get("actual_timesteps") == 4096
                 and progress.get("completed_episodes") == len(episodes)
                 and progress.get("latest_resource_sample") == resources[-1],
                 "passive formal progress cannot be promoted to formal completion")
        validation_counts = validate_validation(summary["validation"], val_ids)
        checkpoint_sha = _validate_checkpoint(raw, summary, prereg, protocol_sha)
        from .analyze_termination_repair import audit_original_l9
        _require(_json(raw / "historical_identity_before.json") == audit_original_l9(),
                 "protected historical L9 identity changed")
        # Recheck every evidence hash after parsing to catch concurrent mutation.
        _require(validate_evidence_manifest(raw, complete) == manifest, "evidence changed while auditing")
        return {"status": "PASS", "actual_timesteps": 4096, "formal_retrain_authorized": False,
                "test_episode_count": 0, "execution_protocol_sha256": protocol_sha,
                "evidence_sha256": manifest, "checkpoint_sha256": checkpoint_sha,
                "episode_audit": accounting, "causal_audit": causal,
                "validation_terminal_counts": validation_counts,
                "protocol": protocol, "summary": summary,
                "audit_method": "READ_ONLY_JSON_JSONL_SHA256_ZIP; NO_MODEL_LOAD_OR_EPISODES",
                "limitations": ["Optimizer/reload equality and finite gradients are hash-bound producer evidence; this audit does not deserialize tensors.",
                                "A single 4096-step Pilot does not prove navigation learning or long-run stability and does not authorize formal retraining."]}
    except (KeyError, TypeError, AttributeError, OSError) as error:
        raise ValueError("incomplete or malformed Pilot evidence") from error


if __name__ == "__main__":
    print(json.dumps(analyze_completed_pilot(), indent=2, allow_nan=False))
