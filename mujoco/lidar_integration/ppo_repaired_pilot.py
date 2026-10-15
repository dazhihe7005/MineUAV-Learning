"""L9S independent one-shot authorization. No changes to the L9R denial gate.

Imports never train. Only the explicit CLI/run request can reserve the one
canonical attempt and construct a fresh PPO. Historical evidence stays read-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import resource
import shutil
import tempfile
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import torch

from .analyze_termination_repair import audit_original_l9
from .failure_evidence import (FailureEvidenceRecorder, NOT_AVAILABLE, simulator_context,
                               transition_failure_context)
from .ppo_formal_checkpoint import CheckpointManager
from .ppo_formal_scheduler import MultiScenarioTrainingEnv, load_l6_prereg
from .ppo_formal_seed0_entry import EXPECTED_SOFTWARE
from .ppo_formal_seed0_health import FormalHealthMetricsEnv, _append_jsonl
from .ppo_formal_validation import LidarPPOValidationEvaluator, _optimizer_states_equal
from .ppo_preflight import make_ppo
from .ppo_termination_repair_protocol import repair_execution_protocol, verify_repair_protocol
from .ppo_training_orchestration import (Milestone, execute_segments, _finite_model_and_gradients,
    _optimizer_step_count, source_hashes, verify_source_hashes)
from .ppo_training_runner import _historical_model_hashes, _weights_sha, _stats, protocol_hash
from .runner import _available_mib
from .scanner import validate_upstream, UPSTREAM_COMMIT, PATTERN_SHA256
from .scene import BASE_MODEL, BASE_SHA256
from .terminal_accounting import TERMINAL_CLASSES, PHYSICAL_REASONS

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "mujoco/reports"
PREREG = REPORTS / "mujoco_lidar_ppo_training_prereg_seed0.json"
OUTPUT = REPORTS / "mujoco_lidar_ppo_repaired_integration_parts/run_01"
EXPERIMENT_ID = "L9S_REPAIRED_PILOT_ONLY_SEED0"
PARENT_SHA256 = "e267406ec4203353b4ee0c7f8349b57759ef64b31bd076c3db57b99b3682678f"
STEPS = 4096
RESOURCE_AUDIT = ContextVar("l9s_resource_audit", default=None)


def verify_parent():
    prereg = load_l6_prereg(PREREG)
    verify_repair_protocol(prereg, expected_sha256=PARENT_SHA256)
    parent = repair_execution_protocol(prereg)
    if parent["formal_retrain_authorized"] is not False:
        raise RuntimeError("L9R formal retraining denial changed")
    report = json.loads((REPORTS / "mujoco_lidar_ppo_termination_repair.json").read_text())
    if (report["FORMAL_RETRAIN_AUTHORIZED"] != "NO"
            or report["TERMINAL_ACCOUNTING_FIXED"] != "YES"
            or report["FAILURE_CAPTURE_READY"] != "YES"):
        raise RuntimeError("L9R acceptance or authorization changed")
    return prereg, parent


def execution_protocol():
    prereg, parent = verify_parent()
    # Algorithm fields are copied from the frozen L6 proposal; orchestration
    # budget is separate and cannot authorize the old formal driver.
    algorithm = {k: v for k, v in prereg["formal_proposal_not_run"].items()
                 if k in ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma",
                          "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
                          "policy", "normalize_images", "network", "ray_count", "physics_hz",
                          "pi_hz", "policy_hz", "lidar_hz", "reward_version")}
    return {"experiment_id": EXPERIMENT_ID, "parent_l9r_sha256": PARENT_SHA256,
            "formal_retrain_authorized": False, "pilot_authorization_scope": "ONE_ATTEMPT_ONLY",
            "seed": 0, "workers": 1, "actual_timesteps": STEPS, "algorithm": algorithm,
            "initialization": "NEW_RANDOM_NETWORK_NO_CHECKPOINT",
            "continuation": "RESTART_REQUIRED; this task does not authorize a restart",
            "validation": "one final diagnostic, six frozen Validation scenes, deterministic",
            "test_episode_budget": 0, "scene_manifest_sha256": prereg["scenario_manifest_sha256"],
            "network_implementation_sha256": prereg["network_implementation_sha256"],
            "historical_checkpoint_sha256": prereg["historical_checkpoint_sha256"],
            "parent_source_sha256": parent["repair_source_sha256"],
            "runtime_source_sha256": source_hashes((Path(__file__),)),
            "minimum_mem_available_mib": 1536., "minimum_disk_free_bytes": 2 * 1024**3,
            "software_versions": EXPECTED_SOFTWARE,
            "original_model_sha256": BASE_SHA256, "upstream_commit": UPSTREAM_COMMIT,
            "upstream_pattern_sha256": PATTERN_SHA256}


def validate_request(*, authorized, seed, workers, timesteps, initializer,
                     expected_sha256, experiment_id):
    if authorized is not True:
        raise PermissionError("explicit L9S pilot authorization required")
    if (type(seed) is not int or seed != 0 or type(workers) is not int or workers != 1
            or type(timesteps) is not int or timesteps != STEPS
            or initializer is not None or experiment_id != EXPERIMENT_ID):
        raise ValueError("only new random L9S seed0, one worker, 4096 steps allowed")
    protocol = execution_protocol()
    if expected_sha256 != protocol_hash(protocol):
        raise ValueError("L9S protocol/source hash mismatch")
    return protocol


def require_output(directory):
    if Path(directory).resolve() != OUTPUT.resolve():
        raise ValueError("L9S permits only its unique canonical output directory")


def check_resources(directory):
    available = float(_available_mib())
    if not np.isfinite(available) or available < 1536.:
        raise MemoryError("MemAvailable below 1.5 GiB")
    probe = Path(directory)
    while not probe.exists():
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    audit = RESOURCE_AUDIT.get()
    if audit is not None:
        audit["samples"] += 1
        audit["min_mem_available_mib"] = min(audit["min_mem_available_mib"], available)
        audit["minimum_disk_free_bytes"] = min(audit["minimum_disk_free_bytes"], free)
    if free < 2 * 1024**3:
        raise OSError("disk below 2 GiB")
    return {"mem_available_mib": available, "disk_free_bytes": free}


def publish_json(path, value):
    """Atomic exclusive publication; an existing artifact is never replaced."""
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary, linked = None, False
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
        linked = True
        temporary.unlink()
        temporary = None
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        # This call created the link exclusively. A caught publication failure
        # must not leave an authoritative complete marker behind.
        if linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reserve_attempt(directory, identity):
    directory = Path(directory)
    # Exclusive directory creation also closes the empty-dir/check race.
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "pilot.attempt"
    publish_json(path, identity)
    return path


def audit_episodes(rows, train_identities, expected_counts):
    counts = {key: 0 for key in TERMINAL_CLASSES}
    physical = Counter()
    previous = 0
    for row in rows:
        cls, raw = row["terminal_class"], row["raw_termination_reason"]
        if (cls not in counts or row["outcome"] != cls
                or row["scenario_identity_sha256"] not in train_identities
                or (cls == "physical_failure" and raw not in (*PHYSICAL_REASONS, "physical_failure"))
                or (cls != "physical_failure" and raw != cls)
                or row["policy_end_step"] <= previous
                or row["episode_policy_steps"] != row["policy_end_step"] - previous
                or not all(np.isfinite(row[key]) for key in ("episode_return", "mean_action_norm"))):
            raise RuntimeError("episode evidence inconsistent, unknown or non-Train")
        counts[cls] += 1
        previous = row["policy_end_step"]
        if cls == "physical_failure":
            physical[raw] += 1
    if counts != expected_counts:
        raise RuntimeError("raw episode count disagrees with metrics")
    return {"terminal_counts": counts, "physical_raw_reason_counts": dict(physical),
            "natural_physical_failure": "OBSERVED" if physical else "NOT OBSERVED"}


class PilotMetricsEnv(FormalHealthMetricsEnv):
    """Additional bounded pilot guards and causal evidence, no plant changes."""
    def __init__(self, env, *, protocol, **kwargs):
        super().__init__(env, **kwargs)
        self.protocol = protocol
        self.causal_rows = 0
        self.lidar_scans = 0
        self.max_lidar_age_s = 0.

    def _audit_observation(self, obs, info):
        plant = self.env.active.unwrapped
        if (set(obs) != {"lidar", "state", "heading", "lidar_age"}
                or not all(np.isfinite(v).all() for v in obs.values())):
            raise RuntimeError("nonfinite or incompatible observation")
        for value in (plant.data.qpos, plant.data.qvel, plant.data.ctrl):
            if not np.isfinite(value).all():
                raise RuntimeError("nonfinite MuJoCo state")
        if info["scenario_split"] != "train" or info["scenario_identity_sha256"] not in self.env.by_identity:
            raise RuntimeError("scene split/identity violation")
        sample, now, age = info["lidar_source_time_s"], float(plant.data.time), info["lidar_age_s"]
        if (sample is None or age is None or not np.isfinite([sample, now, age]).all()
                or sample > now + 1e-9 or age < -1e-9 or abs(now - sample - age) > 1e-8):
            raise RuntimeError("future or inconsistent LiDAR observation")
        self.max_lidar_age_s = max(self.max_lidar_age_s, float(age))
        _append_jsonl(self.health_dir / "causal_observations.jsonl", {
            "policy_step": self.policy_steps, "scenario_identity_sha256": self.env.current_identity,
            "simulation_time": now, "lidar_sample_time": sample, "lidar_age_s": age,
            "lidar_status": info["lidar_status"], "physics_steps": info.get("physics_steps", 0),
            "terminal_class": info.get("terminal_class"),
            "raw_termination_reason": info.get("raw_termination_reason")})
        self.causal_rows += 1

    def reset(self, **kwargs):
        check_resources(self.health_dir)
        result = super().reset(**kwargs)
        self._audit_observation(*result)
        self.lidar_scans += self.env.active.unwrapped.lidar_frame_count
        return result

    def _health_step(self, action):
        if self.policy_steps >= STEPS:
            raise RuntimeError("L9S policy-step budget exhausted")
        check_resources(self.health_dir)
        if self.policy_steps % 128 == 0:
            verify_parent()
            verify_source_hashes((Path(__file__),), self.protocol["runtime_source_sha256"])
        before = self.env.active.unwrapped.lidar_frame_count
        result = super()._health_step(action)
        self.lidar_scans += self.env.active.unwrapped.lidar_frame_count - before
        self._audit_observation(result[0], result[4])
        return result


def _finite_optimizer(model):
    return all(not isinstance(v, torch.Tensor) or bool(torch.isfinite(v).all())
               for state in model.policy.optimizer.state.values() for v in state.values())


@contextmanager
def optimizer_guards(model, output):
    """Observe actual Adam calls without altering losses, gradients or updates."""
    count = [0]
    def before(optimizer, args, kwargs):
        check_resources(output)
        if count[0] >= 128 or not _finite_model_and_gradients(model) or not _finite_optimizer(model):
            raise RuntimeError("optimizer budget or finite-state guard failed")
    def after(optimizer, args, kwargs):
        count[0] += 1
        if not _finite_model_and_gradients(model) or not _finite_optimizer(model):
            raise RuntimeError("nonfinite optimizer result")
    handles = [model.policy.optimizer.register_step_pre_hook(before),
               model.policy.optimizer.register_step_post_hook(after)]
    try:
        yield count
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def preserve_rng():
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


@contextmanager
def guarded_validation(output, recorder):
    """Install a passive single-threaded factory only while frozen evaluator runs.

    The original NavigationEnv methods produce every transition unchanged.
    The scope restores the factory even on errors; no old source is edited.
    """
    from . import ppo_formal_validation as validation
    from .navigation_env import LidarNavigationEnv
    from .terminal_accounting import classify_terminal
    if validation.LidarNavigationEnv is not LidarNavigationEnv:
        raise RuntimeError("unexpected Validation factory binding")
    validation_rows = load_l6_prereg(PREREG)["scenario_manifest"]["validation"]

    class GuardedEnvironment(LidarNavigationEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            matches = [row for row in validation_rows
                       if row["obstacle_configuration_hash"] == self.scene.geometry_hash
                       and row["layout"] == self.scene.layout
                       and row["scene_seed"] == self.scene.scene_seed]
            if len(matches) != 1:
                self.close()
                raise RuntimeError("Validation factory received an unregistered scene")
            self.current_identity = matches[0]["identity_sha256"]
            self._l9s_known_reward = NOT_AVAILABLE

        def _inspect(self, obs, info):
            check_resources(output)
            if (not all(np.isfinite(v).all() for v in obs.values())
                    or not all(np.isfinite(v).all() for v in
                               (self.data.qpos, self.data.qvel, self.data.ctrl))):
                raise RuntimeError("nonfinite Validation observation/physical state")
            sample, age, now = info["lidar_source_time_s"], info["lidar_age_s"], float(self.data.time)
            if (sample is None or age is None or not np.isfinite([sample, age, now]).all()
                    or sample > now + 1e-9 or age < -1e-9 or abs(now-sample-age) > 1e-8):
                raise RuntimeError("invalid Validation LiDAR causality")

        def reset(self, **kwargs):
            check_resources(output)
            result = super().reset(**kwargs)
            self._l9s_completed_steps = 0
            self._l9s_known_reward = NOT_AVAILABLE
            self._inspect(*result)
            return result

        def step(self, action):
            check_resources(output)
            returned = super().step(action)
            try:
                self._inspect(returned[0], returned[4])
            except BaseException as error:
                try:
                    classification = classify_terminal(returned[4], returned[2], returned[3])
                except RuntimeError:
                    classification = None  # retain the actual raw transition even if contradictory
                context = transition_failure_context(self, completed_before=self._l9s_completed_steps,
                    action=action, returned=returned, outcome=classification, phase="VALIDATION",
                    scenario_identity=self.current_identity, known_reward=self._l9s_known_reward)
                recorder.capture(context, error)
                raise
            self._l9s_completed_steps += 1
            self._l9s_known_reward = float(returned[1])
            return returned

    validation.LidarNavigationEnv = GuardedEnvironment
    try:
        yield
    finally:
        validation.LidarNavigationEnv = LidarNavigationEnv


def run(*, upstream, authorized, expected_sha256, seed=0, workers=1,
        timesteps=STEPS, initializer=None, experiment_id=EXPERIMENT_ID):
    protocol = validate_request(authorized=authorized, expected_sha256=expected_sha256,
        seed=seed, workers=workers, timesteps=timesteps, initializer=initializer,
        experiment_id=experiment_id)
    require_output(OUTPUT)
    if OUTPUT.exists():
        raise FileExistsError("unique L9S attempt already exists; no automatic retry")
    resources = check_resources(OUTPUT)
    prereg, _ = verify_parent()
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_SOFTWARE}
    if versions != EXPECTED_SOFTWARE:
        raise RuntimeError("dependency identity mismatch")
    validate_upstream(upstream)
    if hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() != BASE_SHA256:
        raise RuntimeError("original physics model changed")
    original = audit_original_l9()
    attempt_id = "L9S_" + uuid.uuid4().hex
    reserve_attempt(OUTPUT, {"experiment_id": experiment_id, "attempt_id": attempt_id,
                            "execution_protocol_sha256": expected_sha256})
    recorder = FailureEvidenceRecorder(OUTPUT / "failures", run_id=experiment_id,
        attempt_id=attempt_id, origin="NATURAL_L9S_TRAINING_RUNTIME")
    resource_audit = {"samples": 1, "min_mem_available_mib": resources["mem_available_mib"],
                      "minimum_disk_free_bytes": resources["disk_free_bytes"]}
    resource_token = RESOURCE_AUDIT.set(resource_audit)
    scheduler = measured = None
    completed_policy_steps = 0
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        publish_json(OUTPUT / "execution_protocol.json", protocol)
        publish_json(OUTPUT / "historical_identity_before.json", original)
        torch.set_num_threads(1)
        scheduler = MultiScenarioTrainingEnv(prereg, upstream, seed=0)
        measured = PilotMetricsEnv(scheduler, protocol=protocol,
                                  failure_recorder=recorder, diagnostics_dir=OUTPUT)
        model = make_ppo(measured, n_steps=128, seed=0)
        if model.num_timesteps != 0 or model.policy.optimizer.state:
            raise RuntimeError("fresh random initialization required")
        before = {"actor": _weights_sha(model.policy.action_net),
                  "critic": _weights_sha(model.policy.value_net)}
        updates_ms, boundaries = [], []
        with optimizer_guards(model, OUTPUT) as optimizer_calls:
            execute_segments(model, (Milestone(STEPS, STEPS),),
                on_boundary=lambda event: boundaries.append(event.actual_step), update_times_ms=updates_ms)
        if (model.num_timesteps != STEPS or measured.policy_steps != STEPS
                or model._n_updates != 64 or optimizer_calls[0] != 128
                or _optimizer_step_count(model) != 128 or len(updates_ms) != 32
                or boundaries != [STEPS] or not _finite_model_and_gradients(model)):
            raise RuntimeError("incomplete rollout/optimizer/finite-state accounting")
        after = {"actor": _weights_sha(model.policy.action_net),
                 "critic": _weights_sha(model.policy.value_net)}
        if any(before[k] == after[k] for k in before):
            raise RuntimeError("actor/critic not updated")
        health = measured.health_snapshot(model, actual_step=STEPS)
        rows = [json.loads(line) for line in (OUTPUT / "train_episodes.jsonl").read_text().splitlines()]
        if rows != measured.episode_diagnostics:
            raise RuntimeError("persisted episode evidence corrupted")
        accounting = audit_episodes(rows, set(scheduler.by_identity), measured.terminal_counts)
        check_resources(OUTPUT)
        manager = CheckpointManager(OUTPUT, prereg, execution_protocol_sha256=expected_sha256)
        checkpoint = manager.save(model, scheduler, run_id=experiment_id, seed=0)
        state_before_load = scheduler.scheduler_state()
        with preserve_rng():
            restored = manager.load_verified(None, run_id=experiment_id)
        if (_weights_sha(restored.policy) != _weights_sha(model.policy)
                or not _optimizer_states_equal(restored.policy.optimizer.state_dict(), model.policy.optimizer.state_dict())
                or restored.num_timesteps != STEPS or scheduler.scheduler_state() != state_before_load):
            raise RuntimeError("checkpoint reload mismatch")
        # Compare on last real observation already sampled; no new environment step.
        obs = model._last_obs
        np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                      restored.predict(obs, deterministic=True)[0])
        del restored
        train_state = scheduler.scheduler_state()
        train_wall = time.perf_counter() - started
        # Release the sole training scene before sequential Validation scenes.
        scheduler.close()
        closed_train_state = scheduler.scheduler_state()
        with preserve_rng(), guarded_validation(OUTPUT, recorder):
            validation = LidarPPOValidationEvaluator(prereg, upstream, failure_recorder=recorder).evaluate(model)
        if scheduler.scheduler_state() != closed_train_state or validation["probe_only"]:
            raise RuntimeError("Validation contaminated Train state or remained incomplete")
        if audit_original_l9() != original or _historical_model_hashes() != prereg["historical_checkpoint_sha256"]:
            raise RuntimeError("historical evidence changed")
        if protocol_hash(execution_protocol()) != expected_sha256:
            raise RuntimeError("runtime protocol identity changed")
        final_resources = check_resources(OUTPUT)
        result = {"experiment_id": experiment_id, "attempt_id": attempt_id,
            "status": "PASS", "formal_retrain_authorized": False,
            "execution_protocol_sha256": expected_sha256, "actual_timesteps": STEPS,
            "training_physics_ticks": measured.physics_ticks, "rollouts": 32,
            "ppo_n_updates": int(model._n_updates), "optimizer_steps": optimizer_calls[0],
            "finite_gradients_and_optimizer": True, "parameter_hash_before": before,
            "parameter_hash_after": after, "health": health, "accounting": accounting,
            "train_scheduler": train_state, "checkpoint": checkpoint,
            "checkpoint_reload_equal": True, "validation": validation,
            "train_rng_unchanged_during_validation": True, "global_rng_preserved": True,
            "test_episode_count": 0, "lidar_causal_rows": measured.causal_rows,
            "lidar_scans_including_reset": measured.lidar_scans,
            "max_lidar_age_s": measured.max_lidar_age_s,
            "environment_step_ms": _stats(measured.step_ms),
            "lidar_scan_ms": _stats(measured.scan_ms), "lidar_encode_ms": _stats(measured.encode_ms),
            "reset_compile_ms": _stats(measured.reset_ms), "ppo_update_ms": _stats(updates_ms),
            "episode_reset_rss_mib": measured.reset_rss_mib,
            "training_wall_s": train_wall, "total_wall_s": time.perf_counter()-started,
            "total_cpu_s": time.process_time()-cpu_started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
            "min_mem_available_mib": resource_audit["min_mem_available_mib"],
            "minimum_disk_free_bytes": resource_audit["minimum_disk_free_bytes"],
            "resource_guard_samples": resource_audit["samples"],
            "OOM": False, "historical_identity_unchanged": True,
            "runtime_source_sha256": protocol["runtime_source_sha256"]}
        # Required cleanup precedes authoritative completion publication.
        measured.close()
        completed_policy_steps = measured.policy_steps
        measured = None
        scheduler = None
        publish_json(OUTPUT / "pilot_summary.json", result)
        evidence = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in OUTPUT.iterdir() if p.is_file()}
        publish_json(OUTPUT / "pilot.complete.json", {"actual_timesteps": STEPS,
            "execution_protocol_sha256": expected_sha256, "evidence_sha256": evidence})
        return result
    except BaseException as error:
        context = simulator_context(scheduler)
        context.update(context_kind="LAST_KNOWN_CONTEXT",
                       policy_step_attempt=NOT_AVAILABLE,
                       last_completed_policy_step=measured.policy_steps if measured else completed_policy_steps,
                       last_known_reward=measured._last_known_reward if measured else NOT_AVAILABLE)
        recorder.capture(context, error)
        raise
    finally:
        try:
            if measured is not None:
                measured.close()
            elif scheduler is not None:
                scheduler.close()
        finally:
            RESOURCE_AUDIT.reset(resource_token)


def main(argv=None):
    parser = argparse.ArgumentParser(description="One independent L9S 4096-step Pilot; never formal training")
    parser.add_argument("--l9s-pilot", action="store_true", required=True)
    parser.add_argument("--authorize-l9s-pilot", action="store_true", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(upstream=args.upstream, authorized=args.authorize_l9s_pilot,
                 expected_sha256=args.expected_protocol_sha256)
    print(json.dumps({"status": result["status"], "actual_timesteps": result["actual_timesteps"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
