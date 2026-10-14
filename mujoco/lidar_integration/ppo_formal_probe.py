"""Bounded L7 readiness evidence: zero optimizer updates and zero Test episodes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from .ppo_formal_checkpoint import CheckpointManager, formal_readiness_protocol
from .ppo_formal_scheduler import (MultiScenarioTrainingEnv, load_l6_prereg,
                                   require_memory, sample_train_sequence)
from .ppo_formal_validation import LidarPPOValidationEvaluator
from .ppo_preflight import make_ppo
from .ppo_training_runner import _atomic_json, _historical_model_hashes, _stats, protocol_hash


def _rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.
    raise RuntimeError("process RSS unavailable")


def run_readiness_probe(prereg_path: Path, upstream_root: Path, output_dir: Path,
                        *, validation_step_cap: int = 375, train_cycles: int = 2) -> dict:
    """One-shot finite scene/Validation probe; cannot invoke PPO.learn()."""
    if (isinstance(train_cycles, bool) or not isinstance(train_cycles, int)
            or not 1 <= train_cycles <= 3 or isinstance(validation_step_cap, bool)
            or not isinstance(validation_step_cap, int) or not 1 <= validation_step_cap <= 375):
        raise ValueError("bounded probe requires 1..3 Train cycles and 1..375 Validation steps")
    prereg = load_l6_prereg(prereg_path)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "probe_summary.json"
    if report_path.exists():
        raise FileExistsError("readiness probe evidence already exists")
    marker = directory / "probe.attempt"
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    require_memory()
    historical_before = _historical_model_hashes()
    if historical_before != prereg["historical_checkpoint_sha256"]:
        raise RuntimeError("historical checkpoint identity mismatch before probe")
    started = time.perf_counter()
    min_available = require_memory()
    rss_samples = [_rss_mib()]
    training = MultiScenarioTrainingEnv(prereg, upstream_root, seed=0)
    rows = []
    try:
        for cycle in range(train_cycles):
            for row in prereg["scenario_manifest"]["train"]:
                min_available = min(min_available, require_memory())
                begun = time.perf_counter()
                obs, info = training.reset(options={
                    "scenario_identity_sha256": row["identity_sha256"]})
                reset_ms = (time.perf_counter() - begun) * 1000.
                plant = training.active.unwrapped
                if (info["obstacle_configuration_hash"] != row["obstacle_configuration_hash"]
                        or not np.allclose(plant.data.qpos[:3], row["start_xyz"])
                        or not np.allclose(plant.data.qvel, 0.)
                        or not np.allclose(plant.velocity_controller.integral_error, 0.)
                        or plant.lidar_buffer.frame_count != 1 or plant.data.time != 0):
                    raise RuntimeError("Train scene initialization retained stale state")
                scan_ms, encode_ms = plant.last_scan_ms, plant.last_encode_ms
                begun = time.perf_counter()
                next_obs, _, _, _, step_info = training.step(np.zeros(4, dtype=np.float32))
                step_ms = (time.perf_counter() - begun) * 1000.
                if (step_info["physics_steps"] != 20 or next_obs["lidar"].shape != (3, 12, 24)
                        or obs["heading"].shape != (2,)):
                    raise RuntimeError("Train scene step/timing contract changed")
                rss_samples.append(_rss_mib())
                rows.append({"cycle": cycle, "scenario_identity_sha256": row["identity_sha256"],
                             "scenario_split": "train", "layout": row["layout"],
                             "geometry_sha256": info["obstacle_configuration_hash"],
                             "reset_compile_ms": reset_ms, "one_step_ms": step_ms,
                             "initial_scan_ms": scan_ms, "initial_encode_ms": encode_ms,
                             "rss_mib": rss_samples[-1]})
        torch.set_num_threads(1)
        training.reset(options={"scenario_identity_sha256":
                                prereg["scenario_manifest"]["train"][0]["identity_sha256"]})
        model = make_ppo(training, seed=0)
        if model.num_timesteps != 0 or model.policy.optimizer.state:
            raise RuntimeError("probe unexpectedly performed optimizer updates")
        manager = CheckpointManager(directory, prereg)
        checkpoint = manager.save(model, training, run_id="L7_UNTRAINED_CONTRACT", seed=0)
        restored = manager.load_verified(training, run_id="L7_UNTRAINED_CONTRACT")
        sample = training.active.unwrapped._get_obs()
        if restored.num_timesteps != 0 or model.num_timesteps != 0:
            raise RuntimeError("checkpoint contract introduced training steps")
    finally:
        training.close()
    min_available = min(min_available, require_memory())
    evaluator = LidarPPOValidationEvaluator(prereg, upstream_root)
    validation = evaluator.evaluate(model, max_policy_steps=validation_step_cap)
    rss_samples.append(_rss_mib())
    if model.num_timesteps != 0 or model.policy.optimizer.state:
        raise RuntimeError("Validation performed PPO optimizer updates")
    if _historical_model_hashes() != historical_before:
        raise RuntimeError("historical BC/PPO model identity changed")
    samples = sample_train_sequence(prereg, seed=29, draws=1200)
    distribution = {row["identity_sha256"]: samples.count(row["identity_sha256"])
                    for row in prereg["scenario_manifest"]["train"]}
    expected = 1200 / 12
    chi_square = sum((count - expected) ** 2 / expected for count in distribution.values())
    reset_values = [row["reset_compile_ms"] for row in rows]
    steps = [row["one_step_ms"] for row in rows]
    report = {
        "task": "Phase 3E-L7 PPO formal readiness, bounded nontraining probe",
        "train_coverage": {"registered_scene_count": 12,
                           "actual_scene_count": len(set(row["scenario_identity_sha256"] for row in rows)),
                           "cycles": train_cycles, "rows": rows,
                           "visit_counts": training.visit_counts,
                           "uniform_sampler_1200_draw_counts": distribution,
                           "uniform_sampler_descriptive_chi_square_11df": chi_square},
        "validation": validation,
        "checkpoint_contract": checkpoint,
        "l6_manifest_sha256": prereg["scenario_manifest_sha256"],
        "l6_formal_proposal_sha256": prereg["formal_proposal_sha256"],
        "l7_formal_protocol": formal_readiness_protocol(prereg),
        "l7_formal_protocol_sha256": protocol_hash(formal_readiness_protocol(prereg)),
        "l6_prereg_file_sha256": hashlib.sha256(Path(prereg_path).read_bytes()).hexdigest(),
        "software_versions": {name: importlib.metadata.version(name) for name in
                              ("mujoco", "stable-baselines3", "torch", "gymnasium", "numpy")},
        "reset_compile_ms": _stats(reset_values), "train_one_step_ms": _stats(steps),
        "initial_scan_ms": _stats([row["initial_scan_ms"] for row in rows]),
        "initial_encode_ms": _stats([row["initial_encode_ms"] for row in rows]),
        "current_rss_mib_after_switches": rss_samples,
        "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
        "system_min_available_mib": min(min_available, require_memory()),
        "wall_seconds": time.perf_counter() - started,
        "ppo_optimizer_updates": 0, "test_episode_count": 0,
        "formal_training_executed": False,
        "historical_checkpoints_unchanged": True,
        "out_of_memory": False,
    }
    _atomic_json(report_path, report)
    return report
