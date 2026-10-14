"""Identity-checked PPO checkpoint contract; explicitly *not* exact resume."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from .ppo_formal_scheduler import (L6_FORMAL_PROPOSAL_SHA256, L6_MANIFEST_SHA256,
                                   MultiScenarioTrainingEnv, require_memory,
                                   verify_l6_runtime_sources)
from .ppo_training_runner import _atomic_json, _historical_model_hashes, protocol_hash


def formal_readiness_protocol(prereg: dict) -> dict:
    """Explicit L7 revision of L6's NOT IMPLEMENTED scheduler field only."""
    if (protocol_hash(prereg["scenario_manifest"]) != L6_MANIFEST_SHA256
            or protocol_hash(prereg["formal_proposal_not_run"]) != L6_FORMAL_PROPOSAL_SHA256):
        raise ValueError("L6 frozen protocol changed")
    config = copy.deepcopy(prereg["formal_proposal_not_run"])
    config["train_scene_scheduler"] = (
        "uniform registered Train identity draw at each episode reset; "
        "single active MuJoCo environment; L7 IMPLEMENTED")
    config["continuation_policy"] = "interrupted formal runs RESTART_REQUIRED; warm start is new experiment"
    return config


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CheckpointManager:
    """Save model+sidecar atomically; no claim of bitwise rollout continuity."""

    def __init__(self, directory: Path, prereg: dict):
        verify_l6_runtime_sources(prereg)
        self.directory = Path(directory)
        self.prereg = prereg
        self.protocol = formal_readiness_protocol(prereg)
        self.protocol_sha256 = protocol_hash(self.protocol)
        self.manifest_sha256 = L6_MANIFEST_SHA256

    @staticmethod
    def _paths(directory: Path, run_id: str) -> tuple[Path, Path]:
        if not re.fullmatch(r"[A-Z0-9_]{3,80}", run_id):
            raise ValueError("checkpoint run_id must be a safe explicit identifier")
        return directory / f"{run_id}.zip", directory / f"{run_id}.json"

    def save(self, model: PPO, scheduler: MultiScenarioTrainingEnv, *,
             run_id: str, seed: int) -> dict:
        require_memory()
        archive, sidecar = self._paths(self.directory, run_id)
        if archive.exists() or sidecar.exists():
            raise FileExistsError("checkpoint evidence already exists")
        if scheduler.active_environment_count != 1:
            raise ValueError("scheduler must be active when saving checkpoint")
        historical = _historical_model_hashes()
        if historical != self.prereg["historical_checkpoint_sha256"]:
            raise RuntimeError("historical BC/PPO identity changed")
        self.directory.mkdir(parents=True, exist_ok=True)
        partial = self.directory / f".{run_id}.partial.zip"
        if partial.exists():
            raise FileExistsError("incomplete prior checkpoint attempt requires audit")
        try:
            model.save(partial)
            with zipfile.ZipFile(partial) as zipped:
                if "policy.optimizer.pth" not in zipped.namelist():
                    raise RuntimeError("SB3 archive lacks optimizer state")
            os.replace(partial, archive)
            record = {
                "run_id": run_id, "checkpoint_sha256": _sha(archive),
                "training_timesteps": int(model.num_timesteps), "training_seed": int(seed),
                "scheduler_state": scheduler.scheduler_state(),
                "l6_manifest_sha256": self.manifest_sha256,
                "l6_formal_proposal_sha256": L6_FORMAL_PROPOSAL_SHA256,
                "l7_formal_protocol_sha256": self.protocol_sha256,
                "reward_version": self.protocol["reward_version"],
                "network_implementation_sha256": self.prereg["network_implementation_sha256"],
                "historical_checkpoint_sha256": historical,
                "optimizer_state_in_sb3_zip": True,
                "rng_fingerprints": {
                    "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    "numpy_global": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
                    "python_random": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                    "scheduler_full_state_saved": True,
                    "note": "fingerprints are audit evidence, not restorable RNG snapshots",
                },
                "missing_exact_resume_state": ["active MuJoCo MjData", "PI state",
                                               "LiDAR buffer", "partial PPO rollout"],
                "continuation_class": "WARM_START",
                "interrupted_formal_run_policy": "RESTART_REQUIRED",
            }
            record["sidecar_payload_sha256"] = protocol_hash(record)
            _atomic_json(sidecar, record)
            return record
        finally:
            if partial.exists():
                partial.unlink()

    def load_verified(self, env, *, run_id: str, mode: str = "WARM_START") -> PPO:
        verify_l6_runtime_sources(self.prereg)
        if mode != "WARM_START":
            raise ValueError("only WARM_START is implemented; exact/boundary resume is unsupported")
        archive, sidecar = self._paths(self.directory, run_id)
        if not archive.exists() or not sidecar.exists():
            raise FileNotFoundError("complete ZIP and sidecar required")
        record = json.loads(sidecar.read_text())
        digest = record.pop("sidecar_payload_sha256", None)
        if digest != protocol_hash(record):
            raise ValueError("checkpoint sidecar integrity mismatch")
        if (_sha(archive) != record["checkpoint_sha256"]
                or record["l6_manifest_sha256"] != self.manifest_sha256
                or record["l7_formal_protocol_sha256"] != self.protocol_sha256
                or record["reward_version"] != self.protocol["reward_version"]
                or record["network_implementation_sha256"] !=
                self.prereg["network_implementation_sha256"]
                or record["historical_checkpoint_sha256"] != _historical_model_hashes()):
            raise ValueError("checkpoint, protocol, network or historical model identity mismatch")
        with zipfile.ZipFile(archive) as zipped:
            if "policy.optimizer.pth" not in zipped.namelist():
                raise ValueError("optimizer state missing from SB3 checkpoint")
        require_memory()
        model = PPO.load(archive, env=env, device="cpu")
        if model.num_timesteps != record["training_timesteps"]:
            raise ValueError("reloaded timestep count differs from sidecar")
        return model
