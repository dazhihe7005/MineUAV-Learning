"""Single-resident frozen-Train scene scheduler for future PPO runs.

Construction and resets never call an optimizer. A forced reset identity is
reserved for finite coverage diagnostics; ordinary resets draw uniformly.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from .ppo_preflight import HeadingObservation
from .ppo_scenarios import build_ppo_scene
from .navigation_env import LidarNavigationEnv
from .ppo_training_runner import protocol_hash
from .runner import _available_mib


L6_MANIFEST_SHA256 = "f6ee34a153e6cb4d95fa43e7214d6d67c9b534835362be275363e14ec87b75d5"
L6_FORMAL_PROPOSAL_SHA256 = "f12cee995d003cbd502b3bd5477d142d658f1362dae368e5286ee1ddafc49bf4"
L6_PREREG_FILE_SHA256 = "c468490154627ba8f71df9c73bc8c80769f557054d9692313139188971e332c8"
MIN_AVAILABLE_MIB = 1536.


def require_memory() -> float:
    available = float(_available_mib())
    if available < MIN_AVAILABLE_MIB:
        raise MemoryError("MemAvailable below frozen 1.5 GiB guard")
    return available


def load_l6_prereg(path: Path) -> dict:
    """Reject manifest/proposal edits even when inner SHA fields are recomputed."""
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != L6_PREREG_FILE_SHA256:
        raise ValueError("L6 committed preregistration file identity mismatch")
    prereg = json.loads(payload)
    manifest = prereg["scenario_manifest"]
    if (protocol_hash(manifest) != L6_MANIFEST_SHA256
            or prereg["scenario_manifest_sha256"] != L6_MANIFEST_SHA256
            or protocol_hash(prereg["formal_proposal_not_run"]) != L6_FORMAL_PROPOSAL_SHA256
            or prereg["formal_proposal_sha256"] != L6_FORMAL_PROPOSAL_SHA256
            or {key: len(rows) for key, rows in manifest.items()} !=
            {"train": 12, "validation": 6, "test": 6}):
        raise ValueError("L6 frozen manifest/proposal identity mismatch")
    identities = set()
    for split, rows in manifest.items():
        for row in rows:
            identity = row["identity_sha256"]
            value = {key: item for key, item in row.items() if key != "identity_sha256"}
            expected = hashlib.sha256(json.dumps(value, sort_keys=True,
                                                  separators=(",", ":")).encode()).hexdigest()
            if row["split"] != split or identity != expected or identity in identities:
                raise ValueError("L6 scene row identity/split mismatch")
            identities.add(identity)
    verify_l6_runtime_sources(prereg)
    return prereg


def verify_l6_runtime_sources(prereg: dict, *, source_root: Path | None = None) -> None:
    """Compare actual plant, Reward, controller, LiDAR and CNN source bytes."""
    root = Path(__file__).parents[1] if source_root is None else Path(source_root)
    sources = prereg["preflight_source_sha256"]
    if len(sources) != 12:
        raise ValueError("L6 runtime source list is incomplete")
    for relative, expected in sources.items():
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("invalid runtime source path")
        path = root / candidate
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"L6 runtime source identity mismatch: {relative}")


def sample_train_sequence(prereg: dict, *, seed: int, draws: int) -> list[str]:
    """No environments constructed; audit the same uniform integer draw as reset."""
    if (protocol_hash(prereg["scenario_manifest"]) != L6_MANIFEST_SHA256
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or isinstance(draws, bool) or not isinstance(draws, int) or draws < 1):
        raise ValueError("registered scenes, nonnegative seed and positive draw count required")
    rows = prereg["scenario_manifest"]["train"]
    rng = np.random.default_rng(seed)
    return [rows[int(rng.integers(len(rows)))]["identity_sha256"] for _ in range(draws)]


class MultiScenarioTrainingEnv(gym.Env):
    """At most one compiled scene alive; reset-only uniform Train sampler."""

    metadata = {"render_modes": []}

    def __init__(self, prereg: dict, upstream_root: Path, *, seed: int):
        require_memory()
        verify_l6_runtime_sources(prereg)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("nonnegative independent scheduler seed required")
        if protocol_hash(prereg["scenario_manifest"]) != L6_MANIFEST_SHA256:
            raise ValueError("unregistered scene manifest")
        self.rows = tuple(copy.deepcopy(prereg["scenario_manifest"]["train"]))
        if len(self.rows) != 12 or any(row["split"] != "train" for row in self.rows):
            raise ValueError("exactly 12 frozen Train scenes required")
        self.by_identity = {row["identity_sha256"]: row for row in self.rows}
        if len(self.by_identity) != 12:
            raise ValueError("duplicate Train scene identity")
        self.upstream_root = Path(upstream_root)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.visit_counts = {identity: 0 for identity in self.by_identity}
        self.selected_identities: list[str] = []
        self.active: HeadingObservation | None = None
        self.current_identity: str | None = None
        # The first model supplies the actual inherited Gymnasium spaces. It
        # remains the only active scene until a different identity is drawn.
        self._open(self.rows[0])
        self.observation_space = self.active.observation_space
        self.action_space = self.active.action_space

    @property
    def active_environment_count(self) -> int:
        return int(self.active is not None)

    def _open(self, row: dict) -> None:
        require_memory()
        if self.active is not None:
            self.active.close()
            self.active = None
            self.current_identity = None
            gc.collect()
        scene = build_ppo_scene(row)
        if scene.geometry_hash != row["obstacle_configuration_hash"]:
            raise ValueError("compiled obstacle geometry differs from L6 manifest")
        self.active = HeadingObservation(LidarNavigationEnv(
            row["layout"], row["scene_seed"], self.upstream_root, scene=scene))
        self.current_identity = row["identity_sha256"]
        require_memory()

    def reset(self, *, seed=None, options=None):
        require_memory()
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("invalid scheduler reset seed")
            self._rng = np.random.default_rng(seed)
        if options is None:
            row = self.rows[int(self._rng.integers(len(self.rows)))]
        else:
            if not isinstance(options, dict) or set(options) != {"scenario_identity_sha256"}:
                raise ValueError("only registered coverage identity may be specified")
            identity = options["scenario_identity_sha256"]
            if identity not in self.by_identity:
                raise ValueError("coverage reset must use a Train identity")
            row = self.by_identity[identity]
        if self.active is None or self.current_identity != row["identity_sha256"]:
            self._open(row)
        obs, info = self.active.reset(seed=row["scene_seed"], options={
            "start_seed": row["start_seed"], "target_seed": row["target_seed"],
            "start_position": row["start_xyz"], "target_position": row["target_xyz"],
            "start_yaw_rad": row["start_yaw_rad"]})
        if info["obstacle_configuration_hash"] != row["obstacle_configuration_hash"]:
            raise RuntimeError("reset returned wrong obstacle geometry")
        self.visit_counts[row["identity_sha256"]] += 1
        self.selected_identities.append(row["identity_sha256"])
        return obs, {**info, "scenario_identity_sha256": row["identity_sha256"],
                     "scenario_split": "train"}

    def step(self, action):
        require_memory()
        if self.active is None:
            raise RuntimeError("scheduler is closed")
        obs, reward, terminated, truncated, info = self.active.step(action)
        return obs, reward, terminated, truncated, {
            **info, "scenario_identity_sha256": self.current_identity,
            "scenario_split": "train"}

    def scheduler_state(self) -> dict:
        return {"seed": self.seed, "rng_state": copy.deepcopy(self._rng.bit_generator.state),
                "visit_counts": dict(self.visit_counts),
                "selected_identities": list(self.selected_identities),
                "current_identity": self.current_identity}

    def close(self):
        if self.active is not None:
            self.active.close()
            self.active = None
            self.current_identity = None
            gc.collect()
