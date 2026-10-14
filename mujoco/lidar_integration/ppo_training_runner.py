"""One-shot, single-worker PPO training smoke; never a trained navigation model.

Call only after offline tests and independent safety review. This module has
no side effects on import and never reads evaluation episode outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .navigation_env import LidarNavigationEnv
from .ppo_preflight import HeadingObservation, make_ppo
from .ppo_scenarios import build_ppo_scene, make_scenario_manifest
from .runner import _available_mib
from .scanner import PATTERN_SHA256, UPSTREAM_COMMIT
from .scene import BASE_MODEL, BASE_SHA256


SMOKE_PROTOCOL = {
    "experiment_id": "LIDAR_PPO_SMOKE_ONLY_V1", "seed": 0, "workers": 1,
    "total_timesteps": 512, "n_steps": 128, "batch_size": 64, "n_epochs": 2,
    "learning_rate": 3e-4, "gamma": .99, "gae_lambda": .95,
    "clip_range": .2, "ent_coef": 0., "vf_coef": .5, "max_grad_norm": .5,
    "policy": "MultiInputPolicy", "normalize_images": False,
    "network": "Conv(3,16,3,p1)-Conv(16,32,3,s2,p1)-FC64+state7+heading2+age1; pi/vf64x64",
    "ray_count": 24000, "physics_hz": 500, "pi_hz": 100,
    "policy_hz": 25, "lidar_hz": 10,
    "reward_version": "NAV_V2_COLLISION_PREFLIGHT_V1",
    "checkpoint_rule": "final checkpoint only; no validation/test selection",
}

FORMAL_PROPOSAL = {
    **{key: value for key, value in SMOKE_PROTOCOL.items()
       if key not in ("experiment_id", "total_timesteps", "checkpoint_rule")},
    "experiment_id": "LIDAR_PPO_FORMAL_PROPOSAL_NOT_RUN_V1",
    "seeds": [0, 1, 2], "total_timesteps_per_seed": 1_000_000,
    "training_scenario_count": 12, "validation_scenario_count": 6,
    "sealed_test_scenario_count": 6, "checkpoint_every_steps": 100_000,
    "validation_every_steps": 100_000, "selection_rule": "final checkpoint, not test-selected",
    "train_scene_scheduler": "uniform among 12 Train identities at episode reset; NOT IMPLEMENTED",
}


def protocol_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_smoke_budget(config: dict) -> None:
    if (config != SMOKE_PROTOCOL or config["workers"] != 1
            or config["total_timesteps"] > 512
            or config["total_timesteps"] % config["n_steps"]):
        raise ValueError("smoke configuration must match the frozen 512-step protocol")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".partial",
                                     delete=False) as file:
        temporary = Path(file.name)
        json.dump(payload, file, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _weights_sha(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _historical_model_hashes() -> dict[str, str]:
    root = Path(__file__).parents[1] / "rl" / "models"
    names = ("uav_bc_mlp_seed0.pt", "uav_bc_yaw_augmented_seed0.pt",
             *(f"uav_ppo_fixed_protocol_seed{seed}.zip" for seed in range(5)))
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names}


class _ScenarioReset(gym.Wrapper):
    def __init__(self, env: gym.Env, row: dict):
        super().__init__(env)
        self.row = row

    def reset(self, *, seed=None, options=None):
        if options is not None:
            raise ValueError("frozen training scenario does not accept reset overrides")
        return self.env.reset(seed=seed, options={
            "start_seed": self.row["start_seed"],
            "target_seed": self.row["target_seed"],
            "start_position": self.row["start_xyz"],
            "target_position": self.row["target_xyz"],
            "start_yaw_rad": self.row["start_yaw_rad"],
        })


class _StepTiming(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.step_ms = []
        self.scan_ms = []
        self.encode_ms = []
        self.rss_mib = []
        self.available_mib = []
        self.episode_outcomes = []
        self.physics_ticks_total = 0

    def step(self, action):
        before = self.unwrapped.lidar_frame_count
        begun = time.perf_counter()
        result = self.env.step(action)
        self.step_ms.append((time.perf_counter() - begun) * 1000.)
        if self.unwrapped.lidar_frame_count > before:
            self.scan_ms.append(self.unwrapped.last_scan_ms)
            self.encode_ms.append(self.unwrapped.last_encode_ms)
        self.physics_ticks_total += int(result[4]["physics_steps"])
        self.rss_mib.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
        available = _available_mib()
        self.available_mib.append(available)
        if available < 1536:
            raise MemoryError("MemAvailable fell below 1.5 GiB during smoke")
        if result[2] or result[3]:
            self.episode_outcomes.append(result[4]["termination_reason"])
        return result


class _BudgetCallback(BaseCallback):
    def _on_step(self) -> bool:
        if _available_mib() < 1536:
            raise MemoryError("MemAvailable fell below 1.5 GiB between PPO updates")
        return self.num_timesteps <= SMOKE_PROTOCOL["total_timesteps"]


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=float)
    return {"count": len(a), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


def validated_training_diagnostics(values: dict) -> dict[str, float]:
    result = {}
    for key, value in values.items():
        if key.startswith("train/") and isinstance(value, (int, float, np.number)):
            if not np.isfinite(value):
                raise ValueError(f"nonfinite training diagnostic: {key}")
            result[key] = float(value)
    return result


def run_smoke(report_path: Path, prereg_path: Path, checkpoint_path: Path,
              upstream_root: Path) -> dict:
    """Exactly one attempt; an existing attempt marker forbids automatic retry."""
    validate_smoke_budget(SMOKE_PROTOCOL)
    for path in (report_path, prereg_path, checkpoint_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
    if _available_mib() < 1536:
        raise MemoryError("less than 1.5 GiB MemAvailable before smoke")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = checkpoint_path.with_suffix(".attempt")
    fd = os.open(attempt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    manifest = make_scenario_manifest()
    source_paths = [Path(__file__).parent / name for name in
                    ("ppo_preflight.py", "ppo_scenarios.py", "ppo_training_runner.py",
                     "navigation_env.py", "navigation_scene.py", "scanner.py",
                     "observation.py", "async_observation.py")]
    source_paths.append(Path(__file__).parents[1] / "rl" / "mine_uav_env.py")
    source_paths += [Path(__file__).parents[1] / "control" / name for name in
                     ("velocity_command_controller.py", "hover_controller.py",
                      "control_allocator.py")]
    sources = {str(path.relative_to(Path(__file__).parents[1])):
               hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    if hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() != BASE_SHA256:
        raise RuntimeError("historical MuJoCo plant identity changed")
    versions = {name: importlib.metadata.version(name)
                for name in ("stable-baselines3", "torch", "mujoco", "gymnasium", "numpy")}
    prereg = {
        "smoke_protocol": SMOKE_PROTOCOL, "smoke_protocol_sha256": protocol_hash(SMOKE_PROTOCOL),
        "formal_proposal_not_run": FORMAL_PROPOSAL,
        "formal_proposal_sha256": protocol_hash(FORMAL_PROPOSAL),
        "scenario_manifest": manifest, "scenario_manifest_sha256": protocol_hash(manifest),
        "software_versions": versions, "preflight_source_sha256": sources,
        "historical_model_sha256": BASE_SHA256,
        "lidar_pattern_sha256": PATTERN_SHA256, "upstream_lidar_commit": UPSTREAM_COMMIT,
        "reward_implementation_sha256": sources["lidar_integration/navigation_env.py"],
        "network_implementation_sha256": sources["lidar_integration/ppo_preflight.py"],
        "smoke_train_identity_sha256": manifest["train"][1]["identity_sha256"],
        "test_episode_results_used": False,
        "historical_checkpoint_sha256": _historical_model_hashes(),
    }
    prereg["complete_protocol_sha256"] = protocol_hash(prereg)
    _atomic_json(prereg_path, prereg)
    row = manifest["train"][1]  # front-box Train scene; no Test data or tuning
    torch.set_num_threads(1)
    env = _StepTiming(HeadingObservation(_ScenarioReset(
        LidarNavigationEnv(row["layout"], row["scene_seed"], upstream_root,
                           scene=build_ppo_scene(row)), row)))
    try:
        model = make_ppo(env, n_steps=SMOKE_PROTOCOL["n_steps"], seed=0)
        if _available_mib() < 1536:
            raise MemoryError("less than 1.5 GiB MemAvailable after environment/model init")
        actor_initial = _weights_sha(model.policy.action_net)
        critic_initial = _weights_sha(model.policy.value_net)
        parameter_count = sum(p.numel() for p in model.policy.parameters())
        started = time.perf_counter()
        cpu_started = time.process_time()
        model.learn(total_timesteps=512, callback=_BudgetCallback(),
                    reset_num_timesteps=True, progress_bar=False)
        wall_s = time.perf_counter() - started
        cpu_s = time.process_time() - cpu_started
        if model.num_timesteps != 512:
            raise RuntimeError("PPO collected an unexpected number of environment steps")
        actor_final = _weights_sha(model.policy.action_net)
        critic_final = _weights_sha(model.policy.value_net)
        if actor_final == actor_initial or critic_final == critic_initial:
            raise RuntimeError("actor or critic head did not update")
        parameter_gradient_norms = []
        for parameter in model.policy.parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError("trained policy contains nonfinite parameters")
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("trained policy contains nonfinite gradients")
                parameter_gradient_norms.append(float(parameter.grad.norm().item()))
        if not parameter_gradient_norms or max(parameter_gradient_norms) <= 0:
            raise RuntimeError("no finite nonzero training gradient was observed")
        training_diagnostics = validated_training_diagnostics(model.logger.name_to_value)
        temporary = checkpoint_path.with_name(checkpoint_path.stem + ".partial.zip")
        model.save(temporary)
        os.replace(temporary, checkpoint_path)
        restored = PPO.load(checkpoint_path, env=env, device="cpu")
        if _historical_model_hashes() != prereg["historical_checkpoint_sha256"]:
            raise RuntimeError("historical BC/PPO checkpoint identity changed")
        obs, _ = env.reset(seed=0)
        action, _ = model.predict(obs, deterministic=True)
        loaded_action, _ = restored.predict(obs, deterministic=True)
        if not np.array_equal(action, loaded_action) or not np.isfinite(action).all():
            raise RuntimeError("checkpoint roundtrip changed inference")
        report = {
            "status": "PASS", "experiment_id": SMOKE_PROTOCOL["experiment_id"],
            "timesteps": model.num_timesteps, "worker_count": 1,
            "formal_training_executed": False, "navigation_learning_claim": False,
            "scenario_identity_sha256": row["identity_sha256"],
            "prereg_sha256": hashlib.sha256(prereg_path.read_bytes()).hexdigest(),
            "protocol_sha256": prereg["smoke_protocol_sha256"],
            "scenario_manifest_sha256": prereg["scenario_manifest_sha256"],
            "historical_checkpoints_unchanged": True,
            "parameter_count": parameter_count,
            "actor_initial_sha256": actor_initial, "actor_final_sha256": actor_final,
            "critic_initial_sha256": critic_initial, "critic_final_sha256": critic_final,
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "checkpoint_path": str(checkpoint_path), "reload_exact_action": True,
            "wall_seconds": wall_s, "cpu_seconds": cpu_s,
            "env_steps_per_wall_second": 512 / wall_s,
            "physics_ticks_total": env.physics_ticks_total,
            "sim_seconds_per_wall_second": env.physics_ticks_total * .002 / wall_s,
            "non_env_step_wall_seconds": wall_s - sum(env.step_ms) / 1000.,
            "last_gradient_norm_max": max(parameter_gradient_norms),
            "sb3_last_available_training_diagnostics": training_diagnostics,
            "env_step_ms": _stats(env.step_ms), "lidar_scan_ms": _stats(env.scan_ms),
            "lidar_encode_ms": _stats(env.encode_ms),
            "episode_outcomes": env.episode_outcomes,
            "rss_peak_mib": max(max(env.rss_mib, default=0),
                                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.),
            "system_min_available_mib": min(env.available_mib, default=_available_mib()),
            "out_of_memory": False,
        }
        _atomic_json(report_path, report)
        return report
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-shot 512-step LiDAR PPO smoke")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.report, args.prereg, args.checkpoint, args.upstream),
                     indent=2, sort_keys=True))
