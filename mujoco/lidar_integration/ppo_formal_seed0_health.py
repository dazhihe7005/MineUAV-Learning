"""Passive L9 health telemetry around the unchanged L8 formal trainer."""

from __future__ import annotations

import json
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np

from . import ppo_training_orchestration as orchestration
from .ppo_training_orchestration import (FormalPPOTrainingDriver, TrainingMetricsEnv,
                                         _current_rss_mib, _finite_model_and_gradients,
                                         source_hashes, verify_source_hashes)
from .ppo_training_runner import _atomic_json


FORMAL_HEALTH_DIR = Path(__file__).parents[1] / "reports/mujoco_lidar_ppo_formal_seed0_parts/run_01"
MIN_RUNTIME_DISK_FREE_BYTES = 2 * 1024**3
L9_RUNTIME_PATHS = (Path(__file__).with_name("ppo_formal_seed0_entry.py"), Path(__file__))


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class FormalHealthMetricsEnv(TrainingMetricsEnv):
    """Observe terminal motion and bounded resources; return parent transitions verbatim."""

    active_instance: FormalHealthMetricsEnv | None = None

    def __init__(self, env):
        super().__init__(env)
        self.health_dir = FORMAL_HEALTH_DIR
        self.episode_diagnostics: list[dict] = []
        self.resource_samples: list[dict] = []
        self.min_disk_free_bytes = float("inf")
        self._episode_start = np.zeros(3)
        self._previous_position = np.zeros(3)
        self._path_length = 0.
        self._action_norm_sum = 0.
        self._episode_steps = 0
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        self.l9_runtime_sources = source_hashes(L9_RUNTIME_PATHS)
        self._check_disk()
        FormalHealthMetricsEnv.active_instance = self

    def _check_disk(self) -> int:
        free = int(shutil.disk_usage(self.health_dir.parent).free)
        self.min_disk_free_bytes = min(self.min_disk_free_bytes, free)
        if free < MIN_RUNTIME_DISK_FREE_BYTES:
            raise OSError("formal training disk free space below 2 GiB guard")
        return free

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        position = np.asarray(self.env.active.unwrapped.data.qpos[:3], dtype=float).copy()
        self._episode_start = position
        self._previous_position = position.copy()
        self._path_length = 0.
        self._action_norm_sum = 0.
        self._episode_steps = 0
        return result

    def step(self, action):
        if self.policy_steps % 128 == 0:
            verify_source_hashes(L9_RUNTIME_PATHS, self.l9_runtime_sources)
            self._check_disk()
        result = super().step(action)
        obs, reward, terminated, truncated, info = result
        position = np.asarray(self.env.active.unwrapped.data.qpos[:3], dtype=float).copy()
        self._path_length += float(np.linalg.norm(position - self._previous_position))
        self._previous_position = position
        self._action_norm_sum += float(np.linalg.norm(np.asarray(action, dtype=float)))
        self._episode_steps += 1
        if terminated or truncated:
            displacement = float(np.linalg.norm(position - self._episode_start))
            row = {
                "scenario_identity_sha256": info["scenario_identity_sha256"],
                "outcome": info["termination_reason"],
                "policy_end_step": self.policy_steps,
                "episode_policy_steps": self._episode_steps,
                "episode_return": float(self.episode_returns[-1]),
                "trajectory_length_m": self._path_length,
                "net_displacement_m": displacement,
                "mean_action_norm": self._action_norm_sum / self._episode_steps,
                "final_target_distance_m": float(info["distance_m"]),
                "motionless_episode": displacement < .10 and self._path_length < .20,
                "sim_end_time_s": float(self.env.active.unwrapped.data.time),
            }
            if not all(np.isfinite(value) for value in
                       (row["episode_return"], row["trajectory_length_m"],
                        row["net_displacement_m"], row["mean_action_norm"],
                        row["final_target_distance_m"], row["sim_end_time_s"])):
                raise RuntimeError("nonfinite formal episode health telemetry")
            _append_jsonl(self.health_dir / "train_episodes.jsonl", row)
            self.episode_diagnostics.append(row)
        if self.policy_steps % 128 == 0:
            sample = {
                "policy_step": self.policy_steps,
                "wall_elapsed_s": time.perf_counter() - self._start_wall,
                "cpu_elapsed_s": time.process_time() - self._start_cpu,
                "rss_current_mib": _current_rss_mib(),
                "rss_peak_mib": self.rss_peak_mib,
                "mem_available_mib": self.min_available_mib,
                "disk_free_bytes": self._check_disk(),
            }
            _append_jsonl(self.health_dir / "resource_samples.jsonl", sample)
            self.resource_samples.append(sample)
            if self.policy_steps % 1024 == 0:
                _atomic_json(self.health_dir / "formal_progress.json", {
                    "status": "INCOMPLETE", "actual_timesteps": self.policy_steps,
                    "completed_episodes": len(self.episode_diagnostics),
                    "latest_resource_sample": self.resource_samples[-1],
                })
        return result

    def health_snapshot(self, model, *, actual_step: int) -> dict:
        verify_source_hashes(L9_RUNTIME_PATHS, self.l9_runtime_sources)
        if actual_step != self.policy_steps or model.num_timesteps != actual_step:
            raise RuntimeError("formal health snapshot is not at the completed PPO step")
        rows = self.episode_diagnostics
        terminal = dict(self.terminal_counts)
        if sum(terminal.values()) != len(rows):
            raise RuntimeError("formal episode health accounting differs from terminal counts")
        metrics = {}
        for key, value in model.logger.name_to_value.items():
            if key.startswith("train/") and isinstance(value, (int, float, np.number)):
                number = float(value)
                if not np.isfinite(number):
                    raise RuntimeError("nonfinite PPO diagnostic")
                metrics[key] = number
        if not metrics:
            raise RuntimeError("PPO train diagnostics unavailable at post-update milestone")
        return {
            "actual_execution_step": actual_step,
            "optimizer_updates": int(model._n_updates),
            "policy_steps": self.policy_steps,
            "test_episode_count": 0,
            "completed_episodes": len(rows),
            "terminal_counts": terminal,
            "mean_episode_return": float(np.mean(self.episode_returns)) if rows else None,
            "mean_trajectory_length_m": float(np.mean([r["trajectory_length_m"] for r in rows])) if rows else None,
            "mean_final_target_distance_m": float(np.mean([r["final_target_distance_m"] for r in rows])) if rows else None,
            "mean_action_norm": float(np.mean([r["mean_action_norm"] for r in rows])) if rows else None,
            "mean_displacement_m": float(np.mean([r["net_displacement_m"] for r in rows])) if rows else None,
            "motionless_episode_fraction": (sum(r["motionless_episode"] for r in rows) / len(rows)) if rows else None,
            "ppo_diagnostics": metrics,
            "rss_current_mib": _current_rss_mib(),
            "rss_peak_mib": self.rss_peak_mib,
            "mem_available_min_mib": self.min_available_mib,
            "disk_free_bytes": self._check_disk(),
            "wall_elapsed_s": time.perf_counter() - self._start_wall,
            "cpu_elapsed_s": time.process_time() - self._start_cpu,
            "resource_sample_count": len(self.resource_samples),
            "train_episodes_jsonl_rows": len(rows),
            "l9_runtime_source_sha256": self.l9_runtime_sources,
        }

    def close(self):
        if FormalHealthMetricsEnv.active_instance is self:
            FormalHealthMetricsEnv.active_instance = None
        super().close()


class L9FormalPPOTrainingDriver(FormalPPOTrainingDriver):
    """Reuse L8 optimization; add only the passive formal telemetry layer."""

    def __init__(self, prereg_path: Path, upstream_root: Path, output_dir: Path):
        super().__init__(prereg_path, upstream_root, output_dir)
        self.l9_runtime_sources = source_hashes(L9_RUNTIME_PATHS)

    def run_formal(self, *, allow_formal_training: bool = False) -> dict:
        if allow_formal_training is not True:
            raise PermissionError("formal training authorization required")
        verify_source_hashes(L9_RUNTIME_PATHS, self.l9_runtime_sources)
        original = orchestration.TrainingMetricsEnv
        if original is not TrainingMetricsEnv:
            raise RuntimeError("unexpected metrics binding before formal training")
        orchestration.TrainingMetricsEnv = FormalHealthMetricsEnv
        try:
            return super().run_formal(allow_formal_training=True)
        finally:
            orchestration.TrainingMetricsEnv = original

    def persist_milestone(self, model, scheduler, event):
        verify_source_hashes(L9_RUNTIME_PATHS, self.l9_runtime_sources)
        measured = FormalHealthMetricsEnv.active_instance
        if measured is None or measured.env is not scheduler:
            raise RuntimeError("formal health metrics instance missing at milestone")
        if not _finite_model_and_gradients(model):
            raise RuntimeError("nonfinite PPO weights or gradients before checkpoint")
        # Validate diagnostics before L8 writes the checkpoint or runs Validation.
        health = measured.health_snapshot(model, actual_step=event.actual_step)
        result = super().persist_milestone(model, scheduler, event)
        path = self.output_dir / f"L9_HEALTH_STEP{event.actual_step:07d}.json"
        if path.exists():
            raise FileExistsError("formal health snapshot already exists")
        _atomic_json(path, health)
        return result
