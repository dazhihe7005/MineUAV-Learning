"""L8 rollout-aligned PPO orchestration; importing this module never trains."""

from __future__ import annotations

import copy
import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch

from .ppo_formal_checkpoint import formal_readiness_protocol
from .ppo_formal_checkpoint import CheckpointManager
from .ppo_formal_scheduler import MultiScenarioTrainingEnv, load_l6_prereg, require_memory
from .ppo_formal_validation import LidarPPOValidationEvaluator
from .ppo_preflight import make_ppo
from .ppo_training_runner import (_atomic_json, _historical_model_hashes, _stats,
                                  _weights_sha, protocol_hash)


PILOT_ID = "LIDAR_PPO_PILOT_ONLY_L8_SEED0"
PILOT_OUTPUT_DIR = Path(__file__).parents[1] / "reports/mujoco_lidar_ppo_orchestration_parts/run_01"


@dataclass(frozen=True)
class Milestone:
    nominal_step: int
    actual_step: int


def _aligned(step: int, rollout_steps: int) -> int:
    return ((step + rollout_steps - 1) // rollout_steps) * rollout_steps


def aligned_milestones(total: int, interval: int, rollout_steps: int) -> tuple[Milestone, ...]:
    if (any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
            for x in (total, interval, rollout_steps)) or total % interval):
        raise ValueError("positive integer total, divisor interval and rollout size required")
    return tuple(Milestone(nominal, _aligned(nominal, rollout_steps))
                 for nominal in range(interval, total + 1, interval))


class MilestoneLedger:
    """Acknowledge a boundary once; skipped or overshot boundaries fail closed."""

    def __init__(self, milestones: tuple[Milestone, ...]):
        self.milestones = milestones
        self.next_index = 0

    def acknowledge(self, actual_step: int) -> Milestone | None:
        if self.next_index == len(self.milestones):
            return None
        next_event = self.milestones[self.next_index]
        if actual_step < next_event.actual_step:
            return None
        if actual_step != next_event.actual_step:
            raise ValueError("rollout boundary was skipped")
        self.next_index += 1
        return next_event


def l8_execution_protocol(prereg: dict) -> dict:
    """Explicitly revise the L7 proposal for SB3's full-rollout overshoot."""
    l7 = formal_readiness_protocol(prereg)
    config = copy.deepcopy(l7)
    config["parent_l7_protocol_sha256"] = protocol_hash(l7)
    config["experiment_id"] = "LIDAR_PPO_FORMAL_L8_PROTOCOL_NOT_RUN_V1"
    config["nominal_total_timesteps_per_seed"] = l7["total_timesteps_per_seed"]
    config["actual_total_timesteps_per_seed"] = _aligned(
        l7["total_timesteps_per_seed"], l7["n_steps"])
    config["checkpoint_every_nominal_steps"] = l7["checkpoint_every_steps"]
    config["milestone_rule"] = "first complete 128-step rollout at/after nominal threshold; after PPO optimizer"
    config["continuation_policy"] = "interrupted formal runs RESTART_REQUIRED; no exact or boundary resume"
    return config


def require_pilot_budget(*, seed: int, workers: int, total_timesteps: int,
                         experiment_id: str) -> None:
    if (seed != 0 or workers != 1 or total_timesteps != 8192 or
            experiment_id != PILOT_ID or any(isinstance(x, bool) for x in
                                            (seed, workers, total_timesteps))):
        raise ValueError("only the single frozen PILOT_ONLY seed0/1-worker/8192-step run is authorized")


def require_canonical_pilot_output(directory: Path) -> None:
    if Path(directory).resolve() != PILOT_OUTPUT_DIR.resolve():
        raise ValueError("Pilot must use its unique canonical output directory")


def execute_segments(model, milestones: tuple[Milestone, ...], *, on_boundary,
                     update_times_ms: list[float] | None = None) -> None:
    """Each learn() returns after its final PPO optimizer update in SB3 2.9.0."""
    ledger = MilestoneLedger(milestones)
    for planned in milestones:
        before_steps = int(model.num_timesteps)
        before_updates = int(model._n_updates)
        delta = planned.actual_step - before_steps
        if delta <= 0 or (hasattr(model, "n_steps") and delta % model.n_steps):
            raise ValueError("nonpositive or non-rollout-aligned segment")
        before_timing = len(update_times_ms) if update_times_ms is not None else 0
        if update_times_ms is None:
            model.learn(delta, reset_num_timesteps=False)
        else:
            original_train = model.train
            had_override = "train" in vars(model)
            old_override = vars(model).get("train")

            def timed_train():
                begun = time.perf_counter()
                try:
                    return original_train()
                finally:
                    update_times_ms.append((time.perf_counter() - begun) * 1000.)

            model.train = timed_train
            try:
                model.learn(delta, reset_num_timesteps=False)
            finally:
                if had_override:
                    model.train = old_override
                else:
                    del model.train
        expected_updates = 2 * (delta // 128)
        if (model.num_timesteps != planned.actual_step or
                model._n_updates - before_updates != expected_updates):
            raise RuntimeError("SB3 segment did not complete its expected rollout and optimizer update")
        if (update_times_ms is not None and
                len(update_times_ms) - before_timing != delta // 128):
            raise RuntimeError("PPO train() timing count differs from completed rollouts")
        event = ledger.acknowledge(model.num_timesteps)
        if event != planned:
            raise RuntimeError("milestone ledger did not acknowledge the completed update")
        on_boundary(event)


def _current_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.
    raise RuntimeError("process RSS not observable")


def source_hashes(paths: Iterable[Path]) -> dict[str, str]:
    paths = tuple(Path(path) for path in paths)
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("source names must be unique")
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def verify_source_hashes(paths: Iterable[Path], frozen: dict[str, str]) -> None:
    try:
        current = source_hashes(paths)
    except (OSError, ValueError) as exc:
        raise RuntimeError("runtime source identity unavailable") from exc
    if current != frozen:
        raise RuntimeError("runtime source identity changed during run")


def _runtime_paths() -> tuple[Path, ...]:
    root = Path(__file__).parent
    return tuple(root / name for name in
                 ("ppo_training_orchestration.py", "ppo_formal_checkpoint.py",
                  "ppo_formal_scheduler.py", "ppo_formal_validation.py"))


class TrainingMetricsEnv(gym.Wrapper):
    """Observe real train physics and memory without changing control or reward."""

    TERMINALS = ("success", "collision", "time_limit", "physical_failure", "sensor_failure")

    def __init__(self, env: gym.Env):
        available = require_memory()
        super().__init__(env)
        self.step_ms: list[float] = []
        self.scan_ms: list[float] = []
        self.encode_ms: list[float] = []
        self.reset_ms: list[float] = []
        self.reset_rss_mib: list[float] = []
        self.episode_returns: list[float] = []
        self.terminal_counts = {key: 0 for key in self.TERMINALS}
        self.physics_ticks = 0
        self.policy_steps = 0
        self.min_available_mib = available
        self._current_return = 0.

    def reset(self, *, seed=None, options=None):
        self.min_available_mib = min(self.min_available_mib, require_memory())
        started = time.perf_counter()
        result = self.env.reset(seed=seed, options=options)
        self.reset_ms.append((time.perf_counter() - started) * 1000.)
        self.reset_rss_mib.append(_current_rss_mib())
        self._current_return = 0.
        self.min_available_mib = min(self.min_available_mib, require_memory())
        return result

    def step(self, action):
        self.min_available_mib = min(self.min_available_mib, require_memory())
        before_frames = self.env.active.unwrapped.lidar_frame_count
        started = time.perf_counter()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_ms.append((time.perf_counter() - started) * 1000.)
        self.policy_steps += 1
        self.physics_ticks += int(info["physics_steps"])
        self._current_return += float(reward)
        if (info["scenario_split"] != "train" or
                info["scenario_identity_sha256"] not in self.env.by_identity or
                not np.isfinite(reward) or not np.isfinite(action).all()):
            raise RuntimeError("non-Train, unregistered or nonfinite PPO transition")
        plant = self.env.active.unwrapped
        if plant.lidar_frame_count > before_frames:
            self.scan_ms.append(float(plant.last_scan_ms))
            self.encode_ms.append(float(plant.last_encode_ms))
        if terminated or truncated:
            reason = info["termination_reason"]
            if reason not in self.terminal_counts or sum(bool(info.get(key)) for key in
                    ("success", "collision", "timeout", "physical_failure", "sensor_failure")) != 1:
                raise RuntimeError("invalid or nonexclusive training episode outcome")
            self.terminal_counts[reason] += 1
            self.episode_returns.append(self._current_return)
        self.min_available_mib = min(self.min_available_mib, require_memory())
        return obs, reward, terminated, truncated, info

    @property
    def rss_peak_mib(self) -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.


def _optimizer_step_count(model) -> int:
    states = model.policy.optimizer.state.values()
    steps = {int(state["step"].item()) for state in states if "step" in state}
    if len(steps) != 1:
        raise RuntimeError("optimizer states have missing or divergent step counts")
    return steps.pop()


def _finite_model_and_gradients(model) -> bool:
    gradients = 0
    for parameter in model.policy.parameters():
        if not np.isfinite(parameter.detach().cpu().numpy()).all():
            return False
        if parameter.grad is not None:
            gradients += 1
            if not np.isfinite(parameter.grad.detach().cpu().numpy()).all():
                return False
    return gradients > 0


class FormalPPOTrainingDriver:
    """Frozen driver; formal mode is gated and is never launched by pilot CLI."""

    def __init__(self, prereg_path: Path, upstream_root: Path, output_dir: Path):
        self.prereg = load_l6_prereg(prereg_path)
        if _historical_model_hashes() != self.prereg["historical_checkpoint_sha256"]:
            raise RuntimeError("historical checkpoint identity changed")
        self.upstream_root = Path(upstream_root)
        self.output_dir = Path(output_dir)
        self.protocol = l8_execution_protocol(self.prereg)
        self.protocol_sha256 = protocol_hash(self.protocol)
        self.completed_events: list[dict] = []
        self.runtime_sources = source_hashes(_runtime_paths())

    def persist_milestone(self, model, scheduler: MultiScenarioTrainingEnv,
                          event: Milestone) -> dict:
        """Called only after learn() returned from a complete, optimized rollout."""
        require_memory()
        verify_source_hashes(_runtime_paths(), self.runtime_sources)
        if (int(model.num_timesteps) != event.actual_step or event.actual_step % 128
                or int(model._n_updates) != 2 * (event.actual_step // 128)):
            raise RuntimeError("milestone not at a post-optimizer rollout boundary")
        run_id = f"L8_SEED0_STEP{event.actual_step:07d}"
        event_path = self.output_dir / f"{run_id}_event.json"
        if event_path.exists():
            raise FileExistsError("milestone event already exists")
        manager = CheckpointManager(self.output_dir, self.prereg,
                                    execution_protocol_sha256=self.protocol_sha256)
        checkpoint = manager.save(model, scheduler, run_id=run_id, seed=0)
        restored = manager.load_verified(scheduler, run_id=run_id)
        if restored.num_timesteps != model.num_timesteps or _weights_sha(restored.policy) != _weights_sha(model.policy):
            raise RuntimeError("saved checkpoint did not reload the updated policy")
        del restored
        before_rng = scheduler.scheduler_state()
        validation = LidarPPOValidationEvaluator(self.prereg, self.upstream_root).evaluate(model)
        unchanged = scheduler.scheduler_state() == before_rng
        if not unchanged or validation["probe_only"]:
            raise RuntimeError("Validation changed Train scheduler or ended before terminal")
        record = {
            "nominal_step": event.nominal_step, "actual_execution_step": event.actual_step,
            "event_after_optimizer_updates": int(model._n_updates),
            "checkpoint": checkpoint, "validation": validation,
            "train_rng_unchanged_during_validation": unchanged,
            "test_episode_count": 0,
        }
        _atomic_json(event_path, record)
        self.completed_events.append(record)
        return record

    def _reserve_attempt(self, mode: str) -> None:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError("run directory is nonempty; no retry or overwrite allowed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        marker = self.output_dir / f"{mode}.attempt"
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)

    def _run(self, *, mode: str, seed: int, events: tuple[Milestone, ...]) -> dict:
        require_memory()
        if seed != 0 or not events:
            raise ValueError("this driver instance is pinned to independent seed0")
        verify_source_hashes(_runtime_paths(), self.runtime_sources)
        self._reserve_attempt(mode)
        _atomic_json(self.output_dir / "preflight_identity.json", {
            "runtime_source_sha256": self.runtime_sources,
            "l8_protocol_sha256": self.protocol_sha256,
            "l6_manifest_sha256": self.prereg["scenario_manifest_sha256"],
            "network_implementation_sha256": self.prereg["network_implementation_sha256"],
            "reward_version": self.protocol["reward_version"],
            "historical_checkpoint_sha256": self.prereg["historical_checkpoint_sha256"],
        })
        started = time.perf_counter()
        torch.set_num_threads(1)
        scheduler = MultiScenarioTrainingEnv(self.prereg, self.upstream_root, seed=seed)
        measured = TrainingMetricsEnv(scheduler)
        historical_before = _historical_model_hashes()
        try:
            model = make_ppo(measured, n_steps=128, seed=seed)
            if model.num_timesteps != 0 or model.policy.optimizer.state:
                raise RuntimeError("training did not start from random initialization")
            actor_before = _weights_sha(model.policy.action_net)
            critic_before = _weights_sha(model.policy.value_net)
            update_times_ms: list[float] = []
            execute_segments(model, events, on_boundary=lambda event:
                             self.persist_milestone(model, scheduler, event),
                             update_times_ms=update_times_ms)
            if (measured.policy_steps != events[-1].actual_step
                    or model.num_timesteps != events[-1].actual_step
                    or model._n_updates != 2 * len(range(0, model.num_timesteps, 128))
                    or not _finite_model_and_gradients(model)):
                raise RuntimeError("training steps, updates or finite gradients invalid")
            actor_after = _weights_sha(model.policy.action_net)
            critic_after = _weights_sha(model.policy.value_net)
            if actor_before == actor_after or critic_before == critic_after:
                raise RuntimeError("actor or critic did not update")
            if _historical_model_hashes() != historical_before:
                raise RuntimeError("historical model identity changed during run")
            verify_source_hashes(_runtime_paths(), self.runtime_sources)
            outcomes = dict(measured.terminal_counts)
            summary = {
                "experiment_id": PILOT_ID if mode == "pilot" else "L8_FORMAL_SEED0",
                "mode": mode, "seed": seed, "workers": 1,
                "training_actual_timesteps": model.num_timesteps,
                "training_physics_ticks": measured.physics_ticks,
                "rollout_count": model.num_timesteps // 128,
                "ppo_n_updates": int(model._n_updates),
                "optimizer_step_count": _optimizer_step_count(model),
                "train_scenario_visit_counts": scheduler.visit_counts,
                "train_scenario_selected_identities": scheduler.selected_identities,
                "episode_count": len(measured.episode_returns),
                "terminal_counts": outcomes,
                "mean_completed_episode_return": (float(np.mean(measured.episode_returns))
                                                   if measured.episode_returns else None),
                "actor_sha256_before": actor_before, "actor_sha256_after": actor_after,
                "critic_sha256_before": critic_before, "critic_sha256_after": critic_after,
                "finite_policy_and_gradients": True,
                "event_count": len(self.completed_events),
                "events": self.completed_events,
                "reset_compile_ms": _stats(measured.reset_ms),
                "environment_step_ms": _stats(measured.step_ms),
                "lidar_scan_ms": _stats(measured.scan_ms),
                "lidar_encoding_ms": _stats(measured.encode_ms),
                "ppo_train_update_ms": _stats(update_times_ms),
                "episode_reset_rss_mib": measured.reset_rss_mib,
                "rss_peak_mib": measured.rss_peak_mib,
                "system_min_available_mib": measured.min_available_mib,
                "wall_seconds": time.perf_counter() - started,
                "l8_protocol_sha256": self.protocol_sha256,
                "l7_protocol_sha256": self.protocol["parent_l7_protocol_sha256"],
                "l6_manifest_sha256": self.prereg["scenario_manifest_sha256"],
                "l6_formal_proposal_sha256": self.prereg["formal_proposal_sha256"],
                "historical_checkpoint_sha256": historical_before,
                "runtime_source_sha256": dict(self.runtime_sources),
                "software_versions": {name: importlib.metadata.version(name) for name in
                                      ("mujoco", "stable-baselines3", "torch", "gymnasium", "numpy")},
                "test_episode_count": 0,
                "out_of_memory": False,
            }
            summary_path = self.output_dir / "pilot_summary.json" if mode == "pilot" else self.output_dir / "formal_summary.json"
            if summary_path.exists():
                raise FileExistsError("run summary already exists")
            _atomic_json(summary_path, summary)
            _atomic_json(self.output_dir / f"{mode}.complete.json", {
                "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "actual_timesteps": model.num_timesteps,
                "events": len(self.completed_events),
            })
            return summary
        finally:
            measured.close()

    def run_pilot(self) -> dict:
        require_pilot_budget(seed=0, workers=1, total_timesteps=8192,
                             experiment_id=PILOT_ID)
        require_canonical_pilot_output(self.output_dir)
        return self._run(mode="pilot", seed=0, events=(Milestone(8192, 8192),))

    def run_formal(self, *, allow_formal_training: bool = False) -> dict:
        if allow_formal_training is not True:
            raise PermissionError("1M-step formal training needs separate user authorization")
        events = aligned_milestones(1_000_000, 100_000, 128)
        return self._run(mode="formal", seed=0, events=events)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L8 one-shot, 8192-step PPO pilot only")
    parser.add_argument("--pilot", action="store_true", help="execute the uniquely authorized pilot")
    parser.add_argument("--upstream", type=Path, required=True,
                        help="pinned MuJoCo-LiDAR source tree")
    arguments = parser.parse_args(argv)
    if not arguments.pilot:
        parser.error("only explicit --pilot is available in L8")
    prereg = Path(__file__).parents[1] / "reports/mujoco_lidar_ppo_training_prereg_seed0.json"
    driver = FormalPPOTrainingDriver(prereg, arguments.upstream, PILOT_OUTPUT_DIR)
    report = driver.run_pilot()
    print(json.dumps({"actual_timesteps": report["training_actual_timesteps"],
                      "rollout_count": report["rollout_count"],
                      "event_count": report["event_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
