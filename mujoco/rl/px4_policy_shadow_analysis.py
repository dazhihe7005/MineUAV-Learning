"""Reproducible, single-process offline replay; no simulator or PX4 process."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from px4_policy_shadow import ActionAdapter, FrozenPolicy, ObservationGate, shadow_step


ROOT = Path(__file__).resolve().parents[2]
TRACE_ROOT = Path("/home/keyousuke/MineUAV-Learning/mujoco/reports/uav_ppo_bc_nominal_comparison_parts")
REPORT = ROOT / "mujoco/reports/px4_mujoco_policy_shadow_preflight.json"
FIG_DIR = ROOT / "mujoco/reports/px4_mujoco_policy_shadow_figures"
MODELS = {
    "original_bc": ("uav_bc_mlp_seed0.pt", "bc", "7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c"),
    "yaw_bc": ("uav_bc_yaw_augmented_seed0.pt", "bc", "d29b93555ef46f231228c23345761f76b5758f6451648375fcc2b7e4501f03ce"),
    "ppo_seed0": ("uav_ppo_fixed_protocol_seed0.zip", "ppo", "9a1e211611436d058a9fafe218c8f0e965572dfe6389c4dff869a9dcedcfd49b"),
    "ppo_seed1": ("uav_ppo_fixed_protocol_seed1.zip", "ppo", "96d66efd3d1a943d6320d1013abb20cb429cffb1f992ecbb24e94775e91a9a33"),
    "ppo_seed2": ("uav_ppo_fixed_protocol_seed2.zip", "ppo", "61da0c0f000f544f6963254a1f4c3aa086161592988900ff0d4bfd0a3d2e737d"),
    "ppo_seed3": ("uav_ppo_fixed_protocol_seed3.zip", "ppo", "0f0bddee3c711b925b7a9ac7336d23ca47afb250196773a3bf6cda40851c8cd6"),
    "ppo_seed4": ("uav_ppo_fixed_protocol_seed4.zip", "ppo", "80224e40f26faf1802388d9f9f264e5ec120bb40cfdc9efe91c77f921d6a5125"),
}
PINNED_25_TRACE_DIGEST = "927cdd8705f58fb3fccf34cf5f9ed9fec1bc380cf5e8faef1082ec3f7cb3b2ac"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cohort(count: int = 25, trace_root: Path = TRACE_ROOT, expected_digest: str | None = None):
    """First 25 scripted fresh-target episodes, fixed before any model predictions."""
    observations, expert_actions, episode_index, sample_times_us, hashes = [], [], [], [], []
    counts = {"all_policy_samples": 0, "physical_invalid": 0, "gate_rejected": 0}
    for i in range(count):
        row_path = trace_root / f"fresh_{i:03d}_scripted.json"
        trace_path = trace_root / f"fresh_{i:03d}_scripted.npz"
        row = json.loads(row_path.read_text())
        record_payload = {key: value for key, value in row.items() if key != "record_sha256"}
        record_sha = hashlib.sha256(json.dumps(record_payload, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest()
        actual_sha = file_sha(trace_path)
        if (row.get("record_sha256") != record_sha or row.get("trace_sha256") != actual_sha
                or row.get("controller") != "scripted" or row.get("split") != "fresh"
                or row.get("key") != f"fresh/{i:03d}"
                or row.get("target_id") != f"ppo-bc-nominal-2026101501-{i:03d}"):
            raise ValueError(f"trace identity mismatch: {i}")
        hashes.append(actual_sha)
        with np.load(trace_path, allow_pickle=False) as trace:
            obs = np.asarray(trace["observations"][:-1], np.float32)
            valid = np.asarray(trace["physical_state_valid"][:-1], bool)
            action = np.asarray(trace["actions"], np.float32)
            if len(obs) != len(action) or len(valid) != len(obs):
                raise ValueError(f"observation/action indexing mismatch: {i}")
            gate = ObservationGate()
            counts["all_policy_samples"] += len(obs)
            for step, (sample, is_valid) in enumerate(zip(obs, valid)):
                if not is_valid:
                    counts["physical_invalid"] += 1
                    continue
                try:
                    accepted = gate.accept(sample, step * 40_000, step * 40_000)
                except ValueError as exc:
                    counts["gate_rejected"] += 1
                    raise ValueError(f"replay validity gate rejection in episode {i}, step {step}") from exc
                observations.append(accepted)
                expert_actions.append(action[step])
                episode_index.append(i)
                sample_times_us.append(step * 40_000)
    if not observations:
        raise ValueError("empty valid replay cohort")
    counts["accepted"] = len(observations)
    counts["episodes"] = count
    counts["trace_hash_list_sha256"] = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    if expected_digest is None and count == 25:
        expected_digest = PINNED_25_TRACE_DIGEST
    if expected_digest is not None and counts["trace_hash_list_sha256"] != expected_digest:
        raise ValueError("pinned cohort digest mismatch")
    counts["pinned_historical_cohort_verified"] = expected_digest is not None
    return (np.stack(observations), np.stack(expert_actions), np.asarray(episode_index),
            np.asarray(sample_times_us, dtype=np.int64), counts)


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values))}


def process_rss_mib() -> float:
    with Path("/proc/self/status").open() as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS unavailable")


def system_available_mib() -> float:
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("MemAvailable unavailable")


def _write_json_atomic(path: Path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def overall_status(acceptance: dict) -> str:
    required = ("model_load_identity", "observation_action_mapping",
                "shadow_inference_no_transmission", "cpu_25hz_single_inference")
    return "PASS" if all(acceptance.get(key) == "PASS" for key in required) else "FAIL"


def require_deterministic(policy, observation) -> None:
    first = policy.predict(observation)
    second = policy.predict(observation)
    if not np.array_equal(first, second):
        raise ValueError("nondeterministic repeated policy inference")


def run(trace_root: Path = TRACE_ROOT):
    torch.set_num_threads(1)
    available_mib = [system_available_mib()]
    x, expert, ep, sample_times_us, cohort = read_cohort(trace_root=trace_root)
    probe = np.linspace(0, len(x) - 1, min(500, len(x)), dtype=int)
    sample = x[probe]
    models = {}
    action_outputs = {}
    yaw_outputs = {}
    near_outputs = {}
    latency_p95 = {}
    near = np.linalg.norm(x[:, :3], axis=1) < .2
    moving = np.linalg.norm(x[:, 3:6], axis=1) > .05
    for name, (filename, kind, sha) in MODELS.items():
        path = ROOT / "mujoco/rl/models" / filename
        before_load_rss = process_rss_mib()
        started = time.perf_counter()
        policy = FrozenPolicy.load(path, kind=kind, expected_sha256=sha)
        load_ms = 1000 * (time.perf_counter() - started)
        after_load_rss = process_rss_mib()
        available_mib.append(system_available_mib())
        before = policy.parameter_hash()
        outputs = np.stack([policy.predict(obs) for obs in x])
        # Exercise the public no-transport path for every sample, with a fresh
        # causal gate at each genuine historical episode boundary.
        commands = []
        gate = None
        for index, obs in enumerate(x):
            if index == 0 or ep[index] != ep[index - 1]:
                gate = ObservationGate()
            timestamp = int(sample_times_us[index])
            commands.append(shadow_step({name: policy}, obs, timestamp, timestamp,
                                        gate=gate)[name])
        if any(not np.isfinite(command.velocity_ned_m_s).all() for command in commands):
            raise ValueError("nonfinite converted shadow command")
        for obs in sample[:50]:
            policy.predict(obs)
        calls_ms = []
        for obs in sample:
            t0 = time.perf_counter_ns()
            policy.predict(obs)
            calls_ms.append((time.perf_counter_ns() - t0) / 1e6)
        if policy.parameter_hash() != before or file_sha(path) != sha:
            raise ValueError("model identity changed during inference")
        if kind == "bc":
            stats = policy.model.stats
            preprocessing = "checkpoint Train-only observation z-score and output action de-normalization"
            stats_summary = {key: {"mean": stats[key]["mean"], "std": stats[key]["std"]} for key in ("obs", "action")}
            architecture = "7-128ReLU-128ReLU-4"
            training_seed = 0
            training_steps = None
            parameter_count = sum(p.numel() for p in policy.model.actor.parameters())
        else:
            preprocessing = "raw 7D float32, no VecNormalize; deterministic SB3 clipped policy mean"
            stats_summary = None
            architecture = "separate 7-64Tanh-64Tanh actor/critic, diagonal Gaussian policy"
            training_seed = int(policy.model.seed)
            training_steps = int(policy.model.num_timesteps)
            parameter_count = sum(p.numel() for p in policy.model.policy.parameters())
            if training_seed != int(name[-1]) or training_steps != 100352:
                raise ValueError("PPO seed or fixed training-budget provenance mismatch")
        converted = np.stack([command.velocity_ned_m_s for command in commands])
        expected_converted = np.stack([ActionAdapter().convert(a, 0, 0).velocity_ned_m_s for a in outputs])
        if not np.array_equal(converted, expected_converted):
            raise ValueError("public shadow path differs from direct frozen inference")
        yaw_case = {}
        for degrees in (-30, -10, 0, 10, 30):
            altered = x[0].copy()
            altered[6] = np.deg2rad(degrees)
            yaw_case[str(degrees)] = policy.predict(altered).astype(float).tolist()
        n_actions = outputs[near]
        n_vel = x[near, 3:6]
        braking = np.sum(n_actions[:, :3] * np.array([1.5, 1.5, 1.0]) * n_vel, axis=1) < 0
        actual_moving = moving[near]
        near_summary = {"count": int(near.sum()),
                        "mean_command_speed_m_s": float(np.linalg.norm(n_actions[:, :3] * [1.5, 1.5, 1.0], axis=1).mean()) if len(n_actions) else None,
                        "braking_fraction_when_moving": float(braking[actual_moving].mean()) if actual_moving.any() else None}
        deltas = np.linalg.norm(np.diff(outputs, axis=0), axis=1)
        deltas = deltas[ep[1:] == ep[:-1]]
        require_deterministic(policy, sample[0])
        models[name] = {
            "status": "PASS", "checkpoint": filename, "checkpoint_sha256_before_after": sha,
            "parameter_sha256_before_after": before, "kind": kind,
            "training_seed": training_seed, "training_timesteps": training_steps,
            "architecture": architecture, "parameter_count": parameter_count,
            "preprocessing": preprocessing, "checkpoint_statistics": stats_summary,
            "load_ms": load_ms, "inference_ms_cpu": summary(calls_ms),
            "process_rss_mib_before_load": before_load_rss,
            "process_rss_mib_after_load": after_load_rss,
            "process_rss_mib_after_inference": process_rss_mib(),
            "approximate_rss_load_delta_mib": after_load_rss - before_load_rss,
            "25hz_single_inference_budget_pass": max(calls_ms) < 40.,
            "normalized_action_mean": outputs.mean(axis=0).astype(float).tolist(),
            "normalized_action_min": outputs.min(axis=0).astype(float).tolist(),
            "normalized_action_max": outputs.max(axis=0).astype(float).tolist(),
            "saturated_element_fraction_abs_ge_095": float((np.abs(outputs) >= .95).mean()),
            "saturated_command_fraction_any_abs_ge_095": float((np.abs(outputs) >= .95).any(axis=1).mean()),
            "mean_abs_difference_from_recorded_scripted_action": np.mean(np.abs(outputs - expert), axis=0).astype(float).tolist(),
            "within_episode_action_delta_l2": summary(deltas),
            "converted_ned_velocity_mean_m_s": converted.mean(axis=0).astype(float).tolist(),
            "near_target": near_summary,
            "synthetic_yaw_error_probe_normalized_actions": yaw_case,
            "synthetic_yaw_plus30_minus30_yaw_rate_slope": float(yaw_case["30"][3] - yaw_case["-30"][3]),
            "output_repeated_exact": True,
        }
        action_outputs[name] = outputs
        yaw_outputs[name] = [yaw_case[str(d)][3] for d in (-30, -10, 0, 10, 30)]
        near_outputs[name] = near_summary["mean_command_speed_m_s"]
        latency_p95[name] = models[name]["inference_ms_cpu"]["p95"]
        del policy
        gc.collect()
    # Plot only aggregate offline inference statistics; no fabricated trajectories.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(MODELS)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in labels:
        ax.plot([-30, -10, 0, 10, 30], yaw_outputs[name], marker="o", label=name)
    ax.set(xlabel="Synthetic yaw error (degrees)", ylabel="Normalized ENU yaw-rate command",
           title="Counterfactual input probe, not a flown trajectory")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "yaw_response_offline.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, [latency_p95[n] for n in labels])
    axes[0].set_ylim(0, max(latency_p95.values()) * 1.25)
    axes[0].set_ylabel("CPU inference P95 (ms)")
    axes[0].set_title("40 ms policy period is off-scale")
    axes[0].tick_params(axis="x", rotation=60)
    axes[1].bar(labels, [near_outputs[n] for n in labels]); axes[1].set_ylabel("Near-target commanded speed (m/s)")
    axes[1].tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(FIG_DIR / "latency_and_near_target_speed.png", dpi=150); plt.close(fig)
    rss_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    acceptance = {
        "model_load_identity": "PASS", "observation_action_mapping": "PASS",
        "shadow_inference_no_transmission": "PASS",
        "cpu_25hz_single_inference": "PASS" if all(m["25hz_single_inference_budget_pass"] for m in models.values()) else "FAIL",
        "px4_offboard_control": "NOT TESTED", "closed_loop_px4_success": "NOT TESTED",
        "real_slam_odometry": "NOT TESTED", "multi_model_simultaneous_deadline": "NOT TESTED"
    }
    report = {
        "phase": "3B offline policy shadow preflight", "status": overall_status(acceptance),
        "base_commit": "63e504eb1835a53a08d975476ba2dc10cc503d5d",
        "source": {"kind": "recorded nominal PI-environment scripted trajectories, not SLAM/PX4",
                   "selection": "fresh_000 through fresh_024 scripted; fixed before inference",
                   "local_only_path": str(trace_root), **cohort},
        "observation_contract": "7D target minus position ENU m; world ENU velocity m/s; wrapped target-minus-current ENU yaw rad",
        "action_contract": "normalized [-1,1]^4, scales [1.5,1.5,1.0] m/s and 1rad/s; ENU velocity to NED, NED yaw-rate sign reversed",
        "models": models,
        "resource": {"workers": 1, "rss_peak_mib": rss_peak_mib, "cpu_threads_torch": 1,
                     "minimum_system_available_mib_sampled": min(available_mib),
                     "per_model_rss_delta_warning": "process allocator may retain prior model memory; deltas are not isolated model footprint"},
        "verification_record": {
            "scope": "Separate manual verification on 2026-10-13; not run by this replay script",
            "new_offline_policy_tests_passed": 30,
            "inherited_bridge_tests_passed": 356,
            "known_nonfatal_warnings_new_tests": 3,
            "known_nonfatal_warnings_bridge_tests": 2,
            "final_independent_readonly_review": "0 Critical / 0 Important remaining; reviewer did not rerun tests"
        },
        "acceptance": acceptance,
        "limits": ["No PX4 SITL run or command sent", "Only recorded nominal scripted states; no offboard state-estimation validation",
                   "Yaw ±10/30 probes modify input scalar only; not physically replayed", "40ms deadline assessed per model, not seven-model parallel runtime",
                   "Elementwise limits are not a PX4 flight-safety envelope"],
        "figures": [str(p.relative_to(ROOT)) for p in sorted(FIG_DIR.glob("*.png"))],
    }
    _write_json_atomic(REPORT, report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline policy shadow replay; never starts PX4")
    parser.add_argument("--trace-root", type=Path, default=TRACE_ROOT)
    result = run(parser.parse_args().trace_root)
    print(json.dumps({"acceptance": result["acceptance"], "cohort": result["source"],
                      "rss_peak_mib": result["resource"]["rss_peak_mib"]}, indent=2))
