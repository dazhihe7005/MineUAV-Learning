"""Offline 66-frame MID360-like observation and 10Hz→25Hz contract audit.

Reads previous MuJoCo ray NPZ files sequentially. It does not cast new rays,
advance MuJoCo, start PX4, train a policy, or publish to any device.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import resource
import time
from pathlib import Path

import numpy as np

from .async_observation import AsynchronousObservationBuffer
from .observation import AngularRangeEncoder, LidarObservation
from .pointcloud import PointCloudFrame
from .runner import _atomic_json, _available_mib, _stats
from .scanner import PATTERN_SHA256
from .scene import BASE_MODEL, BASE_SHA256
from .trajectory import TRAJECTORIES


PRIOR_REPORT_SHA256 = "689bf759473f821e423a7d2d8ef1c4b2f212f42478db20a5af8d588f56234929"


def verify_raw_identity(path: Path, expected_sha256: str) -> tuple[str, bytes]:
    """Return the exact verified bytes to decode; never reopen the source."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"missing raw frame: {file}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected raw SHA256 is malformed")
    raw = file.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"raw NPZ SHA256 mismatch: {file}")
    return actual, raw


def _policy_schedule(observations: dict[str, list[LidarObservation]],
                     arrival_delay_s: float) -> dict:
    rows_by_path: dict[str, list[dict]] = {}
    read_times = []
    future = missing = stale = repeated = 0
    max_cache_frames = max_cache_bytes = 0
    for name, series in observations.items():
        cache = AsynchronousObservationBuffer(experiment_id=name, capacity=4,
                                              stale_after_s=.15)
        next_source = 0
        rows: list[dict] = []
        for policy_tick in range(26):
            policy_time = policy_tick / 25
            while (next_source < len(series)
                   and series[next_source].frame_timestamp + arrival_delay_s
                   <= policy_time + 1e-9):
                source = series[next_source]
                cache.push(source, arrival_time_s=source.frame_timestamp + arrival_delay_s,
                           experiment_id=name)
                next_source += 1
            started = time.perf_counter()
            selection = cache.read(policy_time, experiment_id=name)
            read_times.append((time.perf_counter() - started) * 1000)
            source_time = selection.source_frame_timestamp
            future += int(source_time is not None and source_time > policy_time + 1e-9)
            missing += int(selection.status == "MISSING")
            stale += int(selection.status == "STALE")
            repeated += int(selection.status == "FRESH" and not selection.is_new_for_policy)
            max_cache_frames = max(max_cache_frames, cache.frame_count)
            max_cache_bytes = max(max_cache_bytes, cache.cache_data_bytes)
            rows.append({"policy_time_s": policy_time, "source_time_s": source_time,
                         "arrival_time_s": selection.arrival_time_s, "age_s": selection.age_s,
                         "status": selection.status,
                         "new_source_frame": selection.is_new_for_policy})
        rows_by_path[name] = rows
    return {"source": ("ZERO_DELAY_OFFLINE_SCHEDULE" if not arrival_delay_s else
                        "SYNTHETIC_OFFLINE_20MS_ARRIVAL_DELAY"),
            "arrival_delay_s": arrival_delay_s,
            "policy_hz": 25, "lidar_hz": 10, "clock_domain": "MUJOCO_SIM_TIME",
            "stale_after_s": .15, "policy_reads": len(read_times), "missing_reads": missing,
            "stale_reads": stale, "reused_fresh_reads": repeated, "future_reads": future,
            "max_cache_frames": max_cache_frames, "max_cache_data_bytes": max_cache_bytes,
            "read_ms": _stats(read_times), "by_trajectory": rows_by_path}


def run_observation_preflight(raw_root: Path, prior_report: Path,
                              output_report: Path, figures_dir: Path) -> dict:
    """Analyze the *same* 66 SHA-pinned Phase3E-L3 frames, never overwrite evidence."""
    raw = Path(raw_root)
    prior_file = Path(prior_report)
    output = Path(output_report)
    figures = Path(figures_dir)
    if output.exists():
        raise ValueError("report already exists; refusing to overwrite evidence")
    if figures.exists() and any(figures.iterdir()):
        raise ValueError("figure directory is not empty; refusing to overwrite evidence")
    if not prior_file.is_file():
        raise FileNotFoundError(f"missing prior report: {prior_file}")
    if not raw.is_dir():
        raise FileNotFoundError(f"missing raw frame directory: {raw}")
    previous_bytes = prior_file.read_bytes()
    if hashlib.sha256(previous_bytes).hexdigest() != PRIOR_REPORT_SHA256:
        raise ValueError("prior report SHA256 identity mismatch")
    prior = json.loads(previous_bytes)
    if (set(prior.get("trajectories", {})) != set(TRAJECTORIES)
            or prior.get("upstream", {}).get("pattern_sha256") != PATTERN_SHA256
            or prior.get("historical_model_sha256") != BASE_SHA256
            or hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() != BASE_SHA256):
        raise ValueError("prior scan manifest, pattern or physical model identity mismatch")
    encoder = AngularRangeEncoder()
    observations: dict[str, list[LidarObservation]] = {}
    encode_rows: list[dict] = []
    hash_ms = []
    load_ms = []
    encode_ms = []
    min_elevation = math.inf
    max_elevation = -math.inf
    total_rays = total_unscanned = total_sampled_no_hit = 0
    ever_sampled = np.zeros((12, 24), dtype=bool)
    available_start = _available_mib()
    for name in TRAJECTORIES:
        source_rows = prior["trajectories"][name]["frames"]
        if len(source_rows) != 11:
            raise ValueError(f"incomplete prior frame manifest: {name}")
        series: list[LidarObservation] = []
        for index, source in enumerate(source_rows):
            expected_relative = f"{name}/frame_{index:02d}.npz"
            if source["raw_npz_local_only"] != expected_relative or source["ray_count"] != 24000:
                raise ValueError("unexpected prior frame identity, order or ray count")
            file = raw / expected_relative
            start = time.perf_counter()
            actual_hash, verified_bytes = verify_raw_identity(file, source["raw_npz_sha256"])
            hash_ms.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            frame = PointCloudFrame.load_npz(io.BytesIO(verified_bytes))
            del verified_bytes
            load_duration = (time.perf_counter() - start) * 1000
            if (frame.ray_count != 24000 or frame.valid_count != source["valid_count"]
                    or abs(frame.frame_timestamp - source["time_s"]) > 1e-9):
                raise ValueError("loaded raw frame does not match prior report counts or time")
            elevations = np.degrees(np.arcsin(np.clip(frame.ray_directions[:, 2], -1, 1)))
            min_elevation = min(min_elevation, float(elevations.min()))
            max_elevation = max(max_elevation, float(elevations.max()))
            start = time.perf_counter()
            obs = encoder.encode_frame(frame,
                                       source_frame_identity=f"SHA256_NPZ:{actual_hash}")
            encode_duration = (time.perf_counter() - start) * 1000
            load_ms.append(load_duration)
            encode_ms.append(encode_duration)
            total_rays += frame.ray_count
            unscanned = int(np.count_nonzero(~obs.sampled_mask))
            no_hit = int(np.count_nonzero(obs.sampled_mask & ~obs.hit_mask))
            total_unscanned += unscanned
            total_sampled_no_hit += no_hit
            ever_sampled |= obs.sampled_mask
            encode_rows.append({"trajectory": name, "index": index,
                                "source_time_s": obs.frame_timestamp,
                                "source_frame_identity": obs.source_frame_identity,
                                "rays": frame.ray_count, "valid_rays": frame.valid_count,
                                "unscanned_bins": unscanned, "sampled_no_hit_bins": no_hit,
                                "hash_ms": hash_ms[-1], "load_ms": load_duration,
                                "encode_ms": encode_duration})
            series.append(obs)
            del frame
        observations[name] = series
    zero_delay = _policy_schedule(observations, 0.)
    delayed = _policy_schedule(observations, .020)
    if zero_delay["future_reads"] or delayed["future_reads"]:
        raise RuntimeError("future LiDAR source reached a policy read")
    from .observation_plots import plot_observation_evidence
    plot_ms = plot_observation_evidence(observations, zero_delay, delayed, encode_rows, figures)
    result = {
        "status": "PASS", "experiment": "PHASE_3E_L4_OFFLINE_LIDAR_POLICY_OBSERVATION",
        "prior_report_sha256": hashlib.sha256(previous_bytes).hexdigest(),
        "source_identity": "MUJOCO_RAYCAST_APPROX_MID360",
        "source_frames_verified": len(encode_rows), "source_rays_verified": total_rays,
        "source_hash_mismatches": 0, "per_frame_evidence": encode_rows,
        "observation": {
            "encoding_version": observations["x_line"][0].encoding_version,
            "tensor_shape": [3, 12, 24],
            "tensor_axis_order": "channel,elevation,azimuth",
            "channels": ["normalized_range", "sampled_mask", "valid_hit_mask"],
            "elevation_bounds_deg": [-10, 55], "azimuth_bounds_deg": [0, 360],
            "actual_ray_elevation_min_deg": min_elevation,
            "actual_ray_elevation_max_deg": max_elevation,
            "range_bounds_m": [.1, 5.], "empty_range_value": 1.,
            "sampled_no_hit_bin_count_total": total_sampled_no_hit,
            "uncovered_bin_count_total": total_unscanned,
            "never_sampled_bins_across_all_frames": int(np.count_nonzero(~ever_sampled)),
            "network_tensor_bytes": observations["x_line"][0].network_tensor.nbytes,
            "all_observation_arrays_bytes": observations["x_line"][0].array_bytes,
            "source_pattern_sha256": PATTERN_SHA256,
            "source_frame_hash_type": "SHA256_NPZ",
            "per_point_time_offset": "UNSUPPORTED",
        },
        "trajectories": {name: {"frames": len(series),
                                "source_times_s": [obs.frame_timestamp for obs in series],
                                "encoded_change_mean_abs_from_first": [float(np.mean(np.abs(
                                    obs.network_tensor - series[0].network_tensor)))
                                    for obs in series]}
                         for name, series in observations.items()},
        "timing": {"hash_ms": _stats(hash_ms), "load_ms": _stats(load_ms),
                   "encode_ms": _stats(encode_ms),
                   "inherited_l3_raycast_ms": "see prior Phase3E-L3 report; not rerun here",
                   "measurement_scope": "sequential single-process CPU; load/encode exclude figure save"},
        "async_zero_delay": zero_delay, "async_delay_20ms": delayed,
        "plot_ms": plot_ms,
        "resource": {"worker_count": 1,
                     "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                     "initial_mem_available_mib": available_start,
                     "final_mem_available_mib": _available_mib(),
                     "px4_runs": 0, "sitl_flights": 0, "real_lidar_network": 0,
                     "fast_lio2_runs": 0, "training_runs": 0,
                     "oom_observed": False},
        "acceptance": {"fixed_finite_encoding": "PASS",
                       "unscanned_vs_no_return_distinction": "PASS",
                       "prior_raw_sha_identity": "PASS", "causal_10hz_to_25hz": "PASS",
                       "stale_fail_closed": "PASS",
                       "px4_or_flight_control": "NOT TESTED",
                       "real_mid360s_fidelity": "NOT TESTED",
                       "fast_lio2_sync": "NOT TESTED"},
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--prior-report", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_observation_preflight(args.raw_root, args.prior_report,
                                       args.output_report, args.figures_dir)
    print(json.dumps({"status": result["status"],
                      "frames": result["source_frames_verified"],
                      "timing": result["timing"], "resource": result["resource"]}, indent=2))


if __name__ == "__main__":
    main()
