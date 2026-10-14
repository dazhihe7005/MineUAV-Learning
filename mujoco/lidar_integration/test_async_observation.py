"""10 Hz scan arrivals and 25 Hz policy reads in one simulation clock."""

import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1] / "reports"
RAW = ROOT / "mid360s_dynamic_scanning_parts" / "run_02" / "x_line"
REPORT = ROOT / "mujoco_mid360s_dynamic_scanning.json"


@pytest.fixture(scope="module")
def observations():
    from lidar_integration.observation import AngularRangeEncoder

    if not (RAW / "frame_00.npz").is_file():
        pytest.skip("local-only real MuJoCo point clouds unavailable")
    rows = json.loads(REPORT.read_text())["trajectories"]["x_line"]["frames"]
    encoder = AngularRangeEncoder()
    return [encoder.encode_npz(RAW / f"frame_{i:02d}.npz", rows[i]["raw_npz_sha256"])
            for i in range(5)]


def test_10hz_frames_feed_only_current_or_past_25hz_policy_ticks(observations):
    from lidar_integration.async_observation import AsynchronousObservationBuffer

    cache = AsynchronousObservationBuffer(experiment_id="A", capacity=4)
    cache.push(observations[0], arrival_time_s=0.0, experiment_id="A")
    reads = [cache.read(t, experiment_id="A") for t in (0., .04, .08)]
    assert [item.observation.frame_timestamp for item in reads] == [0., 0., 0.]
    assert [item.is_new_for_policy for item in reads] == [True, False, False]
    assert [item.status for item in reads] == ["FRESH"] * 3
    cache.push(observations[1], arrival_time_s=.10, experiment_id="A")
    reads = [cache.read(t, experiment_id="A") for t in (.12, .16)]
    assert [round(item.observation.frame_timestamp, 2) for item in reads] == [.10, .10]
    assert [item.is_new_for_policy for item in reads] == [True, False]
    cache.push(observations[2], arrival_time_s=.20, experiment_id="A")
    assert cache.read(.20, experiment_id="A").observation is observations[2]


def test_arrival_delay_never_reveals_unreceived_frame(observations):
    from lidar_integration.async_observation import AsynchronousObservationBuffer

    cache = AsynchronousObservationBuffer(experiment_id="delay")
    cache.push(observations[0], arrival_time_s=0., experiment_id="delay")
    cache.push(observations[1], arrival_time_s=.13, experiment_id="delay")
    before = cache.read(.12, experiment_id="delay")
    after = cache.read(.16, experiment_id="delay")
    assert before.observation is observations[0]
    assert after.observation is observations[1]
    assert before.age_s == pytest.approx(.12)
    assert after.age_s == pytest.approx(.06)


def test_dropouts_age_out_without_silently_reusing_stale_payload(observations):
    from lidar_integration.async_observation import AsynchronousObservationBuffer

    cache = AsynchronousObservationBuffer(experiment_id="drop", stale_after_s=.15)
    assert cache.read(0., experiment_id="drop").status == "MISSING"
    cache.push(observations[0], arrival_time_s=0., experiment_id="drop")
    assert cache.read(.12, experiment_id="drop").status == "FRESH"
    stale = cache.read(.16, experiment_id="drop")
    assert stale.status == "STALE"
    assert stale.observation is None
    assert stale.source_frame_timestamp == 0.
    assert stale.age_s == pytest.approx(.16)
    assert cache.read(.40, experiment_id="drop").status == "STALE"  # consecutive lost frames
    cache.push(observations[4], arrival_time_s=.41, experiment_id="drop")
    assert cache.read(.44, experiment_id="drop").observation is observations[4]


def test_arrival_cannot_be_backdated_after_policy_already_advanced(observations):
    from lidar_integration.async_observation import AsynchronousObservationBuffer

    cache = AsynchronousObservationBuffer(experiment_id="causality")
    cache.push(observations[0], arrival_time_s=0., experiment_id="causality")
    cache.read(.40, experiment_id="causality")
    with pytest.raises(ValueError, match="backdated"):
        cache.push(observations[2], arrival_time_s=.21, experiment_id="causality")


def test_duplicate_arrival_does_not_refresh_age_and_out_of_order_fails(observations):
    from lidar_integration.async_observation import AsynchronousObservationBuffer

    cache = AsynchronousObservationBuffer(experiment_id="order")
    cache.push(observations[0], arrival_time_s=0., experiment_id="order")
    assert cache.push(observations[0], arrival_time_s=.10, experiment_id="order") is False
    assert cache.read(.16, experiment_id="order").status == "STALE"
    cache.push(observations[2], arrival_time_s=.21, experiment_id="order")
    with pytest.raises(ValueError, match="order|regress"):
        cache.push(observations[1], arrival_time_s=.22, experiment_id="order")
    with pytest.raises(ValueError, match="future|arrival"):
        cache.push(observations[3], arrival_time_s=.25, experiment_id="order")


def test_bounded_cache_and_explicit_experiment_reset(observations):
    from lidar_integration.async_observation import AsynchronousObservationBuffer

    cache = AsynchronousObservationBuffer(experiment_id="old", capacity=2)
    for i in range(3):
        cache.push(observations[i], arrival_time_s=i / 10, experiment_id="old")
    assert cache.frame_count == 2
    assert cache.cache_data_bytes <= 2 * observations[0].array_bytes
    assert cache.read(.24, experiment_id="old").observation is observations[2]
    with pytest.raises(ValueError, match="experiment"):
        cache.read(.28, experiment_id="new")
    cache.reset("new")
    assert cache.read(0., experiment_id="new").status == "MISSING"
    with pytest.raises(ValueError, match="experiment"):
        cache.push(observations[0], arrival_time_s=0., experiment_id="old")
    cache.push(observations[0], arrival_time_s=0., experiment_id="new")
    assert cache.read(0., experiment_id="new").is_new_for_policy


def test_bad_clock_and_nonmonotonic_policy_times_fail_closed(observations):
    from lidar_integration.async_observation import AsynchronousObservationBuffer

    cache = AsynchronousObservationBuffer(experiment_id="clock")
    with pytest.raises(ValueError, match="clock"):
        cache.push(observations[0], arrival_time_s=0., experiment_id="clock",
                   arrival_clock_domain="PX4_BOOT_TIME")
    cache.push(observations[0], arrival_time_s=0., experiment_id="clock")
    cache.read(.12, experiment_id="clock")
    with pytest.raises(ValueError, match="policy.*regress"):
        cache.read(.08, experiment_id="clock")
    with pytest.raises(ValueError, match="finite"):
        cache.read(float("nan"), experiment_id="clock")


def test_buffer_rejects_invalid_capacity_and_stale_threshold():
    from lidar_integration.async_observation import AsynchronousObservationBuffer

    with pytest.raises(ValueError):
        AsynchronousObservationBuffer(experiment_id="A", capacity=0)
    with pytest.raises(ValueError):
        AsynchronousObservationBuffer(experiment_id="A", stale_after_s=float("inf"))


def test_cached_observation_cannot_be_mutated_after_push(observations):
    from lidar_integration.async_observation import AsynchronousObservationBuffer

    original = observations[0]
    cache = AsynchronousObservationBuffer(experiment_id="immutable")
    cache.push(original, arrival_time_s=0., experiment_id="immutable")
    before = original.network_tensor.copy()
    for field in ("network_tensor", "range_tensor", "sampled_mask",
                  "hit_mask", "sample_count", "hit_count"):
        with pytest.raises(ValueError):
            getattr(original, field).flags.writeable = True
    with pytest.raises(TypeError):
        original.encoding_configuration["source_pattern_sha256"] = "forged"
    with pytest.raises(TypeError):
        original.encoding_configuration["shape"][0] = 999
    np.testing.assert_array_equal(cache.read(0., experiment_id="immutable")
                                  .observation.network_tensor, before)
