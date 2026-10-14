"""Whole historical 66-frame offline observation audit, without new scans."""

import json
import io
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1] / "reports"
RAW = ROOT / "mid360s_dynamic_scanning_parts" / "run_02"
PRIOR = ROOT / "mujoco_mid360s_dynamic_scanning.json"


def test_historical_66_frame_preflight_and_six_real_observation_figures(tmp_path, monkeypatch):
    from lidar_integration.observation_runner import run_observation_preflight
    from lidar_integration.pointcloud import PointCloudFrame

    if not (RAW / "x_line" / "frame_00.npz").is_file():
        pytest.skip("local-only historical point clouds unavailable")
    report_file = tmp_path / "report.json"
    figure_dir = tmp_path / "figures"
    original = PointCloudFrame.load_npz
    decoded_sources = []

    def checked_decode(source):
        decoded_sources.append(source)
        return original(source)

    monkeypatch.setattr(PointCloudFrame, "load_npz", checked_decode)
    result = run_observation_preflight(RAW, PRIOR, report_file, figure_dir)
    assert len(decoded_sources) == 66
    assert all(isinstance(source, io.BytesIO) for source in decoded_sources)
    assert result["status"] == "PASS"
    assert result["source_frames_verified"] == 66
    assert result["source_rays_verified"] == 66 * 24000
    assert result["source_hash_mismatches"] == 0
    assert len(result["trajectories"]) == 6
    assert result["observation"]["tensor_shape"] == [3, 12, 24]
    assert result["observation"]["sampled_no_hit_bin_count_total"] > 0
    assert result["observation"]["uncovered_bin_count_total"] > 0
    assert result["timing"]["load_ms"]["p95"] >= 0
    assert result["timing"]["encode_ms"]["p95"] >= 0
    assert result["async_zero_delay"]["future_reads"] == 0
    assert result["async_zero_delay"]["policy_reads"] == 6 * 26
    assert result["async_zero_delay"]["stale_reads"] == 0
    assert result["async_zero_delay"]["max_cache_frames"] <= 4
    assert result["async_delay_20ms"]["source"] == "SYNTHETIC_OFFLINE_20MS_ARRIVAL_DELAY"
    assert result["resource"]["peak_rss_mib"] > 0
    assert result["resource"]["px4_runs"] == 0
    assert result["resource"]["training_runs"] == 0
    assert json.loads(report_file.read_text())["source_frames_verified"] == 66
    expected = {"angular_range.png", "valid_hit_mask.png", "unscanned_mask.png",
                "dynamic_encoding.png", "lidar_policy_timeline.png", "encoding_cpu_time.png"}
    assert {file.name for file in figure_dir.glob("*.png")} == expected
    assert all((figure_dir / name).read_bytes().startswith(b"\x89PNG") for name in expected)


def test_corrupt_hash_and_missing_raw_frame_fail_closed(tmp_path):
    from lidar_integration.observation_runner import verify_raw_identity, run_observation_preflight

    if not (RAW / "x_line" / "frame_00.npz").is_file():
        pytest.skip("local-only historical point clouds unavailable")
    with pytest.raises(ValueError, match="SHA256"):
        verify_raw_identity(RAW / "x_line" / "frame_00.npz", "0" * 64)
    with pytest.raises((ValueError, FileNotFoundError), match="missing|frame"):
        run_observation_preflight(tmp_path / "missing", PRIOR,
                                  tmp_path / "report.json", tmp_path / "figures")
    assert not (tmp_path / "report.json").exists()


def test_formal_report_and_figures_are_never_overwritten(tmp_path):
    from lidar_integration.observation_runner import run_observation_preflight

    report = tmp_path / "formal.json"
    report.write_text("historic")
    with pytest.raises(ValueError, match="report"):
        run_observation_preflight(RAW, PRIOR, report, tmp_path / "new_figures")
    assert report.read_text() == "historic"
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "old.png").write_bytes(b"historic")
    with pytest.raises(ValueError, match="figure"):
        run_observation_preflight(RAW, PRIOR, tmp_path / "new.json", figures)
    assert (figures / "old.png").read_bytes() == b"historic"
    assert not (tmp_path / "new.json").exists()


def test_prior_report_identity_cannot_be_replaced_by_edited_manifest(tmp_path):
    from lidar_integration.observation_runner import run_observation_preflight

    prior = json.loads(PRIOR.read_text())
    prior["status"] = "FAIL"
    altered = tmp_path / "altered_prior.json"
    altered.write_text(json.dumps(prior))
    with pytest.raises(ValueError, match="prior report SHA256"):
        run_observation_preflight(RAW, altered, tmp_path / "formal.json",
                                  tmp_path / "figures")
    assert not (tmp_path / "formal.json").exists()
