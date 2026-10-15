"""End-to-end static scan artifact checks (real MuJoCo rays)."""

from research_regression.upstream import resolve_upstream
import json
import os
from pathlib import Path

import numpy as np
import pytest

from lidar_integration.runner import run_experiment


@pytest.fixture
def upstream_root() -> Path:
    return resolve_upstream()


def test_real_scan_artifacts_and_report(tmp_path: Path, upstream_root: Path):
    report = run_experiment(upstream_root, tmp_path, repetitions=2, draw=False,
                            downsample_factors=(24,))
    assert report["actual_scans"] == 3  # two benchmark frames plus one yaw-rotated frame
    assert report["scales"]["1000"]["ray_count"] == 1000
    assert report["scales"]["1000"]["valid_count_min"] > 0
    assert report["geometry_checks"]["z_extent_m"] > 0.5
    assert report["scales"]["1000"]["trace_ms"]["p50"] >= 0
    assert report["environment"]["mujoco_version"] == "3.14.0"
    assert report["model_preservation"]["base_sha256_unchanged"] is True
    assert report["acceptance"]["real_mujoco_ray_cloud"] == "PASS"
    assert report["acceptance"]["full_24000_ray_static_10hz_budget"] == "NOT TESTED"
    assert report["status"] == "NOT TESTED"  # this test intentionally omits full rays
    assert (tmp_path / "local_frame.npz").is_file()
    with np.load(tmp_path / "local_frame.npz", allow_pickle=False) as saved:
        assert saved["valid_mask"].sum() > 0
    assert json.loads((tmp_path / "report.json").read_text())["actual_scans"] == 3
