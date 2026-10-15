"""Bounded real-physics preflight emits only truthful metrics and figures."""

from research_regression.upstream import resolve_upstream
import json
from pathlib import Path

import pytest


UPSTREAM = resolve_upstream()


def test_contact_acceptance_never_passes_without_actual_matching_contact():
    from lidar_integration.navigation_runner import classify_contact_failure

    assert classify_contact_failure([]) == "NOT TESTED"
    front = {"layout": "front_box", "collision": False,
             "collision_kind": None, "contact_pairs": []}
    assert classify_contact_failure([front]) == "FAIL"
    front.update(collision=True, collision_kind="obstacle")
    assert classify_contact_failure([front]) == "FAIL"
    front["contact_pairs"] = [("nav_obstacle_0", "coarse_collision")]
    assert classify_contact_failure([front]) == "PASS"


def test_small_preflight_produces_real_episode_evidence_and_seven_figures(tmp_path):
    from lidar_integration.navigation_runner import run_preflight

    if not UPSTREAM.is_dir():
        pytest.skip("pinned MuJoCo-LiDAR checkout unavailable")
    output = tmp_path / "report.json"
    figures = tmp_path / "figures"
    result = run_preflight(output, figures, UPSTREAM, layouts=("empty",))
    assert output.is_file()
    assert json.loads(output.read_text())["episode_count"] == 1
    assert result["episode_count"] == 1
    assert result["status"] == "NOT TESTED"
    assert result["acceptance"]["contact_failure"] == "NOT TESTED"
    assert result["physics_ticks_total"] > 0
    assert result["lidar_frames_total"] > 0
    assert result["upstream"]["ray_count_per_frame"] == 24000
    assert result["frequencies_hz"] == {"physics": 500, "pi": 100,
                                        "policy": 25, "lidar": 10}
    assert result["resource"]["px4_runs"] == 0
    assert result["resource"]["training_updates"] == 0
    assert all(row["physics_ticks"] > 0 for row in result["episodes"])
    assert all(row["termination_reason"] in {"success", "collision", "time_limit",
                                                   "sensor_failure", "physical_failure"}
               for row in result["episodes"])
    expected = {"obstacle_topdown.png", "real_lidar_pointcloud.png",
                "physical_trajectory.png", "distance_vs_time.png",
                "reward_breakdown.png", "lidar_policy_timeline.png",
                "environment_step_time.png"}
    assert {path.name for path in figures.glob("*.png")} == expected
    assert all((figures / name).read_bytes().startswith(b"\x89PNG") for name in expected)
    parts = tmp_path / "mujoco_lidar_navigation_env_parts" / "preflight_run_01"
    import numpy as np
    with np.load(parts / "empty.npz") as raw:
        assert len(raw["step_wall_ms"]) == result["episodes"][0]["steps"]
        assert len(raw["scan_wall_ms"]) == result["episodes"][0]["lidar_frames"]
        assert len(raw["encode_wall_ms"]) == result["episodes"][0]["lidar_frames"]
        assert np.sum(raw["reward_component_progress"]) == pytest.approx(
            result["episodes"][0]["reward_component_sums"]["progress"])
    with np.load(parts / "evidence_cloud.npz") as cloud:
        assert cloud["world_points"].ndim == 2
        assert cloud["world_points"].shape[1] == 3
        assert len(cloud["world_points"]) > 0


def test_preflight_refuses_overwrite_before_running_any_episode(tmp_path):
    from lidar_integration.navigation_runner import run_preflight

    output = tmp_path / "report.json"
    output.write_text("original")
    with pytest.raises(ValueError, match="report"):
        run_preflight(output, tmp_path / "figures", UPSTREAM)
    assert output.read_text() == "original"
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "historic.png").write_bytes(b"historic")
    with pytest.raises(ValueError, match="figure"):
        run_preflight(tmp_path / "new.json", figures, UPSTREAM)
    assert (figures / "historic.png").read_bytes() == b"historic"
    assert not (tmp_path / "new.json").exists()


def test_old_scene_geometry_and_model_hash_are_preserved_after_real_preflight(tmp_path):
    import hashlib
    from lidar_integration.navigation_runner import run_preflight
    from lidar_integration.scene import BASE_MODEL, BASE_SHA256

    if not UPSTREAM.is_dir():
        pytest.skip("pinned MuJoCo-LiDAR checkout unavailable")
    before = hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest()
    result = run_preflight(tmp_path / "report.json", tmp_path / "figures", UPSTREAM,
                           layouts=("front_box",))
    assert result["status"] == "PASS"
    assert result["acceptance"]["contact_failure"] == "PASS"
    assert before == hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() == BASE_SHA256
