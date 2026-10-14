"""Physical scene identity, feasible seeded endpoints and real LiDAR obstacles."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from lidar_integration.scene import BASE_MODEL, BASE_SHA256, set_uav_pose


def test_five_seeded_scenes_preserve_original_plant_and_have_real_contact_geoms():
    from lidar_integration.navigation_scene import build_scene

    expected_interior = {"empty": 0, "front_box": 1, "side_pair": 2,
                         "detour": 2, "channel": 2}
    assert hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() == BASE_SHA256
    base = mujoco.MjModel.from_xml_path(str(BASE_MODEL))
    for layout, count in expected_interior.items():
        scene = build_scene(layout, 1101)
        assert scene.layout == layout
        assert len(scene.obstacle_geom_ids) == count
        assert scene.model.opt.timestep == pytest.approx(.002)
        assert mujoco.mj_getTotalmass(scene.model) == pytest.approx(7.)
        np.testing.assert_allclose(scene.model.body("mine_uav").ipos,
                                   base.body("mine_uav").ipos)
        np.testing.assert_allclose(scene.model.actuator_gear, base.actuator_gear)
        assert scene.model.site("lidar_site").id >= 0
        assert scene.model.geom("coarse_collision").id >= 0
        for geom_id in scene.obstacle_geom_ids:
            assert scene.model.geom_contype[geom_id] != 0
            assert scene.model.geom_conaffinity[geom_id] != 0
        assert len(scene.boundary_geom_ids) == 4


def test_scene_seeds_and_split_manifest_do_not_reuse_layout_identity():
    from lidar_integration.navigation_scene import build_scene, fixed_split_manifest

    a = build_scene("front_box", 1101)
    b = build_scene("front_box", 1101)
    c = build_scene("front_box", 2101)
    assert a.geometry_hash == b.geometry_hash
    assert a.geometry_hash != c.geometry_hash
    manifest = fixed_split_manifest()
    assert set(manifest) == {"train", "validation", "test"}
    seen = set()
    for split, rows in manifest.items():
        assert len(rows) == 5
        assert {row["layout"] for row in rows} == {
            "empty", "front_box", "side_pair", "detour", "channel"}
        for row in rows:
            identity = (row["scene_seed"], row["start_seed"], row["target_seed"])
            assert identity not in seen
            seen.add(identity)
            assert row["split"] == split
    hashes = [build_scene(row["layout"], row["scene_seed"]).geometry_hash
              for rows in manifest.values() for row in rows]
    assert len(set(hashes)) == 15


def test_seeded_start_and_target_are_feasible_and_reproducible():
    from lidar_integration.navigation_scene import build_scene, sample_start_target

    for layout in ("empty", "front_box", "side_pair", "detour", "channel"):
        scene = build_scene(layout, 1101)
        first = sample_start_target(scene, 1201, 1301)
        second = sample_start_target(scene, 1201, 1301)
        np.testing.assert_array_equal(first.start_xyz, second.start_xyz)
        np.testing.assert_array_equal(first.target_xyz, second.target_xyz)
        assert first.start_yaw_rad == second.start_yaw_rad
        assert first.start_xyz[2] > .7 and first.target_xyz[2] > .7
        assert not scene.overlaps_obstacle(first.start_xyz, clearance_m=.20)
        assert not scene.overlaps_obstacle(first.target_xyz, clearance_m=.20)
        assert np.linalg.norm(first.target_xyz - first.start_xyz) > 1.5


def test_real_cpu_rays_see_front_box_that_physically_collides():
    from lidar_integration.navigation_scene import build_scene
    from lidar_integration.scanner import Mid360Scanner

    upstream = Path("/tmp/mineuav-mujoco-lidar-audit")
    if not upstream.is_dir():
        pytest.skip("pinned upstream MuJoCo-LiDAR source unavailable")
    scene = build_scene("front_box", 1101)
    set_uav_pose(scene.model, scene.data, [0., 0., 1.], [1., 0., 0., 0.])
    frame = Mid360Scanner(scene.model, upstream).scan_next(scene.data, downsample=24)
    assert frame.ray_count == 1000
    box = scene.obstacle_aabbs[0]
    xyz = frame.world_points
    front_face = xyz[(abs(xyz[:, 0] - box[0][0]) < .02)
                     & (xyz[:, 1] > box[1][0]) & (xyz[:, 1] < box[1][1])
                     & (xyz[:, 2] > box[2][0]) & (xyz[:, 2] < box[2][1])]
    assert len(front_face) > 0


def test_each_initial_obstacle_has_real_lidar_returns_in_fixed_layouts():
    from lidar_integration.navigation_scene import build_scene, sample_start_target
    from lidar_integration.scanner import Mid360Scanner

    upstream = Path("/tmp/mineuav-mujoco-lidar-audit")
    if not upstream.is_dir():
        pytest.skip("pinned upstream MuJoCo-LiDAR source unavailable")
    for layout in ("front_box", "side_pair", "detour", "channel"):
        scene = build_scene(layout, 1001)
        endpoints = sample_start_target(scene, 1101, 1201)
        set_uav_pose(scene.model, scene.data, endpoints.start_xyz, [1., 0., 0., 0.])
        frame = Mid360Scanner(scene.model, upstream).scan_next(scene.data, downsample=1)
        hits = frame.world_points
        for bounds in scene.obstacle_aabbs:
            low = np.array([axis[0] for axis in bounds])
            high = np.array([axis[1] for axis in bounds])
            assert np.any(np.all((hits >= low - .02) & (hits <= high + .02), axis=1)), layout


def test_invalid_layout_or_seed_fails_instead_of_reusing_a_scene():
    from lidar_integration.navigation_scene import build_scene

    with pytest.raises(ValueError):
        build_scene("unknown", 1)
    with pytest.raises(ValueError):
        build_scene("empty", -1)
