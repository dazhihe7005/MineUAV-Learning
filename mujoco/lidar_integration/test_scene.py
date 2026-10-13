"""Real-MuJoCo isolated scene invariants (no PX4 or flight)."""

import hashlib
import importlib
import importlib.metadata
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest


def test_real_mujoco_distribution_not_repository_namespace():
    assert importlib.metadata.version("mujoco") == "3.14.0"
    assert "site-packages/mujoco/__init__.py" in str(Path(mujoco.__file__).resolve())
    assert hasattr(mujoco, "MjModel") and hasattr(mujoco, "mj_multiRay")


def scene_api_module():
    assert importlib.util.find_spec("lidar_integration.scene") is not None, "scene module missing"
    return importlib.import_module("lidar_integration.scene")


def test_scene_preserves_original_vehicle_physics_and_adds_only_isolated_geometry():
    scene_api = scene_api_module()
    BASE_MODEL = scene_api.BASE_MODEL
    make_scene = scene_api.make_scene
    assert hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() == (
        "cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520"
    )
    base = mujoco.MjModel.from_xml_path(str(BASE_MODEL))
    model, data = make_scene()
    base_body = base.body("mine_uav")
    body = model.body("mine_uav")
    assert body.mass == base_body.mass == 7.0
    np.testing.assert_allclose(body.ipos, base_body.ipos, atol=0)
    np.testing.assert_allclose(body.inertia, base_body.inertia, atol=0)
    np.testing.assert_allclose(model.actuator_gear, base.actuator_gear, atol=0)
    np.testing.assert_allclose(model.actuator_ctrlrange, base.actuator_ctrlrange, atol=0)
    assert model.opt.timestep == base.opt.timestep == 0.002
    assert model.nu == base.nu == 4
    np.testing.assert_allclose(data.site("lidar_site").xpos, [0.1, 0.0, 0.8], atol=1e-12)
    assert model.geom("front_wall").id >= 0
    assert model.geom("left_wall").id >= 0
    assert model.geom("right_wall").id >= 0
    assert model.geom("box_obstacle").id >= 0


def test_scene_site_moves_and_rotates_with_vehicle():
    scene_api = scene_api_module()
    make_scene, set_uav_pose = scene_api.make_scene, scene_api.set_uav_pose
    model, data = make_scene()
    set_uav_pose(model, data, [1, 2, 1], [2**-0.5, 0, 0, 2**-0.5])
    np.testing.assert_allclose(data.site("lidar_site").xpos, [1, 2.1, 1.1], atol=1e-10)
    np.testing.assert_allclose(data.site("lidar_site").xmat.reshape(3, 3) @ [1, 0, 0],
                               [0, 1, 0], atol=1e-10)


@pytest.mark.parametrize("xyz,quat", [([float("nan"), 0, 1], [1, 0, 0, 0]),
                                       ([0, 0, 1], [0, 0, 0, 0]),
                                       ([0, 0, 1], [2, 0, 0, 0])])
def test_invalid_pose_rejected(xyz, quat):
    scene_api = scene_api_module()
    make_scene, set_uav_pose = scene_api.make_scene, scene_api.set_uav_pose
    model, data = make_scene()
    with pytest.raises(ValueError):
        set_uav_pose(model, data, xyz, quat)
