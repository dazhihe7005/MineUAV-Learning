"""Derive a LiDAR test scene in memory without editing the historical MJCF."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


MODELS = Path(__file__).resolve().parents[1] / "models"
BASE_MODEL = MODELS / "mine_uav_dynamics_v2.xml"
BASE_SHA256 = "cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520"
SITE_POS_BODY_FLU_M = (0.10, 0.0, 0.10)
SITE_RPY_BODY_RAD = (0.0, 0.0, 0.0)
EXTRINSIC_PROVENANCE = "SIM_CONFIG"


def _geom(parent: ET.Element, name: str, pos: str, size: str) -> None:
    ET.SubElement(parent, "geom", {"name": name, "type": "box", "pos": pos,
                                   "size": size, "rgba": "0.45 0.5 0.55 1"})


def make_scene(*, include_box: bool = True,
               include_self_marker: bool = False) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile a static geometry scene around the exact original vehicle.

    Only the in-memory XML copy gains a SIM_CONFIG LiDAR site and obstacle geoms.
    The original inertia, motors, time step and source file remain untouched.
    """
    if hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() != BASE_SHA256:
        raise RuntimeError("historical UAV MJCF identity changed")
    root = ET.parse(BASE_MODEL).getroot()
    root.find("compiler").set("meshdir", str(MODELS.parent / "assets"))
    include = root.find("include")
    if include is None or include.get("file") != "rotor_actuators.xml":
        raise RuntimeError("unexpected rotor include")
    root.remove(include)
    rotor_root = ET.parse(MODELS / "rotor_actuators.xml").getroot()
    root.append(rotor_root.find("actuator"))

    world = root.find("worldbody")
    body = world.find("body[@name='mine_uav']")
    ET.SubElement(body, "site", {"name": "lidar_site", "pos": "0.10 0 0.10",
                                "euler": "0 0 0", "size": "0.003",
                                "rgba": "0.9 0.1 0.9 1"})
    # Inner faces: x=3.0, y=+2.0 and y=-2.0; all rise from z=0 to 3m.
    _geom(world, "front_wall", "3.05 0 1.5", "0.05 2 1.5")
    _geom(world, "left_wall", "0 2.05 1.5", "3 0.05 1.5")
    _geom(world, "right_wall", "0 -2.05 1.5", "3 0.05 1.5")
    if include_box:
        _geom(world, "box_obstacle", "1.5 0.5 0.8", "0.25 0.25 0.25")
    if include_self_marker:
        ET.SubElement(body, "geom", {"name": "self_test_marker", "type": "sphere",
                                     "pos": "0.4 0 0.1", "size": "0.08", "mass": "0",
                                     "contype": "0", "conaffinity": "0"})
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def set_uav_pose(model: mujoco.MjModel, data: mujoco.MjData,
                 xyz: object, quat_wxyz: object) -> None:
    position = np.asarray(xyz, dtype=np.float64)
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position must be finite XYZ")
    if quat.shape != (4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat) - 1) > 1e-6:
        raise ValueError("quaternion must be finite unit wxyz")
    data.qpos[:3] = position
    data.qpos[3:7] = quat
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
