"""Small physical obstacle layouts built only in memory from the v2 UAV MJCF.

The inherited `coarse_collision` is a TEMPORARY_PLACEHOLDER central body box,
not a measured propeller/arm collision envelope. Geometry and extrinsic are
simulation configuration, never real MID360S calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import _base_tree, _compile


LAYOUTS = ("empty", "front_box", "side_pair", "detour", "channel")
_BOXES = {
    "empty": (),
    "front_box": ((.90, .00, 1., .24, .24, .35),),
    "side_pair": ((.95, -.68, 1., .24, .27, .35),
                  (.95, +.68, 1., .24, .27, .35)),
    "detour": ((.75, .00, 1., .22, .35, .35),
               (1.45, -.85, 1., .22, .50, .35)),
    "channel": ((1.00, -.68, 1., .50, .32, .50),
                (1.00, +.68, 1., .50, .32, .50)),
}


@dataclass(frozen=True)
class NavigationScene:
    model: mujoco.MjModel
    data: mujoco.MjData
    layout: str
    scene_seed: int
    obstacle_geom_ids: frozenset[int]
    boundary_geom_ids: frozenset[int]
    obstacle_aabbs: tuple[tuple[tuple[float, float], ...], ...]
    boundary_half_extent_m: float
    geometry_hash: str

    def overlaps_obstacle(self, xyz: object, *, clearance_m: float = 0.) -> bool:
        point = np.asarray(xyz, dtype=float)
        if (point.shape != (3,) or not np.isfinite(point).all()
                or not math.isfinite(clearance_m) or clearance_m < 0):
            raise ValueError("finite xyz and nonnegative clearance required")
        return any(all(low - clearance_m <= point[axis] <= high + clearance_m
                       for axis, (low, high) in enumerate(box))
                   for box in self.obstacle_aabbs)


@dataclass(frozen=True)
class NavigationEndpoints:
    start_xyz: np.ndarray
    target_xyz: np.ndarray
    start_yaw_rad: float
    start_seed: int
    target_seed: int


def _validated_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("scene/start/target seed must be a nonnegative integer")
    return seed


def _add_box(world: ET.Element, name: str, center: tuple[float, float, float],
             half_size: tuple[float, float, float], color: str) -> None:
    ET.SubElement(world, "geom", {
        "name": name, "type": "box", "pos": " ".join(map(str, center)),
        "size": " ".join(map(str, half_size)), "rgba": color,
        "contype": "1", "conaffinity": "1", "friction": "1 0.005 0.0001",
    })


def build_scene(layout: str, scene_seed: int,
                obstacle_variation: tuple[tuple[float, float, float], ...] | None = None) -> NavigationScene:
    """Compile five seeded layouts; never write or alter the original MJCF."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown navigation layout: {layout}")
    if obstacle_variation is not None:
        if len(obstacle_variation) != len(_BOXES[layout]) or any(
            len(item) != 3 or not all(math.isfinite(v) for v in item)
            or not (-.25 <= item[0] <= .25 and -.30 <= item[1] <= .30
                    and .75 <= item[2] <= 1.25)
            for item in obstacle_variation
        ):
            raise ValueError("obstacle variation outside preflight geometry limits")
    seed = _validated_seed(scene_seed)
    rng = np.random.default_rng(seed)
    extent = 3.0 + float(rng.uniform(-.05, .05))
    boxes = []
    root, world = _base_tree()
    boundary = (
        ("nav_wall_x_pos", (extent + .05, 0., 1.5), (.05, extent + .10, 1.5)),
        ("nav_wall_x_neg", (-extent - .05, 0., 1.5), (.05, extent + .10, 1.5)),
        ("nav_wall_y_pos", (0., extent + .05, 1.5), (extent + .10, .05, 1.5)),
        ("nav_wall_y_neg", (0., -extent - .05, 1.5), (extent + .10, .05, 1.5)),
    )
    for name, center, half in boundary:
        _add_box(world, name, center, half, ".35 .4 .45 1")
    for index, template in enumerate(_BOXES[layout]):
        x, y, z, sx, sy, sz = template
        # Small fixed-seed layout variation; the designed gap remains open.
        if obstacle_variation is None:
            center = (x + float(rng.uniform(-.025, .025)),
                      y + float(rng.uniform(-.025, .025)), z)
            size = (sx, sy, sz)
        else:
            dx, dy, scale = obstacle_variation[index]
            center = (x + dx, y + dy, z)
            size = (sx * scale, sy * scale, sz * scale)
        _add_box(world, f"nav_obstacle_{index}", center, size, ".75 .25 .15 1")
        boxes.append(tuple((center[axis] - size[axis], center[axis] + size[axis])
                           for axis in range(3)))
    model, data = _compile(root)
    geom_ids = frozenset(model.geom(f"nav_obstacle_{i}").id for i in range(len(boxes)))
    boundary_ids = frozenset(model.geom(item[0]).id for item in boundary)
    config = {"layout": layout, "scene_seed": seed, "boundary_half_extent_m": extent,
              "obstacle_aabbs": boxes}
    if obstacle_variation is not None:
        config["obstacle_variation"] = obstacle_variation
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return NavigationScene(model, data, layout, seed, geom_ids, boundary_ids,
                           tuple(boxes), extent, digest)


def sample_start_target(scene: NavigationScene, start_seed: int,
                        target_seed: int) -> NavigationEndpoints:
    """Seeded nearby endpoints in the existing target range, outside all boxes."""
    srng = np.random.default_rng(_validated_seed(start_seed))
    trng = np.random.default_rng(_validated_seed(target_seed))
    start = np.array([srng.uniform(-.04, .04), srng.uniform(-.04, .04),
                      srng.uniform(.97, 1.03)])
    target = np.array([trng.uniform(1.81, 1.89), trng.uniform(-.04, .04),
                       trng.uniform(.97, 1.03)])
    yaw = float(srng.uniform(-.05, .05))
    if (scene.overlaps_obstacle(start, clearance_m=.20)
            or scene.overlaps_obstacle(target, clearance_m=.20)):
        raise ValueError("seeded endpoint overlaps obstacle")
    return NavigationEndpoints(start, target, yaw, start_seed, target_seed)


def fixed_split_manifest() -> dict[str, list[dict]]:
    """Small preflight split identity; no training or test-driven tuning."""
    groups: dict[str, list[dict]] = {}
    for split, base in (("train", 1000), ("validation", 2000), ("test", 3000)):
        groups[split] = [
            {"split": split, "layout": layout, "scene_seed": base + index + 1,
             "start_seed": base + 100 + index + 1,
             "target_seed": base + 200 + index + 1}
            for index, layout in enumerate(LAYOUTS)
        ]
    return groups
