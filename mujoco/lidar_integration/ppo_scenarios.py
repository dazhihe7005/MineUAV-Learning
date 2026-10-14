"""Pre-registered, bounded scene splits; geometry is never a policy input."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque

import numpy as np

from .navigation_scene import LAYOUTS, NavigationScene, _BOXES, build_scene


def build_ppo_scene(row: dict) -> NavigationScene:
    return build_scene(row["layout"], int(row["scene_seed"]),
                       tuple(tuple(float(v) for v in item)
                             for item in row["obstacle_variation"]))


def path_exists(scene: NavigationScene, start: object, target: object,
                *, clearance_m: float = .20, grid_m: float = .10) -> bool:
    """Independent 2D AABB grid reachability at the nominal flight altitude.

    This rejects fully blocked plans; it is not a flight path or policy input.
    """
    a, b = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite start and target xyz required")
    if abs(a[2] - b[2]) > .2:
        return False
    extent = scene.boundary_half_extent_m - clearance_m
    width = int(2 * extent / grid_m) + 1
    def cell(xyz):
        return tuple(int(round((float(v) + extent) / grid_m)) for v in xyz[:2])
    first, last = cell(a), cell(b)
    def valid(index):
        ix, iy = index
        if not (0 <= ix < width and 0 <= iy < width):
            return False
        x, y = ix * grid_m - extent, iy * grid_m - extent
        if abs(x) > extent or abs(y) > extent:
            return False
        return not scene.overlaps_obstacle([x, y, a[2]], clearance_m=clearance_m)
    if not valid(first) or not valid(last):
        return False
    pending = deque([first])
    seen = {first}
    while pending:
        current = pending.popleft()
        if current == last:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = current[0] + dx, current[1] + dy
            if nxt not in seen and valid(nxt):
                seen.add(nxt)
                pending.append(nxt)
    return False


def make_scenario_manifest() -> dict[str, list[dict]]:
    """Separate 12/6/6 identities. No Test observation is consulted by training."""
    result = {}
    for split, base, count in (("train", 41000, 12),
                               ("validation", 51000, 6), ("test", 61000, 6)):
        result[split] = []
        for index in range(count):
            seed = base + index
            rng = np.random.default_rng(seed)
            layout = LAYOUTS[index % len(LAYOUTS)]
            variation = [[round(float(rng.uniform(-.20, .20)), 7),
                          round(float(rng.uniform(-.07 if layout == "channel" else -.22,
                                                  .07 if layout == "channel" else .22)), 7),
                          round(float(rng.uniform(.8, 1.05 if layout == "channel" else 1.2)), 7)]
                         for _ in _BOXES[layout]]
            row = {
                "split": split, "layout": layout, "scene_seed": seed,
                "start_seed": seed + 100000, "target_seed": seed + 200000,
                "start_xyz": [round(float(rng.uniform(-.20, .10)), 7),
                              round(float(rng.uniform(-.30, .30)), 7), 1.],
                "target_xyz": [round(float(rng.uniform(1.93, 1.99)), 7),
                               round(float(rng.uniform(-.18, .18)), 7), 1.],
                "start_yaw_rad": round(float(rng.uniform(-math.pi, math.pi)), 7),
                "obstacle_variation": variation,
            }
            scene = build_ppo_scene(row)
            if (scene.overlaps_obstacle(row["start_xyz"], clearance_m=.20)
                    or scene.overlaps_obstacle(row["target_xyz"], clearance_m=.20)
                    or not path_exists(scene, row["start_xyz"], row["target_xyz"])):
                raise ValueError(f"seeded scenario is geometrically infeasible: {seed}")
            row["obstacle_configuration_hash"] = scene.geometry_hash
            row["identity_sha256"] = hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            result[split].append(row)
    return result
