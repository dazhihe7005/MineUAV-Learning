"""Real MuJoCo horizontal-ray evidence for an ideal uniform vertical shaft."""

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pytest

from lidar_integration.scanner import Mid360Scanner
from lidar_integration.scene import BASE_MODEL, set_uav_pose


UPSTREAM = Path(os.environ["MINEUAV_LIDAR_UPSTREAM"])


def test_uniform_shaft_horizontal_ranges_are_z_invariant_but_hits_move_in_world_z():
    from lidar_integration.scene import make_shaft_scene

    model, data = make_shaft_scene()
    scanner = Mid360Scanner(model, UPSTREAM)
    angles = np.array([0, math.pi/2, math.pi, -math.pi/2])
    low = scanner.scan_angles(data, angles, np.zeros(4))
    set_uav_pose(model, data, [0, 0, 1.7], [1, 0, 0, 0])
    high = scanner.scan_angles(data, angles, np.zeros(4))
    np.testing.assert_allclose(low.ranges, [2.4, 2.5, 2.6, 2.5], atol=1e-6)
    np.testing.assert_allclose(high.ranges, low.ranges, atol=1e-6)
    np.testing.assert_allclose(high.local_points, low.local_points, atol=1e-6)
    np.testing.assert_allclose(high.world_points[:, :2], low.world_points[:, :2], atol=1e-6)
    np.testing.assert_allclose(high.world_points[:, 2] - low.world_points[:, 2],
                               np.ones(4), atol=1e-6)
    assert hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() == (
        "cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520")


def test_shaft_has_no_upper_cap_and_self_echo_remains_excluded():
    from lidar_integration.scene import make_shaft_scene

    model, data = make_shaft_scene(include_self_marker=True)
    scanner = Mid360Scanner(model, UPSTREAM)
    frame = scanner.scan_angles(data, [0, 0], [0, math.pi/2])
    assert frame.ranges[0] == pytest.approx(2.4, abs=1e-6)
    assert frame.ranges[1] == -1
    assert not frame.valid_mask[1]


@pytest.mark.parametrize("sides,radius,height", [(3, 2.5, 6), (24, 0, 6),
                                                (24, 2.5, -1), (24, float("nan"), 6)])
def test_invalid_shaft_geometry_rejected(sides, radius, height):
    from lidar_integration.scene import make_shaft_scene

    with pytest.raises(ValueError):
        make_shaft_scene(sides=sides, radius_m=radius, height_m=height)
