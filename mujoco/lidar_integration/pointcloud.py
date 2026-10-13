"""Validated, ROS-independent point cloud frame for approximate MuJoCo rays."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SOURCE_IDENTITY = "MUJOCO_RAYCAST_APPROX_MID360"


def _rigid_pose(value: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be finite 4x4")
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-10)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or abs(np.linalg.det(rotation) - 1) > 1e-6):
        raise ValueError(f"{name} must be a proper rigid transform")
    return pose


@dataclass(frozen=True)
class PointCloudFrame:
    frame_timestamp: float
    clock_domain: str
    sensor_frame_id: str
    points_xyz: np.ndarray
    ranges: np.ndarray
    valid_mask: np.ndarray
    ray_directions: np.ndarray
    sensor_pose: np.ndarray
    T_body_lidar: np.ndarray
    extrinsic_provenance: str
    scan_configuration: dict
    source_identity: str = SOURCE_IDENTITY

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xyz, dtype=np.float64)
        ranges = np.asarray(self.ranges, dtype=np.float64)
        valid = np.asarray(self.valid_mask)
        directions = np.asarray(self.ray_directions, dtype=np.float64)
        n = len(ranges)
        if (points.shape != (n, 3) or directions.shape != (n, 3)
                or valid.shape != (n,) or valid.dtype != np.bool_ or n == 0):
            raise ValueError("point/range/mask/direction shapes or mask dtype invalid")
        if (not np.isfinite(ranges).all() or not np.isfinite(directions).all()
                or not np.allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-6)):
            raise ValueError("ranges or directions invalid")
        if (not np.isfinite(points[valid]).all() or not np.isnan(points[~valid]).all()
                or np.any(ranges[valid] < 0)
                or np.any((ranges[~valid] != -1) & (ranges[~valid] < 0))):
            raise ValueError("valid points or no-return representation invalid")
        if not np.allclose(points[valid], directions[valid] * ranges[valid, None], atol=1e-6):
            raise ValueError("point coordinates disagree with ray distances")
        if not np.isfinite(self.frame_timestamp) or self.frame_timestamp < 0:
            raise ValueError("simulation timestamp invalid")
        if self.clock_domain != "MUJOCO_SIM_TIME" or self.sensor_frame_id != "lidar_site":
            raise ValueError("unsupported clock or sensor frame")
        if self.source_identity != SOURCE_IDENTITY or self.extrinsic_provenance != "SIM_CONFIG":
            raise ValueError("source and extrinsic provenance must remain explicit")
        if (self.scan_configuration.get("ray_count") != n
                or self.scan_configuration.get("per_point_time_offset") != "UNSUPPORTED"):
            raise ValueError("scan metadata must disclose ray count and unsupported timing")
        object.__setattr__(self, "points_xyz", points)
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "ray_directions", directions)
        object.__setattr__(self, "sensor_pose", _rigid_pose(self.sensor_pose, "sensor_pose"))
        object.__setattr__(self, "T_body_lidar", _rigid_pose(self.T_body_lidar, "T_body_lidar"))

    @property
    def ray_count(self) -> int:
        return int(self.ranges.size)

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def local_points(self) -> np.ndarray:
        return self.points_xyz[self.valid_mask]

    @property
    def world_points(self) -> np.ndarray:
        return self.local_points @ self.sensor_pose[:3, :3].T + self.sensor_pose[:3, 3]

    def save_npz(self, path: str | Path) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "frame_timestamp": self.frame_timestamp, "clock_domain": self.clock_domain,
            "sensor_frame_id": self.sensor_frame_id,
            "extrinsic_provenance": self.extrinsic_provenance,
            "scan_configuration": self.scan_configuration,
            "source_identity": self.source_identity,
        }
        with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".npz", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            np.savez_compressed(temporary, points_xyz=self.points_xyz, ranges=self.ranges,
                                valid_mask=self.valid_mask, ray_directions=self.ray_directions,
                                sensor_pose=self.sensor_pose, T_body_lidar=self.T_body_lidar,
                                metadata=np.array(json.dumps(metadata, sort_keys=True)))
            os.replace(temporary, dest)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load_npz(cls, path: str | Path) -> "PointCloudFrame":
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            arrays = {name: archive[name].copy() for name in
                      ("points_xyz", "ranges", "valid_mask", "ray_directions",
                       "sensor_pose", "T_body_lidar")}
        return cls(**metadata, **arrays)
