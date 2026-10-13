"""Pinned MuJoCo-LiDAR CPU adapter; no PX4, ROS, or GPU dependency."""

from __future__ import annotations

import hashlib
import math
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from .pointcloud import PointCloudFrame


UPSTREAM_COMMIT = "e6b879df4b96daaed30ef10fe02650d29048fa9b"
PATTERN_SHA256 = "9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3"
PATTERN_REL = Path("src/mujoco_lidar/scan_mode/mid360.npy")
_SOURCE_FILES = ("src/mujoco_lidar/scan_gen.py", "src/mujoco_lidar/lidar_wrapper.py",
                 "src/mujoco_lidar/core_cpu/mjlidar_cpu.py", str(PATTERN_REL))


def validate_upstream(source_root: Path) -> Path:
    root = Path(source_root).resolve()
    pattern = root / PATTERN_REL
    if not pattern.is_file():
        raise ValueError("pinned MID360 angle file missing")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    if commit.returncode != 0 or commit.stdout.strip() != UPSTREAM_COMMIT:
        raise ValueError("MuJoCo-LiDAR source commit is not pinned version")
    diff = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--",
                           *_SOURCE_FILES], check=False)
    if diff.returncode != 0:
        raise ValueError("pinned MuJoCo-LiDAR runtime files are modified")
    if hashlib.sha256(pattern.read_bytes()).hexdigest() != PATTERN_SHA256:
        raise ValueError("MID360 angle file SHA256 mismatch")
    return root


class Mid360Scanner:
    """Approximate 10 Hz frame scanner using unchanged upstream CPU rays."""

    backend = "cpu"

    def __init__(self, model: mujoco.MjModel, upstream_root: str | Path,
                 min_range_m: float = 0.1, max_range_m: float = 5.0) -> None:
        root = validate_upstream(Path(upstream_root))
        import mujoco_lidar
        from mujoco_lidar import MjLidarWrapper, scan_gen

        imported = Path(mujoco_lidar.__file__).resolve()
        if root / "src" not in imported.parents or mujoco_lidar.__version__ != "0.3.5":
            raise ValueError("imported MuJoCo-LiDAR does not match pinned source")
        minimum, maximum = float(min_range_m), float(max_range_m)
        if not (math.isfinite(minimum) and math.isfinite(maximum)
                and 0 < minimum < maximum):
            raise ValueError("range limits must be finite and ordered")
        self.model = model
        self.min_range_m = minimum
        self.max_range_m = maximum
        self.site_id = model.site("lidar_site").id
        self.body_id = model.body("mine_uav").id
        if model.site_bodyid[self.site_id] != self.body_id:
            raise ValueError("lidar_site must be attached to mine_uav body")
        self.wrapper = MjLidarWrapper(model, "lidar_site", backend="cpu",
                                     cutoff_dist=maximum,
                                     args={"bodyexclude": self.body_id})
        self.generator = scan_gen.LivoxGenerator("mid360")
        self.pattern_samples = int(self.generator.samples)
        self._last_generated_time = -math.inf
        self.last_trace_ms = 0.0
        self.last_conversion_ms = 0.0

    def scan_angles(self, data: mujoco.MjData, theta: object, phi: object,
                    *, downsample: int | None = None) -> PointCloudFrame:
        theta = np.asarray(theta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        if (theta.ndim != 1 or phi.shape != theta.shape or theta.size == 0
                or not np.isfinite(theta).all() or not np.isfinite(phi).all()
                or np.any(np.abs(theta) > 2 * math.pi + 1e-6)
                or np.any(np.abs(phi) > math.pi / 2 + 1e-6)):
            raise ValueError("ray angle vectors invalid")
        if not math.isfinite(float(data.time)) or data.time < 0:
            raise ValueError("MuJoCo simulation time invalid")
        start = time.perf_counter()
        distances = np.asarray(self.wrapper.trace_rays(data, theta, phi), dtype=np.float64).copy()
        trace_end = time.perf_counter()
        if distances.shape != theta.shape or not np.isfinite(distances).all():
            raise RuntimeError("MuJoCo ray backend returned invalid distances")
        directions = np.stack((np.cos(phi) * np.cos(theta),
                               np.cos(phi) * np.sin(theta), np.sin(phi)), axis=-1)
        valid = (distances >= self.min_range_m) & (distances <= self.max_range_m)
        points = np.full((theta.size, 3), np.nan, dtype=np.float64)
        points[valid] = directions[valid] * distances[valid, None]
        sensor_pose = np.eye(4)
        sensor_pose[:3, :3] = self.wrapper.sensor_rotation
        sensor_pose[:3, 3] = self.wrapper.sensor_position
        T_body_lidar = np.eye(4)
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, self.model.site_quat[self.site_id])
        T_body_lidar[:3, :3] = rot.reshape(3, 3)
        T_body_lidar[:3, 3] = self.model.site_pos[self.site_id]
        frame = PointCloudFrame(
            frame_timestamp=float(data.time), clock_domain="MUJOCO_SIM_TIME",
            sensor_frame_id="lidar_site", points_xyz=points, ranges=distances,
            valid_mask=valid, ray_directions=directions,
            sensor_pose=sensor_pose, T_body_lidar=T_body_lidar,
            extrinsic_provenance="SIM_CONFIG",
            scan_configuration={"mode": "mid360" if downsample is not None else "DIAGNOSTIC_RAYS",
                                "backend": "cpu", "downsample": downsample,
                                "ray_count": int(theta.size), "min_range_m": self.min_range_m,
                                "max_range_m": self.max_range_m,
                                "per_point_time_offset": "UNSUPPORTED",
                                "pattern_sha256": PATTERN_SHA256},
        )
        self.last_trace_ms = (trace_end - start) * 1000
        self.last_conversion_ms = (time.perf_counter() - trace_end) * 1000
        return frame

    def scan_next(self, data: mujoco.MjData, downsample: int = 1) -> PointCloudFrame:
        if (isinstance(downsample, bool) or not isinstance(downsample, int)
                or downsample < 1 or downsample > self.pattern_samples):
            raise ValueError("downsample must be an integer in [1, 24000]")
        if not math.isfinite(float(data.time)) or data.time <= self._last_generated_time:
            raise ValueError("generated frame timestamps must strictly increase")
        theta, phi = self.generator.sample_ray_angles(downsample=downsample)
        result = self.scan_angles(data, theta, phi, downsample=downsample)
        self._last_generated_time = result.frame_timestamp
        return result
