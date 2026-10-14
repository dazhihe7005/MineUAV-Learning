"""Fixed-size policy observation from local MID360-like MuJoCo ray returns.

No scene geometry, target, vehicle world pose, future frame, or estimator state
enters the encoder. This is not a calibrated real MID360S observation model.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .pointcloud import PointCloudFrame, SOURCE_IDENTITY
from .scanner import PATTERN_SHA256


ENCODING_VERSION = "ANGULAR_RANGE_V1"


def _frozen(value: np.ndarray, dtype: type) -> np.ndarray:
    # A merely non-writeable owning ndarray can be switched back to writeable
    # after it enters the asynchronous buffer. Immutable bytes prevent that.
    packed = np.array(value, dtype=dtype, copy=True, order="C")
    return np.frombuffer(packed.tobytes(), dtype=packed.dtype).reshape(packed.shape)


@dataclass(frozen=True)
class LidarObservation:
    """Network channels are [normalized closest range, sampled, valid hit]."""

    range_tensor: np.ndarray  # float32 [12,24], 1 means no measured hit
    sampled_mask: np.ndarray  # bool [12,24], at least one ray sampled this bin
    hit_mask: np.ndarray  # bool [12,24], at least one valid obstacle return
    sample_count: np.ndarray  # int32 [12,24]
    hit_count: np.ndarray  # int32 [12,24]
    network_tensor: np.ndarray  # float32 [3,12,24]
    frame_timestamp: float  # source capture time, not policy read time
    clock_domain: str
    source_identity: str
    encoding_configuration: Mapping[str, object]
    encoding_version: str
    source_frame_identity: str

    def __post_init__(self) -> None:
        expected = (("range_tensor", (12, 24), np.float32),
                    ("sampled_mask", (12, 24), np.bool_),
                    ("hit_mask", (12, 24), np.bool_),
                    ("sample_count", (12, 24), np.int32),
                    ("hit_count", (12, 24), np.int32),
                    ("network_tensor", (3, 12, 24), np.float32))
        for field, shape, dtype in expected:
            value = _frozen(getattr(self, field), dtype)
            if value.shape != shape or (np.issubdtype(value.dtype, np.number)
                                        and not np.isfinite(value).all()):
                raise ValueError(f"{field} must be finite with shape {shape}")
            object.__setattr__(self, field, value)
        if (not math.isfinite(self.frame_timestamp) or self.frame_timestamp < 0
                or self.clock_domain != "MUJOCO_SIM_TIME"
                or self.source_identity != SOURCE_IDENTITY
                or self.encoding_version != ENCODING_VERSION
                or not self.source_frame_identity):
            raise ValueError("unsupported observation time or provenance")
        config = {key: tuple(value) if isinstance(value, list) else value
                  for key, value in self.encoding_configuration.items()}
        object.__setattr__(self, "encoding_configuration", MappingProxyType(config))
        if (np.any(self.sample_count < 0) or np.any(self.hit_count < 0)
                or np.any(self.hit_count > self.sample_count)
                or not np.array_equal(self.sampled_mask, self.sample_count > 0)
                or not np.array_equal(self.hit_mask, self.hit_count > 0)
                or not np.allclose(self.network_tensor[0], self.range_tensor)
                or not np.array_equal(self.network_tensor[1], self.sampled_mask.astype(np.float32))
                or not np.array_equal(self.network_tensor[2], self.hit_mask.astype(np.float32))
                or np.any(self.range_tensor < 0) or np.any(self.range_tensor > 1)
                or np.any(self.range_tensor[~self.hit_mask] != 1)):
            raise ValueError("observation channels or masks disagree")

    @property
    def array_bytes(self) -> int:
        return sum(getattr(self, field).nbytes for field in
                   ("range_tensor", "sampled_mask", "hit_mask", "sample_count",
                    "hit_count", "network_tensor"))


class AngularRangeEncoder:
    """24 azimuth × 12 elevation bins over the *observed* mode envelope.

    Historical 66 frames have elevation [-7.2075,52.1641] degrees. The
    fixed [-10,55] degree grid encloses that range, not a full sphere.
    """

    azimuth_bins = 24
    elevation_bins = 12
    elevation_min_deg = -10.0
    elevation_max_deg = 55.0
    min_range_m = 0.1
    max_range_m = 5.0

    def bin_directions(self, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray(directions, dtype=np.float64)
        if (vector.ndim != 2 or vector.shape[1] != 3 or not len(vector)
                or not np.isfinite(vector).all()
                or not np.allclose(np.linalg.norm(vector, axis=1), 1, atol=1e-6)):
            raise ValueError("ray directions must be finite unit XYZ vectors")
        elevation = np.degrees(np.arcsin(np.clip(vector[:, 2], -1, 1)))
        if (np.any(elevation < self.elevation_min_deg - 1e-8)
                or np.any(elevation > self.elevation_max_deg + 1e-8)):
            raise ValueError("ray elevation outside pinned scan envelope")
        azimuth = np.mod(np.degrees(np.arctan2(vector[:, 1], vector[:, 0])), 360)
        rows = np.floor((elevation - self.elevation_min_deg)
                        / (self.elevation_max_deg - self.elevation_min_deg)
                        * self.elevation_bins).astype(np.int64)
        cols = np.floor(azimuth / 360 * self.azimuth_bins).astype(np.int64)
        return np.clip(rows, 0, self.elevation_bins - 1), np.clip(cols, 0, self.azimuth_bins - 1)

    def _validate_frame(self, frame: PointCloudFrame) -> None:
        config = frame.scan_configuration
        if (frame.source_identity != SOURCE_IDENTITY or frame.clock_domain != "MUJOCO_SIM_TIME"
                or config.get("mode") != "mid360" or config.get("backend") != "cpu"
                or config.get("per_point_time_offset") != "UNSUPPORTED"
                or config.get("pattern_sha256") != PATTERN_SHA256):
            raise ValueError("source scan mode or pattern identity mismatch")
        if (config.get("min_range_m") != self.min_range_m
                or config.get("max_range_m") != self.max_range_m
                or config.get("ray_count") != frame.ray_count):
            raise ValueError("scan range limits or ray count mismatch")
        if (frame.points_xyz.shape != (frame.ray_count, 3)
                or frame.ray_directions.shape != (frame.ray_count, 3)
                or frame.valid_mask.shape != (frame.ray_count,)
                or not np.isfinite(frame.ranges).all()
                or not np.isfinite(frame.ray_directions).all()
                or not np.array_equal(frame.valid_mask,
                                      (frame.ranges >= self.min_range_m)
                                      & (frame.ranges <= self.max_range_m))
                or not np.isfinite(frame.points_xyz[frame.valid_mask]).all()
                or not np.isnan(frame.points_xyz[~frame.valid_mask]).all()
                or not np.allclose(frame.points_xyz[frame.valid_mask],
                                   frame.ray_directions[frame.valid_mask]
                                   * frame.ranges[frame.valid_mask, None], atol=1e-6)):
            raise ValueError("point/ray order, range or valid mask mismatch")

    def encode_frame(self, frame: PointCloudFrame, *,
                     source_frame_identity: str | None = None) -> LidarObservation:
        self._validate_frame(frame)
        rows, cols = self.bin_directions(frame.ray_directions)
        shape = (self.elevation_bins, self.azimuth_bins)
        sample_count = np.zeros(shape, dtype=np.int32)
        hit_count = np.zeros(shape, dtype=np.int32)
        nearest = np.full(shape, self.max_range_m, dtype=np.float64)
        np.add.at(sample_count, (rows, cols), 1)
        valid = frame.valid_mask
        np.add.at(hit_count, (rows[valid], cols[valid]), 1)
        np.minimum.at(nearest, (rows[valid], cols[valid]), frame.ranges[valid])
        sampled = sample_count > 0
        hit = hit_count > 0
        normalized = ((nearest - self.min_range_m)
                      / (self.max_range_m - self.min_range_m)).astype(np.float32)
        normalized[~hit] = 1.0
        network = np.stack((normalized, sampled.astype(np.float32),
                            hit.astype(np.float32)), axis=0)
        if source_frame_identity is None:
            digest = hashlib.sha256()
            for array in (frame.points_xyz, frame.ranges, frame.valid_mask,
                          frame.ray_directions):
                digest.update(np.ascontiguousarray(array).tobytes())
            digest.update(str(frame.frame_timestamp).encode("ascii"))
            digest.update(json.dumps(frame.scan_configuration, sort_keys=True).encode("utf-8"))
            source_frame_identity = f"SHA256_CONTENT:{digest.hexdigest()}"
        return LidarObservation(
            range_tensor=normalized, sampled_mask=sampled, hit_mask=hit,
            sample_count=sample_count, hit_count=hit_count, network_tensor=network,
            frame_timestamp=frame.frame_timestamp, clock_domain=frame.clock_domain,
            source_identity=frame.source_identity, encoding_version=ENCODING_VERSION,
            source_frame_identity=source_frame_identity,
            encoding_configuration={
                "axis_order": "channels,elevation,azimuth", "channels":
                ["normalized_range", "sampled_mask", "valid_hit_mask"],
                "elevation_bounds_deg": [self.elevation_min_deg, self.elevation_max_deg],
                "azimuth_bounds_deg": [0.0, 360.0], "shape": [3, 12, 24],
                "range_bounds_m": [self.min_range_m, self.max_range_m],
                "range_normalization": "(clipped_closest_valid_m - 0.1)/4.9",
                "no_return_or_unsampled_range_value": 1.0,
                "source_pattern_sha256": PATTERN_SHA256,
                "per_point_time_offset": "UNSUPPORTED",
            })

    def encode_npz(self, path: str | Path, expected_sha256: str) -> LidarObservation:
        file = Path(path)
        if len(expected_sha256) != 64:
            raise ValueError("expected SHA256 must contain 64 hex digits")
        raw = file.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise ValueError("source NPZ SHA256 does not match manifest")
        try:
            # Decode the exact bytes whose hash was checked; do not reopen a
            # path that could have changed between identity and content reads.
            frame = PointCloudFrame.load_npz(io.BytesIO(raw))
        except (KeyError, zipfile.BadZipFile, EOFError, OSError) as exc:
            raise ValueError("incomplete or corrupt NPZ frame") from exc
        return self.encode_frame(frame, source_frame_identity=f"SHA256_NPZ:{actual}")
