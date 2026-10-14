"""Causal bounded LiDAR-to-policy handoff; no policy or PX4 side effects."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .observation import LidarObservation


_CLOCK_EPS_S = 1e-9  # MuJoCo integrated 0.1s timestamps have ~1e-15s roundoff.


@dataclass(frozen=True)
class ObservationRead:
    status: str  # FRESH, STALE or MISSING
    observation: LidarObservation | None
    age_s: float | None
    source_frame_timestamp: float | None
    arrival_time_s: float | None
    is_new_for_policy: bool


class AsynchronousObservationBuffer:
    """Latest arrived source sample at each policy tick; stale payload is withheld.

    All times share MUJOCO_SIM_TIME in this offline experiment. The source
    timestamp is never replaced by arrival or policy time. This class does
    not authorize vehicle control or certify a hardware safety deadline.
    """

    def __init__(self, *, experiment_id: str, capacity: int = 4,
                 stale_after_s: float = .15,
                 clock_domain: str = "MUJOCO_SIM_TIME") -> None:
        if (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1
                or not math.isfinite(stale_after_s) or stale_after_s <= 0
                or clock_domain != "MUJOCO_SIM_TIME"):
            raise ValueError("invalid bounded LiDAR buffer config or clock domain")
        self.capacity = capacity
        self.stale_after_s = float(stale_after_s)
        self.clock_domain = clock_domain
        self._frames: deque[tuple[LidarObservation, float]] = deque(maxlen=capacity)
        self.reset(experiment_id)

    def reset(self, experiment_id: str) -> None:
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("experiment_id must be nonempty")
        self.experiment_id = experiment_id
        self._frames.clear()
        self._last_source_time: float | None = None
        self._last_arrival_time: float | None = None
        self._last_policy_time: float | None = None
        self._last_delivered_identity: str | None = None

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def cache_data_bytes(self) -> int:
        return sum(obs.array_bytes for obs, _ in self._frames)

    def _check_experiment(self, experiment_id: str) -> None:
        if experiment_id != self.experiment_id:
            raise ValueError("experiment identity mismatch; reset before reusing buffer")

    def push(self, observation: LidarObservation, *, arrival_time_s: float,
             experiment_id: str, arrival_clock_domain: str = "MUJOCO_SIM_TIME") -> bool:
        self._check_experiment(experiment_id)
        if not isinstance(observation, LidarObservation):
            raise TypeError("only encoded LidarObservation frames are accepted")
        if observation.clock_domain != self.clock_domain or arrival_clock_domain != self.clock_domain:
            raise ValueError("clock domain mismatch; no implicit time conversion")
        arrival = float(arrival_time_s)
        source = observation.frame_timestamp
        if not math.isfinite(arrival) or arrival < 0:
            raise ValueError("arrival time must be finite and nonnegative")
        if self._last_policy_time is not None and arrival < self._last_policy_time - _CLOCK_EPS_S:
            raise ValueError("backdated arrival after an already processed policy tick")
        if source > arrival + _CLOCK_EPS_S:
            raise ValueError("future source sample cannot arrive before capture")
        if self._last_source_time is not None:
            delta = source - self._last_source_time
            if abs(delta) <= _CLOCK_EPS_S:
                if observation.source_frame_identity == self._frames[-1][0].source_frame_identity:
                    return False  # duplicate does not refresh arrival or age
                raise ValueError("conflicting duplicate source timestamp")
            if delta < 0:
                raise ValueError("source timestamp order regressed")
        if self._last_arrival_time is not None and arrival < self._last_arrival_time - _CLOCK_EPS_S:
            raise ValueError("arrival timestamp order regressed")
        self._frames.append((observation, arrival))
        self._last_source_time = source
        self._last_arrival_time = arrival
        return True

    def read(self, policy_time_s: float, *, experiment_id: str) -> ObservationRead:
        self._check_experiment(experiment_id)
        policy_time = float(policy_time_s)
        if not math.isfinite(policy_time) or policy_time < 0:
            raise ValueError("policy time must be finite and nonnegative")
        if self._last_policy_time is not None and policy_time < self._last_policy_time - _CLOCK_EPS_S:
            raise ValueError("policy time regressed")
        self._last_policy_time = policy_time
        for obs, arrival in reversed(self._frames):
            if (arrival <= policy_time + _CLOCK_EPS_S
                    and obs.frame_timestamp <= policy_time + _CLOCK_EPS_S):
                age = max(0., policy_time - obs.frame_timestamp)
                if age > self.stale_after_s + _CLOCK_EPS_S:
                    return ObservationRead("STALE", None, age, obs.frame_timestamp,
                                           arrival, False)
                new = obs.source_frame_identity != self._last_delivered_identity
                self._last_delivered_identity = obs.source_frame_identity
                return ObservationRead("FRESH", obs, age, obs.frame_timestamp,
                                       arrival, new)
        return ObservationRead("MISSING", None, None, None, None, False)
