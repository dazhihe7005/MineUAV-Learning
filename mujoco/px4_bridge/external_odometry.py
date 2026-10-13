"""Offline-only simulated-ground-truth MAVLink ODOMETRY adapter.

Not wired into the flight runner. Its source-time guard is essential because
the pinned PX4 SimulatorMavlink receiver replaces ODOMETRY.time_usec with
hrt_absolute_time() before publishing vehicle_visual_odometry.
"""
from dataclasses import dataclass

import numpy as np

from .frame_conversions import ENU_TO_NED, FLU_TO_FRD, attitude_to_px4, rotation, vector

POSE_FRAME_LOCAL_NED = 1
VELOCITY_FRAME_BODY_FRD = 12
ESTIMATOR_UNKNOWN = 0  # Simulated ground truth, never claim VIO/SLAM.
MAX_SAMPLE_AGE_USEC = 100_000
MAX_SOURCE_GAP_USEC = 200_000  # Pinned EKF requires strictly less than this.


@dataclass(frozen=True)
class GroundTruthOdomSample:
    """Capture-time truth; covariances in [ENU position, FLU attitude]
    and [ENU world velocity, FLU angular velocity] coordinates respectively.
    Covariances are declared uncertainty, not estimated from MuJoCo truth.
    """
    time_usec: int
    position_enu_m: object
    quaternion_enu_flu: object
    velocity_enu_m_s: object
    angular_velocity_flu_rad_s: object
    pose_covariance: object
    twist_covariance: object
    reset_counter: int
    reset_event: bool
    quality: int


def _uint(value, max_value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value <= max_value:
        raise ValueError(f"invalid {name}")
    return int(value)


def _covariance(value, name):
    c = np.asarray(value, dtype=float)
    if c.shape != (6, 6) or not np.isfinite(c).all():
        raise ValueError(f"invalid {name} shape/finite values")
    # An absolute 1e-10 tolerance silently accepted strongly asymmetric
    # covariances when the declared variances were much smaller than that.
    if not np.allclose(c, c.T, rtol=1e-9, atol=0):
        raise ValueError(f"asymmetric {name}")
    c = (c + c.T) * .5
    scale = np.max(np.abs(c))
    eig_tolerance = 16 * np.finfo(float).eps * scale
    if (np.min(np.diag(c)) < np.finfo(np.float32).tiny
            or np.min(np.linalg.eigvalsh(c)) < -eig_tolerance):
        raise ValueError(f"nonpositive {name}")
    return _wire_f32(c, name)


def _wire_f32(value, name):
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all() or np.max(np.abs(a)) > np.finfo(np.float32).max:
        raise ValueError(f"{name} cannot be represented by MAVLink float32")
    return a


def _packed_upper(c):
    return [float(c[i, j]) for i in range(6) for j in range(i, 6)]


class OdometryAdapter:
    """Fail-closed emitter. One instance per source/boot epoch; no retransmission.

    A latched outage requires a new, explicitly audited source instance. This
    class does not infer PX4 EKF fusion or change GNSS/safety configuration.
    """
    def __init__(self):
        self.last_sample_usec = None
        self.last_now_usec = None
        self.last_reset_counter = None
        self.fault = None

    def _fail(self, reason):
        self.fault = reason
        raise ValueError(reason)

    def check_outage(self, now_usec):
        now = _uint(now_usec, 2**64 - 1, "current simulation time")
        if self.fault is not None:
            raise ValueError(f"odometry source latched: {self.fault}")
        if self.last_now_usec is not None and now < self.last_now_usec:
            self._fail("simulation clock regressed")
        if self.last_sample_usec is not None and now - self.last_sample_usec >= MAX_SOURCE_GAP_USEC:
            self._fail("odometry source interrupted")
        self.last_now_usec = now

    def make_message(self, sample, mavlink, *, now_usec):
        self.check_outage(now_usec)
        now = _uint(now_usec, 2**64 - 1, "current simulation time")
        stamp = _uint(sample.time_usec, 2**64 - 1, "sample timestamp")
        if stamp > now:
            self._fail("future odometry sample")
        if now - stamp > MAX_SAMPLE_AGE_USEC:
            self._fail("stale odometry sample")
        if self.last_sample_usec is not None and stamp <= self.last_sample_usec:
            self._fail("duplicate or regressed odometry sample")
        if self.last_sample_usec is not None and stamp - self.last_sample_usec >= MAX_SOURCE_GAP_USEC:
            self._fail("odometry source gap")
        counter = _uint(sample.reset_counter, 255, "reset counter")
        if not isinstance(sample.reset_event, bool):
            raise ValueError("reset event must be explicit bool")
        if self.last_reset_counter is None and sample.reset_event:
            raise ValueError("initial sample cannot claim prior reset")
        if self.last_reset_counter is not None:
            next_counter = (self.last_reset_counter + 1) % 256
            if sample.reset_event != (counter == next_counter):
                raise ValueError("reset event/counter mismatch")
            if not sample.reset_event and counter != self.last_reset_counter:
                raise ValueError("unexpected reset counter change")
        quality = _uint(sample.quality, 100, "quality")
        if quality == 0:
            raise ValueError("quality zero cannot authorize source emission")
        p = _wire_f32(vector(sample.position_enu_m), "position")
        v = _wire_f32(vector(sample.velocity_enu_m_s), "velocity")
        w = _wire_f32(vector(sample.angular_velocity_flu_rad_s), "angular velocity")
        r = rotation(sample.quaternion_enu_flu)
        q = attitude_to_px4(sample.quaternion_enu_flu)
        cp = _covariance(sample.pose_covariance, "pose covariance")
        cv = _covariance(sample.twist_covariance, "twist covariance")
        tp = np.zeros((6, 6));tp[:3, :3] = ENU_TO_NED;tp[3:, 3:] = FLU_TO_FRD
        tv = np.zeros((6, 6));tv[:3, :3] = FLU_TO_FRD @ r.T;tv[3:, 3:] = FLU_TO_FRD
        pn = _wire_f32(ENU_TO_NED @ p, "converted position")
        vb = _wire_f32(FLU_TO_FRD @ r.T @ v, "converted velocity")
        wb = _wire_f32(FLU_TO_FRD @ w, "converted angular velocity")
        cpn = _wire_f32(tp @ cp @ tp.T, "converted pose covariance")
        cvn = _wire_f32(tv @ cv @ tv.T, "converted velocity covariance")
        if (np.min(np.diag(cpn)) < np.finfo(np.float32).tiny
                or np.min(np.diag(cvn)) < np.finfo(np.float32).tiny):
            raise ValueError("converted covariance diagonal underflows float32")
        msg = mavlink.MAVLink_odometry_message(
            stamp, POSE_FRAME_LOCAL_NED, VELOCITY_FRAME_BODY_FRD,
            *[float(x) for x in pn], [float(x) for x in q],
            *[float(x) for x in vb], *[float(x) for x in wb],
            _packed_upper(cpn), _packed_upper(cvn),
            counter, ESTIMATOR_UNKNOWN, quality)
        self.last_sample_usec = stamp
        self.last_now_usec = now
        self.last_reset_counter = counter
        return msg


def fusion_evidence(flags, aid_times, previous_flags, *, require_gnss_denied=False):
    """Conservative offline predicate over two successive aid fuse timestamps.

    Does not certify an actual flight: callers must obtain authentic PX4 uORB
    samples, prove target instance/parameter identity and valid local estimates.
    """
    for flag, aid in (("cs_ev_pos", "ev_pos"), ("cs_ev_vel", "ev_vel")):
        times = aid_times.get(aid)
        if flags.get(flag) is not True or times is None or len(times) != 2:
            return False
        if not all(isinstance(t, int) and t >= 0 for t in times) or times[1] <= times[0]:
            return False
    if require_gnss_denied:
        for flag in ("cs_gnss_pos", "cs_gnss_vel", "cs_gps_hgt", "cs_gnss_yaw"):
            if flags.get(flag) is not False or previous_flags.get(flag) is not False:
                return False
        for aid in ("gnss_pos", "gnss_vel", "gnss_hgt", "gnss_yaw"):
            times = aid_times.get(aid)
            if times is None or len(times) != 2 or times[1] != times[0]:
                return False
    return True
