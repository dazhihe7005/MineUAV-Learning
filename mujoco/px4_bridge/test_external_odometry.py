"""Offline odometry wire/geometry and fail-closed source contracts; no SITL."""
import numpy as np
import pytest

from .test_interfaces import module
from .test_transport import SEARCH


def sample(**overrides):
    o = module("external_odometry")
    values = dict(
        time_usec=1_000_000,
        position_enu_m=(1., 2., 3.),
        quaternion_enu_flu=(1., 0., 0., 0.),
        velocity_enu_m_s=(1., 2., 3.),
        angular_velocity_flu_rad_s=(.1, .2, .3),
        pose_covariance=np.diag([1., 4., 9., .01, .04, .09]),
        twist_covariance=np.diag([1., 4., 9., .01, .04, .09]),
        reset_counter=0,
        reset_event=False,
        quality=100,
    )
    values.update(overrides)
    return o.GroundTruthOdomSample(**values)


def test_frames_body_velocity_covariance_and_wire_roundtrip():
    o = module("external_odometry")
    mv = module("transport").load_mavlink(str(SEARCH))
    adapter = o.OdometryAdapter()
    msg = adapter.make_message(sample(), mv, now_usec=1_000_000)
    assert (msg.frame_id, msg.child_frame_id, msg.estimator_type) == (1, 12, 0)
    assert (msg.x, msg.y, msg.z) == pytest.approx((2, 1, -3))
    assert msg.q == pytest.approx([2**-.5, 0, 0, 2**-.5])
    assert (msg.vx, msg.vy, msg.vz) == pytest.approx((1, -2, -3))
    assert (msg.rollspeed, msg.pitchspeed, msg.yawspeed) == pytest.approx((.1, -.2, -.3))
    assert [msg.pose_covariance[i] for i in (0, 6, 11, 15, 18, 20)] == pytest.approx([4, 1, 9, .01, .04, .09])
    assert [msg.velocity_covariance[i] for i in (0, 6, 11, 15, 18, 20)] == pytest.approx([1, 4, 9, .01, .04, .09])
    wire = msg.pack(mv.MAVLink(None, srcSystem=42, srcComponent=51))
    decoded = mv.MAVLink(None).parse_buffer(wire)[0]
    assert decoded.get_type() == "ODOMETRY"
    assert decoded.time_usec == 1_000_000
    assert decoded.pose_covariance == pytest.approx(msg.pose_covariance)


def test_rotated_world_velocity_is_transformed_to_body_frd():
    q = (2**-.5, 0., 0., 2**-.5)
    msg = module("external_odometry").OdometryAdapter().make_message(
        sample(quaternion_enu_flu=q, velocity_enu_m_s=(1., 0., 0.)),
        module("transport").load_mavlink(str(SEARCH)), now_usec=1_000_000)
    assert (msg.vx, msg.vy, msg.vz) == pytest.approx((0., 1., 0.), abs=1e-6)


@pytest.mark.parametrize("change", [
    {"quaternion_enu_flu": (2., 0., 0., 0.)},
    {"position_enu_m": (float("nan"), 0., 0.)},
    {"velocity_enu_m_s": (float("inf"), 0., 0.)},
    {"quality": 0},
    {"quality": 101},
    {"pose_covariance": np.diag([-1., 1., 1., 1., 1., 1.])},
    {"twist_covariance": np.full((6, 6), float("nan"))},
    {"pose_covariance": np.triu(np.ones((6, 6)))},
    {"position_enu_m": (1e40, 0., 0.)},
    {"pose_covariance": np.diag([1e40, 1., 1., 1., 1., 1.])},
    {"pose_covariance": np.diag([1e-50, 1., 1., 1., 1., 1.])},
    {"pose_covariance": np.eye(6)*1e-11 + np.array([
        [0., 6e-11, 0., 0., 0., 0.],
        [6e-11, 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]])},
])
def test_invalid_sample_never_emitted(change):
    o = module("external_odometry")
    with pytest.raises(ValueError):
        o.OdometryAdapter().make_message(sample(**change),
            module("transport").load_mavlink(str(SEARCH)), now_usec=1_000_000)


def test_tiny_covariance_asymmetry_cannot_hide_behind_absolute_tolerance():
    covariance = np.eye(6) * 1e-12
    covariance[0, 1] = 2e-13
    covariance[1, 0] = 3e-13
    with pytest.raises(ValueError, match="asymmetric"):
        module("external_odometry").OdometryAdapter().make_message(
            sample(pose_covariance=covariance),
            module("transport").load_mavlink(str(SEARCH)), now_usec=1_000_000)


def test_sample_clock_duplicate_backwards_stale_future_and_outage_latch():
    o = module("external_odometry")
    mv = module("transport").load_mavlink(str(SEARCH))
    for stamp, now in ((1_000_000,1_020_000),(999_999,1_020_000),
                       (1_020_000,1_200_001),(1_040_000,1_020_000)):
        a = o.OdometryAdapter()
        a.make_message(sample(), mv, now_usec=1_000_000)
        with pytest.raises(ValueError): a.make_message(sample(time_usec=stamp), mv, now_usec=now)
        with pytest.raises(ValueError): a.make_message(sample(time_usec=1_020_000), mv, now_usec=1_020_000)
    a = o.OdometryAdapter()
    a.make_message(sample(), mv, now_usec=1_000_000)
    a.make_message(sample(time_usec=1_020_000), mv, now_usec=1_020_000)
    with pytest.raises(ValueError): a.check_outage(now_usec=1_220_001)
    with pytest.raises(ValueError): a.make_message(sample(time_usec=1_220_001), mv, now_usec=1_220_001)


def test_reset_counter_requires_explicit_event_and_wraps():
    o = module("external_odometry")
    mv = module("transport").load_mavlink(str(SEARCH))
    a = o.OdometryAdapter()
    a.make_message(sample(reset_counter=255), mv, now_usec=1_000_000)
    with pytest.raises(ValueError):
        a.make_message(sample(time_usec=1_020_000,reset_counter=0), mv, now_usec=1_020_000)
    a.make_message(sample(time_usec=1_020_000,reset_counter=0,reset_event=True), mv, now_usec=1_020_000)
    with pytest.raises(ValueError):
        a.make_message(sample(time_usec=1_040_000,reset_counter=0,reset_event=True), mv, now_usec=1_040_000)


def test_ekf_strict_less_than_200ms_boundary_is_respected():
    o = module("external_odometry")
    mv = module("transport").load_mavlink(str(SEARCH))
    a = o.OdometryAdapter()
    a.make_message(sample(), mv, now_usec=1_000_000)
    with pytest.raises(ValueError):
        a.make_message(sample(time_usec=1_200_000), mv, now_usec=1_200_000)


def test_clock_check_records_every_successful_observation():
    o = module("external_odometry")
    mv = module("transport").load_mavlink(str(SEARCH))
    a = o.OdometryAdapter()
    a.make_message(sample(), mv, now_usec=1_000_000)
    a.check_outage(now_usec=1_050_000)
    with pytest.raises(ValueError): a.check_outage(now_usec=1_040_000)


def test_live_receiver_semantics_are_not_false_fusion_evidence():
    o = module("external_odometry")
    assert not o.fusion_evidence({"cs_ev_pos": True}, {"ev_pos": (100, 100)},
                                  {"cs_gnss_pos": False}, require_gnss_denied=True)
    flags = {"cs_ev_pos": True, "cs_ev_vel": True, "cs_gnss_pos": False,
             "cs_gnss_vel": False, "cs_gps_hgt": False, "cs_gnss_yaw": False}
    aids = {"ev_pos": (100, 200), "ev_vel": (100, 200),
            "gnss_pos": (100, 100), "gnss_vel": (100, 100),
            "gnss_hgt": (100, 100), "gnss_yaw": (100, 100)}
    assert o.fusion_evidence(
        flags, aids, flags, require_gnss_denied=True)
    old_flags = dict(flags, cs_gnss_pos=True)
    assert not o.fusion_evidence(
        flags, aids, old_flags, require_gnss_denied=True)


def test_gnss_denial_cannot_ignore_height_or_dual_antenna_yaw_aids():
    o = module("external_odometry")
    flags = {"cs_ev_pos": True, "cs_ev_vel": True, "cs_gnss_pos": False,
             "cs_gnss_vel": False, "cs_gps_hgt": False, "cs_gnss_yaw": False}
    aids = {"ev_pos": (100, 200), "ev_vel": (100, 200),
            "gnss_pos": (100, 100), "gnss_vel": (100, 100),
            "gnss_hgt": (100, 101), "gnss_yaw": (100, 100)}
    assert not o.fusion_evidence(flags, aids, flags, require_gnss_denied=True)
    aids["gnss_hgt"] = (100, 100)
    flags["cs_gnss_yaw"] = True
    assert not o.fusion_evidence(flags, aids, flags, require_gnss_denied=True)
