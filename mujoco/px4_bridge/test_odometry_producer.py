"""50 Hz COM truth emission without PX4 or an extra physics environment."""
import mujoco
import numpy as np
import pytest

from .simulation_runner import MODEL
from .sensor_adapter import SensorAdapter
from .test_interfaces import module
from .test_transport import SEARCH


def test_com_and_body_velocity_not_freejoint_origin_are_emitted():
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    d.qpos[:3] = [1., 2., 3.]
    d.qvel[:6] = [0., 0., 0., 0., 0., 1.]
    mujoco.mj_forward(m, d)
    producer = module("odometry_producer").GroundTruthOdometryProducer()
    sensor = SensorAdapter(m).sample(d, 1_000_000)
    msg = producer.tick(sensor, module("transport").load_mavlink(str(SEARCH)))
    body = m.body("mine_uav").id
    assert (msg.x, msg.y, msg.z) == pytest.approx(
        [d.xipos[body, 1], d.xipos[body, 0], -d.xipos[body, 2]])
    assert np.linalg.norm(np.asarray([msg.x, msg.y, msg.z]) - np.asarray([2., 1., -3.])) > .01
    jp = np.empty((3, m.nv)); jr = np.empty_like(jp)
    mujoco.mj_jacBodyCom(m, d, jp, jr, body)
    world_velocity = jp @ d.qvel
    assert (msg.vx, msg.vy, msg.vz) == pytest.approx(
        [world_velocity[0], -world_velocity[1], -world_velocity[2]])
    assert msg.estimator_type == 0


def test_producer_sends_exactly_50hz_and_rejects_clock_gap():
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    producer = module("odometry_producer").GroundTruthOdometryProducer()
    mv = module("transport").load_mavlink(str(SEARCH))
    sensors = SensorAdapter(m)
    sent = [producer.tick(sensors.sample(d, 1_000_000 + 4000*i), mv) for i in range(11)]
    assert [x.time_usec for x in sent if x is not None] == [1_000_000, 1_020_000, 1_040_000]
    with pytest.raises(ValueError):
        producer.tick(sensors.sample(d, 1_048_000), mv)
