"""50 Hz COM truth from the existing 250 Hz MuJoCo SensorAdapter output."""
import numpy as np

from .external_odometry import GroundTruthOdomSample, OdometryAdapter
from .frame_conversions import attitude_from_px4, enu_to_ned, flu_to_frd

SAMPLE_PERIOD_USEC = 20_000
PHYSICS_SENSOR_PERIOD_USEC = 4_000
BOOT_EPOCH_USEC = 1_000_000
# Declared synthetic uncertainty, not a measured sensor specification.
POSE_VARIANCE = np.diag([.01, .01, .01, .01, .01, .01])
TWIST_VARIANCE = np.diag([.01, .01, .01, .01, .01, .01])
MAX_20MS_POSITION_JUMP_M = .5


class GroundTruthOdometryProducer:
    def __init__(self):
        self.adapter = OdometryAdapter()
        self.last_tick_usec = None
        self.last_position = None
        self.samples = []

    def tick(self, sensor, mavlink):
        """Called for each real SensorAdapter COM sample, before HIL send."""
        time_usec = sensor.time_usec
        if not isinstance(time_usec, int) or time_usec < BOOT_EPOCH_USEC:
            raise ValueError("invalid odometry simulation clock")
        if self.last_tick_usec is None:
            if time_usec != BOOT_EPOCH_USEC:
                raise ValueError("odometry source missed first sample")
        elif time_usec - self.last_tick_usec != PHYSICS_SENSOR_PERIOD_USEC:
            raise ValueError("odometry source missed or repeated sensor tick")
        self.adapter.check_outage(time_usec)
        self.last_tick_usec = time_usec
        if (time_usec - BOOT_EPOCH_USEC) % SAMPLE_PERIOD_USEC:
            return None
        # SensorAdapter sampled xipos and mj_jacBodyCom at the integrated
        # physics state. Convert its NED/FRD truth back to Phase3A's ENU/FLU
        # input contract; OdometryAdapter performs the MAVLink conversion.
        position = enu_to_ned(sensor.position_ned_m)
        velocity = enu_to_ned(sensor.velocity_ned_m_s)
        angular_velocity_flu = flu_to_frd(sensor.gyro_frd)
        if self.last_position is not None and np.linalg.norm(position-self.last_position) > MAX_20MS_POSITION_JUMP_M:
            raise ValueError("implausible odometry COM position jump")
        sample = GroundTruthOdomSample(
            time_usec=time_usec,
            position_enu_m=position,
            quaternion_enu_flu=attitude_from_px4(sensor.quaternion_ned_frd),
            velocity_enu_m_s=velocity,
            angular_velocity_flu_rad_s=angular_velocity_flu,
            pose_covariance=POSE_VARIANCE,
            twist_covariance=TWIST_VARIANCE,
            reset_counter=0,
            reset_event=False,
            quality=100)
        message = self.adapter.make_message(sample, mavlink, now_usec=time_usec)
        self.last_position = position
        self.samples.append({"sample_time_usec":time_usec, "position_enu_m":position.tolist(),
                             "velocity_enu_m_s":velocity.tolist(),
                             "position_ned_m":[message.x,message.y,message.z],
                             "velocity_frd_m_s":[message.vx,message.vy,message.vz],
                             "quaternion_ned_frd":list(message.q)})
        return message
