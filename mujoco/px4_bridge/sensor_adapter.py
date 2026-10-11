"""Virtual IMU at model COM; optional seeded synthetic mag/baro noise."""
from dataclasses import dataclass
import mujoco
import numpy as np
from .frame_conversions import enu_to_ned, flu_to_frd, attitude_to_px4

@dataclass(frozen=True)
class SensorSample:
    time_usec: int
    accel_frd: np.ndarray
    gyro_frd: np.ndarray
    mag_frd_gauss: np.ndarray
    position_ned_m: np.ndarray
    velocity_ned_m_s: np.ndarray
    quaternion_ned_frd: np.ndarray
    pressure_hpa: float
    altitude_m: float

class SensorAdapter:
    def __init__(self,model,noise_seed=0,mag_noise_gauss=0.,baro_noise_hpa=0.):
        self.model=model
        self.body=model.body("mine_uav").id
        self.jp=np.empty((3,model.nv));self.jr=np.empty_like(self.jp)
        self.jpd=np.empty_like(self.jp)
        # Synthetic earth field, NED gauss, known to magnetometer only.
        self.field_ned=np.array([.215,0.,.427])
        if not np.isfinite([mag_noise_gauss,baro_noise_hpa]).all() or min(mag_noise_gauss,baro_noise_hpa)<0:
            raise ValueError("finite nonnegative synthetic noise scales required")
        self.rng=np.random.default_rng(noise_seed)
        self.mag_noise_gauss=mag_noise_gauss;self.baro_noise_hpa=baro_noise_hpa

    def sample(self,data,time_usec):
        if not isinstance(time_usec,int) or time_usec<=0:
            raise ValueError("positive integer simulator microseconds required")
        # Recompute at CURRENT integrated state, not mj_step's preceding qacc.
        mujoco.mj_forward(self.model,data)
        r=data.xmat[self.body].reshape(3,3)
        mujoco.mj_jacBodyCom(self.model,data,self.jp,self.jr,self.body)
        mujoco.mj_jacDot(self.model,data,self.jpd,None,data.xipos[self.body],self.body)
        acceleration=self.jp@data.qacc+self.jpd@data.qvel
        velocity=self.jp@data.qvel
        gyro_body=r.T@(self.jr@data.qvel)
        q=attitude_to_px4(data.xquat[self.body])
        specific=flu_to_frd(r.T@(acceleration-self.model.opt.gravity))
        mag=flu_to_frd(r.T@enu_to_ned(self.field_ned)) # E is self-inverse
        altitude=488.+float(data.xipos[self.body,2])
        pressure=1013.25*(1-altitude/44330.)**5.255
        mag=mag+self.rng.normal(0,self.mag_noise_gauss,3)
        pressure+=self.rng.normal(0,self.baro_noise_hpa)
        values=np.r_[specific,gyro_body,velocity,data.xipos[self.body],mag,q,pressure]
        if not np.isfinite(values).all():
            raise ValueError("nonfinite simulated sensor state")
        return SensorSample(time_usec,specific,flu_to_frd(gyro_body),mag,
                            enu_to_ned(data.xipos[self.body]),enu_to_ned(velocity),
                            q,float(pressure),altitude)
