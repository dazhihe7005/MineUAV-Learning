"""Ideal normalized thrust → target RPM → omega²; no ESC lag claimed."""
import json
from pathlib import Path
import numpy as np
import mujoco
from .frame_conversions import flu_to_frd

class ActuatorAdapter:
    def __init__(self,model,config_path,max_rpm=8500.):
        self.config=json.loads(Path(config_path).read_text())
        self.kf=float(self.config["k_f_n_per_rad_s_squared"])
        self.km=float(self.config["k_m_nm_per_rad_s_squared"])
        self.signs=np.asarray(self.config["rotation_signs"],dtype=float)
        if model.nu!=4 or self.signs.shape!=(4,) or not np.isin(self.signs,[-1,1]).all():
            raise ValueError("expected four nonreversible model rotors")
        if not np.isfinite([self.kf,self.km,max_rpm]).all() or min(self.kf,self.km,max_rpm)<=0:
            raise ValueError("invalid motor coefficients")
        self.max_rpm=float(max_rpm)
        self.u_max=(self.max_rpm*2*np.pi/60)**2
        com=model.body("mine_uav").ipos
        self.arms_frd=np.array([flu_to_frd(model.site(f"motor_{i+1}_site").pos-com) for i in range(4)])
        for i,sign in enumerate(self.signs):
            site=model.site(f"motor_{i+1}_site")
            if (model.actuator_trntype[i]!=mujoco.mjtTrn.mjTRN_SITE or model.actuator_trnid[i,0]!=site.id
                or model.actuator_dyntype[i]!=mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[i]!=mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_gainprm[i,0]!=1 or model.actuator_biastype[i]!=mujoco.mjtBias.mjBIAS_NONE
                or not np.array_equal(site.quat,[1,0,0,0])):
                raise ValueError("requires identity-frame combined site motors without hidden dynamics")
            if model.actuator(i).name!=f"motor_{i+1}_wrench" or not np.allclose(
                model.actuator_gear[i],[0,0,self.kf,0,0,sign*self.km],rtol=1e-10,atol=1e-12):
                raise ValueError("model/config actuator identity or gear mismatch")

    def apply(self,controls,armed):
        c=np.asarray(controls,dtype=float)
        if c.ndim!=1 or c.size<4 or not np.isfinite(c[:4]).all() or np.any(c[:4]<0) or np.any(c[:4]>1):
            raise ValueError("four finite nonreversible controls in [0,1] required")
        # PX4 THR_MDL_FAC=0: output is linear normalized thrust, not RPM.
        omega_target=np.sqrt(c[:4])*np.sqrt(self.u_max) if armed else np.zeros(4)
        return omega_target**2

    def px4_geometry(self):
        params={"CA_AIRFRAME":0,"CA_ROTOR_COUNT":4,"THR_MDL_FAC":0}
        for i,(arm,sign) in enumerate(zip(self.arms_frd,self.signs)):
            for axis,value in zip("XYZ",arm): params[f"CA_ROTOR{i}_P{axis}"]=float(value)
            params[f"CA_ROTOR{i}_CT"]=1.
            # PX4 moment=ct*r×axis - ct*km*axis, axis FRD −z.
            params[f"CA_ROTOR{i}_KM"]=-float(sign)*self.km/self.kf
            params[f"PWM_MAIN_FUNC{i+1}"]=101+i
        return params
