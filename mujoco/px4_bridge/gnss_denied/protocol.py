"""DISARMED observation only. Old GNSS flight requirements are untouched."""
import math
from ..telemetry_freshness import TelemetryFreshness
from ..allocation_runner import AllocationYawProtocol

GNSS_FLAGS=("cs_gnss_pos","cs_gnss_vel","cs_gps_hgt","cs_gnss_yaw")
SENSOR_FAULTS=("cs_mag_fault","cs_baro_fault","cs_gnss_fault","cs_gnss_yaw_fault","cs_gnss_hgt_fault",
               "fs_bad_mag_x","fs_bad_mag_y","fs_bad_mag_z","fs_bad_hdg",
               "fs_bad_acc_vertical","fs_bad_acc_clipping")


class DisarmedFreshness(TelemetryFreshness):
    REQUIRED={**TelemetryFreshness.REQUIRED,
              "estimator_status_flags":tuple(dict.fromkeys(TelemetryFreshness.REQUIRED["estimator_status_flags"]+
                 GNSS_FLAGS+("cs_ev_pos","cs_ev_vel","cs_baro_hgt","cs_ev_yaw")+SENSOR_FAULTS))}

    def inspect_health(self,name,fields,required_values=None,require_mag_fusion=False,clock=None):
        if clock is not None and not self.validate_source(name,fields,clock):return
        if name not in self.REQUIRED:self._fail("unexpected disarmed safety topic");return
        for key in self.REQUIRED[name]:
            value=fields.get(key)
            if key in ("heading","arming_state","nav_state"):
                if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
                    self._fail("invalid disarmed numeric field: "+name+"."+key)
            elif not isinstance(value,bool):self._fail("missing/invalid disarmed flag: "+name+"."+key)
        if name=="vehicle_local_position":
            if any(fields.get(k) is not True for k in ("xy_valid","z_valid","v_xy_valid","v_z_valid")):
                self._fail("disarmed local estimate invalid")
        elif name=="vehicle_status":
            if fields.get("arming_state")!=1 or fields.get("failsafe") is not False:
                self._fail("unexpected armed/failsafe state in disarmed-only mode")
        else:
            if any(fields.get(k) is not False for k in GNSS_FLAGS+SENSOR_FAULTS+("cs_in_air","cs_ev_yaw")):
                self._fail("unexpected GNSS/airborne/EV-yaw fusion or EKF fault")
            if any(fields.get(k) is not True for k in ("cs_tilt_align","cs_yaw_align","cs_ev_pos","cs_ev_vel","cs_baro_hgt")):
                self._fail("disarmed EV/barometer/alignment unavailable")
            if not (fields.get("cs_mag_hdg") is True or fields.get("cs_mag_3d") is True):
                self._fail("disarmed magnetic heading source unavailable")
        # Preflight checks, in-flight mag alignment and control-quality heading
        # remain recorded but do not authorize anything in this ground test.

    def decision(self,clock,wall,require_flight=False):
        if require_flight:self._fail("GNSS-denied DISARMED mode cannot authorize flight")
        result=super().decision(clock,wall,require_flight=False)
        result["allow_new_maneuver"]=False
        result["flight_authorized"]=False
        result["scope"]="GNSS_DENIED_DISARMED_OBSERVATION_ONLY"
        return result


class GnssDeniedProtocol(AllocationYawProtocol):
    @property
    def active(self):return False

    @active.setter
    def active(self,value):
        if value is not False:raise ValueError("GNSS-denied protocol is permanently disarmed-only")

    def __init__(self,field,expected):
        super().__init__(field,expected,active=False)
        self.monitor=DisarmedFreshness()

    def validate_proof(self,proof):
        raise ValueError("GNSS-denied DISARMED mode rejects every flight proof")

    def setpoint(self,*args,**kwargs):
        raise ValueError("GNSS-denied DISARMED mode cannot emit setpoints")

    def metadata(self):
        return {**super().metadata(),"scope":"GNSS_DENIED_DISARMED_OBSERVATION_ONLY",
                "flight_authorized":False,"not_flight_proof":True}
