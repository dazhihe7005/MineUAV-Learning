"""Independent cold-input, nonfusion and valid-EV predicates."""
import numpy as np
from ..analyze_odometry_fusion import assess_aid_source,_end_flags
from .protocol import GNSS_FLAGS


def aggregate(checks):
    if not set("ABCDEFGHIJKLMN")<=set(checks):return "BLOCKED"
    if all(v=="PASS" for v in checks.values()):return "PASS"
    return "FAIL" if "FAIL" in checks.values() else "BLOCKED"


def assess_boundary(raw):
    counts=raw.get("messages_sent",{})
    seconds=raw.get("simulation_seconds",np.nan)
    physical=raw.get("final_physical_state",{}).get("time_s",np.nan)
    expected={"HIL_SENSOR":5000,"HIL_GPS":0,"ODOMETRY":1000}
    passed=bool(raw.get("complete") is True and np.isfinite(seconds) and np.isfinite(physical)
                and 0<=seconds<=20. and 0<=physical<=20.
                and abs(seconds-20.)<1e-9 and abs(physical-seconds)<1e-9
                and all(counts.get(k)==v for k,v in expected.items()))
    return {"status":"PASS" if passed else "FAIL","simulation_seconds":seconds,
            "physical_seconds":physical,"expected_counts":expected,"actual_counts":counts,
            "sensor_cycles":5000,"comparison_guard_s":1e-6}


def assess_gnss_absence(topics,params,counts,*,end_usec):
    flags=topics.get("estimator_status_flags")
    result={"status":"BLOCKED","flags":{},"aid_sources":{},"gps_input_records":{},
            "parameters_disabled":params.get("EKF2_GPS_CTRL")==0,"hil_gps_count":counts.get("HIL_GPS")}
    if not flags or any(k not in flags for k in ("timestamp",)+GNSS_FLAGS):
        return {**result,"reason":"missing GNSS control-state evidence"}
    stamps=np.asarray(flags["timestamp"],dtype=float)
    n=len(stamps)
    fresh=(n>=3 and np.isfinite(stamps).all() and np.all(stamps>0) and np.all(np.diff(stamps)>0)
           and 0<=end_usec-stamps[-1]<=1_650_000 and np.all(stamps<=end_usec))
    good=bool(fresh and result["parameters_disabled"] and result["hil_gps_count"]==0)
    for name in GNSS_FLAGS:
        values=np.asarray(flags[name],dtype=float)
        ok=values.ndim==1 and len(values)==n and np.isfinite(values).all() and np.all(values==0)
        result["flags"][name]={"all_false":bool(ok),"true_samples":int(np.sum(values==1))}
        good &= ok
    for key in ("pos","vel","hgt","yaw"):
        data=topics.get("estimator_aid_src_gnss_"+key)
        if not data or not len(data.get("timestamp",())):
            result["aid_sources"][key]={"status":"ABSENT","samples":0};continue
        if any(name not in data for name in ("timestamp","fused","time_last_fuse")):
            return {**result,"status":"BLOCKED","reason":"incomplete recorded GNSS aid source"}
        t=np.asarray(data["timestamp"],dtype=float);f=np.asarray(data["fused"],dtype=float);last=np.asarray(data["time_last_fuse"],dtype=float)
        ok=(len(t)==len(f)==len(last) and np.isfinite(np.r_[t,f,last]).all()
            and np.all((t>0)&(t<=end_usec)) and np.all(np.diff(t)>0) and np.all(f==0) and np.all(last==0))
        result["aid_sources"][key]={"status":"NEVER_FUSED" if ok else "FAIL","samples":len(t),
            "fused_samples":int(np.sum(f==1)),"last_fuse_max_usec":float(last.max()) if len(last) else None}
        good &= ok
    for name in ("vehicle_gps_position","sensor_gps"):
        count=len(topics.get(name,{}).get("timestamp",()))
        result["gps_input_records"][name]=count;good &= count==0
    return {**result,"status":"PASS" if good else "FAIL",
            "reason":None if good else "GPS input, active/stale GNSS state, parameter or aid evidence"}


def assess_non_gnss_sources(topics,params,*,end_usec):
    aids={key:assess_aid_source(topics.get("estimator_aid_src_"+key),end_usec=end_usec)
          for key in ("baro_hgt","mag")}
    flags=topics.get("estimator_status_flags")
    source_flags=_end_flags(flags,("cs_tilt_align","cs_yaw_align","cs_baro_hgt"),end_usec=end_usec)
    magnetic=bool(flags and len(flags.get("timestamp",()))>=3
        and all(np.all(np.asarray(flags.get(key,[1]))[-3:]==0) for key in ("cs_mag_fault","cs_baro_fault"))
        and np.all((np.asarray(flags.get("cs_mag_hdg",[0]))[-3:]==1)|(np.asarray(flags.get("cs_mag_3d",[0]))[-3:]==1)))
    configured=params.get("EKF2_HGT_REF")==0 and params.get("EKF2_BARO_CTRL")==1 and params.get("EKF2_MAG_TYPE") in (0,1)
    if any(r["status"]=="BLOCKED" for r in aids.values()):status="BLOCKED"
    else:status="PASS" if configured and source_flags and magnetic and all(r["pass"] for r in aids.values()) else "FAIL"
    return {"status":status,"aid_sources":aids,"configured":configured,
            "ground_alignment_and_baro_flags_valid":source_flags,"magnetic_heading_flags_valid":magnetic,
            "flight_heading_readiness":"NOT TESTED"}
