"""No GNSS input != no GNSS fusion != valid local EV estimation."""
import numpy as np
import pytest
from .evidence import assess_gnss_absence,assess_non_gnss_sources,assess_boundary,aggregate


def topics():
    stamps=np.array([19_000_000,20_000_000,20_900_000])
    flags={"timestamp":stamps,"cs_gnss_pos":np.zeros(3),"cs_gnss_vel":np.zeros(3),
           "cs_gps_hgt":np.zeros(3),"cs_gnss_yaw":np.zeros(3),"cs_baro_hgt":np.ones(3),
           "cs_tilt_align":np.ones(3),"cs_yaw_align":np.ones(3),"cs_mag_hdg":np.ones(3),
           "cs_mag_3d":np.zeros(3),"cs_mag_fault":np.zeros(3),"cs_baro_fault":np.zeros(3)}
    aid={"timestamp":np.array([20_940_000,20_960_000,20_980_000]),
         "time_last_fuse":np.array([20_800_000,20_820_000,20_840_000]),
         "fused":np.ones(3),"innovation_rejected":np.zeros(3)}
    return {"estimator_status_flags":flags,"estimator_aid_src_baro_hgt":aid.copy(),
            "estimator_aid_src_mag":aid.copy()}


def params():return {"EKF2_EV_CTRL":5,"EKF2_GPS_CTRL":0,"EKF2_HGT_REF":0,"EKF2_BARO_CTRL":1,"EKF2_MAG_TYPE":0}


def test_absent_gnss_aids_are_not_false_failures():
    result=assess_gnss_absence(topics(),params(),{"HIL_GPS":0},end_usec=21_000_000)
    assert result["status"]=="PASS"
    assert all(row["status"]=="ABSENT" for row in result["aid_sources"].values())


@pytest.mark.parametrize("change",["packet","parameter","flag_pos","flag_vel","flag_height","flag_yaw","aid", "other_gps", "missing_flag", "stale_flags"])
def test_cold_start_absence_requires_corroboration(change):
    t=topics();p=params();counts={"HIL_GPS":0}
    if change=="packet":counts["HIL_GPS"]=1
    elif change=="parameter":p["EKF2_GPS_CTRL"]=7
    elif change.startswith("flag_"):
        flag={"flag_pos":"cs_gnss_pos","flag_vel":"cs_gnss_vel","flag_height":"cs_gps_hgt","flag_yaw":"cs_gnss_yaw"}[change]
        t["estimator_status_flags"][flag][0]=1
    elif change=="aid":t["estimator_aid_src_gnss_pos"]={"timestamp":np.array([20_000_000]),"time_last_fuse":np.array([19_900_000]),"fused":np.array([0])}
    elif change=="other_gps":t["sensor_gps"]={"timestamp":np.array([1_100_000])}
    elif change=="missing_flag":del t["estimator_status_flags"]["cs_gps_hgt"]
    elif change=="stale_flags":t["estimator_status_flags"]["timestamp"]-=5_000_000
    assert assess_gnss_absence(t,p,counts,end_usec=21_000_000)["status"]!="PASS"


def test_published_but_unfused_zero_gnss_aid_is_not_three_sample_requirement():
    t=topics();t["estimator_aid_src_gnss_vel"]={"timestamp":np.array([1_500_000]),"time_last_fuse":np.array([0]),"fused":np.array([0])}
    assert assess_gnss_absence(t,params(),{"HIL_GPS":0},end_usec=21_000_000)["status"]=="PASS"


def test_missing_packet_counter_is_not_zero_evidence():
    assert assess_gnss_absence(topics(),params(),{},end_usec=21_000_000)["status"]!="PASS"


def test_boundary_zero_gps_positive_and_input_immutable():
    raw={"complete":True,"simulation_seconds":20.,"final_physical_state":{"time_s":20.},
         "messages_sent":{"HIL_SENSOR":5000,"HIL_GPS":0,"ODOMETRY":1000}}
    assert assess_boundary(raw)["status"]=="PASS"
    assert raw["messages_sent"]["HIL_GPS"]==0


def test_actual_non_gnss_source_evidence():
    assert assess_non_gnss_sources(topics(),params(),end_usec=21_000_000)["status"]=="PASS"


@pytest.mark.parametrize("change",["gps_height","no_baro","no_mag","missing_aid","old_fuse","flag_only"])
def test_non_gnss_height_heading_cannot_be_assumed(change):
    t=topics();p=params()
    if change=="gps_height":p["EKF2_HGT_REF"]=1
    if change=="no_baro":p["EKF2_BARO_CTRL"]=0
    if change=="no_mag":p["EKF2_MAG_TYPE"]=5
    if change=="missing_aid":del t["estimator_aid_src_baro_hgt"]
    if change=="old_fuse":t["estimator_aid_src_mag"]["time_last_fuse"]=np.array([1,2,3])
    if change=="flag_only":t.pop("estimator_aid_src_mag")
    assert assess_non_gnss_sources(t,p,end_usec=21_000_000)["status"]!="PASS"


@pytest.mark.parametrize("seconds",[20.004,21.,19.996])
def test_boundary_remains_strict_without_gps(seconds):
    raw={"complete":True,"simulation_seconds":seconds,"final_physical_state":{"time_s":seconds},
         "messages_sent":{"HIL_SENSOR":5000,"HIL_GPS":0,"ODOMETRY":1000}}
    assert assess_boundary(raw)["status"]=="FAIL"


def test_all_fourteen_items_required_for_overall():
    checks={letter:"PASS" for letter in "ABCDEFGHIJKLMN"}
    assert aggregate(checks)=="PASS"
    checks["E"]="BLOCKED";assert aggregate(checks)=="BLOCKED"
    del checks["E"];assert aggregate(checks)=="BLOCKED"
