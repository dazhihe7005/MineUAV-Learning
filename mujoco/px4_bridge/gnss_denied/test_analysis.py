"""Recorded flags cannot replace direct receipt and advancing EV fusion."""
import numpy as np
import pytest
import json
from pathlib import Path
from .analyze import fusion_evidence, fresh_run_identity, global_origin_evidence,assess_safety
from .test_evidence import topics,params
from ..test_odometry_reverification import visual,sent_for


def complete():
    t=topics();t["vehicle_visual_odometry"]=visual()
    t["estimator_status_flags"].update(cs_ev_pos=np.ones(3),cs_ev_vel=np.ones(3))
    t["vehicle_local_position"]={"timestamp":np.array([20_800_000,20_850_000,20_900_000]),
        **{key:np.ones(3) for key in ("xy_valid","z_valid","v_xy_valid","v_z_valid")}}
    for key in ("ev_pos","ev_vel"):t["estimator_aid_src_"+key]=t["estimator_aid_src_mag"].copy()
    return t


def test_complete_fusion_chain():
    t=complete();result=fusion_evidence(t,params(),sent_for(t["vehicle_visual_odometry"]))
    assert all(result[key]=="PASS" for key in ("direct_status","ev_pos_status","ev_vel_status","local_status"))


@pytest.mark.parametrize("missing",["vehicle_visual_odometry","estimator_aid_src_ev_pos","estimator_aid_src_ev_vel","vehicle_local_position"])
def test_missing_chain_evidence_cannot_pass(missing):
    t=complete();sent=sent_for(t["vehicle_visual_odometry"]);t.pop(missing)
    result=fusion_evidence(t,params(),sent)
    assert any(result[key]!="PASS" for key in ("direct_status","ev_pos_status","ev_vel_status","local_status"))


def test_ev_flags_with_frozen_fuse_times_rejected():
    t=complete();t["estimator_aid_src_ev_pos"]["time_last_fuse"]=np.array([20_000_000]*3)
    assert fusion_evidence(t,params(),sent_for(t["vehicle_visual_odometry"]))["ev_pos_status"]=="FAIL"


def test_historical_smoke_not_new_evidence():
    assert not fresh_run_identity({"kind":"disarmed_odometry_fusion_reverification","protocol_sha256":"x","not_flight_proof":True},"x")


def test_unknown_global_initialization_rejected():
    t={"vehicle_local_position":{"ref_timestamp":np.array([1]),"xy_global":np.array([1]),"z_global":np.array([0])}}
    assert global_origin_evidence(t)["status"]!="PASS"


def test_local_frame_without_global_origin():
    t={"vehicle_local_position":{"ref_timestamp":np.array([0]),"xy_global":np.array([0]),"z_global":np.array([0])}}
    assert global_origin_evidence(t)["status"]=="PASS"


def recorded_timing_fixture():
    path=Path(__file__).parents[2]/"reports/px4_odometry_reverification_parts/disarmed_01/smoke_summary.json"
    if not path.is_file():pytest.skip("historical timings are local-only")
    raw=json.loads(path.read_text())
    # Synthetic new-mode decision fields; timing data remain historical.
    raw["phase2"]["telemetry_fix"]["last_decision"].update(allow_new_maneuver=False,flight_authorized=False)
    return raw


def test_recorded_continuous_query_timings():
    assert assess_safety(recorded_timing_fixture())["status"]=="PASS"


@pytest.mark.parametrize("change",["missing_fault","missing_decision","early_only","sim_late","dropped","stale_source","allow_flight"])
def test_missing_late_or_inconsistent_safety_evidence(change):
    raw=recorded_timing_fixture();fix=raw["phase2"]["telemetry_fix"];metrics=raw["runtime_measurements"]
    if change=="missing_fault":fix.pop("fault")
    elif change=="missing_decision":fix.pop("last_decision")
    elif change=="early_only":metrics["queries"]=metrics["queries"][:2]
    elif change=="sim_late":metrics["queries"][-1]["end_clock_usec"]+=200000
    elif change=="dropped":metrics["dropped"]={"queries":1}
    elif change=="stale_source":fix["last_decision"]["topics"]["vehicle_local_position"]["publication_usec"]-=1000000
    elif change=="allow_flight":fix["last_decision"]["flight_authorized"]=True
    assert assess_safety(raw)["status"]!="PASS"


@pytest.mark.parametrize("missing",[None,"vehicle_visual_odometry","estimator_aid_src_ev_pos","estimator_aid_src_ev_vel"])
def test_saved_run_analysis_without_launch(tmp_path,monkeypatch,missing):
    from . import analyze as module
    runtime=Path(__file__).parents[2]/"reports/px4_gnss_denied_disarmed_parts/disarmed_01"
    if not (runtime/"odometry_source.json").is_file():pytest.skip("single authorized run is local-only")
    original=module.read_log
    def read(path):
        t,p,m=original(path)
        if missing:t.pop(missing,None)
        return t,p,m
    monkeypatch.setattr(module,"read_log",read)
    monkeypatch.setattr(module,"plots",lambda *args:[])
    result=module.analyze(runtime,tmp_path/"report.json",tmp_path/"figures")
    assert result["overall"]=="PASS" if missing is None else result["overall"]!="PASS"
