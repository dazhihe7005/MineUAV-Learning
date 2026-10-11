"""Regressions: source publication is not a query, receipt or heartbeat."""
import json
from pathlib import Path
import numpy as np
import pytest
from .test_interfaces import module
from .test_yaw_validation import safety_console


def monitor():
    m=module("telemetry_freshness")
    assert hasattr(m,"TelemetryFreshness"), "old publication-age-only gate must be replaced by an independent monitor"
    return m.TelemetryFreshness()


def deliver(m,clock,wall,source=None,text=None):
    m.begin_query(clock,wall)
    m.observe(safety_console(clock if source is None else source) if text is None else text,clock,wall)
    m.actuator(clock,wall)


def test_normal_1hz_publication_2hz_queries_250hz_checks_with_jitter():
    m=monitor();last_query=-1;last_act=-1
    for clock in range(33000000,37000001,4000):
        wall=(clock-33000000)/1e6
        if clock>=33000000+(last_query+1)*500000:
            last_query+=1
            source=33000000+int((clock-33000000)/1040000)*1040000
            text=safety_console(clock)
            text=text.replace(f"estimator_status_flags\n estimator_status_flags\n    timestamp: {clock}",f"estimator_status_flags\n estimator_status_flags\n    timestamp: {source}")
            m.begin_query(clock,wall);m.observe(text,clock,wall)
        if clock>last_act:m.actuator(clock,wall);last_act=clock
        assert not m.decision(clock,wall)["safe_stop"]


def test_original_phase2_abort_snapshot_is_not_expired_when_transport_is_live():
    m=monitor();p=Path(__file__).resolve().parents[1]/"reports/px4_yaw_phase2_parts/yaw_03/smoke_summary.json"
    if not p.exists():pytest.skip("local-only immutable Phase2 regression log")
    s=json.loads(p.read_text());topics=module("yaw_validation").latest_topics(s["px4_console"][-65536:])
    assert topics["estimator_status_flags"]["timestamp"]==32192000
    # Simulate a complete query containing exactly the captured source fields.
    text="".join(f"TOPIC: {name}\n"+"".join(f"    {k}: {v}\n" for k,v in fields.items())+"\n" for name,fields in topics.items())
    deliver(m,33196000,1.,text=text)
    assert m.decision(33196000,1.)["allow_new_maneuver"]
    assert not module("yaw_validation").heading_topics_ready(topics,33196000)


def test_duplicates_do_not_renew_source_time_even_with_live_communication():
    m=monitor();deliver(m,33000000,0.)
    for clock in range(33004000,34656001,4000):
        wall=(clock-33000000)/1e6
        if (clock-33000000)%500000==0:
            m.begin_query(clock,wall);m.observe(safety_console(33000000),clock,wall)
        m.actuator(clock,wall)
        result=m.decision(clock,wall)
        if result["safe_stop"]:break
    assert result["safe_stop"] and clock==33652000
    assert result["reason"]=="source publication expired: vehicle_local_position"
    assert m.topics["estimator_status_flags"]["timestamp"]==33000000


@pytest.mark.parametrize("change",[
    ("cs_mag_fault: False","cs_mag_fault: True"),
    ("cs_gnss_fault: False","cs_gnss_fault: True"),
    ("cs_gnss_yaw_fault: False","cs_gnss_yaw_fault: True"),
    ("cs_gnss_hgt_fault: False","cs_gnss_hgt_fault: True"),
    ("failsafe: False","failsafe: True"),
    ("heading_good_for_control: True","heading_good_for_control: False"),
    ("cs_mag_aligned_in_flight: True","cs_mag_aligned_in_flight: False"),
    ("xy_valid: True","xy_valid: False"),
])
def test_real_bad_health_stops_without_grace_and_latches(change):
    m=monitor();deliver(m,33000000,0.)
    deliver(m,33500000,.5,text=safety_console(33500000).replace(*change))
    result=m.decision(33500000,.5)
    assert result["safe_stop"] and not result["allow_new_maneuver"]
    deliver(m,34000000,1.)
    assert m.decision(34000000,1.)["safe_stop"]


@pytest.mark.parametrize("stamp",[32900000,33600000,float("nan")])
def test_backwards_future_or_invalid_publication_stops(stamp):
    m=monitor();deliver(m,33000000,0.)
    deliver(m,33500000,.5,stamp)
    assert m.decision(33500000,.5)["safe_stop"]


def test_partial_query_forbids_new_maneuver_then_times_out_without_reusing_health():
    m=monitor();deliver(m,33000000,0.)
    m.begin_query(33500000,.5);m.observe("TOPIC: estimator_status_flags\n    timestamp: 33500000\n",33500000,.5)
    m.actuator(33500000,.5)
    assert not m.decision(33500000,.5)["allow_new_maneuver"]
    m.actuator(33652000,.652)
    assert m.decision(33652000,.652)["safe_stop"]


def test_complete_missing_health_field_is_not_a_fragment_and_stops():
    m=monitor();deliver(m,33000000,0.)
    deliver(m,33500000,.5,text=safety_console(33500000).replace("    cs_gnss_fault: False\n",""))
    assert m.decision(33500000,.5)["safe_stop"]


def test_communication_interruption_cannot_reuse_cached_healthy_state():
    m=monitor();deliver(m,33000000,0.)
    result=m.decision(33252000,.252)
    assert result["safe_stop"] and "actuator" in result["reason"]


def test_query_publication_and_receipt_times_are_separate():
    m=monitor();m.begin_query(33000000,0.)
    m.observe(safety_console(32900000),33020000,.02);m.actuator(33020000,.02)
    r=m.decision(33020000,.02)
    assert r["allow_new_maneuver"]
    assert r["topics"]["estimator_status_flags"]=={"publication_usec":32900000.,"sample_usec":None,"received_clock_usec":33020000,"received_wall_s":.02,"source_age_s":.12}


@pytest.mark.parametrize("clock,wall",[(32999999,.01),(34000000,.01),(33004000,-.01)])
def test_clock_rollback_or_jump_stops(clock,wall):
    m=monitor();deliver(m,33000000,0.);m.decision(33000000,0.)
    assert m.decision(clock,wall)["safe_stop"]


def test_duplicate_changed_payload_is_not_a_new_publication():
    m=monitor();deliver(m,33000000,0.)
    deliver(m,33500000,.5,text=safety_console(33000000,heading=False))
    assert m.decision(33500000,.5)["safe_stop"]


def test_sample_clock_regression_and_future_sample_stops():
    for sample in (32900000,33510000):
        m=monitor()
        deliver(m,33000000,0.,text=safety_console(33000000).replace("    timestamp: 33000000","    timestamp: 33000000\n    timestamp_sample: 32950000"))
        deliver(m,33500000,.5,text=safety_console(33500000).replace("    timestamp: 33500000",f"    timestamp: 33500000\n    timestamp_sample: {sample}"))
        assert m.decision(33500000,.5)["safe_stop"]


def test_new_protocol_pending_query_cannot_start_yaw_and_rejects_bad_health():
    from types import SimpleNamespace
    r=module("yaw_telemetry_runner")
    p=r.TelemetryYawProtocol({"field_ned_gauss":[.2,.01,.4]},active=True)
    data=SimpleNamespace(time=32.,qpos=np.array([0,0,2.051,1,0,0,0.]),qvel=np.zeros(6))
    deliver(p.monitor,33000000,0.)
    p.monitor.begin_query(33000000,0.)
    p.before_step(data,33000000,"",wall=0.)
    assert p.step_time is None
    p.monitor.observe(safety_console(33000000),33000000,0.)
    p.before_step(data,33000000,"",wall=0.)
    assert p.step_time==32. and p.command_yaw==pytest.approx(np.deg2rad(100))
    deliver(p.monitor,33004000,.004,text=safety_console(33004000,aligned=False))
    with pytest.raises(ValueError):p.before_step(data,33004000,"",wall=.004)


def test_observed_unhealthy_partial_batch_stops_before_batch_completes():
    m=monitor();deliver(m,33000000,0.)
    m.begin_query(33004000,.004)
    # Complete unsafe first topic, other two not yet returned.
    text=safety_console(33004000,heading=False).split("TOPIC: estimator_status_flags")[0]
    m.observe(text,33004000,.004);m.actuator(33004000,.004)
    assert m.decision(33004000,.004)["safe_stop"]


def test_flags_only_stall_is_detected_in_1654ms_despite_fresh_other_topics():
    m=monitor();deliver(m,33000000,0.)
    for clock in range(33004000,34656001,4000):
        wall=(clock-33000000)/1e6
        if (clock-33000000)%500000==0:
            text=safety_console(clock).replace(f"estimator_status_flags\n estimator_status_flags\n    timestamp: {clock}","estimator_status_flags\n estimator_status_flags\n    timestamp: 33000000")
            m.begin_query(clock,wall);m.observe(text,clock,wall)
        m.actuator(clock,wall)
        decision=m.decision(clock,wall)
        if decision["safe_stop"]:break
    assert clock==34652000 and decision["safe_stop"]
    assert "publication expired" in decision["reason"]


def test_saved_phase2_ulog_replay_separates_sampling_publication_and_query():
    import sys
    sys.path.append("/home/keyousuke/.local/lib/python3.10/site-packages")
    from pyulog import ULog
    p=Path(__file__).resolve().parents[1]/"reports/px4_yaw_phase2_parts/yaw_03"
    if not p.exists():pytest.skip("local-only Phase2 ULog replay")
    from .telemetry_freshness import TelemetryFreshness
    u=ULog(str(next(p.rglob("*.ulg"))))
    sources={d.name:d.data for d in u.data_list if d.multi_id==0 and d.name in TelemetryFreshness.SOURCE_LIMITS}
    m=monitor();false_old=0
    # Last5s before the old32.196s interruption; no inferred/faked health values.
    for clock in range(28000000,33196001,4000):
        wall=(clock-28000000)/1e6
        if (clock-28000000)%500000==0:
            text=""
            for name,data in sources.items():
                i=np.searchsorted(data["timestamp"],clock,side="right")-1
                assert i>=0
                text+=f"TOPIC: {name}\n"
                for key,values in data.items():
                    if "[" in key:continue
                    value=values[i]
                    if key.startswith(("cs_","fs_")) or key in ("xy_valid","z_valid","v_xy_valid","v_z_valid","heading_good_for_control","pre_flight_checks_pass","failsafe"):
                        value=bool(value)
                    text+=f"    {key}: {value}\n"
                text+="\n"
            m.begin_query(clock,wall);m.observe(text,clock,wall)
        m.actuator(clock,wall)
        assert not m.decision(clock,wall)["safe_stop"]
        false_old+=not module("yaw_validation").heading_topics_ready(m.topics,clock)
    assert false_old>0


def test_no_query_delivery_times_out_in154ms_at250hz():
    m=monitor();deliver(m,33000000,0.);m.decision(33000000,0.)
    m.begin_query(33004000,.004)
    for clock in range(33004000,33164001,4000):
        wall=(clock-33000000)/1e6;m.actuator(clock,wall)
        if m.decision(clock,wall)["safe_stop"]:break
    assert clock==33156000


def test_frozen_simulation_clock_still_detects_dead_transport_in_wall_time():
    m=monitor();deliver(m,33000000,0.)
    assert m.decision(33000000,.252)["safe_stop"]


@pytest.mark.parametrize("fragment",["TOPIC: estimator_status_f","TOPIC: estimator_status_flags 3 inst","TOP"])
def test_unsafe_completed_topic_survives_fragmented_following_title(fragment):
    m=monitor();deliver(m,33000000,0.)
    m.begin_query(33004000,.004)
    first=safety_console(33004000,heading=False).split("TOPIC: estimator_status_flags")[0]
    m.observe(first+fragment,33004000,.004);m.actuator(33004000,.004)
    assert m.decision(33004000,.004)["safe_stop"]
    assert not m.topics["vehicle_local_position"]["heading_good_for_control"]


def test_late_complete_response_cannot_erase_expired_wall_deadline():
    m=monitor();deliver(m,33000000,0.)
    m.begin_query(33004000,.004)
    m.observe(safety_console(33008000),33008000,.205)
    m.actuator(33008000,.205)
    r=m.decision(33008000,.205)
    assert r["safe_stop"] and not r["allow_new_maneuver"]
    assert "query response timeout" in r["reason"]


def test_late_actuator_cannot_renew_expired_liveness():
    m=monitor();deliver(m,33000000,0.)
    m.actuator(33004000,.3)
    assert m.decision(33004000,.3)["safe_stop"]


def test_blocking_receive_uses_pending_query_deadline_and_services_console():
    from types import SimpleNamespace
    r=module("yaw_telemetry_runner")
    protocol=r.TelemetryYawProtocol({"field_ned_gauss":[.2,.01,.4]},active=True)
    deliver(protocol.monitor,33000000,0.)
    protocol.monitor.begin_query(33004000,.004)
    protocol.before_step(SimpleNamespace(time=32.,qpos=np.array([0,0,2.051,1,0,0,0.]),qvel=np.zeros(6)),33004000,"",wall=.004)
    runner=module("simulation_runner")
    # Model lockstep clock stalled; wall advances. Real receive function must
    # poll diagnostics without letting a250ms actuator wait hide a150ms query.
    class Clock:
        now=100.004
        def monotonic(self):return self.now
    clock=Clock();calls=[]
    class Wire:
        def receive_until(self,kind,timeout):
            calls.append(timeout);clock.now+=timeout;raise TimeoutError("slice expired")
    def service():
        wall=clock.now-100
        if wall>=.205:protocol.monitor.observe(safety_console(33008000),33008000,wall)
        protocol.before_step(SimpleNamespace(time=32.,qpos=np.array([0,0,2.051,1,0,0,0.]),qvel=np.zeros(6)),33004000,"",wall=wall)
    from unittest.mock import patch
    with patch.object(runner.time,"monotonic",clock.monotonic):
        with pytest.raises((TimeoutError,ValueError)):
            runner.receive_current_actuator(Wire(),33004000,True,100.25,
                service=service,deadline_provider=lambda:100+protocol.monitor.next_wall_deadline())
    assert clock.now<=100.158001
    assert calls and max(calls)<=.008000001
