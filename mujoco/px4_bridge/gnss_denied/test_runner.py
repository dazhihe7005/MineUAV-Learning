"""No network peer/PX4 launch: message, configuration and identity boundaries."""
import json
from pathlib import Path
import numpy as np
import pytest

from . import runner
from .. import simulation_runner as sim
from ..test_transport import SEARCH
from ..transport import load_mavlink


def test_no_gps_even_on_original_gps_scheduled_ticks():
    import mujoco
    from ..sensor_adapter import SensorAdapter
    from ..odometry_producer import GroundTruthOdometryProducer
    m=mujoco.MjModel.from_xml_path(str(sim.MODEL));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    adapter=SensorAdapter(m);producer=GroundTruthOdometryProducer();mv=load_mavlink(str(SEARCH))
    counts={}
    for tick in range(5000):
        sample=adapter.sample(d,1_000_000+4000*tick)
        for msg in runner.sensor_messages_no_gps(sample,mv,producer):
            counts[msg.get_type()]=counts.get(msg.get_type(),0)+1
    assert counts=={"HIL_SENSOR":5000,"ODOMETRY":1000}
    assert producer.samples[0]["sample_time_usec"]==1_000_000
    assert producer.samples[-1]["sample_time_usec"]==20_980_000


def test_send_whitelist_rejects_gps_and_control_before_wire_write():
    mv=load_mavlink(str(SEARCH));sent=[]
    with runner.GnssDeniedTransport(port=0,mavlink=mv,sent_samples=sent) as wire:
        for kind in ("HIL_GPS","GPS_INPUT","COMMAND_LONG","SET_POSITION_TARGET_LOCAL_NED"):
            class Message:
                def get_type(self):return kind
            with pytest.raises(ValueError,match="forbidden"):
                wire.send(Message())
        assert wire.sent.get("HIL_GPS")==0 and sent==[]


def test_hooks_restore_and_use_disarmed_identity_and_private_logger(tmp_path):
    from ..odometry_producer import GroundTruthOdometryProducer
    root=tmp_path/"rootfs";root.mkdir();(root/"rcS").write_text("untouched\n")
    before=(sim.sensor_messages,sim.px4_environment,sim.prepare_runtime,sim.LoopbackTransport,sim.atomic_json)
    metadata={};producer=GroundTruthOdometryProducer()
    with pytest.raises(RuntimeError):
        with runner.hooks(root,producer,[],metadata,"protocol"):
            runtime=sim.prepare_runtime(tmp_path/"runtime")
            text=(runtime/"etc/logging/logger_topics.txt").read_text()
            assert "estimator_aid_src_gnss_hgt 0 0" in text
            assert "estimator_aid_src_baro_hgt 0 0" in text
            assert not (root/"logging/logger_topics.txt").exists()
            sim.atomic_json(runtime/"smoke_summary.json",{"kind":"disarmed_communication_smoke","flight_command_sent":False})
            raw=json.loads((runtime/"smoke_summary.json").read_text())
            assert raw["kind"]==runner.KIND and raw["not_flight_proof"] is True
            assert raw["protocol_sha256"]=="protocol"
            assert raw["gps_hz"]==0
            raise RuntimeError("probe")
    assert before==(sim.sensor_messages,sim.px4_environment,sim.prepare_runtime,sim.LoopbackTransport,sim.atomic_json)


def test_startup_configuration_keeps_only_audited_fusion_changes(monkeypatch):
    monkeypatch.setattr(runner,"base_environment",lambda adapter:{"PX4_PARAM_EKF2_GPS_CTRL":"7","PX4_PARAM_EKF2_HGT_REF":"1"})
    env=runner.environment(None)
    assert env["PX4_PARAM_EKF2_GPS_CTRL"]=="0"
    assert env["PX4_PARAM_EKF2_HGT_REF"]=="0"
    assert env["PX4_PARAM_EKF2_EV_CTRL"]=="5"
    assert env["PX4_PARAM_SDLOG_PROFILE"]==env["PX4_PARAM_SDLOG_MISSION"]=="0"


def test_current_protocol_rejects_changed_runtime_model_identity(monkeypatch):
    monkeypatch.setattr(runner,"protocol_content",lambda *args:{"runtime_hashes":{"new":"hash"}})
    with pytest.raises(ValueError,match="frozen"):
        runner.verify_protocol({"runtime_hashes":{"old":"hash"}},None,None)


def test_exclusive_attempt_cannot_be_reused(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"ATTEMPT_PATH",tmp_path/"campaign.attempt.json")
    p=tmp_path/"protocol.json";p.write_text("{}")
    runner.reserve_attempt(p,tmp_path/"runtime")
    with pytest.raises(FileExistsError):runner.reserve_attempt(p,tmp_path/"another_runtime")


def test_alternate_protocol_filename_cannot_bypass_campaign_lock(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"ATTEMPT_PATH",tmp_path/"campaign.attempt.json")
    first=tmp_path/"one.json";second=tmp_path/"two.json"
    first.write_text("{}");second.write_text("{}")
    runner.reserve_attempt(first,tmp_path/"one")
    with pytest.raises(FileExistsError):runner.reserve_attempt(second,tmp_path/"two")


def test_historical_runtime_inventory_remains_unchanged():
    from ..odometry_reverification_runner import verify_protocol
    from ..compatibility_identity import MANIFEST
    p=sim.ROOT/"reports/px4_odometry_reverification_parts/protocol.json"
    if not p.is_file():pytest.skip("historical proof is local-only")
    verify_protocol(json.loads(p.read_text()),Path("/home/keyousuke/PX4-Autopilot-mineuav-yaw"),MANIFEST)


def test_full_logger_list_is_finite_explicit_and_contains_sources():
    from ..logger_configuration import audit_topics
    text=runner.logger_text();names={line.split()[0] for line in text.splitlines()}
    audit=audit_topics(text,names)
    assert audit["subscriptions"]==27 and audit["instances"]==[0]
    assert "sensor_gps" in names and "estimator_aid_src_mag" in names
