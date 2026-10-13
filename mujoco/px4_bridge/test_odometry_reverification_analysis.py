"""Corrupt or cut logs cannot certify direct receipt and fusion."""
import struct
import pytest

from .analyze_odometry_reverification import ulog_framing,log_is_complete,artifact_hashes_match


def encoded():
    header=b"ULog\x01\x12\x35\x01"+struct.pack("<Q",1_000_000)
    return header+struct.pack("<HB",5,ord("I"))+b"hello"


def test_complete_bounded_record_walk(tmp_path):
    p=tmp_path/"log.ulg";p.write_bytes(encoded())
    result=ulog_framing(p)
    assert result["complete"] and result["records"]==1


@pytest.mark.parametrize("cut",[1,2,4,7,9,19])
def test_partial_record_or_header_is_rejected(tmp_path,cut):
    p=tmp_path/"log.ulg";p.write_bytes(encoded()[:-cut])
    assert not ulog_framing(p)["complete"]


def test_pyulog_corruption_with_zero_dropout_not_accepted():
    metadata={"framing":{"complete":True},"file_corruption":True,"dropouts":0}
    assert not log_is_complete(metadata)
    metadata["file_corruption"]=False
    assert log_is_complete(metadata)


def test_pyulog_short_payload_without_corrupt_flag_not_accepted():
    metadata={"framing":{"complete":False},"file_corruption":False,"dropouts":0}
    assert not log_is_complete(metadata)


def test_complete_record_boundary_truncation_breaks_closed_raw_identity(tmp_path):
    from .diagnostics import sha256
    p=tmp_path/"log.ulg";p.write_bytes(encoded()+struct.pack("<HB",5,ord("I"))+b"world")
    (tmp_path/"px4_console.bin").write_bytes(b"immutable")
    expected={"log.ulg":sha256(p),"px4_console.bin":sha256(tmp_path/"px4_console.bin")}
    assert artifact_hashes_match(tmp_path,expected)
    p.write_bytes(encoded())
    assert ulog_framing(p)["complete"]  # framing alone cannot detect this case
    assert not artifact_hashes_match(tmp_path,expected)


def test_console_only_manifest_cannot_certify_log_identity(tmp_path):
    from .diagnostics import sha256
    p=tmp_path/"px4_console.bin";p.write_bytes(b"log")
    assert not artifact_hashes_match(tmp_path,{"px4_console.bin":sha256(p)})


@pytest.mark.parametrize("missing",[None,"vehicle_visual_odometry","estimator_aid_src_ev_pos","estimator_aid_src_ev_vel"])
def test_saved_new_run_recomputes_and_missing_direct_evidence_blocks(tmp_path,monkeypatch,missing):
    from pathlib import Path
    from . import analyze_odometry_reverification as analysis
    runtime=Path(__file__).resolve().parents[1]/"reports/px4_odometry_reverification_parts/disarmed_01"
    if not (runtime/"odometry_source.json").is_file():pytest.skip("one authorized run is local-only")
    original=analysis.read_log
    def changed(path):
        topics,params,metadata=original(path)
        if missing:topics.pop(missing,None)
        return topics,params,metadata
    monkeypatch.setattr(analysis,"read_log",changed)
    monkeypatch.setattr(analysis,"plots",lambda *args:[])
    report=analysis.analyze(runtime,tmp_path/"report.json",tmp_path/"figures")
    if missing:
        assert report["overall"]!="PASS"
    else:
        assert report["overall"]=="PASS"
        assert report["identity"]["summary_sha256"]=="b03fd4e09ed79794f91e7d9923c4e714690139fb6a6911791a697a4ae134aeae"
        assert report["evidence"]["direct"]["samples"]==985
        assert report["runtime"]["messages_sent"]=={"HIL_SENSOR":5000,"HIL_GPS":200,"ODOMETRY":1000}


@pytest.mark.parametrize("missing",["offboard","failsafe"])
def test_no_arm_flag_alone_does_not_prove_no_active_control(missing):
    from .analyze_odometry_reverification import passive_status
    topics={"vehicle_control_mode":{"flag_control_offboard_enabled":[0,0]},
            "vehicle_status":{"failsafe":[0,0]}}
    assert passive_status(topics)
    if missing=="offboard":topics["vehicle_control_mode"]["flag_control_offboard_enabled"]=[0,1]
    else:topics["vehicle_status"]["failsafe"]=[0,1]
    assert not passive_status(topics)
