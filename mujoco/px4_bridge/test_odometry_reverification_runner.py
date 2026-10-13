"""Wrapper isolation and single-use protocol; never starts SITL in tests."""
import json
from pathlib import Path
import pytest

from . import odometry_reverification_runner as runner
from . import simulation_runner as sim


def test_runtime_hooks_restore_and_private_logger(tmp_path,monkeypatch):
    rootfs=tmp_path/"approved";rootfs.mkdir();(rootfs/"rcS").write_text("immutable\n")
    original=(sim.prepare_runtime,sim.px4_environment,sim.atomic_json,sim.LoopbackTransport)
    monkeypatch.setattr(sim,"px4_environment",lambda adapter:{"PX4_PARAM_SDLOG_PROFILE":"131"})
    before_env=sim.px4_environment
    metadata={};sent=[]
    with pytest.raises(RuntimeError,match="probe"):
        with runner.reverification_hooks(rootfs,metadata,sent,"protocol-id"):
            runtime=sim.prepare_runtime(tmp_path/"runtime")
            assert (runtime/"etc/logging/logger_topics.txt").is_file()
            env=sim.px4_environment(None)
            assert env["PX4_PARAM_SDLOG_PROFILE"]==env["PX4_PARAM_SDLOG_MISSION"]=="0"
            assert not (rootfs/"logging/logger_topics.txt").exists()
            sim.atomic_json(runtime/"smoke_summary.json",{"kind":"disarmed_communication_smoke","flight_command_sent":False})
            summary=json.loads((runtime/"smoke_summary.json").read_text())
            assert summary["kind"]==runner.KIND and summary["not_flight_proof"] is True
            assert summary["protocol_sha256"]=="protocol-id"
            raise RuntimeError("probe")
    assert sim.prepare_runtime is original[0] and sim.px4_environment is before_env
    assert sim.atomic_json is original[2] and sim.LoopbackTransport is original[3]


def test_freeze_check_rejects_runtime_mutation(monkeypatch):
    monkeypatch.setattr(runner,"protocol_content",lambda *args:{"runtime_hashes":{"source":"new"}})
    with pytest.raises(ValueError,match="frozen"):
        runner.verify_protocol({"runtime_hashes":{"source":"old"}},Path("root"),Path("manifest"))


def test_existing_runtime_cannot_consume_another_attempt(tmp_path):
    runtime=tmp_path/"attempt";runtime.mkdir();(runtime/"attempt.json").write_text("started\n")
    with pytest.raises(ValueError):runner.require_new_runtime(runtime)


def test_historical_protocol_and_old_flight_proof_rejected():
    for kind in ("disarmed_communication_smoke","disarmed_odometry_fusion_verification"):
        assert not runner.is_new_evidence({"kind":kind,"protocol_sha256":"p"},"p")
    assert not runner.is_new_evidence({"kind":runner.KIND,"protocol_sha256":"old"},"p")
    assert runner.is_new_evidence({"kind":runner.KIND,"protocol_sha256":"p","not_flight_proof":True},"p")


def test_no_arm_or_offboard_path_in_wrapper():
    import inspect
    code=inspect.getsource(runner.run)
    assert "hover_proof" not in code
    assert "SMOKE_DURATION_S" in code
    assert "active=False" in code


def test_second_attempt_for_frozen_protocol_rejected_even_in_new_runtime(tmp_path,monkeypatch):
    protocol=tmp_path/"protocol.json";protocol.write_text("{}")
    protocol.with_suffix(".attempt.json").write_text('{"attempt":1}')
    monkeypatch.setattr(runner,"verify_protocol",lambda *args:None)
    monkeypatch.setattr(sim,"require_startup_memory",lambda:None)
    with pytest.raises(FileExistsError):
        runner.run(Path("root"),tmp_path/"new_runtime",Path("manifest"),protocol)
