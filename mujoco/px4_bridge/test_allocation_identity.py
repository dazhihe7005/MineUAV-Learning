"""Identity and allocation proof regressions: no PX4 launch."""
from pathlib import Path
import json
import numpy as np
import pytest
from .test_interfaces import module,fixture,ROOT


def test_original_identity_remains_available_and_custom_requires_explicit_manifest():
    original=Path.home()/"PX4-Autopilot"
    assert module("simulation_runner").source_identity(original)["commit"]==module("simulation_runner").PX4_COMMIT
    with pytest.raises(ValueError):
        module("simulation_runner").source_identity(Path.home()/"PX4-Autopilot-mineuav-yaw")


def test_missing_or_incomplete_custom_manifest_is_rejected(tmp_path):
    f=tmp_path/"identity.json";f.write_text("{}")
    with pytest.raises(ValueError,match="identity"):
        module("compatibility_identity").verified_identity(tmp_path,f)


def test_allocation_status_requires_real_signed_yaw_row():
    r=module("allocation_runner")
    expected=np.array([-.017207708,.017207708,-.017207708,.017207708])
    prefix="INFO  [control_allocator] MineUAV yaw axis: "
    assert r.allocation_status(prefix+" ".join(map(str,expected)),expected)
    for text in ("",prefix+"0 0 0 0",prefix+"nan .0172 -.0172 .0172",prefix+"-.0172 .0172"):
        assert not r.allocation_status(text,expected)
    # A later zero/invalid diagnostic may not be hidden by an earlier good row.
    assert not r.allocation_status(prefix+" ".join(map(str,expected))+"\n"+prefix+"0 0 0 0",expected)


def test_allocation_status_accepts_real_px4_ansi_but_rejects_later_invalid_row():
    r=module("allocation_runner");expected=np.array([-.017207708,.017207708,-.017207708,.017207708])
    # Exact numeric line shape observed in disarmed_01; color bytes are not values.
    line="\x1b[0mINFO  \x1b[37m[control_allocator] \x1b[0mMineUAV yaw axis: -0.017207708 0.017207708 -0.017207708 0.017207708\x1b[0m\r\n"
    assert r.allocation_status(line,expected)
    assert not r.allocation_status(line+"MineUAV yaw axis: 0 0 0 0\x1b[0m\r\n",expected)
    assert not r.allocation_status(line+"MineUAV yaw axis: -.017207708 .017207708 -.017207708 nan\x1b[0m\r\n",expected)


def test_disarmed_proof_requires_same_binary_model_and_allocator_matrix():
    m,_=fixture();a=module("actuator_adapter").ActuatorAdapter(m,ROOT/"models/rotor_config.json")
    b=module("allocation_math").effectiveness(a)
    identity={"custom_sitl":True,"commit":"verified","rootfs_sha256":"root"};history={"model":"frozen"}
    protocol=module("allocation_runner").AllocationYawProtocol({"field_ned_gauss":[.2,.01,.4]},b[2],identity=identity,history=history)
    proof=valid_proof(identity,history)
    proof["phase2"]["field_sha256"]=protocol.field_hash
    with pytest.raises(ValueError,match="allocation"):protocol.validate_proof(proof)
    proof["px4_console"]="MineUAV yaw axis: "+" ".join(f"{x:.9f}" for x in b[2])
    protocol.validate_proof(proof)
    assert "control_allocator status" in protocol.queries()


def valid_proof(identity,history):
    return {"kind":"disarmed_communication_smoke","complete":True,"simulation_seconds":16.,
        "final_physical_state":{"time_s":16.},"flight_command_sent":False,"exception":None,
        "px4_exit_code":0,"history_unchanged":True,"px4":identity,
        "history_hashes_before":history,"history_hashes_after":history,
        "messages_sent":{"HIL_SENSOR":4000,"HIL_GPS":160},
        "messages_received":{"HIL_ACTUATOR_CONTROLS":3900},
        "phase2":{"active":False,"telemetry_fix":{"fault":None,"last_decision":{"safe_stop":False,"query_pending":False,"topics":{"vehicle_local_position":{},"estimator_status_flags":{},"vehicle_status":{}}}}},
        "px4_console":""}


@pytest.mark.parametrize("key",["identity","history","duration","fault","active","missing_status","missing_actuator","incomplete","exception"])
def test_custom_proof_rejects_partial_faulted_or_mismatched_protocol_before_flight(key):
    r=module("allocation_runner");identity={"custom_sitl":True,"commit":"verified"};history={"model":"hash"}
    proof=valid_proof(identity,history)
    if key=="identity":proof["px4"]={**identity,"rootfs_sha256":"changed"}
    elif key=="history":proof["history_hashes_before"]={"model":"changed"}
    elif key=="duration":proof["simulation_seconds"]=8.
    elif key=="fault":proof["phase2"]["telemetry_fix"]["fault"]="fault"
    elif key=="active":proof["flight_command_sent"]=True
    elif key=="missing_status":proof["phase2"]["telemetry_fix"]["last_decision"]["topics"]={}
    elif key=="missing_actuator":proof["messages_received"]={}
    elif key=="incomplete":proof["complete"]=False
    elif key=="exception":proof["exception"]={"type":"failure"}
    with pytest.raises(ValueError,match="proof"):r.validate_disarmed_evidence(proof,identity,history)


def test_complete_matching_custom_disarmed_evidence_is_admitted():
    r=module("allocation_runner");identity={"custom_sitl":True,"commit":"verified"};history={"model":"hash"}
    r.validate_disarmed_evidence(valid_proof(identity,history),identity,history)


def test_active_protocol_fails_closed_when_actual_yaw_row_is_missing_or_removed():
    m,d=fixture();d.time=10
    a=module("actuator_adapter").ActuatorAdapter(m,ROOT/"models/rotor_config.json")
    b=module("allocation_math").effectiveness(a)
    r=module("allocation_runner");protocol=r.AllocationYawProtocol({"field_ned_gauss":[.2,.01,.4]},b[2],active=True)
    with pytest.raises(ValueError,match="allocation"):protocol.before_step(d,11000000,"",wall=1.)
    good="MineUAV yaw axis: "+" ".join(f"{x:.9f}" for x in b[2])
    protocol.before_step(d,11000000,good,wall=1.)
    with pytest.raises(ValueError,match="allocation"):
        protocol.before_step(d,11000000,good+"\nMineUAV yaw axis: 0 0 0 0",wall=1.)


def test_pinned_custom_manifest_rejects_each_identity_tamper(tmp_path):
    # Runs after compilation; all checks inspect the installed isolated binary.
    f=Path(__file__).parent/"compat/px4_yaw_build_identity.json"
    assert f.is_file(),"verified custom binary manifest missing"
    c=module("compatibility_identity");root=Path.home()/"PX4-Autopilot-mineuav-yaw"
    manifest=json.loads(f.read_text());ident=c.verified_identity(root,f)
    assert ident["custom_sitl"] and ident["commit"]==manifest["commit"]
    for key in ("commit","base_commit","binary_sha256","build_version_header_sha256","patch_sha256","rootfs_sha256","boardconfig_sha256"):
        bad=dict(manifest);bad[key]="0"*64
        p=tmp_path/(key+".json");p.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match="identity"):c.verified_identity(root,p)
    with pytest.raises(ValueError,match="identity"):
        c.verified_identity(Path.home()/"PX4-Autopilot",f)
