"""Offline checks of the disarmed-only odometry wiring and frozen parameters."""
import json
import mujoco
import pytest

from .simulation_runner import MODEL, CONFIG
from .actuator_adapter import ActuatorAdapter
from .sensor_adapter import SensorAdapter
from .test_interfaces import module
from .compatibility_identity import MANIFEST
from .test_closed_loop import GOOD
from .diagnostics import sha256


def test_odometry_startup_params_keep_gnss_and_restore_historical_runner():
    runner = module("simulation_runner")
    fusion = module("odometry_fusion_runner")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    adapter = ActuatorAdapter(model, CONFIG)
    original = (runner.px4_environment, runner.sensor_messages, runner.LoopbackTransport)
    with fusion.isolated_odometry_hooks(fusion.GroundTruthOdometryProducer(), []):
        env = runner.px4_environment(adapter)
    assert (runner.px4_environment, runner.sensor_messages, runner.LoopbackTransport) == original
    assert env["PX4_PARAM_EKF2_EV_CTRL"] == "5"
    assert env["PX4_PARAM_EKF2_GPS_CTRL"] == "7"
    assert env["PX4_PARAM_EKF2_HGT_REF"] == "1"
    assert env["PX4_PARAM_SDLOG_PROFILE"] == "131"
    assert "PX4_PARAM_EKF2_EV_CTRL" not in runner.px4_environment(adapter)


def test_odometry_packet_is_appended_without_replacing_hil_gps():
    runner = module("simulation_runner")
    fusion = module("odometry_fusion_runner")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    data.qpos[2] = .051
    sensor = SensorAdapter(model).sample(data, 1_000_000)
    mavlink = module("transport").load_mavlink(str(module("test_transport").SEARCH))
    with fusion.isolated_odometry_hooks(fusion.GroundTruthOdometryProducer(), []):
        messages = runner.sensor_messages(sensor, mavlink, include_gps=True)
    assert [message.get_type() for message in messages] == ["HIL_SENSOR", "HIL_GPS", "ODOMETRY"]
    assert messages[-1].time_usec == sensor.time_usec


def test_scoped_hooks_restore_all_historical_runtime_bindings_on_error():
    runner = module("simulation_runner")
    fusion = module("odometry_fusion_runner")
    original = (runner.px4_environment, runner.sensor_messages,
                runner.LoopbackTransport, runner.atomic_json)
    with pytest.raises(RuntimeError, match="injected stop"):
        with fusion.isolated_odometry_hooks(fusion.GroundTruthOdometryProducer(), []):
            raise RuntimeError("injected stop")
    assert (runner.px4_environment, runner.sensor_messages,
            runner.LoopbackTransport, runner.atomic_json) == original


def test_odom_summary_is_nonflight_kind_and_cannot_rearm_as_old_proof(tmp_path, monkeypatch):
    runner = module("simulation_runner")
    fusion = module("odometry_fusion_runner")
    summary_path = tmp_path/"smoke_summary.json"
    history = {"model_xml":sha256(MODEL), "rotor_config":sha256(CONFIG),
               "rotor_actuators":sha256(MODEL.parent/"rotor_actuators.xml")}
    result = {"kind":"disarmed_communication_smoke", "flight_command_sent":False,
              "complete":True, "history_hashes_after":history,
              "px4":{"binary_sha256":"test-identity"}, "px4_console":GOOD}
    with fusion.isolated_odometry_hooks(fusion.GroundTruthOdometryProducer(), []):
        runner.atomic_json(summary_path, result)
    assert json.loads(summary_path.read_text())["kind"] == "disarmed_odometry_fusion_verification"
    assert result["kind"] == "disarmed_odometry_fusion_verification"
    assert result["complete"] and result["history_hashes_after"] == history
    assert module("closed_loop").estimator_ready(result["px4_console"])
    monkeypatch.setattr(runner, "source_identity", lambda *args: {"binary_sha256":"test-identity"})
    def never_launch(*args, **kwargs):
        raise AssertionError("proof must be rejected before launching PX4")
    monkeypatch.setattr(runner.subprocess, "Popen", never_launch)
    with pytest.raises(ValueError, match="flight requires successful matching disarmed smoke"):
        runner.smoke(tmp_path/"unused_px4", tmp_path/"never_start_px4",
                     seconds=22., hover_proof=summary_path, compatibility_manifest=MANIFEST)


def test_float_loop_cannot_add_a_sensor_cycle_beyond_twenty_seconds():
    fusion=module("odometry_fusion_runner")
    def cycles(stop):
        time=0.;count=0
        while time<stop:
            time+=.002;time+=.002;count+=1
        return count
    assert cycles(20.)==5001 # reproduces the saved single-run boundary bug
    assert cycles(fusion.SMOKE_DURATION_S)==5000
    assert cycles(fusion.SMOKE_DURATION_S)*4000==20_000_000
