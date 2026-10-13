"""One disarmed, GNSS-retained PX4/MuJoCo odometry run; no arm path."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time

import mujoco

from .actuator_adapter import ActuatorAdapter
from .allocation_math import effectiveness
from .allocation_runner import AllocationYawProtocol, allocation_status
from .magnetic_model import field_from_px4
from .odometry_producer import GroundTruthOdometryProducer
from . import simulation_runner as sim
from .diagnostics import atomic_json, sha256

PARAM_OVERRIDES = {"PX4_PARAM_EKF2_EV_CTRL": "5", "PX4_PARAM_EKF2_GPS_CTRL": "7",
                   "PX4_PARAM_EKF2_HGT_REF": "1", "PX4_PARAM_SDLOG_PROFILE": "131"}
# The unchanged historical smoke loops on accumulated floating d.time. A 1us
# comparison guard (<4ms sensor period) prevents the observed 5001st cycle;
# the last completed integer-clock cycle is exactly 20,000,000us of physics.
# This post-run fix has offline evidence only; the sole run used literal20.
SMOKE_DURATION_S=20.-1e-6
RUNTIME_FILES=("odometry_fusion_runner.py", "odometry_producer.py", "external_odometry.py",
               "frame_conversions.py", "simulation_runner.py")
CHECKPOINTS=("uav_bc_mlp_seed0.pt", "uav_bc_yaw_augmented_seed0.pt",
             "joint_latent_world_model_v3_autonomous_consistency.pt") + tuple(
                 f"uav_ppo_fixed_protocol_seed{seed}.zip" for seed in range(5))


def runtime_hashes():
    return {name:sha256(Path(__file__).with_name(name)) for name in RUNTIME_FILES}


def checkpoint_hashes():
    root=sim.ROOT/"rl/models"
    return {name:sha256(root/name) for name in CHECKPOINTS}


@contextmanager
def isolated_odometry_hooks(producer, sent):
    """Extend only this single-process invocation; historical smoke bytes stay intact.

    The original loop applies the same bounded sensor send and safety service
    to each appended ODOMETRY packet. Hooks are restored even after failure.
    """
    if threading.active_count() != 1:
        raise RuntimeError("odometry hooks require a single-threaded runner")
    original_messages = sim.sensor_messages
    original_environment = sim.px4_environment
    original_transport = sim.LoopbackTransport
    original_atomic_json = sim.atomic_json

    def messages(sample, mavlink, include_gps=True):
        output = original_messages(sample, mavlink, include_gps=include_gps)
        odometry = producer.tick(sample, mavlink)
        if odometry is not None:
            output.append(odometry)
        return output

    def environment(adapter):
        output = original_environment(adapter)
        if any(name in output and output[name] != value for name, value in PARAM_OVERRIDES.items()):
            raise ValueError("conflicting PX4 startup parameter")
        output.update(PARAM_OVERRIDES)
        return output

    class ObservedTransport(original_transport):
        def send(self, msg, timeout=None):
            super().send(msg, timeout=timeout)
            if msg.get_type() == "ODOMETRY":
                sent.append({"sample_time_usec": int(msg.time_usec),
                             "send_monotonic_s": time.monotonic()})

    def save_summary(path, value):
        if Path(path).name == "smoke_summary.json":
            if value.get("kind") != "disarmed_communication_smoke" or value.get("flight_command_sent") is not False:
                raise ValueError("odometry hook may only save an unarmed disarmed summary")
            # Set at creation, not as a post-hoc report edit: every old
            # hover-proof gate rejects this new kind before any PX4 launch.
            value["kind"] = "disarmed_odometry_fusion_verification"
            value["not_flight_proof"] = True
        return original_atomic_json(path, value)

    try:
        sim.sensor_messages = messages
        sim.px4_environment = environment
        sim.LoopbackTransport = ObservedTransport
        sim.atomic_json = save_summary
        yield
    finally:
        sim.sensor_messages = original_messages
        sim.px4_environment = original_environment
        sim.LoopbackTransport = original_transport
        sim.atomic_json = original_atomic_json


def run(px4_root, runtime, compatibility_manifest, pymavlink_search_path=None):
    # Strict allowlisted source/build verification occurs again inside smoke.
    identity = sim.source_identity(px4_root, compatibility_manifest)
    runtime_before=runtime_hashes();checkpoints_before=checkpoint_hashes()
    model = mujoco.MjModel.from_xml_path(str(sim.MODEL))
    adapter = ActuatorAdapter(model, sim.CONFIG)
    yaw_row = effectiveness(adapter)[2]
    del adapter, model
    protocol = AllocationYawProtocol(field_from_px4(px4_root), yaw_row, active=False)
    producer = GroundTruthOdometryProducer()
    sent = []
    with isolated_odometry_hooks(producer, sent):
        result = sim.smoke(px4_root, runtime, SMOKE_DURATION_S, pymavlink_search_path,
                           phase2=protocol, compatibility_manifest=compatibility_manifest)
    # Raw runtime data are local-only; preserve the historical smoke summary
    # verbatim and bind this independent source manifest to its exact bytes.
    identity_error=None
    try:identity_after=sim.source_identity(px4_root, compatibility_manifest)
    except ValueError as error:identity_after=None;identity_error=str(error)
    runtime_after=runtime_hashes();checkpoints_after=checkpoint_hashes()
    source = {"label": "SIMULATED GROUND TRUTH ODOMETRY", "sample_hz": 50,
              "pose_variance_diagonal": [.01]*6, "twist_variance_diagonal": [.01]*6,
              "source_samples": producer.samples, "sent_samples": sent,
              "summary_sha256": sha256(Path(runtime)/"smoke_summary.json"),
              "runtime_source_sha256": runtime_before,
              "runtime_source_hashes_after":runtime_after,
              "runtime_unchanged":runtime_before==runtime_after,
              "checkpoint_hashes_before":checkpoints_before,
              "checkpoint_hashes_after":checkpoints_after,
              "checkpoints_unchanged":checkpoints_before==checkpoints_after,
              "px4_identity_after":identity_after,"px4_identity_error":identity_error}
    atomic_json(Path(runtime)/"odometry_source.json", source)
    result["odometry"] = source
    result["actual_yaw_row_valid"] = allocation_status(result.get("px4_console", ""), yaw_row)
    result["source_identity_rechecked"] = result["px4"] == identity == identity_after
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True, help="new empty runtime directory")
    parser.add_argument("--compatibility-manifest", type=Path, required=True)
    parser.add_argument("--pymavlink-search-path")
    args = parser.parse_args()
    result = run(args.px4_root, args.runtime, args.compatibility_manifest,
                 args.pymavlink_search_path)
    print(json.dumps({key: result.get(key) for key in (
        "complete", "exception", "simulation_seconds", "messages_sent",
        "messages_received", "flight_command_sent", "actual_yaw_row_valid",
        "source_identity_rechecked")}, ensure_ascii=False))
    return 0 if (result["complete"] and not result["flight_command_sent"]
                 and result["actual_yaw_row_valid"] and result["source_identity_rechecked"]
                 and result.get("messages_sent", {}).get("ODOMETRY") == 1000) else 1


if __name__ == "__main__":
    raise SystemExit(main())
