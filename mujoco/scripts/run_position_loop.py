"""Six XY/Z/yaw closed-loop tests using v2 model, shared attitude PD and allocator.

No PID integral, motor lag, controller learning, or lateral force actuator.
The logged position is the free-joint/body origin, not the estimated COM.
"""

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "control"))
from control_allocator import ControlAllocator, MODEL  # noqa: E402
from hover_controller import HoverController  # noqa: E402
from position_controller import PositionController  # noqa: E402
from hover_trim_test import euler_zyx  # noqa: E402


REPORTS = ROOT / "reports"
MASS_REPORT = REPORTS / "dynamics_v2_report.json"
SUMMARY = REPORTS / "position_loop_summary.json"
DURATION_S = 16.0
LOG_EVERY_STEPS = 10
EVENT_START_S = 2.0
EVENT_END_S = 2.1
HORIZONTAL_FORCE_N = 21.0
YAW_STEP_DEG = 45.0
FIELDS = (
    "time", "target_x", "target_y", "target_z", "target_yaw_deg",
    "x", "y", "z", "vx", "vy", "vz",
    "desired_ax", "desired_ay", "desired_az",
    "desired_roll_deg", "desired_pitch_deg", "desired_yaw_deg",
    "roll_deg", "pitch_deg", "yaw_deg", "wx", "wy", "wz",
    "Fz", "Tx", "Ty", "Tz", "rpm1", "rpm2", "rpm3", "rpm4",
    "position_error_norm", "speed_norm", "tilt_deg", "allocator_saturated",
)

CASES = {
    "A_x_plus_2": {"target": [2.0, 0.0, 1.0]},
    "B_y_plus_2": {"target": [0.0, 2.0, 1.0]},
    "C_xy_plus_2": {"target": [2.0, 2.0, 1.0]},
    "D_xyz_mixed": {"target": [-2.0, 1.0, 1.5]},
    "E_horizontal_impulse": {"target": [0.0, 0.0, 1.0], "impulse": True},
    "F_yaw_step": {"target": [0.0, 0.0, 1.0], "yaw_step": True},
}


def build_controllers(model: mujoco.MjModel) -> tuple[PositionController, ControlAllocator]:
    report = json.loads(MASS_REPORT.read_text(encoding="utf-8"))
    inertia = np.asarray(report["estimated_inertia_v2_kg_m2"], dtype=float)
    if not np.allclose(model.body("mine_uav").ipos, report["estimated_com_v2_m"], atol=1e-10):
        raise ValueError("Model COM differs from inertia report")
    if not np.allclose(model.opt.gravity, [0, 0, -9.81], atol=1e-12):
        raise ValueError("Expected world +Z up and gravity [0,0,-9.81]")
    allocator = ControlAllocator.from_v2_model()
    attitude = HoverController(
        mass_kg=mujoco.mj_getTotalmass(model), inertia_kg_m2=inertia,
        max_total_thrust_n=float(np.sum(allocator.B[0]) * allocator.u_max),
        gravity_m_s2=abs(float(model.opt.gravity[2])),
    )
    return PositionController(attitude, model.opt.gravity), allocator


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _sample(data: mujoco.MjData, command, allocation, target: np.ndarray,
            target_yaw: float) -> dict:
    actual_rpy = euler_zyx(data.qpos[3:7])
    desired_rpy = euler_zyx(command.desired_quaternion_wxyz)
    q = data.qpos[3:7]
    tilt = math.degrees(math.acos(float(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))))
    sample = {
        "time": float(data.time),
        "target_x": float(target[0]), "target_y": float(target[1]),
        "target_z": float(target[2]), "target_yaw_deg": math.degrees(target_yaw),
        "x": float(data.qpos[0]), "y": float(data.qpos[1]), "z": float(data.qpos[2]),
        "vx": float(data.qvel[0]), "vy": float(data.qvel[1]), "vz": float(data.qvel[2]),
        "desired_ax": float(command.acceleration_world[0]),
        "desired_ay": float(command.acceleration_world[1]),
        "desired_az": float(command.acceleration_world[2]),
        "desired_roll_deg": math.degrees(desired_rpy[0]),
        "desired_pitch_deg": math.degrees(desired_rpy[1]),
        "desired_yaw_deg": math.degrees(desired_rpy[2]),
        "roll_deg": math.degrees(actual_rpy[0]),
        "pitch_deg": math.degrees(actual_rpy[1]),
        "yaw_deg": math.degrees(actual_rpy[2]),
        "wx": float(data.qvel[3]), "wy": float(data.qvel[4]), "wz": float(data.qvel[5]),
        "Fz": float(command.desired_wrench_body[0]),
        "Tx": float(command.desired_wrench_body[1]),
        "Ty": float(command.desired_wrench_body[2]),
        "Tz": float(command.desired_wrench_body[3]),
        "rpm1": float(allocation.rpm[0]), "rpm2": float(allocation.rpm[1]),
        "rpm3": float(allocation.rpm[2]), "rpm4": float(allocation.rpm[3]),
        "position_error_norm": float(np.linalg.norm(data.qpos[:3] - target)),
        "speed_norm": float(np.linalg.norm(data.qvel[:3])),
        "tilt_deg": tilt,
        "allocator_saturated": int(allocation.saturated),
    }
    if not all(math.isfinite(value) for value in sample.values()):
        raise FloatingPointError(f"Non-finite state at t={data.time:.3f}")
    return sample


def simulate_case(model: mujoco.MjModel, controller: PositionController,
                  allocator: ControlAllocator, config: dict) -> tuple[list[dict], dict]:
    data = mujoco.MjData(model)
    data.qpos[:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    target = np.asarray(config["target"], dtype=float)
    body_id = model.body("mine_uav").id
    rows = []
    saturation_steps = 0
    max_rpm = 0.0
    max_tilt = 0.0
    total_steps = round(DURATION_S / model.opt.timestep)
    for step in range(total_steps):
        t = step * model.opt.timestep
        target_yaw = math.radians(YAW_STEP_DEG) if config.get("yaw_step") and t >= EVENT_START_S else 0.0
        data.xfrc_applied.fill(0.0)
        if config.get("impulse") and EVENT_START_S <= t < EVENT_END_S:
            data.xfrc_applied[body_id, 0] = HORIZONTAL_FORCE_N
        command = controller.compute(
            data.qpos[:3], data.qvel[:3], data.qpos[3:7], data.qvel[3:6],
            target_position=target, target_velocity=np.zeros(3), target_yaw=target_yaw)
        allocation = allocator.allocate(command.desired_wrench_body)
        data.ctrl[:] = allocation.u
        mujoco.mj_step(model, data)
        saturation_steps += int(allocation.saturated)
        max_rpm = max(max_rpm, float(np.max(allocation.rpm)))
        q = data.qpos[3:7]
        max_tilt = max(max_tilt, math.degrees(math.acos(
            float(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1)))))
        if step % LOG_EVERY_STEPS == 0 or step == total_steps - 1:
            rows.append(_sample(data, command, allocation, target, target_yaw))
    return rows, {"saturation_steps": saturation_steps,
                  "saturation_fraction": saturation_steps / total_steps,
                  "max_motor_rpm": max_rpm, "max_tilt_deg": max_tilt}


def summarize_case(rows: list[dict], extra: dict, config: dict) -> dict:
    event_end = EVENT_END_S if config.get("impulse") else EVENT_START_S if config.get("yaw_step") else 0.0
    suffix_in_band = True
    settling_time = None
    final_yaw = math.radians(YAW_STEP_DEG) if config.get("yaw_step") else 0.0
    for row in reversed(rows):
        yaw_ok = abs(_wrap_angle(math.radians(row["yaw_deg"]) - final_yaw)) < math.radians(2)
        suffix_in_band = (suffix_in_band
                          and row["position_error_norm"] < 0.05
                          and row["speed_norm"] < 0.05 and yaw_ok)
        if suffix_in_band and row["time"] >= event_end:
            settling_time = row["time"]
    final = rows[-1]
    return {
        "target_position_m": config["target"],
        "target_yaw_deg": YAW_STEP_DEG if config.get("yaw_step") else 0.0,
        "final_position_m": [final[k] for k in ("x", "y", "z")],
        "final_velocity_m_s": [final[k] for k in ("vx", "vy", "vz")],
        "final_position_error_norm_m": final["position_error_norm"],
        "final_speed_norm_m_s": final["speed_norm"],
        "final_yaw_error_deg": math.degrees(_wrap_angle(math.radians(final["yaw_deg"]) - final_yaw)),
        "settling_time_s": settling_time,
        "peak_position_error_m": max(row["position_error_norm"] for row in rows),
        "peak_post_event_error_m": (max(row["position_error_norm"] for row in rows
                                         if row["time"] >= event_end)
                                    if event_end else None),
        "max_abs_actual_roll_pitch_yaw_deg": [max(abs(row[k]) for row in rows)
                                               for k in ("roll_deg", "pitch_deg", "yaw_deg")],
        "max_abs_desired_roll_pitch_yaw_deg": [max(abs(row[k]) for row in rows)
                                                for k in ("desired_roll_deg", "desired_pitch_deg",
                                                          "desired_yaw_deg")],
        "has_nonfinite_state": False,
        **extra,
    }


def save_csv(name: str, rows: list[dict]) -> None:
    with (REPORTS / f"position_loop_{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(series: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for axis, coordinate in zip(axes, ("x", "y", "z")):
        for name, rows in series.items():
            axis.plot([r["time"] for r in rows], [r[coordinate] for r in rows], label=name)
        axis.set(ylabel=f"{coordinate} (m)")
        axis.grid(alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].set_title("Position response: XY/Z targets and impulse recovery")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(REPORTS / "position_response.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 7))
    for name, rows in series.items():
        axis.plot([r["x"] for r in rows], [r["y"] for r in rows], label=name)
        axis.plot(rows[-1]["target_x"], rows[-1]["target_y"], "x", markersize=8)
    axis.set(xlabel="World x (m)", ylabel="World y (m)", title="XY trajectory")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "xy_trajectory.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for axis, key in zip(axes, ("roll", "pitch", "yaw")):
        for name in ("A_x_plus_2", "B_y_plus_2", "F_yaw_step"):
            rows = series[name]
            axis.plot([r["time"] for r in rows], [r[f"desired_{key}_deg"] for r in rows],
                      linestyle="--", label=f"{name} target")
            axis.plot([r["time"] for r in rows], [r[f"{key}_deg"] for r in rows],
                      label=f"{name} actual")
        axis.set(ylabel=f"{key} (deg)")
        axis.grid(alpha=0.3)
    axes[0].legend(ncol=3, fontsize=7)
    axes[0].set_title("Desired vs actual attitude: X, Y and yaw steps")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(REPORTS / "attitude_command_response.png", dpi=160)
    plt.close(fig)


def run() -> dict:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    controller, allocator = build_controllers(model)
    series = {}
    metrics = {}
    for name, config in CASES.items():
        rows, extra = simulate_case(model, controller, allocator, config)
        series[name] = rows
        metrics[name] = summarize_case(rows, extra, config)
        save_csv(name, rows)
    save_plots(series)
    summary = {
        "model": str(MODEL), "duration_s": DURATION_S,
        "frames": "World XYZ has +Z up; thrust is body +Z; qpos quaternion body-to-world wxyz; qvel angular rate body frame",
        "controller": {
            "kp_position_s_inv2": controller.kp_position.tolist(),
            "kd_position_s_inv": controller.kd_position.tolist(),
            "horizontal_accel_limit_m_s2": controller.horizontal_accel_limit,
            "vertical_accel_limit_m_s2": controller.vertical_accel_limit,
            "kp_attitude_s_inv2": controller.attitude_controller.kp_attitude.tolist(),
            "kd_attitude_s_inv": controller.attitude_controller.kd_attitude.tolist(),
            "integral_gain": 0.0,
        },
        "event": {"time_s": EVENT_START_S, "horizontal_force_n": HORIZONTAL_FORCE_N,
                  "force_duration_s": EVENT_END_S - EVENT_START_S,
                  "yaw_step_deg": YAW_STEP_DEG},
        "settling_band": {"position_error_norm_m": 0.05, "speed_norm_m_s": 0.05,
                          "yaw_error_deg": 2.0},
        "cases": metrics,
        "warning": "Mass/COM/inertia remain estimated; rotor signs temporary. Ideal motors and no aerodynamic disturbances.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = run()
    for name, result in summary["cases"].items():
        print(f"{name}: error={result['final_position_error_norm_m']:.4f} m, "
              f"speed={result['final_speed_norm_m_s']:.4f} m/s, "
              f"settling={result['settling_time_s']} s, "
              f"tilt={result['max_tilt_deg']:.1f} deg, "
              f"rpm={result['max_motor_rpm']:.0f}, sat={result['saturation_steps']}")
    print(f"Saved {SUMMARY} and three PNGs")


if __name__ == "__main__":
    main()
