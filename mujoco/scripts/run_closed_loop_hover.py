"""Run six MuJoCo PD hover recoveries through the bounded four-rotor allocator."""

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
from hover_controller import (HoverController, DEFAULT_KP_Z, DEFAULT_KD_Z,
                              DEFAULT_KP_ATTITUDE, DEFAULT_KD_ATTITUDE)  # noqa: E402
from hover_trim_test import euler_zyx  # noqa: E402


REPORTS = ROOT / "reports"
MASS_REPORT = REPORTS / "dynamics_v2_report.json"
SUMMARY = REPORTS / "closed_loop_hover_summary.json"
TARGET_Z_M = 1.0
DURATION_S = 10.0
DISTURBANCE_START_S = 2.0
DISTURBANCE_END_S = 2.1
DISTURBANCE_FORCE_Z_N = -21.0
LOG_FIELDS = (
    "time", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw",
    "wx", "wy", "wz", "Fz", "Tx", "Ty", "Tz", "rpm1", "rpm2", "rpm3", "rpm4",
    "allocator_saturated", "actual_Fz", "actual_Tx", "actual_Ty", "actual_Tz",
)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.r_[a[0] * b[0] - np.dot(a[1:], b[1:]),
                 a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])]


def initial_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """ZYX initial orientation, wxyz format."""
    def axis_quat(axis: int, angle_deg: float) -> np.ndarray:
        half = math.radians(angle_deg) / 2
        q = np.zeros(4)
        q[0] = math.cos(half)
        q[axis + 1] = math.sin(half)
        return q

    return _quat_multiply(_quat_multiply(axis_quat(2, yaw_deg),
                                        axis_quat(1, pitch_deg)),
                          axis_quat(0, roll_deg))


def simulate_case(model: mujoco.MjModel, controller: HoverController,
                  allocator: ControlAllocator, z0: float, roll0: float = 0.0,
                  pitch0: float = 0.0, yaw0: float = 0.0,
                  external_vertical_force: bool = False) -> list[dict]:
    data = mujoco.MjData(model)
    data.qpos[:3] = [0.0, 0.0, z0]
    data.qpos[3:7] = initial_quaternion(roll0, pitch0, yaw0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    body_id = model.body("mine_uav").id
    steps = round(DURATION_S / model.opt.timestep)
    rows = []
    for step in range(steps):
        t = step * model.opt.timestep
        data.xfrc_applied.fill(0.0)
        if external_vertical_force and DISTURBANCE_START_S <= t < DISTURBANCE_END_S:
            data.xfrc_applied[body_id, 2] = DISTURBANCE_FORCE_Z_N
        desired = controller.compute(
            position=data.qpos[:3], velocity=data.qvel[:3],
            quaternion=data.qpos[3:7], angular_velocity=data.qvel[3:6],
            target_z=TARGET_Z_M, target_yaw=0.0,
        )
        allocation = allocator.allocate(desired)
        data.ctrl[:] = allocation.u
        mujoco.mj_step(model, data)
        angles = euler_zyx(data.qpos[3:7])
        sample = {
            "time": float(data.time),
            "x": float(data.qpos[0]), "y": float(data.qpos[1]), "z": float(data.qpos[2]),
            "vx": float(data.qvel[0]), "vy": float(data.qvel[1]), "vz": float(data.qvel[2]),
            "roll": angles[0], "pitch": angles[1], "yaw": angles[2],
            "wx": float(data.qvel[3]), "wy": float(data.qvel[4]), "wz": float(data.qvel[5]),
            "Fz": float(desired[0]), "Tx": float(desired[1]),
            "Ty": float(desired[2]), "Tz": float(desired[3]),
            "rpm1": float(allocation.rpm[0]), "rpm2": float(allocation.rpm[1]),
            "rpm3": float(allocation.rpm[2]), "rpm4": float(allocation.rpm[3]),
            "allocator_saturated": int(allocation.saturated),
            "actual_Fz": float(allocation.achieved_wrench[0]),
            "actual_Tx": float(allocation.achieved_wrench[1]),
            "actual_Ty": float(allocation.achieved_wrench[2]),
            "actual_Tz": float(allocation.achieved_wrench[3]),
        }
        if not all(math.isfinite(value) for value in sample.values()):
            raise FloatingPointError(f"Nonfinite simulation value at t={data.time:.3f}s")
        rows.append(sample)
    return rows


def _inside_settling_band(row: dict) -> bool:
    return (abs(row["z"] - TARGET_Z_M) <= 0.03
            and abs(row["vz"]) <= 0.05
            and max(abs(row[axis]) for axis in ("roll", "pitch", "yaw")) <= math.radians(2)
            and max(abs(row[axis]) for axis in ("wx", "wy", "wz")) <= 0.10)


def summarize_case(rows: list[dict], z0: float, external_vertical_force: bool) -> dict:
    start_time = DISTURBANCE_END_S if external_vertical_force else 0.0
    suffix_inside = True
    settling_time = None
    for row in reversed(rows):
        suffix_inside = suffix_inside and _inside_settling_band(row)
        if suffix_inside and row["time"] >= start_time:
            settling_time = row["time"]
    final = rows[-1]
    if z0 < TARGET_Z_M:
        overshoot = max(0.0, max(row["z"] - TARGET_Z_M for row in rows))
    elif z0 > TARGET_Z_M:
        overshoot = max(0.0, max(TARGET_Z_M - row["z"] for row in rows))
    else:
        overshoot = max(abs(row["z"] - TARGET_Z_M) for row in rows)
    return {
        "initial_z_m": z0,
        "final_position_m": [final["x"], final["y"], final["z"]],
        "final_velocity_m_s": [final["vx"], final["vy"], final["vz"]],
        "final_attitude_error_deg": [math.degrees(final[axis]) for axis in ("roll", "pitch", "yaw")],
        "final_angular_velocity_rad_s": [final[axis] for axis in ("wx", "wy", "wz")],
        "final_z_error_m": final["z"] - TARGET_Z_M,
        "max_attitude_error_deg": [max(abs(math.degrees(row[axis])) for row in rows)
                                   for axis in ("roll", "pitch", "yaw")],
        "max_abs_height_error_m": max(abs(row["z"] - TARGET_Z_M) for row in rows),
        "height_overshoot_m": overshoot,
        "settling_time_s": settling_time,
        "max_motor_rpm": max(row[f"rpm{i}"] for row in rows for i in range(1, 5)),
        "saturation_count": sum(row["allocator_saturated"] for row in rows),
        "saturation_fraction": sum(row["allocator_saturated"] for row in rows) / len(rows),
        "has_nonfinite_state": False,
        "peak_post_disturbance_height_error_m": (
            max(abs(row["z"] - TARGET_Z_M) for row in rows
                if row["time"] >= DISTURBANCE_END_S) if external_vertical_force else None
        ),
    }


def save_csv(name: str, rows: list[dict]) -> None:
    path = REPORTS / f"closed_loop_{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(series: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, rows in series.items():
        ax.plot([row["time"] for row in rows], [row["z"] for row in rows], label=name, linewidth=1.2)
    ax.axhline(TARGET_Z_M, color="black", linestyle="--", linewidth=1, label="target z")
    ax.set(xlabel="Time (s)", ylabel="Body-origin z (m)", title="Closed-loop height recovery")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "height_response.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for axis, key in zip(axes, ("roll", "pitch", "yaw")):
        for name, rows in series.items():
            axis.plot([row["time"] for row in rows],
                      [math.degrees(row[key]) for row in rows], label=name, linewidth=1.1)
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set(ylabel=f"{key} (deg)")
        axis.grid(alpha=0.25)
    axes[0].set_title("Closed-loop attitude recovery")
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "attitude_response.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for axis, name in zip(axes, ("combined", "external_vertical_force")):
        rows = series[name]
        for i in range(1, 5):
            axis.plot([row["time"] for row in rows], [row[f"rpm{i}"] for row in rows],
                      label=f"motor {i}", linewidth=1.1)
        axis.axhline(8500, color="red", linestyle="--", linewidth=0.8, label="8500 rpm cap")
        axis.set(ylabel="RPM", title=name)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "motor_rpm_response.png", dpi=160)
    plt.close(fig)


def run() -> dict:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    mass_report = json.loads(MASS_REPORT.read_text(encoding="utf-8"))
    inertia = np.asarray(mass_report["estimated_inertia_v2_kg_m2"])
    if not np.allclose(model.body("mine_uav").ipos, mass_report["estimated_com_v2_m"], atol=1e-10):
        raise ValueError("Model COM and estimated mass report disagree")
    allocator = ControlAllocator.from_v2_model()
    controller = HoverController(
        mass_kg=mujoco.mj_getTotalmass(model), inertia_kg_m2=inertia,
        max_total_thrust_n=float(np.sum(allocator.B[0]) * allocator.u_max),
        gravity_m_s2=abs(float(model.opt.gravity[2])),
    )
    configurations = {
        "height_low": dict(z0=0.5),
        "height_high": dict(z0=1.5),
        "roll_15": dict(z0=1.0, roll0=15.0),
        "pitch_minus_15": dict(z0=1.0, pitch0=-15.0),
        "combined": dict(z0=0.7, roll0=10.0, pitch0=-10.0, yaw0=15.0),
        "external_vertical_force": dict(z0=1.0, external_vertical_force=True),
    }
    series = {}
    metrics = {}
    for name, config in configurations.items():
        rows = simulate_case(model, controller, allocator, **config)
        series[name] = rows
        metrics[name] = summarize_case(rows, config["z0"], name == "external_vertical_force")
        save_csv(name, rows)
    save_plots(series)
    summary = {
        "model": str(MODEL), "duration_s": DURATION_S, "target_z_m": TARGET_Z_M,
        "controller": {
            "kp_z_n_per_m": controller.kp_z, "kd_z_n_per_m_s": controller.kd_z,
            "kp_attitude_per_s2": controller.kp_attitude.tolist(),
            "kd_attitude_per_s": controller.kd_attitude.tolist(),
            "integral_gain": 0.0,
            "rigid_body_compensation": "omega cross (I omega)",
        },
        "settling_band": {
            "z_m": 0.03, "vz_m_s": 0.05, "attitude_deg": 2.0,
            "angular_velocity_rad_s": 0.10,
        },
        "external_disturbance": {
            "type": "world-frame vertical force at body COM",
            "force_z_n": DISTURBANCE_FORCE_Z_N,
            "start_s": DISTURBANCE_START_S, "end_s": DISTURBANCE_END_S,
        },
        "cases": metrics,
        "warning": "COM/inertia and rotor signs remain estimated/temporary; no x/y controller or integral term.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = run()
    for name, result in summary["cases"].items():
        print(f"{name}: z_error={result['final_z_error_m']:+.4f} m, "
              f"rpy_final_deg={result['final_attitude_error_deg']}, "
              f"settling={result['settling_time_s']} s, "
              f"max_rpm={result['max_motor_rpm']:.1f}, sat={result['saturation_count']}")
    print(f"Saved {SUMMARY} and three response PNGs")


if __name__ == "__main__":
    main()
