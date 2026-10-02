"""Five-second open-loop hover trim and equal-speed comparison, no controller."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "control"))
from control_allocator import ControlAllocator, MODEL  # noqa: E402


REPORT = ROOT / "reports" / "hover_trim_report.json"
TRAJECTORY = ROOT / "reports" / "hover_trim_trajectory.csv"


def euler_zyx(q: np.ndarray) -> list[float]:
    """Return roll, pitch, yaw in radians from MuJoCo's wxyz quaternion."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [roll, pitch, yaw]


def state(data: mujoco.MjData) -> dict:
    return {
        "time_s": float(data.time),
        "position_m": data.qpos[:3].tolist(),
        "linear_velocity_m_s": data.qvel[:3].tolist(),
        "euler_rad": euler_zyx(data.qpos[3:7]),
        "angular_velocity_rad_s": data.qvel[3:6].tolist(),
    }


def simulate(model: mujoco.MjModel, u: np.ndarray, duration_s: float,
             comparison_s: float = 0.3, record_trajectory: bool = False) -> tuple[dict, list[dict]]:
    steps = round(duration_s / model.opt.timestep)
    comparison_step = round(comparison_s / model.opt.timestep)
    if steps < comparison_step or steps <= 0:
        raise ValueError("Simulation duration must be positive and include comparison time")

    data = mujoco.MjData(model)
    data.qpos[:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.ctrl[:] = u
    mujoco.mj_forward(model, data)
    initial_qacc = data.qacc.copy()
    initial_wrench = data.qfrc_actuator.copy()
    com = model.body("mine_uav").ipos.copy()
    initial_wrench[3:6] -= np.cross(com, initial_wrench[:3])
    initial_state = state(data)
    samples = [initial_state] if record_trajectory else []
    pitch_at_comparison = None
    sample_stride = max(1, round(0.1 / model.opt.timestep))

    for step in range(1, steps + 1):
        mujoco.mj_step(model, data)
        if step == comparison_step:
            pitch_at_comparison = euler_zyx(data.qpos[3:7])[1]
        if record_trajectory and (step % sample_stride == 0 or step == steps):
            samples.append(state(data))

    result = {
        "duration_s": float(data.time),
        "initial_state": initial_state,
        "initial_linear_acceleration_m_s2": initial_qacc[:3].tolist(),
        "initial_angular_acceleration_rad_s2": initial_qacc[3:6].tolist(),
        "initial_mujoco_wrench_about_com": initial_wrench.tolist(),
        "comparison_s": comparison_s,
        "pitch_at_comparison_s_rad": pitch_at_comparison,
        "final_state": state(data),
        "contact_count_final": int(data.ncon),
    }
    return result, samples


def run() -> tuple[dict, list[dict]]:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    allocator = ControlAllocator.from_v2_model()
    gravity = abs(float(model.opt.gravity[2]))
    mg = mujoco.mj_getTotalmass(model) * gravity
    desired = np.array([mg, 0.0, 0.0, 0.0])
    trim = allocator.allocate(desired)
    if not trim.feasible or trim.saturated:
        raise ValueError("Hover trim is not feasible inside 0..8500 rpm")

    trim_sim, trajectory = simulate(model, trim.u, 5.0, record_trajectory=True)
    equal_u = np.full(4, mg / np.sum(allocator.B[0]))
    equal_sim, _ = simulate(model, equal_u, 0.3)

    requested_cases = {
        "hover": [mg, 0, 0, 0],
        "roll": [mg, 0.5, 0, 0],
        "pitch": [mg, 0, 0.5, 0],
        "yaw": [mg, 0, 0, 0.05],
        "combined": [mg, 0.2, -0.2, 0.02],
    }
    allocation_cases = {}
    for name, wrench in requested_cases.items():
        allocation = allocator.allocate(wrench)
        allocation_cases[name] = {
            "requested_wrench": allocation.requested_wrench.tolist(),
            "achieved_wrench": allocation.achieved_wrench.tolist(),
            "error": allocation.error.tolist(),
            "error_norm": float(np.linalg.norm(allocation.error)),
            "u": allocation.u.tolist(),
            "saturated": allocation.saturated,
        }
    impossible = allocator.allocate([1.2 * np.sum(allocator.B[0]) * allocator.u_max, 0, 0, 0])
    report = {
        "model": str(MODEL), "mass_kg": float(mujoco.mj_getTotalmass(model)),
        "gravity_m_s2": gravity, "rpm_cap": allocator.max_rpm, "u_max": allocator.u_max,
        "trim": {
            "desired_wrench": desired.tolist(), "u": trim.u.tolist(),
            "omega_rad_s": trim.omega_rad_s.tolist(), "rpm": trim.rpm.tolist(),
            "achieved_wrench": trim.achieved_wrench.tolist(),
            "error": trim.error.tolist(), "saturated": trim.saturated,
        },
        "trim_simulation": trim_sim,
        "equal_speed_comparison": {
            "u": equal_u.tolist(), **equal_sim,
        },
        "allocation_cases": allocation_cases,
        "unreachable_stress_case": {
            "requested_wrench": impossible.requested_wrench.tolist(),
            "achieved_wrench": impossible.achieved_wrench.tolist(),
            "error": impossible.error.tolist(),
            "saturated": impossible.saturated,
            "active_limits": impossible.active_limits,
        },
        "warning": "Open-loop numerical equilibrium only; COM/inertia/rotation signs are estimated or temporary.",
    }
    return report, trajectory


def save(report: dict, trajectory: list[dict]) -> None:
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ("time_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
              "roll_rad", "pitch_rad", "yaw_rad", "wx_rad_s", "wy_rad_s", "wz_rad_s")
    with TRAJECTORY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for item in trajectory:
            writer.writerow([
                item["time_s"], *item["position_m"], *item["linear_velocity_m_s"],
                *item["euler_rad"], *item["angular_velocity_rad_s"],
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report, trajectory = run()
    save(report, trajectory)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        trim = report["trim"]
        print(f"u_hover={trim['u']} (rad/s)^2")
        print(f"omega={trim['omega_rad_s']} rad/s")
        print(f"rpm={trim['rpm']}")
        print(f"B*u_hover={trim['achieved_wrench']} [Fz,Tx,Ty,Tz]")
        print(f"initial angular acceleration={report['trim_simulation']['initial_angular_acceleration_rad_s2']} rad/s^2")
        print(f"5s final={report['trim_simulation']['final_state']}")
        print(f"equal-speed pitch@0.3s={report['equal_speed_comparison']['pitch_at_comparison_s_rad']} rad")
        print(f"trim pitch@0.3s={report['trim_simulation']['pitch_at_comparison_s_rad']} rad")
        for name, case in report["allocation_cases"].items():
            print(f"{name} allocation error={case['error']} saturated={case['saturated']}")
        print(f"unreachable stress saturated={report['unreachable_stress_case']['saturated']}")
        print(f"Saved {REPORT} and {TRAJECTORY}")


if __name__ == "__main__":
    main()
