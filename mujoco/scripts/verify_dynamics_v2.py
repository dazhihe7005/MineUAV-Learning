"""Verify the v2 four-rotor mixer about ESTIMATED_COM_V2 against MuJoCo."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from build_rotor_model import CONFIG, read_config
from estimate_dynamics_v2 import NEW_MODEL, REPORT as MASS_REPORT
from verify_rotor_wrench import analytic_wrench, make_cases, mixing_matrix, motor_positions


OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "mixing_matrix_v2.txt"


def inspect_case(model: mujoco.MjModel, sites: np.ndarray, com: np.ndarray,
                 u: np.ndarray, B: np.ndarray, config: dict) -> dict:
    k_f = config["k_f_n_per_rad_s_squared"]
    k_m = config["k_m_nm_per_rad_s_squared"]
    signs = config["rotation_signs"]
    theory = analytic_wrench(sites - com, u, k_f, k_m, signs)
    origin_theory = analytic_wrench(sites, u, k_f, k_m, signs)
    matrix_result = B @ u
    if not np.allclose(np.array([theory[2], *theory[3:]]), matrix_result, atol=1e-8):
        raise AssertionError("V2 mixing matrix differs from direct COM-based cross products")

    data = mujoco.MjData(model)
    data.ctrl[:] = u
    mujoco.mj_forward(model, data)
    # Freejoint generalized torque is about the body frame origin. Shift its
    # reference point to the estimated COM for comparison with the v2 mixer.
    mujoco_origin = data.qfrc_actuator.copy()
    mujoco_com = mujoco_origin.copy()
    mujoco_com[3:] -= np.cross(com, mujoco_origin[:3])
    if not np.allclose(mujoco_origin, origin_theory, atol=1e-8, rtol=1e-9):
        raise AssertionError("MuJoCo generalized wrench differs from body-origin theory")
    if not np.allclose(mujoco_com, theory, atol=1e-8, rtol=1e-9):
        raise AssertionError("MuJoCo COM-shifted wrench differs from COM theory")
    for _ in range(10):
        mujoco.mj_step(model, data)
    return {
        "u_omega_squared": u.tolist(), "theoretical": theory.tolist(),
        "mujoco": mujoco_com.tolist(), "mujoco_body_origin": mujoco_origin.tolist(),
        "simulated_qvel": data.qvel.tolist(), "simulated_duration_s": float(data.time),
    }


def verify() -> dict:
    if not NEW_MODEL.exists() or not MASS_REPORT.exists():
        raise FileNotFoundError("Generate v2 model first: estimate_dynamics_v2.py")
    mass_report = json.loads(MASS_REPORT.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(NEW_MODEL))
    config = read_config(CONFIG)
    if model.nu != 4 or not np.isclose(mujoco.mj_getTotalmass(model), 7.0):
        raise ValueError("V2 model must have four rotor inputs and exactly 7 kg mass")
    com = model.body("mine_uav").ipos.copy()
    if not np.allclose(com, mass_report["estimated_com_v2_m"], atol=1e-10):
        raise ValueError("V2 MJCF COM is stale relative to dynamics_v2_report.json")
    sites = motor_positions(model)
    arms = sites - com
    k_f = config["k_f_n_per_rad_s_squared"]
    k_m = config["k_m_nm_per_rad_s_squared"]
    signs = config["rotation_signs"]
    for i, sign in enumerate(signs):
        if not np.allclose(model.actuator_gear[i], [0, 0, k_f, 0, 0, sign * k_m], atol=1e-12):
            raise ValueError(f"Rotor {i + 1} actuator differs from rotor_config.json")
    B = mixing_matrix(arms, k_f, k_m, signs)
    u0 = float(7.0 * abs(model.opt.gravity[2]) / (4 * k_f))
    cases = {name: inspect_case(model, sites, com, u, B, config)
             for name, u in make_cases(u0, signs).items()}
    collective = cases["collective"]["theoretical"]
    for case in cases.values():
        case["angular_velocity_delta_vs_collective"] = (
            np.asarray(case["simulated_qvel"])[3:] -
            np.asarray(cases["collective"]["simulated_qvel"])[3:]
        ).tolist()
    old_origin_collective = analytic_wrench(
        sites, np.full(4, u0), k_f, k_m, signs,
    )
    return {
        "model": str(NEW_MODEL), "mass_kg": 7.0,
        "torque_reference": "ESTIMATED_COM_V2, not measured COM",
        "estimated_com_v2_m": com.tolist(), "motor_arms_from_com_m": arms.tolist(),
        "k_f": k_f, "k_m": k_m, "rotation_signs": signs,
        "mixing_matrix": B.tolist(), "row_order": ["Fz", "Tx", "Ty", "Tz"],
        "collective_body_origin_torque_nm": old_origin_collective[3:].tolist(),
        "collective_v2_com_torque_nm": collective[3:],
        "tests": cases,
        "note": "Direction/sign only: mass, COM and inertia are engineering estimates; spin assignment temporary.",
    }


def save_report(report: dict) -> None:
    lines = [
        "MineUAV dynamics v2: [Fz,Tx,Ty,Tz]^T = B [omega_1^2,...,omega_4^2]^T",
        "Torque reference: ESTIMATED_COM_V2 (not measured COM).",
        f"COM (body frame, m): {report['estimated_com_v2_m']}",
        f"k_f={report['k_f']:.17g} N/(rad/s)^2; k_m={report['k_m']:.17g} N*m/(rad/s)^2",
        f"rotation signs={report['rotation_signs']} TEMPORARY_ROTATION_CONFIG",
        "Motor arms relative to COM (m):",
    ]
    for i, arm in enumerate(report["motor_arms_from_com_m"], start=1):
        lines.append(f"  motor_{i}: {arm}")
    lines.append("B rows, columns u1 u2 u3 u4:")
    for name, row in zip(report["row_order"], report["mixing_matrix"]):
        lines.append(f"  {name}: " + " ".join(f"{x:+.12e}" for x in row))
    lines.extend([
        "Fz=sum(k_f*u_i); Tx=sum(r_y*k_f*u_i); Ty=sum(-r_x*k_f*u_i); Tz=sum(s_i*k_m*u_i).",
        f"Collective body-origin torque (Nm): {report['collective_body_origin_torque_nm']}",
        f"Collective COM-based torque (Nm): {report['collective_v2_com_torque_nm']}",
        "Sanity cases: wrench order [Fx,Fy,Fz,Tx,Ty,Tz] about ESTIMATED_COM_V2:",
    ])
    for name, case in report["tests"].items():
        lines.append(f"  {name}: u={case['u_omega_squared']}")
        lines.append(f"    theory={case['theoretical']}")
        lines.append(f"    MuJoCo shifted to COM={case['mujoco']}")
        lines.append(f"    qvel@{case['simulated_duration_s']:.3f}s={case['simulated_qvel']}")
        lines.append(f"    delta angular qvel vs collective={case['angular_velocity_delta_vs_collective']}")
    lines.append(report["note"])
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = verify()
    save_report(report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Saved {OUTPUT}")
        for name, row in zip(report["row_order"], report["mixing_matrix"]):
            print(f"{name}: {row}")
        for name, case in report["tests"].items():
            print(f"{name}: theory={case['theoretical']} MuJoCo={case['mujoco']}")


if __name__ == "__main__":
    main()
