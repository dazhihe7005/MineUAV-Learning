"""Compare the four-rotor analytic wrench and mixing matrix with MuJoCo."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from build_rotor_model import CONFIG, MODEL, OUTPUT, actuator_xml, read_config


REPORT = Path(__file__).resolve().parents[1] / "reports" / "mixing_matrix.txt"


def motor_positions(model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.site(f"motor_{i}_site").pos.copy() for i in range(1, 5)])


def mixing_matrix(positions: np.ndarray, k_f: float, k_m: float, signs: list[int]) -> np.ndarray:
    return np.array([
        np.full(4, k_f),
        positions[:, 1] * k_f,
        -positions[:, 0] * k_f,
        np.array(signs) * k_m,
    ])


def analytic_wrench(positions: np.ndarray, u: np.ndarray, k_f: float, k_m: float, signs: list[int]) -> np.ndarray:
    force = np.zeros(3)
    torque = np.zeros(3)
    for r, squared_speed, sign in zip(positions, u, signs):
        rotor_force = np.array([0.0, 0.0, k_f * squared_speed])
        force += rotor_force
        torque += np.cross(r, rotor_force) + np.array([0.0, 0.0, sign * k_m * squared_speed])
    return np.concatenate((force, torque))


def inspect_case(model: mujoco.MjModel, positions: np.ndarray, u: np.ndarray, B: np.ndarray, config: dict) -> dict:
    theory = analytic_wrench(
        positions, u, config["k_f_n_per_rad_s_squared"],
        config["k_m_nm_per_rad_s_squared"], config["rotation_signs"],
    )
    from_matrix = B @ u
    data = mujoco.MjData(model)
    data.ctrl[:] = u
    mujoco.mj_forward(model, data)
    actual = data.qfrc_actuator.copy()
    if not np.allclose(theory, actual, atol=1e-8, rtol=1e-9):
        raise AssertionError(f"MuJoCo wrench differs from theory: {actual} vs {theory}")
    if not np.allclose(np.array([theory[2], *theory[3:]]), from_matrix, atol=1e-8, rtol=1e-9):
        raise AssertionError("Mixing matrix differs from direct cross-product calculation")
    # Short direction-only simulation: mass/COM/inertia are placeholders.
    for _ in range(10):
        mujoco.mj_step(model, data)
    return {
        "u_omega_squared": u.tolist(),
        "theoretical": theory.tolist(),
        "mujoco": actual.tolist(),
        "simulated_qvel": data.qvel.tolist(),
        "simulated_duration_s": float(data.time),
    }


def make_cases(u0: float, signs: list[int]) -> dict[str, np.ndarray]:
    delta = 0.1 * u0
    return {
        "collective": np.full(4, u0),
        "roll": np.array([u0 + delta, u0 - delta, u0 - delta, u0 + delta]),
        "pitch": np.array([u0 + delta, u0 + delta, u0 - delta, u0 - delta]),
        "yaw": np.array([u0 + delta * sign for sign in signs]),
    }


def verify() -> dict:
    config = read_config(CONFIG)
    if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != actuator_xml(config):
        raise ValueError("Generated rotor actuators are stale; run build_rotor_model.py")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    if model.nu != 4:
        raise ValueError(f"Expected four rotor actuator inputs, found {model.nu}")
    positions = motor_positions(model)
    k_f = config["k_f_n_per_rad_s_squared"]
    k_m = config["k_m_nm_per_rad_s_squared"]
    signs = config["rotation_signs"]
    for i, sign in enumerate(signs):
        gear = model.actuator_gear[i]
        if not np.allclose(gear, [0, 0, k_f, 0, 0, sign * k_m], atol=1e-12, rtol=1e-9):
            raise ValueError(f"Actuator {i} gear does not match rotor_config.json")
    B = mixing_matrix(positions, k_f, k_m, signs)
    # The chosen baseline balances the TEMPORARY 1 kg mass; its RPM is below
    # the manufacturer data range and is used only for sign/symmetry checks.
    u0 = float(mujoco.mj_getTotalmass(model) * abs(model.opt.gravity[2]) / (4 * k_f))
    tests = make_cases(u0, signs)
    results = {name: inspect_case(model, positions, u, B, config) for name, u in tests.items()}
    return {
        "model": str(MODEL),
        "k_f": k_f,
        "k_m": k_m,
        "motor_sites_m": positions.tolist(),
        "rotation_signs": signs,
        "rotation_config_note": config["note"],
        "torque_reference": "TEMPORARY body origin, not measured COM",
        "mixing_matrix": B.tolist(),
        "row_order": ["Fz (N)", "Tx (N*m)", "Ty (N*m)", "Tz (N*m)"],
        "tests": results,
    }


def save_text_report(report: dict, path: Path) -> None:
    lines = [
        "MineUAV four-rotor mixing matrix; input u_i = omega_i^2 [(rad/s)^2]",
        f"k_f = {report['k_f']:.12g} N/(rad/s)^2",
        f"k_m = {report['k_m']:.12g} N*m/(rad/s)^2",
        f"rotation signs = {report['rotation_signs']} (TEMPORARY ROTATION CONFIG)",
        "Torque is about the TEMPORARY body origin, not a measured COM.",
        "[Fz, Tx, Ty, Tz]^T = B [u1, u2, u3, u4]^T",
        "B rows:",
    ]
    for name, row in zip(report["row_order"], report["mixing_matrix"]):
        lines.append(f"  {name:10s} " + " ".join(f"{entry:+.12e}" for entry in row))
    lines.extend([
        "Fz: sum of four upward rotor thrusts.",
        "Tx: roll moment from each rotor's Y offset times upward thrust.",
        "Ty: pitch moment from negative X offset times upward thrust.",
        "Tz: signed propeller reaction torque, independent of X/Y offsets.",
        "motor sites (body frame, m):",
    ])
    for i, site in enumerate(report["motor_sites_m"], start=1):
        lines.append(f"  motor_{i}_site: {site}")
    lines.append("Sanity tests: wrench order [Fx, Fy, Fz, Tx, Ty, Tz]")
    for name, case in report["tests"].items():
        lines.append(f"  {name}: u={case['u_omega_squared']}")
        lines.append(f"    theory={case['theoretical']}")
        lines.append(f"    mujoco={case['mujoco']}")
        lines.append(f"    qvel@{case['simulated_duration_s']:.3f}s={case['simulated_qvel']}")
    lines.append("WARNING: mass, COM, inertia, spin assignment, and collision are TEMPORARY_PLACEHOLDER; no motor lag or ESC dynamics.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = verify()
    save_text_report(report, REPORT)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Saved {REPORT}")
        for name, row in zip(report["row_order"], report["mixing_matrix"]):
            print(f"{name}: {row}")
        for name, result in report["tests"].items():
            print(f"{name}: theory={result['theoretical']} MuJoCo={result['mujoco']}")


if __name__ == "__main__":
    main()
