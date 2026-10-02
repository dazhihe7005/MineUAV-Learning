"""Three open-loop thrust trials with the Menagerie Crazyflie 2 model."""

import argparse
import json
from pathlib import Path

import mujoco


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "references" / "crazyflie" / "cf2.xml"


def run_trial(model: mujoco.MjModel, thrust: float, duration: float = 0.15) -> dict:
    data = mujoco.MjData(model)
    data.ctrl[:] = [thrust, 0.0, 0.0, 0.0]
    samples = [{"time": data.time, "z": float(data.qpos[2]), "vz": float(data.qvel[2])}]
    next_print = 0.05
    while data.time < duration - model.opt.timestep / 2:
        mujoco.mj_step(model, data)
        if data.time + 1e-9 >= next_print:
            samples.append({"time": float(data.time), "z": float(data.qpos[2]), "vz": float(data.qvel[2])})
            next_print += 0.05
    if samples[-1]["time"] != data.time:
        samples.append({"time": float(data.time), "z": float(data.qpos[2]), "vz": float(data.qvel[2])})
    return {
        "ctrl": data.ctrl.tolist(),
        "initial_z": samples[0]["z"],
        "final_z": float(data.qpos[2]),
        "final_vz": float(data.qvel[2]),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    expected = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    if expected != ["body_thrust", "x_moment", "y_moment", "z_moment"]:
        raise ValueError(f"Unexpected actuator order: {expected}")
    mg = mujoco.mj_getTotalmass(model) * abs(float(model.opt.gravity[2]))
    cases = {
        "zero": run_trial(model, 0.0),
        "hover": run_trial(model, mg),
        "high": run_trial(model, 1.2 * mg),
    }
    if args.json:
        print(json.dumps(cases, indent=2))
        return
    print(f"mass={mujoco.mj_getTotalmass(model):.6f} kg, g={abs(model.opt.gravity[2]):.3f} m/s², mg={mg:.6f} N")
    print("This model has one body-thrust actuator, not four individual motor actuators.")
    for name, case in cases.items():
        print(f"\n{name}: ctrl={case['ctrl']}")
        print("time (s)    z (m)       vz (m/s)   ctrl")
        for s in case["samples"]:
            print(f"{s['time']:8.3f}    {s['z']:9.5f}   {s['vz']:9.5f}   {case['ctrl']}")


if __name__ == "__main__":
    main()
