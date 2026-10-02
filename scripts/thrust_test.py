"""Compare three constant, equal thrust settings without feedback control."""

import math
from pathlib import Path

import mujoco


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "simple_quadrotor.xml"
DURATION_SECONDS = 1.0
REPORT_INTERVAL_SECONDS = 0.25


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    if model.nu != 4:
        raise RuntimeError(f"Expected 4 rotor actuators, found {model.nu}")

    body_id = model.body("quadrotor").id
    mass = model.body_mass[body_id]
    gravity = abs(model.opt.gravity[2])
    hover_thrust = mass * gravity / model.nu
    print(f"mass={mass:.3f} kg  gravity={gravity:.3f} m/s^2")
    print(f"calculated hover thrust={hover_thrust:.4f} N per rotor")

    total_steps = round(DURATION_SECONDS / model.opt.timestep)
    report_every = round(REPORT_INTERVAL_SECONDS / model.opt.timestep)
    final_heights = {}

    for label, scale in (("low", 0.8), ("near_hover", 1.0), ("high", 1.2)):
        data = mujoco.MjData(model)
        data.ctrl[:] = scale * hover_thrust
        print(f"\n{label}: thrust={data.ctrl[0]:.4f} N per rotor")

        for step in range(total_steps + 1):
            if step % report_every == 0:
                print(
                    f"t={data.time:.2f} s  z={data.qpos[2]:.4f} m  "
                    f"vz={data.qvel[2]:.4f} m/s"
                )
            if not math.isfinite(data.qpos[2]) or not math.isfinite(data.qvel[2]):
                raise RuntimeError(f"Non-finite state in {label} test")
            if step < total_steps:
                mujoco.mj_step(model, data)

        final_heights[label] = data.qpos[2]

    initial_height = model.qpos0[2]
    if not (
        final_heights["low"] < initial_height - 0.1
        and abs(final_heights["near_hover"] - initial_height) < 0.05
        and final_heights["high"] > initial_height + 0.1
    ):
        raise RuntimeError(f"Unexpected thrust response: {final_heights}")


if __name__ == "__main__":
    main()
