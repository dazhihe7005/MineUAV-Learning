"""Open the MuJoCo viewer and let the quadrotor fall with zero control."""

import argparse
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "simple_quadrotor.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the quadrotor fall freely.")
    parser.add_argument("--duration", type=float, default=5.0, help="simulation seconds")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    print(f"t={data.time:.2f} s  z={data.qpos[2]:.4f} m  vz={data.qvel[2]:.4f} m/s", flush=True)

    next_report = 0.2
    threads_before_viewer = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer_threads = set(threading.enumerate()) - threads_before_viewer
    with viewer:
        while viewer.is_running() and data.time < args.duration:
            step_start = time.monotonic()
            mujoco.mj_step(model, data)

            if data.time + 1e-9 >= next_report:
                print(
                    f"t={data.time:.2f} s  z={data.qpos[2]:.4f} m  "
                    f"vz={data.qvel[2]:.4f} m/s",
                    flush=True,
                )
                next_report += 0.2

            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    # The passive viewer closes on a background thread. Wait before Python
    # tears down the X11 connection used by that thread.
    for thread in viewer_threads:
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError("MuJoCo viewer thread did not close")


if __name__ == "__main__":
    main()
