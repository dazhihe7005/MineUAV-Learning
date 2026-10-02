"""Inspect the unmodified Menagerie Crazyflie 2 MJCF."""

import argparse
import json
from pathlib import Path

import mujoco


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "references" / "crazyflie" / "scene.xml"


def inspect(path: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    free_joints = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free_joints) != 1:
        raise ValueError(f"Expected one free joint, found {len(free_joints)}")
    joint = free_joints[0]
    qstart = int(model.jnt_qposadr[joint])
    vstart = int(model.jnt_dofadr[joint])
    actuators = []
    for i in range(model.nu):
        actuators.append({
            "index": i,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            "gear": model.actuator_gear[i].tolist(),
            "ctrlrange": model.actuator_ctrlrange[i].tolist(),
        })
    return {
        "model": str(path),
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nsensor": model.nsensor,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "ctrl": data.ctrl.tolist(),
        "qpos_xyz_indices": list(range(qstart, qstart + 3)),
        "qpos_xyz": data.qpos[qstart:qstart + 3].tolist(),
        "qpos_quaternion_wxyz_indices": list(range(qstart + 3, qstart + 7)),
        "qpos_quaternion_wxyz": data.qpos[qstart + 3:qstart + 7].tolist(),
        "qvel_linear_indices": list(range(vstart, vstart + 3)),
        "qvel_angular_indices": list(range(vstart + 3, vstart + 6)),
        "actuator_names": [a["name"] for a in actuators],
        "actuators": actuators,
        "sensor_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--json", action="store_true", help="Machine-readable inspection result")
    args = parser.parse_args()
    result = inspect(args.model)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    for key in ("model", "nq", "nv", "nu", "nsensor", "nbody", "ngeom", "qpos", "qvel", "ctrl"):
        print(f"{key}: {result[key]}")
    print("qpos[0:3] = xyz; qpos[3:7] = quaternion (w,x,y,z)")
    print("qvel[0:3] = world-frame linear velocity; qvel[3:6] = local-frame angular velocity")
    for a in result["actuators"]:
        print(f"ctrl[{a['index']}] = {a['name']}, gear={a['gear']}, ctrlrange={a['ctrlrange']}")
    print(f"sensors: {result['sensor_names']}")


if __name__ == "__main__":
    main()
