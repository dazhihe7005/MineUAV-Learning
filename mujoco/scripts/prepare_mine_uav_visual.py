"""Build three visual-only meshes from selected assembly-space STL parts.

The source directory is read-only. The output is a working copy in metres,
rotated from CAD Y-up to MuJoCo Z-up. No physical parameters are inferred.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh


SOURCE = Path.home() / "Documents/xwechat_files/wxid_rlp4qxgja6mi22_9f7a/msg/file/2026-09/stl2"
ASSETS = Path(__file__).resolve().parents[1] / "assets"
# Geometry-frame origin is for visualization only; it is NOT measured CoM.
ORIGIN_MM = np.array([508.23989868, 340.0, 1100.0])


def category(filename: str) -> str | None:
    if "MID-360" in filename:
        return "lidar"
    if "四目鱼眼相机" in filename:
        return "camera"
    major = (
        "机臂-", "U7-", "P15x5桨叶", "电池", "nuc-", "中心版", "飞控-", "电调-",
        "保护笼长横", "保护笼弯", "桨保碳管", "脚架2-", "26碳管", "保护壳1",
        "保护壳盖", "nuc隔板",
    )
    return "body" if any(token in filename for token in major) else None


def cad_to_mujoco(vertices: np.ndarray) -> np.ndarray:
    relative = vertices - ORIGIN_MM
    # +90 degrees about CAD X: X -> X, CAD Y -> MuJoCo Z, CAD Z -> -MuJoCo Y.
    return np.column_stack((relative[:, 0], -relative[:, 2], relative[:, 1])) * 0.001


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    args = parser.parse_args()
    paths = sorted(path for path in args.source.iterdir() if path.is_file() and path.suffix.lower() == ".stl")
    if not paths:
        raise FileNotFoundError(f"No STL files in {args.source}")
    grouped = {"body": [], "lidar": [], "camera": []}
    for path in paths:
        label = category(path.name)
        if label is None:
            continue
        mesh = trimesh.load(path, force="mesh", process=False)
        if mesh.is_empty:
            raise ValueError(f"Empty STL: {path.name}")
        mesh.vertices = cad_to_mujoco(mesh.vertices)
        grouped[label].append(mesh)
    args.assets.mkdir(parents=True, exist_ok=True)
    for label, meshes in grouped.items():
        if not meshes:
            raise ValueError(f"No {label} parts selected")
        merged = trimesh.util.concatenate(meshes)
        # MuJoCo's STL decoder caps each mesh at 200,000 faces. The MID360
        # source alone exceeds that, so keep all its triangles in OBJ format.
        extension = ".obj" if label == "lidar" else ".stl"
        output = args.assets / f"mine_uav_{label}_visual{extension}"
        merged.export(output)
        print(f"{label}: {len(meshes)} parts, {len(merged.faces)} faces -> {output}")
        print(f"  bounds_m={merged.bounds.tolist()}")
    print(f"source={len(paths)} STL files; originals unchanged")


if __name__ == "__main__":
    main()
