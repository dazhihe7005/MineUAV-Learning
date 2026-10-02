"""Estimate 7 kg MineUAV COM/inertia from assembly-space STL geometry.

This is an engineering estimate, not a measurement. Original STL files are
read-only; generated artifacts live under mujoco/models and mujoco/reports.
"""

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from prepare_mine_uav_visual import SOURCE, cad_to_mujoco


ROOT = Path(__file__).resolve().parents[1]
OLD_MODEL = ROOT / "models" / "mine_uav_visual.xml"
NEW_MODEL = ROOT / "models" / "mine_uav_dynamics_v2.xml"
REPORT = ROOT / "reports" / "dynamics_v2_report.json"
MANIFEST = ROOT / "reports" / "dynamics_v2_component_manifest.csv"
TOTAL_MASS_KG = 7.0
MASS_KG = {
    "battery_1": 1.60, "battery_2": 1.60,
    "nuc": 0.5775, "camera": 0.880, "mid360": 0.265,
    "motor_1": 0.299, "motor_2": 0.299,
    "motor_3": 0.299, "motor_4": 0.299,
    "propeller_1": 0.013, "propeller_2": 0.013,
    "propeller_3": 0.013, "propeller_4": 0.013,
    "flight_controller": 0.0593,
}


@dataclass(frozen=True)
class Moments:
    method: str
    center_m: np.ndarray
    inertia_per_kg: np.ndarray
    volume_m3: float | None
    area_m2: float


def classify_stl_name(name: str) -> str:
    lower = name.lower()
    for token, role, count in (
        ("12000mah格氏电池", "battery", 2),
        ("u7", "motor", 4),
        ("p15x5桨叶", "propeller", 4),
    ):
        if token in lower:
            match = re.search(r"-(\d+)\.stl$", lower)
            if match and 1 <= int(match.group(1)) <= count:
                return f"{role}_{match.group(1)}"
            raise ValueError(f"Unrecognized {role} STL instance: {name}")
    if "mid-360" in lower:
        return "mid360"
    if "四目鱼眼相机" in name:
        return "camera"
    if "nuc-" in lower:
        return "nuc"
    if "飞控-" in name:
        return "flight_controller"
    return "structure"


def geometry_moments(mesh: trimesh.Trimesh) -> Moments:
    """Return geometry centroid and unit-mass inertia in the mesh's metre frame.

    Closed meshes use solid-volume moments. Open meshes use a uniform-area
    triangle surface proxy; their true internal mass distribution is unknown.
    """
    if mesh.is_empty or not np.isfinite(mesh.vertices).all():
        raise ValueError("Empty or nonfinite STL geometry")
    if mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0:
        prop = mesh.mass_properties
        volume = float(prop["volume"])
        return Moments(
            "watertight_volume", np.asarray(prop["center_mass"], dtype=float),
            np.asarray(prop["inertia"], dtype=float) / volume,
            volume, float(mesh.area),
        )

    triangles = mesh.triangles
    areas = mesh.area_faces
    area = float(areas.sum())
    if not math.isfinite(area) or area <= 0:
        raise ValueError("Open STL has no finite triangle area")
    triangle_centers = triangles.mean(axis=1)
    center = np.einsum("n,ni->i", areas, triangle_centers) / area
    offsets = triangle_centers - center
    within = triangles - triangle_centers[:, None, :]
    covariance = (
        np.einsum("n,ni,nj->ij", areas, offsets, offsets)
        + np.einsum("n,nvi,nvj->ij", areas, within, within) / 12.0
    ) / area
    inertia = np.trace(covariance) * np.eye(3) - covariance
    return Moments("triangle_area_surface", center, inertia, None, area)


def estimate_dynamics(source: Path = SOURCE) -> dict:
    paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".stl")
    if len(paths) != 177:
        raise ValueError(f"Expected 177 source STL files; found {len(paths)} in {source}")

    entries = []
    for path in paths:
        mesh = trimesh.load(path, force="mesh", process=True)
        mesh.vertices = cad_to_mujoco(mesh.vertices)
        entries.append({"filename": path.name, "role": classify_stl_name(path.name),
                        "moments": geometry_moments(mesh)})

    role_counts = {role: sum(item["role"] == role for item in entries)
                   for role in set(item["role"] for item in entries)}
    for role in MASS_KG:
        expected = 8 if role == "mid360" else 1
        if role_counts.get(role, 0) != expected:
            raise ValueError(f"Expected {expected} STL file(s) for {role}; found {role_counts.get(role, 0)}")
    mid_assemblies = [item for item in entries if item["role"] == "mid360"
                      and "MID-360_4_ASM_2_ASM-0" in item["filename"]]
    if len(mid_assemblies) != 1:
        raise ValueError("Expected one MID360 aggregate assembly STL")
    mid_assembly = mid_assemblies[0]["filename"]

    known_mass = sum(MASS_KG.values())
    structure_mass = TOTAL_MASS_KG - known_mass
    if structure_mass <= 0:
        raise ValueError("Known component masses exceed total mass")
    structural = [item for item in entries if item["role"] == "structure"]
    if not structural or any(item["moments"].method != "watertight_volume" for item in structural):
        raise ValueError("Structure volume weighting requires all remaining STL meshes to be watertight")
    structure_volume = sum(item["moments"].volume_m3 for item in structural)
    if structure_volume <= 0:
        raise ValueError("Structure STL volume sum must be positive")

    components = []
    for item in entries:
        role, moments = item["role"], item["moments"]
        if role == "structure":
            mass = structure_mass * moments.volume_m3 / structure_volume
            reason = "structure_mass allocated by watertight mesh volume"
        elif role == "mid360" and item["filename"] != mid_assembly:
            mass = 0.0
            reason = "MID360 constituent reference; excluded to avoid aggregate double count"
        else:
            mass = MASS_KG[role]
            reason = "assigned component mass"
        components.append({
            "filename": item["filename"], "role": role, "mass_kg": float(mass),
            "geometry_method": moments.method, "volume_m3": moments.volume_m3,
            "area_m2": moments.area_m2, "center_m": moments.center_m.tolist(),
            "mass_reason": reason, "inertia_per_kg": moments.inertia_per_kg.tolist(),
        })

    structure_com = sum((c["mass_kg"] * np.asarray(c["center_m"]) for c in components
                         if c["role"] == "structure"), np.zeros(3)) / structure_mass
    total_from_parts = sum(c["mass_kg"] for c in components)
    if not math.isclose(total_from_parts, TOTAL_MASS_KG, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Mass manifest sums to {total_from_parts}, not {TOTAL_MASS_KG}")
    com = sum((c["mass_kg"] * np.asarray(c["center_m"]) for c in components), np.zeros(3)) / TOTAL_MASS_KG
    inertia = np.zeros((3, 3))
    for component in components:
        mass = component["mass_kg"]
        if mass == 0:
            continue
        delta = np.asarray(component["center_m"]) - com
        intrinsic = mass * np.asarray(component["inertia_per_kg"])
        parallel = mass * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
        inertia += intrinsic + parallel
    inertia = 0.5 * (inertia + inertia.T)
    if np.min(np.linalg.eigvalsh(inertia)) <= 0:
        raise ValueError("Estimated inertia is not positive definite")

    method_counts = {method: sum(c["geometry_method"] == method for c in components)
                     for method in {c["geometry_method"] for c in components}}
    return {
        "label_com": "ESTIMATED_COM_V2", "label_inertia": "ESTIMATED_INERTIA_V2",
        "source_stl_directory": str(source), "stl_count": len(paths),
        "structure_mesh_count": len(structural), "known_mass_kg": known_mass,
        "structure_mass_kg": structure_mass, "total_mass_kg": TOTAL_MASS_KG,
        "structure_volume_m3": structure_volume,
        "structure_com_m": structure_com.tolist(), "estimated_com_v2_m": com.tolist(),
        "estimated_inertia_v2_kg_m2": inertia.tolist(),
        "geometry_method_counts": method_counts, "mid360_mass_geometry": mid_assembly,
        "components": components,
        "caveats": [
            "COM and inertia are geometry-based estimates, not measured values.",
            "All 156 structure STL meshes are watertight; residual mass is allocated by enclosed volume.",
            "Open U7 and MID360 aggregate meshes use uniform triangle-area surface inertia proxies.",
            "MID360 constituent files are excluded from mass to avoid aggregate double counting.",
            "Each part is treated as uniform-density geometry; material composition is not known.",
        ],
    }


def write_model(report: dict) -> None:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(OLD_MODEL, parser=parser).getroot()
    root.set("model", "Mine UAV dynamics v2 estimated mass properties")
    for child in root:
        if child.tag is ET.Comment and "TEMPORARY mass/COM/inertia" in child.text:
            child.text = " Mass 7 kg; ESTIMATED_COM_V2 and ESTIMATED_INERTIA_V2 are engineering estimates, not measurements. "
    body = root.find(".//body[@name='mine_uav']")
    if body is None:
        raise ValueError("No mine_uav body in existing model")
    for child in body:
        if child.tag is ET.Comment and "mass = TEMPORARY_PLACEHOLDER" in child.text:
            child.text = " ESTIMATED_COM_V2 and ESTIMATED_INERTIA_V2 from 177 assembly STL files; not measured. "
    inertial = body.find("inertial")
    if inertial is None:
        raise ValueError("No inertial element in existing model")
    com = report["estimated_com_v2_m"]
    tensor = report["estimated_inertia_v2_kg_m2"]
    inertial.attrib.clear()
    inertial.set("pos", " ".join(f"{x:.17g}" for x in com))
    inertial.set("mass", f"{report['total_mass_kg']:.17g}")
    # MJCF fullinertia order: Ixx Iyy Izz Ixy Ixz Iyz.
    full = (tensor[0][0], tensor[1][1], tensor[2][2],
            tensor[0][1], tensor[0][2], tensor[1][2])
    inertial.set("fullinertia", " ".join(f"{x:.17g}" for x in full))
    ET.indent(root, space="  ")
    NEW_MODEL.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def write_reports(report: dict) -> None:
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ("filename", "role", "mass_kg", "geometry_method", "volume_m3",
              "area_m2", "center_x_m", "center_y_m", "center_z_m", "mass_reason")
    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for c in report["components"]:
            writer.writerow({
                "filename": c["filename"], "role": c["role"], "mass_kg": c["mass_kg"],
                "geometry_method": c["geometry_method"], "volume_m3": c["volume_m3"],
                "area_m2": c["area_m2"], "center_x_m": c["center_m"][0],
                "center_y_m": c["center_m"][1], "center_z_m": c["center_m"][2],
                "mass_reason": c["mass_reason"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print full report to stdout")
    args = parser.parse_args()
    report = estimate_dynamics()
    write_model(report)
    write_reports(report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"known_mass={report['known_mass_kg']:.7f} kg")
        print(f"structure_mass={report['structure_mass_kg']:.7f} kg")
        print(f"structure_COM={report['structure_com_m']} m")
        print(f"ESTIMATED_COM_V2={report['estimated_com_v2_m']} m")
        print(f"ESTIMATED_INERTIA_V2={report['estimated_inertia_v2_kg_m2']} kg*m^2")
        print(f"Saved {NEW_MODEL}, {REPORT}, {MANIFEST}")


if __name__ == "__main__":
    main()
