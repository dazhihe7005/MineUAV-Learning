"""Inspect STL coordinates and render the SolidWorks export without alignment."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


DEFAULT_SOURCE = (
    Path.home()
    / "Documents/xwechat_files/wxid_rlp4qxgja6mi22_9f7a/msg/file/2026-09/stl2"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent
CSV_FIELDS = [
    "filename", "status", "error", "vertex_count", "face_count",
    "bbox_min_x", "bbox_min_y", "bbox_min_z",
    "bbox_max_x", "bbox_max_y", "bbox_max_z",
    "extent_x", "extent_y", "extent_z",
    "centroid_x", "centroid_y", "centroid_z",
]
COLORS = {
    "arms": "#f39c12",
    "motors": "#d62728",
    "propellers": "#2ca02c",
    "battery": "#1f77b4",
    "nuc / controller": "#17becf",
    "MID360": "#9467bd",
    "fisheye camera": "#e377c2",
    "other parts": "#818b96",
}


def discover_stls(source_dir: Path) -> list[Path]:
    """Discover only immediate children, accepting any STL extension case."""
    return sorted(
        (path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".stl"),
        key=lambda path: path.name.casefold(),
    )


def validate_output_dir(source_dir: Path, output_dir: Path) -> None:
    """Refuse any output location within the read-only source tree."""
    source = source_dir.resolve()
    output = output_dir.resolve()
    if output == source or source in output.parents:
        raise ValueError(f"Output directory must be outside STL source: {output}")


def inspect_files(paths: list[Path]):
    """Return CSV rows, unchanged meshes, and the union of their bounds."""
    records = []
    loaded = []
    union_min = np.full(3, np.inf)
    union_max = np.full(3, -np.inf)

    for path in paths:
        row = {field: "" for field in CSV_FIELDS}
        row["filename"] = path.name
        try:
            mesh = trimesh.load_mesh(path, process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Expected one mesh, got {type(mesh).__name__}")
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                raise ValueError("STL contains no triangles")
            if not np.isfinite(mesh.vertices).all():
                raise ValueError("STL contains non-finite vertices")

            bounds = np.asarray(mesh.bounds)
            centroid = np.asarray(mesh.centroid)
            if not np.isfinite(bounds).all() or not np.isfinite(centroid).all():
                raise ValueError("STL has invalid bounds or centroid")
            extent = bounds[1] - bounds[0]
            row.update(status="ok", vertex_count=len(mesh.vertices), face_count=len(mesh.faces))
            for axis, index in zip("xyz", range(3)):
                row[f"bbox_min_{axis}"] = float(bounds[0, index])
                row[f"bbox_max_{axis}"] = float(bounds[1, index])
                row[f"extent_{axis}"] = float(extent[index])
                row[f"centroid_{axis}"] = float(centroid[index])
            union_min = np.minimum(union_min, bounds[0])
            union_max = np.maximum(union_max, bounds[1])
            loaded.append((path, mesh))
        except Exception as exc:  # One damaged STL must not abort the batch.
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        records.append(row)

    bounds = np.stack((union_min, union_max)) if loaded else None
    return records, loaded, bounds


def category(filename: str) -> str:
    name = filename.casefold()
    if "机臂" in name:
        return "arms"
    if "u7-" in name:
        return "motors"
    if "p15x5" in name:
        return "propellers"
    if "电池" in name:
        return "battery"
    if "mid-360" in name or "mid360" in name:
        return "MID360"
    if "四目鱼眼相机" in name:
        return "fisheye camera"
    if "nuc" in name or "飞控" in name:
        return "nuc / controller"
    return "other parts"


def project_2d(triangles: np.ndarray, view: str) -> np.ndarray:
    """Project raw coordinates onto the rotor plane or a vertical plane."""
    if view == "top":
        return triangles[:, :, [0, 2]]  # X-Z is this export's rotor plane.
    if view == "side":
        return triangles[:, :, [0, 1]]  # X-Y shows the vertical spacing.
    raise ValueError(f"Unknown 2D view: {view}")


def render_previews(loaded, bounds, output_dir: Path, max_faces: int = 75000) -> None:
    """Sample faces for PNG speed, never changing mesh coordinates or geometry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    face_counts = np.array([len(mesh.faces) for _, mesh in loaded])
    weights = np.sqrt(face_counts)
    quotas = np.minimum(
        face_counts,
        np.maximum(12, np.floor(max_faces * weights / weights.sum()).astype(int)),
    )
    drawables = []
    for (path, mesh), quota in zip(loaded, quotas):
        indices = np.linspace(0, len(mesh.faces) - 1, int(quota), dtype=int)
        drawables.append((mesh.triangles[indices], category(path.name)))

    center = bounds.mean(axis=0)
    radius = max(float(np.max(bounds[1] - bounds[0])) * 0.55, 1.0)
    used_categories = set(kind for _, kind in drawables)
    for name in ("top", "side", "iso"):
        fig = plt.figure(figsize=(12, 10))
        if name == "iso":
            ax = fig.add_subplot(111, projection="3d", proj_type="ortho")
            for triangles, kind in drawables:
                ax.add_collection3d(
                    Poly3DCollection(
                        triangles, facecolors=COLORS[kind], edgecolors="none",
                        linewidths=0, alpha=0.85 if kind != "other parts" else 0.45,
                    )
                )
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=26, azim=-55)
            ax.set_xlabel("X (native STL units)")
            ax.set_ylabel("Y (native STL units)")
            ax.set_zlabel("Z (native STL units)")
        else:
            ax = fig.add_subplot(111)
            for triangles, kind in drawables:
                ax.add_collection(
                    PolyCollection(
                        project_2d(triangles, name), facecolors=COLORS[kind],
                        edgecolors="none", linewidths=0,
                        alpha=0.85 if kind != "other parts" else 0.45,
                    )
                )
            vertical_index = 2 if name == "top" else 1
            ax.set_xlim(bounds[0, 0] - 30, bounds[1, 0] + 30)
            ax.set_ylim(bounds[0, vertical_index] - 30, bounds[1, vertical_index] + 30)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("X (native STL units)")
            ax.set_ylabel(f"{'Z' if name == 'top' else 'Y'} (native STL units)")
            ax.grid(alpha=0.25)
        ax.set_title(f"SolidWorks STL assembly — {name} view (unaligned)")
        fig.legend(
            handles=[Patch(facecolor=COLORS[kind], label=kind) for kind in COLORS if kind in used_categories],
            loc="lower center", ncol=4, frameon=False,
        )
        fig.savefig(output_dir / f"assembly_preview_{name}.png", dpi=150)
        if name == "iso":
            fig.savefig(output_dir / "assembly_preview.png", dpi=150)
        plt.close(fig)


def export_raw_assembly(loaded, output_path: Path) -> None:
    """Concatenate triangle geometry only; no transforms, scaling, or repair."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = trimesh.util.concatenate([mesh for _, mesh in loaded])
    combined.export(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--export-raw", action="store_true",
        help="Export raw concatenation only after manually confirming assembly placement.",
    )
    args = parser.parse_args()
    validate_output_dir(args.source, args.output)
    paths = discover_stls(args.source)
    if not paths:
        raise SystemExit(f"No STL files found in {args.source}")

    records, loaded, bounds = inspect_files(paths)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "stl_report.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"STL files: {len(paths)}; loaded: {len(loaded)}; failed: {len(paths) - len(loaded)}")
    for row in records:
        if row["status"] != "ok":
            print(f"FAILED {row['filename']}: {row['error']}")
    if not loaded:
        raise SystemExit("No readable STL files; report contains the errors.")

    print(f"Bounds min: {bounds[0]}")
    print(f"Bounds max: {bounds[1]}")
    print(f"Extents: {bounds[1] - bounds[0]} (native STL units)")
    render_previews(loaded, bounds, args.output)
    if args.export_raw:
        export_raw_assembly(loaded, args.output / "assembly_raw.stl")
        print(f"Raw assembly: {args.output / 'assembly_raw.stl'}")


if __name__ == "__main__":
    main()
