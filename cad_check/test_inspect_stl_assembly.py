"""Small synthetic STL checks; never writes to the SolidWorks export directory."""

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import trimesh

from inspect_stl_assembly import (
    discover_stls,
    export_raw_assembly,
    inspect_files,
    project_2d,
    render_previews,
    validate_output_dir,
)


class AssemblyInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "output"
        self.source.mkdir()
        self.output.mkdir()

    def write_box(self, name, center):
        mesh = trimesh.creation.box(extents=[2, 2, 2])
        mesh.apply_translation(center)
        path = self.source / name
        mesh.export(path)
        return path

    def test_mixed_case_discovery_and_centroids_keep_world_positions(self):
        self.write_box("arm-1.STL", [12, -3, 7])
        self.write_box("arm-2.stl", [-8, 4, 2])
        (self.source / "ignore.txt").write_text("not an STL")

        files = discover_stls(self.source)
        self.assertEqual([path.name for path in files], ["arm-1.STL", "arm-2.stl"])
        records, loaded, bounds = inspect_files(files)

        self.assertEqual(len(loaded), 2)
        self.assertTrue(all(row["status"] == "ok" for row in records))
        self.assertTrue(np.allclose(records[0]["centroid_x"], 12))
        self.assertTrue(np.allclose(records[1]["centroid_x"], -8))
        self.assertTrue(np.allclose(bounds[0], [-9, -4, 1]))
        self.assertTrue(np.allclose(bounds[1], [13, 5, 8]))

    def test_bad_stl_is_recorded_without_blocking_good_stl(self):
        self.write_box("good.STL", [3, 0, 0])
        (self.source / "bad.stl").write_bytes(b"not a valid STL")

        records, loaded, bounds = inspect_files(discover_stls(self.source))

        self.assertEqual(len(records), 2)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(records[0]["status"], "error")
        self.assertTrue(records[0]["error"])
        self.assertEqual(records[1]["status"], "ok")
        self.assertTrue(np.allclose(bounds[0], [2, -1, -1]))

    def test_raw_export_and_preview_do_not_translate_parts(self):
        self.write_box("first.STL", [12, -3, 7])
        self.write_box("second.stl", [-8, 4, 2])
        _, loaded, bounds = inspect_files(discover_stls(self.source))

        export_path = self.output / "assembly_raw.stl"
        export_raw_assembly(loaded, export_path)
        combined = trimesh.load_mesh(export_path, process=False)
        self.assertEqual(len(combined.faces), sum(len(mesh.faces) for _, mesh in loaded))
        expected = sorted(
            tuple(face.ravel())
            for _, mesh in loaded
            for face in mesh.triangles
        )
        actual = sorted(tuple(face.ravel()) for face in combined.triangles)
        self.assertEqual(actual, expected)

        render_previews(loaded, bounds, self.output, max_faces=100)
        for name in (
            "assembly_preview.png",
            "assembly_preview_top.png",
            "assembly_preview_side.png",
            "assembly_preview_iso.png",
        ):
            self.assertGreater((self.output / name).stat().st_size, 1000)

    def test_output_must_be_outside_source(self):
        with self.assertRaises(ValueError):
            validate_output_dir(self.source, self.source)
        with self.assertRaises(ValueError):
            validate_output_dir(self.source, self.source / "child")
        validate_output_dir(self.source, self.output)

    def test_top_view_uses_xz_and_side_view_uses_xy(self):
        triangle = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        self.assertEqual(project_2d(triangle, "top").tolist(), [[[1, 3], [4, 6], [7, 9]]])
        self.assertEqual(project_2d(triangle, "side").tolist(), [[[1, 2], [4, 5], [7, 8]]])


if __name__ == "__main__":
    unittest.main()
