import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
DURABLE=ROOT/'.research_local/dependencies/MuJoCo-LiDAR'


def api():
    assert importlib.util.find_spec('research_regression') and importlib.util.find_spec('research_regression.upstream'), 'portable source resolver missing'
    from research_regression import upstream
    return upstream


class PortabilityTests(unittest.TestCase):
    def test_default_is_durable_and_does_not_fall_back_to_tmp(self):
        m=api()
        self.assertEqual(m.resolve_upstream(ROOT, {}),DURABLE.resolve())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):m.resolve_upstream(Path(d),{})

    def test_explicit_path_is_validated_and_missing_source_is_rejected(self):
        m=api()
        self.assertEqual(m.resolve_upstream(ROOT,{'MINEUAV_LIDAR_UPSTREAM':str(DURABLE)}),DURABLE.resolve())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):m.resolve_upstream(ROOT,{'MINEUAV_LIDAR_UPSTREAM':d})

    def test_source_identity_records_pinned_commit_and_pattern(self):
        m=api();value=m.source_identity(ROOT,require_import=False)
        self.assertEqual(value['commit'],'e6b879df4b96daaed30ef10fe02650d29048fa9b')
        self.assertEqual(value['pattern_sha256'],'9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3')
        self.assertEqual(value['source_directory'],str(DURABLE.resolve()))


if __name__=='__main__':unittest.main()
