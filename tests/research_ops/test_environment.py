import json
from pathlib import Path
import tempfile
import unittest

from support import git,project
from research_ops.environment import inspect_environment,restore_upstream
from research_ops.storage import sha256


class EnvironmentTests(unittest.TestCase):
    def test_memory_and_disk_guards_reject_insufficient_resources(self):
        from unittest.mock import patch
        from research_ops.environment import enforce_resources
        for available,disk in ((1535.9,3*1024**3),(3000.,2*1024**3-1)):
            with patch('research_ops.environment.resource_status',return_value={'mem_available_mib':available,'disk_free_bytes':disk}):
                with self.assertRaises(Exception):enforce_resources(Path.cwd())

    def test_fixed_dependency_is_restored_durably_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root=project(Path(d)/'repo');up=project(Path(d)/'upstream')
            p=up/'pattern.npy';p.write_bytes(b'fixed scan pattern')
            git(up,'add','pattern.npy');git(up,'commit','-m','pattern')
            lock=json.loads((root/'research_ops/environment_lock.json').read_text())
            lock['upstream']={'directory':'.research_local/dependencies/lidar','url':str(up),
                'commit':git(up,'rev-parse','HEAD'),'pattern_path':'pattern.npy','pattern_sha256':sha256(p)}
            result=restore_upstream(root,lock,source=up)
            self.assertEqual(result['commit'],git(up,'rev-parse','HEAD'))
            self.assertEqual(inspect_environment(root,lock)['upstream']['pattern_sha256'],sha256(p))
            destination=root/lock['upstream']['directory']/'pattern.npy'
            destination.write_bytes(b'user change')
            with self.assertRaises(Exception):restore_upstream(root,lock,source=up)
            self.assertEqual(destination.read_bytes(),b'user change')

    def test_dependency_from_wrong_commit_or_pattern_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=project(Path(d)/'repo');up=project(Path(d)/'upstream')
            p=up/'pattern.npy';p.write_bytes(b'pattern');git(up,'add','pattern.npy');git(up,'commit','-m','pattern')
            lock=json.loads((root/'research_ops/environment_lock.json').read_text())
            lock['upstream']={'directory':'.research_local/dependencies/lidar','url':str(up),
                'commit':git(up,'rev-parse','HEAD'),'pattern_path':'pattern.npy','pattern_sha256':'0'*64}
            with self.assertRaises(Exception):restore_upstream(root,lock,source=up)
            self.assertFalse((root/lock['upstream']['directory']).exists())

    def test_untracked_startup_code_is_rejected_before_python_probe(self):
        with tempfile.TemporaryDirectory() as d:
            root=project(Path(d)/'repo');up=project(Path(d)/'upstream')
            (up/'src').mkdir();p=up/'pattern.npy';p.write_bytes(b'pattern')
            git(up,'add','pattern.npy');git(up,'commit','-m','pattern')
            lock=json.loads((root/'research_ops/environment_lock.json').read_text())
            lock['upstream']={'directory':'.research_local/dependencies/lidar','url':str(up),
                'commit':git(up,'rev-parse','HEAD'),'pattern_path':'pattern.npy','pattern_sha256':sha256(p)}
            restore_upstream(root,lock,source=up)
            dest=root/lock['upstream']['directory'];(dest/'src').mkdir()
            marker=root/'executed'
            (dest/'src/sitecustomize.py').write_text('from pathlib import Path;Path('+repr(str(marker))+').write_text("executed")')
            with self.assertRaises(Exception):inspect_environment(root,lock)
            self.assertFalse(marker.exists())


if __name__=='__main__':unittest.main()
