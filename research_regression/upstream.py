"""Single source of upstream paths for test fixtures and bounded regression."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

from research_ops.environment import inspect_upstream
from research_ops.storage import OpsError, read_json

ROOT=Path(__file__).resolve().parents[1]


def resolve_upstream(root=ROOT, environment=None):
    environment=os.environ if environment is None else environment
    lock=read_json(ROOT/'research_ops/environment_lock.json')['upstream']
    path=Path(environment.get('MINEUAV_LIDAR_UPSTREAM',str(Path(root)/lock['directory']))).resolve()
    # The same pinned identity applies to an explicitly selected directory.
    inspect_upstream(path.parent,{**lock,'directory':path.name})
    return path


def source_identity(root=ROOT,require_import=True):
    path=resolve_upstream(root)
    lock=read_json(ROOT/'research_ops/environment_lock.json')['upstream']
    value={'source_directory':str(path),'commit':lock['commit'],
           'pattern_sha256':lock['pattern_sha256'],'imported_package_directory':'NOT_TESTED'}
    if require_import:
        module=importlib.import_module('mujoco_lidar')
        imported=Path(module.__file__).resolve()
        if imported.parent != path/'src/mujoco_lidar':
            raise OpsError('imported MuJoCo-LiDAR and selected test source differ')
        value['imported_package_directory']=str(imported.parent)
    return value
