"""Pinned dependency inspection/restoration without package installation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from .identity import git, safe_path
from .storage import OpsError, sha256


def resource_status(root):
    fields = {line.split(':', 1)[0]: line.split(':', 1)[1].split()[0]
              for line in Path('/proc/meminfo').read_text().splitlines()}
    return {'mem_available_mib': int(fields['MemAvailable'])/1024.,
            'disk_free_bytes': shutil.disk_usage(root).free}


def enforce_resources(root):
    resources = resource_status(root)
    if resources['mem_available_mib'] < 1536. or resources['disk_free_bytes'] < 2*1024**3:
        raise OpsError('resource guard: need >=1536MiB MemAvailable and >=2GiB free disk')
    return resources


def runtime_environment(root, lock):
    env = dict(os.environ)
    paths = [str(Path(root)/'mujoco'), str(Path(root)/'mujoco/rl')]
    if lock.get('upstream'):
        paths.append(str(safe_path(root, lock['upstream']['directory'])/'src'))
    if lock.get('package_root'):
        paths.append(lock['package_root'])
    # Do not inherit arbitrary PYTHONPATH or Python startup injection.
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env.pop('PYTHONHOME', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.update(lock.get('environment_variables', {}))
    if lock.get('upstream'):
        env['MINEUAV_LIDAR_UPSTREAM'] = str(safe_path(root, lock['upstream']['directory']))
    return env


def inspect_upstream(root, upstream):
    if upstream is None:
        return 'NOT_APPLICABLE'
    path = safe_path(root, upstream['directory'])
    if not path.is_dir() or git(path, 'rev-parse', 'HEAD').stdout.strip() != upstream['commit']:
        raise OpsError('pinned durable MuJoCo-LiDAR dependency missing or wrong commit')
    if git(path, 'status', '--porcelain', '--untracked-files=no').stdout.strip():
        raise OpsError('pinned dependency has modified tracked files')
    for name in git(path, 'ls-files', '--others', '-z').stdout.split('\0'):
        if not name:
            continue
        extra = safe_path(path, name)
        if '__pycache__' not in extra.parts or extra.suffix != '.pyc':
            raise OpsError('unexpected untracked content in pinned dependency')
    if sha256(path/upstream['pattern_path']) != upstream['pattern_sha256']:
        raise OpsError('LiDAR scan pattern identity mismatch')
    return {'commit': upstream['commit'], 'directory': str(path),
            'pattern_sha256': upstream['pattern_sha256']}


def inspect_environment(root, lock):
    upstream = inspect_upstream(root, lock.get('upstream'))
    executable = Path(lock['interpreter'])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise OpsError('locked Python interpreter missing')
    if lock.get('interpreter_sha256') and sha256(executable) != lock['interpreter_sha256']:
        raise OpsError('locked Python executable identity changed')
    script = '''import importlib,importlib.metadata,json,sys
request=json.loads(sys.argv[1]); modules={}
for name in request['modules']:
 m=importlib.import_module(name)
 modules[name]={'file':getattr(m,'__file__',None),'version':getattr(m,'__version__',None)}
 if name=='mujoco':
  assert hasattr(m,'MjModel') and callable(m.MjModel.from_xml_string)
  modules[name]['real_MjModel_API']=True
print(json.dumps({'executable':sys.executable,'python':sys.version.split()[0],
 'packages':{p:importlib.metadata.version(p) for p in request['packages']},'modules':modules}))'''
    result = subprocess.run([str(executable), '-c', script, json.dumps(lock)],
                            capture_output=True, text=True, timeout=30,
                            env=runtime_environment(root, lock))
    if result.returncode:
        raise OpsError('locked environment probe failed; no package installation attempted')
    value = json.loads(result.stdout)
    if value['packages'] != lock['packages']:
        raise OpsError('locked software package versions differ')
    if lock.get('python_version') and value['python'] != lock['python_version']:
        raise OpsError('locked Python version differs')
    if 'mujoco' in value['modules']:
        module_file = value['modules']['mujoco']['file']
        if not module_file or not Path(module_file).resolve().is_relative_to(Path(lock['package_root']).resolve()):
            raise OpsError('MuJoCo import is not the actual locked Python package')
    value['upstream'] = upstream
    return value


def restore_upstream(root, lock, source=None):
    upstream = lock.get('upstream')
    if not upstream:
        raise OpsError('no restorable upstream declared')
    destination = safe_path(root, upstream['directory'])
    if destination.exists():
        return inspect_upstream(root, upstream)
    enforce_resources(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent/('.fetch-'+uuid.uuid4().hex)
    # Retain failed fetches for diagnosis; never remove an existing dependency.
    origin = str(source) if source is not None else upstream['url']
    subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', origin, str(temporary)],
                   check=True, capture_output=True, timeout=180)
    git(temporary, 'checkout', '--detach', upstream['commit'])
    if sha256(temporary/upstream['pattern_path']) != upstream['pattern_sha256']:
        raise OpsError('restored scan pattern is not authorized')
    # mkdir reservation prevents an overlapping restore from replacing a dependency.
    if destination.exists():
        raise OpsError('dependency appeared during restoration; no overwrite')
    temporary.rename(destination)
    return inspect_upstream(root, upstream)
