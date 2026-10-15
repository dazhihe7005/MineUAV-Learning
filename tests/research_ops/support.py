import json
from pathlib import Path
import subprocess
import sys


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def project(root):
    root.mkdir()
    git(root, 'init', '-b', 'feat/ops-test')
    git(root, 'config', 'user.name', 'OPS Test')
    git(root, 'config', 'user.email', 'ops@example.invalid')
    (root / 'research_ops').mkdir()
    (root / 'docs').mkdir()
    (root / 'history').mkdir()
    (root / 'history/failure.json').write_text('{"result":"FAIL"}\n')
    (root / 'runtime.py').write_text('# no training\n')
    (root / 'README.md').write_text('# Test project\n')
    (root / '.gitignore').write_text('.research_local/\n')
    environment = {'interpreter': sys.executable, 'package_root': None, 'packages': {},
                   'modules': [], 'upstream': None, 'environment_variables': {}}
    (root / 'research_ops/environment_lock.json').write_text(json.dumps(environment))
    config = {'project_id': 'MineUAV-Learning', 'phase_id': 'OPS-01',
              'runtime_paths': ['runtime.py'], 'protected_paths': ['history'],
              'protected_exclusions': [], 'history_worktrees': [],
              'original_l9': None, 'checkpoint': None,
              'scenario_manifest_sha256': 'NOT_APPLICABLE',
              'sync_files': ['README.md', 'docs/PROJECT_PROGRESS.md', 'docs/CODEX_NEXT_ACTION.md',
                             'docs/DAILY_PROGRESS.md', 'docs/RESEARCH_STATUS.json'],
              'remote': 'origin'}
    (root / 'research_ops/operations_config.json').write_text(json.dumps(config))
    git(root, 'add', 'README.md', 'runtime.py', 'history', 'research_ops', '.gitignore')
    git(root, 'commit', '-m', 'fixture baseline')
    return root
