"""Streaming historical inventories; never repair or overwrite protected data."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

from .storage import OpsError, digest, read_json, sha256


def git(root, *args, check=True, timeout=45):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                            text=True, timeout=timeout, check=False,
                            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    if check and result.returncode:
        # Git diagnostics can include credential-bearing URLs; do not echo them.
        raise OpsError('git operation failed: ' + args[0])
    return result


def safe_path(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise OpsError('path outside project')
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise OpsError('symlink in managed path')
    if not current.resolve().is_relative_to(root):
        raise OpsError('path escapes project')
    return current


def file_record(path):
    path = Path(path)
    if path.is_symlink():
        target = os.readlink(path)
        return {'kind': 'symlink', 'size': len(target.encode()), 'target_sha256': digest(target)}
    before = path.stat()
    value = {'kind': 'file', 'size': before.st_size, 'sha256': sha256(path)}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise OpsError('file changed while hashing')
    return value


def tree_records(root, paths, exclusions=()):
    records = {}
    for relative in paths:
        path = safe_path(root, relative)
        if not path.exists():
            raise OpsError('required identity path missing: ' + relative)
        files = sorted(path.rglob('*')) if path.is_dir() else [path]
        for item in files:
            rel = str(item.relative_to(root))
            if '__pycache__' in item.parts or rel in exclusions or item.is_dir():
                continue
            records[rel] = file_record(item)
    return records


def source_identity(root, paths):
    return tree_records(root, paths)


def untracked_records(root):
    # NUL separators retain spaces/newlines in historical filenames.
    names = git(root, 'ls-files', '--others', '--exclude-standard', '-z').stdout.split('\0')
    return {name: file_record(Path(root)/name) for name in names if name}


def capture_baseline(root, config):
    return {'protected_files': tree_records(root, config['protected_paths'], config['protected_exclusions']),
            'historical_untracked': {str(Path(p).resolve()): untracked_records(p)
                                     for p in config['history_worktrees']},
            'historical_scoped_files': {str(Path(p).resolve()): tree_records(Path(p).resolve(), paths)
                for p, paths in config.get('historical_ignored_paths', {}).items()}}


def verify_baseline(root, baseline):
    for relative, expected in baseline['protected_files'].items():
        path = safe_path(root, relative)
        if not path.exists() or file_record(path) != expected:
            raise OpsError('protected historical identity changed: ' + relative)
    additions = {}
    for worktree, previous in baseline['historical_untracked'].items():
        current = untracked_records(worktree)
        for name, expected in previous.items():
            if current.get(name) != expected:
                raise OpsError('historical untracked file changed or moved: ' + name)
        additions[worktree] = sorted(set(current) - set(previous))
    for worktree, previous in baseline.get('historical_scoped_files', {}).items():
        for name, expected in previous.items():
            path = safe_path(worktree, name)
            if not path.exists() or file_record(path) != expected:
                raise OpsError('scoped historical data changed or missing: ' + name)
    return {'unchanged': True, 'protected_count': len(baseline['protected_files']),
            'historical_untracked_counts': {p: len(v) for p, v in baseline['historical_untracked'].items()},
            'new_untracked_names': additions,
            'historical_scoped_counts': {p: len(v) for p, v in baseline.get('historical_scoped_files', {}).items()}}


def verify_original_l9(root, config):
    spec = config.get('original_l9')
    if spec is None:
        return {'status': 'NOT_APPLICABLE'}
    report = read_json(safe_path(root, spec['report']))
    if report['overall_result'] != 'FAIL':
        raise OpsError('original L9 must remain FAIL')
    raw = safe_path(root, spec['raw_directory'])
    actual = {p.name: sha256(p) for p in raw.iterdir() if p.is_file()}
    if actual != report['raw_evidence_sha256']:
        raise OpsError('original L9 raw evidence changed or acquired extra files')
    return {'status': 'PASS', 'original_result': 'FAIL', 'raw_file_count': len(actual)}


def verify_checkpoint(root, config):
    checkpoint = config.get('checkpoint')
    if checkpoint is None:
        return 'NOT_VERIFIED'
    path = safe_path(root, checkpoint['path'])
    if not path.is_file() or sha256(path) != checkpoint['sha256']:
        raise OpsError('checkpoint identity mismatch')
    # Identity only, not a tensor deserialization or resume-equivalence claim.
    return {**checkpoint, 'verification': 'FILE_SHA256_ONLY', 'continuation_authorized': False}
