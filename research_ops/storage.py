"""Strict local state, durable publication and kernel-owned writer exclusion."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import tempfile
from datetime import datetime, timezone


class OpsError(RuntimeError):
    pass


STATUSES = {'PENDING', 'RUNNING', 'STOP_REQUESTED', 'SAFE_STOPPED', 'INTERRUPTED',
            'BLOCKED', 'PASS', 'FAIL', 'SYNC_PENDING'}
TERMINAL = {'PENDING', 'SAFE_STOPPED', 'INTERRUPTED', 'BLOCKED', 'PASS', 'FAIL', 'SYNC_PENDING'}
AUTHORIZATION = {'levels': [0, 1], 'optimizer_updates': False, 'PX4': False,
                 'real_devices': False, 'automatic_training': 'DISABLED'}
ZERO_BUDGET = {'authorized_training_timesteps': 0, 'authorized_training_wall_time': 0,
               'consumed_training_timesteps': 0, 'consumed_training_wall_time': 0,
               'max_workers': 1, 'authorized_training_seeds': []}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or path.stat().st_size > 16 * 1024**2:
        raise OpsError('unsafe or oversized state file')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise OpsError('duplicate JSON key')
            result[key] = value
        return result
    def number(text):
        value = float(text)
        if not math.isfinite(value):
            raise OpsError('nonfinite JSON number')
        return value
    def reject(text):
        raise OpsError('invalid JSON constant')
    try:
        return json.loads(path.read_text(), object_pairs_hook=pairs,
                          parse_float=number, parse_constant=reject)
    except (ValueError, UnicodeError) as error:
        raise OpsError('corrupt JSON: ' + str(path)) from error


def atomic_json(path, value, *, exclusive=False):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    atomic_bytes(path, data, exclusive=exclusive)


def atomic_bytes(path, data, *, exclusive=False):
    path = Path(path)
    if len(data) > 16 * 1024**2 or any(p.is_symlink() for p in (path, *path.parents)):
        raise OpsError('unsafe output file')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    temporary = Path(name)
    linked = False
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path); linked = True
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


class FileLock:
    def __init__(self, path):
        self.path, self.fd = Path(path), None

    def __enter__(self):
        if any(p.is_symlink() for p in (self.path, *self.path.parents)):
            raise OpsError('symlink in lock path')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self.fd); self.fd = None
            raise OpsError('another operations writer holds flock') from error
        return self

    def __exit__(self, *_):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN); os.close(self.fd); self.fd = None


def lock_held(path):
    try:
        with FileLock(path):
            return False
    except OpsError:
        return True


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': int(pid), 'start_ticks': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, ValueError, IndexError):
        return 'NOT_VERIFIED'


def initial_state(project, branch, commit, protocol_sha):
    return {'schema_version': 1, 'project_id': project, 'phase_id': 'OPS-01',
            'experiment_id': 'OPS01_MAINTENANCE', 'run_id': 'NOT_STARTED',
            'current_branch': branch, 'base_commit': commit, 'latest_commit': commit,
            'protocol_sha256': protocol_sha, 'scenario_manifest_sha256': 'NOT_APPLICABLE',
            'current_stage': 'OPERATIONS_PREFLIGHT', 'stage_status': 'PENDING',
            'authorization_scope': copy.deepcopy(AUTHORIZATION),
            'budget_remaining': copy.deepcopy(ZERO_BUDGET), 'training_seed': 'NOT_APPLICABLE',
            'nominal_timesteps': 0, 'last_verified_actual_timesteps': 0,
            'progress_evidence': 'OPS has executed zero training steps; historical experiments are separate.',
            'last_valid_checkpoint': 'NOT_VERIFIED', 'checkpoint_resume_class': 'RESTART_REQUIRED',
            'latest_valid_report': 'NOT_VERIFIED', 'latest_successful_push': 'NOT_VERIFIED',
            'next_action': 'ops-audit', 'last_update_time': now(),
            'last_shutdown_status': 'NOT_VERIFIED', 'owner': 'NOT_VERIFIED',
            'phase_complete': False}


def validate_state(state):
    if not isinstance(state, dict) or state.get('schema_version') != 1:
        raise OpsError('unknown state schema')
    required = set(initial_state('', '', '', ''))
    if not required <= state.keys() or state['stage_status'] not in STATUSES:
        raise OpsError('incomplete state or invalid status')
    if digest(state['authorization_scope']) != digest(AUTHORIZATION) or digest(state['budget_remaining']) != digest(ZERO_BUDGET):
        raise OpsError('OPS V1 cannot install or restore training authority/budget')
    if state['nominal_timesteps'] != 0 or state['last_verified_actual_timesteps'] != 0:
        raise OpsError('maintenance cannot claim training progress')
    if state['phase_id'] != 'OPS-01' or state['checkpoint_resume_class'] != 'RESTART_REQUIRED':
        raise OpsError('unknown phase or unsupported continuation')
    run = state['run_id']
    if run == 'NOT_STARTED':
        valid_experiment = state['experiment_id'] == 'OPS01_MAINTENANCE'
    else:
        valid_experiment = (isinstance(run, str) and re.fullmatch('OPS01_[0-9a-f]{32}', run)
                            and state['experiment_id'] == 'OPS01_AUDIT_'+run[6:])
    if not valid_experiment:
        raise OpsError('unrecognized experiment/run identity')
    if state['next_action'] not in ('ops-audit', 'sync', 'BLOCKED_TRAINING_AUTHORIZATION', 'BLOCKED_REVIEW_REQUIRED'):
        raise OpsError('unregistered next action')
    return state


def transition(state, target, **updates):
    validate_state(state)
    source = state['stage_status']
    allowed = {'RUNNING': {'STOP_REQUESTED', 'PASS', 'FAIL', 'INTERRUPTED', 'BLOCKED'},
               'STOP_REQUESTED': {'SAFE_STOPPED', 'INTERRUPTED', 'FAIL'}}
    if target not in STATUSES or (source in allowed and target not in allowed[source]):
        raise OpsError(f'invalid transition {source} -> {target}')
    if source in TERMINAL and target == 'STOP_REQUESTED':
        raise OpsError('no running task to stop')
    result = copy.deepcopy(state)
    result.update(updates, stage_status=target, last_update_time=now())
    return validate_state(result)
