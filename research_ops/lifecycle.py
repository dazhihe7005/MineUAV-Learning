"""Foreground maintenance lifecycle. There is deliberately no training executor."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time
import uuid

from . import documents
from .environment import enforce_resources, inspect_environment, restore_upstream
from .identity import (capture_baseline, git, safe_path, source_identity, verify_baseline,
                       verify_checkpoint, verify_original_l9)
from .storage import (AUTHORIZATION, ZERO_BUDGET, FileLock, OpsError, atomic_json, digest,
                      initial_state, lock_held, now, process_identity, read_json,
                      sha256, transition, validate_state)


class Operations:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = read_json(self.root/'research_ops/operations_config.json')
        self.environment_lock = read_json(self.root/'research_ops/environment_lock.json')
        self.local = safe_path(self.root, '.research_local')
        common = Path(git(self.root, 'rev-parse', '--git-common-dir').stdout.strip())
        self.common = (self.root/common).resolve()
        self.lock_path = self.common/'research-operations-v1.lock'
        self.state_path = self.local/'research_state.json'

    def branch(self):
        branch = git(self.root, 'branch', '--show-current').stdout.strip()
        if not branch or branch in ('main', 'master'):
            raise OpsError('operations requires an independent named feature branch')
        return branch

    def head(self):
        return git(self.root, 'rev-parse', 'HEAD').stdout.strip()

    def _write(self, state):
        validate_state(state)
        atomic_json(self.state_path, {'state': state, 'state_sha256': digest(state)})
        return state

    def _read(self):
        envelope = read_json(self.state_path)
        state = envelope['state']
        if envelope['state_sha256'] != digest(state):
            raise OpsError('state checksum mismatch; recover cannot invent missing state')
        validate_state(state)
        return state

    def initialize(self):
        with FileLock(self.lock_path):
            self.branch()
            enforce_resources(self.root)
            if self.local.exists() and any((self.local/name).exists() for name in
                                           ('research_state.json', 'initialization.attempt.json')):
                raise OpsError('initialization already attempted; do not overwrite history')
            self.local.mkdir(parents=True, exist_ok=True)
            atomic_json(self.local/'initialization.attempt.json', {'time': now(), 'owner': process_identity()}, exclusive=True)
            baseline = capture_baseline(self.root, self.config)
            atomic_json(self.local/'history_baseline.json', baseline, exclusive=True)
            protocol = {'version': 'OPS_PROTOCOL_V1', 'phase_id': 'OPS-01',
                        'source_identity': source_identity(self.root, self.config['runtime_paths']),
                        'configuration_sha256': sha256(self.root/'research_ops/operations_config.json'),
                        'environment_lock_sha256': sha256(self.root/'research_ops/environment_lock.json'),
                        'baseline_sha256': sha256(self.local/'history_baseline.json'),
                        'authorization_scope': AUTHORIZATION, 'budget': ZERO_BUDGET,
                        'registered_tasks': {'ops-audit': {'level': 1, 'max_wall_seconds': 30}},
                        'training_executor': 'UNSUPPORTED', 'continuation': 'RESTART_REQUIRED'}
            atomic_json(self.local/'protocol.json', protocol, exclusive=True)
            atomic_json(self.local/'budget_ledger.json', {'scope': 'OPS_ONLY_NO_TRAINING',
                        'authorization': ZERO_BUDGET, 'training_experiments': []}, exclusive=True)
            base = self.config.get('trusted_base_commit', self.head())
            if git(self.root, 'merge-base', '--is-ancestor', base, 'HEAD', check=False).returncode:
                raise OpsError('trusted repository baseline is not an ancestor')
            state = initial_state(self.config['project_id'], self.branch(), base, digest(protocol))
            state['latest_commit'] = self.head()
            state['scenario_manifest_sha256'] = self.config['scenario_manifest_sha256']
            state['last_valid_checkpoint'] = verify_checkpoint(self.root, self.config)
            self._write(state)
            atomic_json(self.local/'initialization.complete.json',
                        {'protocol_sha256': digest(protocol), 'base_commit': self.head()}, exclusive=True)
            documents.refresh(self.root, state)
            return self.status()

    def status(self):
        state = copy.deepcopy(self._read())
        state['latest_commit'] = self.head()
        held = lock_held(self.lock_path)
        request = self.local/'stop_requests'/(state['run_id']+'.json')
        if held and state['stage_status'] == 'RUNNING' and request.exists():
            value = read_json(request)
            if value.get('run_id') == state['run_id']:
                state['stage_status'] = 'STOP_REQUESTED'
        owner_alive = (isinstance(state['owner'], dict)
                       and process_identity(state['owner']['pid']) == state['owner'])
        return {'state': state, 'lock_held': held, 'owner_process_identity_matches': owner_alive,
                'next_action': state['next_action'], 'AUTO_TRAINING_AUTHORIZED': 'NO',
                'TRAINING_BUDGET_AVAILABLE': 0}

    def verify(self, *, check_environment=True):
        state = self._read()
        if self.branch() != state['current_branch'] or self.config['project_id'] != state['project_id']:
            raise OpsError('worktree branch/project identity differs from initialization')
        protocol = read_json(self.local/'protocol.json')
        complete = read_json(self.local/'initialization.complete.json')
        if digest(protocol) != state['protocol_sha256'] or complete['protocol_sha256'] != state['protocol_sha256']:
            raise OpsError('execution protocol identity mismatch')
        if (sha256(self.root/'research_ops/operations_config.json') != protocol['configuration_sha256']
                or sha256(self.root/'research_ops/environment_lock.json') != protocol['environment_lock_sha256']
                or source_identity(self.root, self.config['runtime_paths']) != protocol['source_identity']):
            raise OpsError('runtime/configuration identity changed')
        ledger = read_json(self.local/'budget_ledger.json')
        if digest(ledger) != digest({'scope': 'OPS_ONLY_NO_TRAINING', 'authorization': ZERO_BUDGET, 'training_experiments': []}):
            raise OpsError('training budget ledger mismatch; no automatic grants')
        baseline_path = self.local/'history_baseline.json'
        if sha256(baseline_path) != protocol['baseline_sha256']:
            raise OpsError('historical baseline integrity mismatch')
        history = verify_baseline(self.root, read_json(baseline_path))
        original = verify_original_l9(self.root, self.config)
        checkpoint = verify_checkpoint(self.root, self.config)
        resources = enforce_resources(self.root)
        environment = inspect_environment(self.root, self.environment_lock) if check_environment else 'NOT TESTED'
        if state['run_id'] != 'NOT_STARTED':
            attempt = read_json(self.local/'runs'/state['run_id']/'attempt.json')
            if (attempt.get('run_id') != state['run_id'] or attempt.get('protocol_sha256') != state['protocol_sha256']
                    or attempt.get('experiment_id') != state['experiment_id']):
                raise OpsError('attempt identity mismatch')
        runs = safe_path(self.root, '.research_local/runs')
        if runs.exists():
            for run in runs.iterdir():
                if not run.is_dir() or run.is_symlink():
                    raise OpsError('unexpected operations run path')
                attempt = read_json(run/'attempt.json')
                if (attempt.get('run_id') != run.name or attempt.get('protocol_sha256') != state['protocol_sha256']
                        or not str(attempt.get('experiment_id', '')).startswith('OPS01_AUDIT_')):
                    raise OpsError('orphan or incompatible attempt identity')
                markers = [run/name for name in ('complete.json', 'stopped.json') if (run/name).exists()]
                if len(markers) > 1:
                    raise OpsError('contradictory run completion')
                if markers:
                    marker = read_json(markers[0])
                    if (marker.get('run_id') != run.name or marker.get('audit_sha256') != sha256(run/'audit.json')
                            or read_json(run/'audit.json').get('status') != 'PASS'):
                        raise OpsError('completed audit integrity mismatch')
                elif run.name != state['run_id'] or state['stage_status'] not in ('RUNNING', 'STOP_REQUESTED', 'INTERRUPTED', 'FAIL'):
                    raise OpsError('incomplete previous attempt requires recovery')
        public = safe_path(self.root, 'docs/RESEARCH_STATUS.json')
        if public.exists():
            snapshot = read_json(public)
            if (snapshot.get('project_id') != state['project_id'] or snapshot.get('protocol_sha256') != state['protocol_sha256']
                    or digest(snapshot.get('authorization_scope')) != digest(AUTHORIZATION)
                    or digest(snapshot.get('budget_remaining')) != digest(ZERO_BUDGET)):
                raise OpsError('public snapshot identity/authority inconsistent; cannot restore it')
            snapshot_commit = snapshot.get('latest_commit', '')
            if not isinstance(snapshot_commit, str) or len(snapshot_commit) != 40 or git(self.root, 'merge-base', '--is-ancestor', snapshot_commit, 'HEAD', check=False).returncode:
                raise OpsError('public snapshot references unrelated Git history')
        return {'status': 'PASS', 'history': history, 'original_l9': original,
                'checkpoint': checkpoint, 'resources': resources, 'environment': environment,
                'protocol_sha256': state['protocol_sha256']}

    def doctor(self, *, restore_dependencies=False, source=None):
        with FileLock(self.lock_path):
            if restore_dependencies:
                self.verify(check_environment=False)
                restore_upstream(self.root, self.environment_lock, source)
            return self.verify()

    def start(self, task='ops-audit', *, max_seconds=0):
        # Reject unsupported execution before creating a run or importing any ML module.
        if task != 'ops-audit':
            raise OpsError('UNAUTHORIZED/UNSUPPORTED: only L1 ops-audit; training budget is zero')
        if type(max_seconds) not in (float, int) or not math.isfinite(max_seconds) or not 0 <= max_seconds <= 30:
            raise OpsError('bounded maintenance duration must be0..30 seconds')
        with FileLock(self.lock_path):
            state = self._read()
            if state['stage_status'] in ('RUNNING', 'STOP_REQUESTED', 'INTERRUPTED', 'FAIL', 'SYNC_PENDING'):
                raise OpsError('recover or synchronize previous task before start')
            if state['next_action'] != 'ops-audit':
                raise OpsError('the next task is not authorized maintenance')
            self.verify()
            run_id = 'OPS01_'+uuid.uuid4().hex
            run = self.local/'runs'/run_id
            run.mkdir(parents=True, exist_ok=False)
            state = transition(state, 'RUNNING', run_id=run_id,
                               experiment_id='OPS01_AUDIT_'+run_id[6:],
                               owner=process_identity(), last_shutdown_status='RUNNING', phase_complete=False)
            atomic_json(run/'attempt.json', {'run_id': run_id, 'experiment_id': state['experiment_id'],
                        'protocol_sha256': state['protocol_sha256'], 'owner': state['owner'],
                        'started_at': now(), 'max_wall_seconds': max_seconds}, exclusive=True)
            self._write(state)
            try:
                deadline = time.monotonic() + max_seconds
                stop = self.local/'stop_requests'/(run_id+'.json')
                stopped = False
                while True:
                    enforce_resources(self.root)
                    if stop.exists():
                        request = read_json(stop)
                        if request.get('run_id') != run_id:
                            raise OpsError('stop request identity mismatch')
                        state = transition(state, 'STOP_REQUESTED'); self._write(state)
                        stopped = True; break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(min(.05, max(0., deadline-time.monotonic())))
                evidence = self.verify()
                atomic_json(run/'audit.json', evidence, exclusive=True)
                target = 'SAFE_STOPPED' if stopped else 'PASS'
                state = transition(state, target, latest_valid_report=str((run/'audit.json').relative_to(self.root)),
                                   last_shutdown_status='SAFE_STOPPED' if stopped else 'TASK_COMPLETED',
                                   latest_commit=self.head())
                # Run completion proves the audit/stop, not whole-phase Git delivery.
                atomic_json(run/('stopped.json' if stopped else 'complete.json'),
                            {'run_id': run_id, 'status': target, 'audit_sha256': sha256(run/'audit.json'),
                             'training_timesteps': 0, 'optimizer_updates': 0}, exclusive=True)
                self._write(state); documents.refresh(self.root, state)
                return self.status()
            except BaseException as error:
                state = transition(state, 'FAIL', last_shutdown_status='FAILED', next_action='BLOCKED_REVIEW_REQUIRED')
                state['failure_type'] = type(error).__name__
                try:
                    atomic_json(run/'failure.json', {'run_id': run_id, 'observed_at': now(),
                                'exception_type': type(error).__name__, 'training_steps': 0,
                                'message': str(error)[:1000] if isinstance(error, OpsError) else 'NOT_AVAILABLE'}, exclusive=True)
                except Exception:
                    state['failure_capture'] = 'NOT_AVAILABLE'
                self._write(state)
                raise

    def request_stop(self, *, sync=True, wait_seconds=10):
        state = self._read()
        if lock_held(self.lock_path):
            if state['stage_status'] not in ('RUNNING', 'STOP_REQUESTED'):
                raise OpsError('another writer is busy; no matching running task')
            path = self.local/'stop_requests'/(state['run_id']+'.json')
            try:
                atomic_json(path, {'run_id': state['run_id'], 'requested_at': now()}, exclusive=True)
            except FileExistsError:
                if read_json(path).get('run_id') != state['run_id']:
                    raise OpsError('existing stop request is inconsistent')
            deadline = time.monotonic() + wait_seconds
            while lock_held(self.lock_path) and time.monotonic() < deadline:
                time.sleep(.05)
            if lock_held(self.lock_path):
                return {'status': 'STOP_REQUESTED', 'state': self._read(), 'sync': 'NOT TESTED: writer not stopped yet'}
        else:
            with FileLock(self.lock_path):
                state = self._read()
                if state['stage_status'] in ('RUNNING', 'STOP_REQUESTED'):
                    raise OpsError('unclean process exit; run recover before claiming safe shutdown')
                self.verify()
                if state['stage_status'] not in ('INTERRUPTED', 'FAIL'):
                    state = transition(state, 'SAFE_STOPPED', last_shutdown_status='SAFE_STOPPED', latest_commit=self.head())
                self._write(state); documents.refresh(self.root, state)
        return self.sync() if sync else self.status()

    def recover(self):
        if lock_held(self.lock_path):
            return {'status': 'BLOCKED', 'reason': 'live writer holds flock; no takeover', **self.status()}
        with FileLock(self.lock_path):
            state = self._read()
            audit = self.verify()
            if state['stage_status'] in ('RUNNING', 'STOP_REQUESTED'):
                # A matching live process with no flock is ambiguous, never authorize a replacement.
                owner = state['owner']
                if isinstance(owner, dict) and process_identity(owner['pid']) == owner:
                    raise OpsError('owner process alive without lock; ambiguous state')
                state = transition(state, 'INTERRUPTED', last_shutdown_status='UNCLEAN_INTERRUPTION',
                                   next_action='BLOCKED_REVIEW_REQUIRED', phase_complete=False)
                self._write(state)
            else:
                state['latest_commit'] = self.head(); state['last_update_time'] = now(); self._write(state)
            if state['stage_status'] != 'SYNC_PENDING':
                documents.refresh(self.root, state)
            return {'audit': audit, **self.status()}

    def sync(self):
        from .synchronization import synchronize
        with FileLock(self.lock_path):
            return synchronize(self)
