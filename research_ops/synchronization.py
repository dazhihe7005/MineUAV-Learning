"""Explicit-file Git synchronization with a durable retry receipt."""
from __future__ import annotations

import re
from pathlib import Path

from . import documents
from .identity import git, safe_path
from .storage import OpsError, digest, now, read_json, sha256, transition


def check_public_file(root, relative):
    path = safe_path(root, relative)
    if not path.is_file() or path.stat().st_size > 2*1024**2:
        raise OpsError('public file missing, not regular or >2MiB: '+relative)
    if path.suffix.lower() in ('.pt', '.pth', '.zip', '.npz', '.bag', '.npy', '.log'):
        raise OpsError('raw/model artifacts cannot be synchronized')
    if path.name == '.env' or path.name.endswith(('.pem', '.key')):
        raise OpsError('credential file cannot be synchronized')
    check_public_text(relative, path.read_text())


def check_public_text(relative, text):
    if Path(relative).suffix.lower() in ('.pt', '.pth', '.zip', '.npz', '.bag', '.npy', '.log') or len(text.encode()) > 2*1024**2:
        raise OpsError('raw/oversized historical blob cannot be published')
    signatures = (r'gh[pousr]_[A-Za-z0-9]{25,}', r'github_pat_[A-Za-z0-9_]{30,}',
                  r'sk-[A-Za-z0-9_-]{25,}', r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                  r'https?://[^\s/@]+:[^\s/@]+@',
                  r'(?i)(?:api_key|access_token|password)\s*[:=]\s*[\"\'][A-Za-z0-9/+_=.-]{16,}')
    if any(re.search(pattern, text) for pattern in signatures):
        raise OpsError('possible credential in public file; refusing automatic commit')


def audit_outgoing_history(manager, state):
    base = state['base_commit']
    if git(manager.root, 'merge-base', '--is-ancestor', base, 'HEAD', check=False).returncode:
        raise OpsError('delivery HEAD is not descended from trusted base')
    allowed = set(manager.config['sync_files'])
    for commit in git(manager.root, 'rev-list', '--reverse', base+'..HEAD').stdout.split():
        if len(git(manager.root, 'rev-list', '--parents', '-n', '1', commit).stdout.split()) != 2:
            raise OpsError('unreviewed merge history cannot be auto-published')
        paths = git(manager.root, 'diff-tree', '--no-commit-id', '--name-only', '-r', '-z', commit).stdout.split('\0')
        for name in filter(None, paths):
            if name not in allowed:
                raise OpsError('outgoing commit contains a path outside sync allowlist')
            blob = git(manager.root, 'show', commit+':'+name, check=False)
            if blob.returncode == 0:
                check_public_text(name, blob.stdout)


def _ready(manager):
    config = manager.config
    path = config.get('readiness_report')
    if not path:
        return False
    for relative in config['required_delivery_paths']:
        if not safe_path(manager.root, relative).is_file():
            raise OpsError('required delivery file missing')
    report = read_json(safe_path(manager.root, path))
    if (report.get('implementation_status') != 'PASS' or report.get('tests', {}).get('status') != 'PASS'
            or report.get('review', {}).get('status') != 'GO'):
        raise OpsError('stage implementation/tests/review not ready for completion')
    state = manager._read()
    protocol = read_json(manager.local/'protocol.json')
    source_sha = digest(protocol['source_identity'])
    if (report.get('project_id') != state['project_id'] or report.get('phase_id') != 'OPS-01'
            or report.get('protocol_sha256') != state['protocol_sha256']
            or report.get('runtime_identity_sha256') != source_sha
            or report['review'].get('runtime_identity_sha256') != source_sha
            or report['tests'].get('runtime_identity_sha256') != source_sha):
        raise OpsError('readiness report does not cover this runtime/protocol')
    evidence = report['tests'].get('evidence', {})
    p = safe_path(manager.root, evidence.get('path', ''))
    if not p.is_file() or sha256(p) != evidence.get('sha256'):
        raise OpsError('readiness test evidence missing or changed')
    tests = read_json(p)
    if (tests.get('complete') is not True or tests.get('failures') != 0 or tests.get('errors') != 0
            or tests.get('optimizer_updates') != 0 or tests.get('runtime_identity_sha256') != source_sha
            or tests.get('test_source_identity') != report['tests'].get('test_source_identity')
            or not tests.get('results') or any(r.get('outcome') != 'PASS' for r in tests['results'])):
        raise OpsError('readiness test results are incomplete or do not cover current source')
    from .identity import source_identity
    if source_identity(manager.root, ['tests/research_ops']) != tests['test_source_identity']:
        raise OpsError('tests changed after readiness verification')
    if state.get('pending_sync') and git(manager.root, 'show', state['pending_sync']['commit']+':'+path).stdout != p_report_text(manager.root, path):
        raise OpsError('readiness report differs from pending commit')
    return True


def p_report_text(root, path):
    return safe_path(root, path).read_text()


def synchronize(manager):
    state = manager._read()
    manager.verify()
    branch = manager.branch()
    if state['stage_status'] in ('RUNNING', 'STOP_REQUESTED'):
        raise OpsError('recover/stop before synchronization')
    if git(manager.root, 'diff', '--cached', '--name-only').stdout.strip():
        raise OpsError('existing staged files belong to user; refusing automatic commit')
    audit_outgoing_history(manager, state)
    pending = state.get('pending_sync')
    ready = _ready(manager)
    if pending:
        if pending['commit'] != manager.head() or pending['branch'] != branch:
            raise OpsError('pending delivery HEAD changed; explicit audit required')
        if git(manager.root, 'diff', '--name-only', 'HEAD', '--', *manager.config['sync_files']).stdout.strip():
            raise OpsError('pending delivery documents/source changed; preserve and audit before retry')
        target = pending['commit']
        prior_status = pending['prior_status']
    else:
        prior_status = state['stage_status']
        state['latest_commit'] = manager.head(); state['last_update_time'] = now()
        documents.refresh(manager.root, state)
        allowed = manager.config['sync_files']
        if len(set(allowed)) != len(allowed):
            raise OpsError('duplicate sync path')
        protected = read_json(manager.local/'history_baseline.json')['protected_files']
        paths = []
        for relative in allowed:
            path = safe_path(manager.root, relative)
            if relative in protected:
                raise OpsError('protected history cannot be committed by operations')
            if path.exists():
                check_public_file(manager.root, relative)
                paths.append(relative)
        # Only changed allowlisted paths are staged. No add-A, no user index manipulation.
        if paths:
            git(manager.root, 'add', '--', *paths)
        staged = set(git(manager.root, 'diff', '--cached', '--name-only', '-z').stdout.strip('\0').split('\0')) - {''}
        if not staged <= set(paths):
            raise OpsError('unexpected index content; leaving it untouched for inspection')
        if staged:
            git(manager.root, 'diff', '--cached', '--check')
            git(manager.root, 'commit', '-m', 'Build research operations and persistence infrastructure')
        target = manager.head()
        audit_outgoing_history(manager, state)
        state = transition(state, 'SYNC_PENDING', next_action='sync', latest_commit=target, phase_complete=False)
        state['pending_sync'] = {'commit': target, 'branch': branch, 'prior_status': prior_status,
                                 'created_at': now()}
        manager._write(state)
    try:
        remote = manager.config['remote']
        url = git(manager.root, 'remote', 'get-url', remote).stdout.strip()
        if manager.config.get('expected_remote_url') and url != manager.config['expected_remote_url']:
            raise OpsError('remote repository identity changed')
        push_urls = git(manager.root, 'remote', 'get-url', '--push', '--all', remote).stdout.splitlines()
        if push_urls != [url]:
            raise OpsError('effective push destination differs from verified fetch destination')
        if re.search(r'https?://[^\s/]*@', url):
            raise OpsError('credential-bearing remote URL is not allowed')
        pushed = git(manager.root, 'push', '-u', remote, f'HEAD:refs/heads/{branch}', check=False, timeout=60)
        if pushed.returncode:
            raise OpsError('push failed; local commit retained')
        received = git(manager.root, 'ls-remote', '--heads', remote, 'refs/heads/'+branch, timeout=45).stdout.split()
        if not received or received[0] != target:
            raise OpsError('remote SHA not verified')
        manager.verify()
    except Exception as error:
        state['sync_error_type'] = type(error).__name__
        state['last_update_time'] = now(); manager._write(state)
        return {'status': 'SYNC_PENDING', 'commit': target,
                'reason': 'remote synchronization not verified; no further task starts',
                'state': manager.status()['state']}
    state.pop('pending_sync', None); state.pop('sync_error_type', None)
    status = prior_status if prior_status in ('SAFE_STOPPED', 'INTERRUPTED', 'FAIL', 'BLOCKED') else 'PASS'
    state = transition(state, status,
                       next_action='BLOCKED_REVIEW_REQUIRED' if status in ('INTERRUPTED','FAIL','BLOCKED') else 'ops-audit',
                       latest_commit=target, phase_complete=ready and status not in ('INTERRUPTED','FAIL','BLOCKED'))
    state['latest_successful_push'] = {'commit': target, 'branch': branch, 'remote': remote, 'verified_at': now()}
    manager._write(state)
    return {'status': 'PASS', 'commit': target, 'remote_sha_verified': True, 'state': state}
