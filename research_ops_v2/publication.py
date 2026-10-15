"""Version-bound explicit-file publication; no V1 allowlist bypass."""
from __future__ import annotations
import re
import struct
import zlib
from pathlib import Path
from research_ops.identity import git,safe_path
from research_ops.storage import OpsError,atomic_json,now,read_json,sha256,digest
from research_ops.synchronization import check_public_text
from .registration import delivery_path,verify_review


def check_blob(relative,data):
    if len(data)>2*1024**2:raise OpsError('public artifact exceeds 2MiB')
    if Path(relative).suffix=='.png':
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):raise OpsError('invalid PNG signature')
        offset=8;types=[]
        while offset<len(data):
            if offset+12>len(data):raise OpsError('truncated PNG')
            size=struct.unpack('>I',data[offset:offset+4])[0];kind=data[offset+4:offset+8]
            end=offset+12+size
            if end>len(data) or kind not in (b'IHDR',b'PLTE',b'tRNS',b'IDAT',b'IEND'):
                raise OpsError('PNG contains unsupported metadata or payload')
            if zlib.crc32(data[offset+4:end-4])&0xffffffff!=struct.unpack('>I',data[end-4:end])[0]:
                raise OpsError('PNG checksum invalid')
            if kind==b'IHDR':
                if size!=13 or types:raise OpsError('invalid PNG header')
                w,h=struct.unpack('>II',data[offset+8:offset+16])
                if not 0<w<=4096 or not 0<h<=4096:raise OpsError('PNG dimensions too large')
            types.append(kind);offset=end
            if kind==b'IEND':
                if size or end!=len(data):raise OpsError('PNG has trailing data')
        if not types or types[0]!=b'IHDR' or types[-1]!=b'IEND' or b'IDAT' not in types:raise OpsError('incomplete PNG')
    else:
        try:check_public_text(relative,data.decode('utf8'))
        except UnicodeError as error:raise OpsError('non-text public source/report') from error


def refresh(manager):
    _,protocol,state,_=manager._load()
    value={'operations_version':'RESEARCH_OPS_V2','phase_id':state['phase_id'],
           'protocol_sha256':state['protocol_sha256'],'registration_sha256':digest(protocol['registration']),
           'stage_status':state['stage_status'],'head_at_snapshot':manager.legacy.head(),
           'training_budget':0,'automatic_training':'DISABLED','resume':'RESTART_REQUIRED',
           'authority':'INFORMATION_ONLY; not a grant or restorable execution state'}
    if 'docs/RESEARCH_STATUS_V2.json' not in protocol['registration']['sync_files']:
        raise OpsError('public snapshot not registered')
    atomic_json(manager.root/'docs/RESEARCH_STATUS_V2.json',value)


def audit_history(manager,registration):
    base=manager._load()[1]['publication_anchor']['commit'];allowed=set(registration['sync_files'])
    for commit in git(manager.root,'rev-list','--reverse',base+'..HEAD').stdout.split():
        if len(git(manager.root,'rev-list','--parents','-n','1',commit).stdout.split())!=2:
            raise OpsError('unreviewed merge commits cannot be published')
        names=git(manager.root,'diff-tree','--no-commit-id','--name-only','-r','-z',commit).stdout.split('\0')
        for name in filter(None,names):
            if name not in allowed:raise OpsError('outgoing commit includes unregistered path')
            # Git helper is text-only: inspect binary blobs separately and bounded.
            import subprocess
            size=git(manager.root,'cat-file','-s',commit+':'+name,check=False)
            if size.returncode:raise OpsError('deletion of delivery/history is not authorized')
            if int(size.stdout)>2*1024**2:raise OpsError('oversized outgoing blob')
            blob=subprocess.run(['git','-C',str(manager.root),'show',commit+':'+name],capture_output=True,check=True,timeout=15).stdout
            check_blob(name,blob)


def readiness(manager,registration):
    path=registration['report_path']
    if path not in registration['sync_files']:raise OpsError('readiness report not registered')
    report=read_json(manager.root/path)
    if report.get('registration_sha256')!=digest(registration) or report.get('source_identity_sha256')!=digest(registration['source_identity']):
        raise OpsError('readiness report not bound to this reviewed registration')
    tests=report.get('tests',{});review=report.get('review',{})
    if (tests.get('status')!='PASS' or review.get('status') not in ('GO','NO_GO')
            or review.get('source_identity_sha256')!=digest(registration['source_identity'])):
        raise OpsError('test/review evidence incomplete or unbound')
    for evidence in (tests.get('evidence',{}),review.get('evidence',{})):
        p=safe_path(manager.root,evidence.get('path',''))
        if not p.is_file() or sha256(p)!=evidence.get('sha256'):raise OpsError('readiness evidence missing or changed')
    actual=read_json(safe_path(manager.root,tests['evidence']['path']))
    from research_ops.identity import source_identity
    if (not actual.get('complete') or actual.get('source_identity_sha256')!=digest(registration['source_identity'])
            or actual.get('optimizer_updates')!=0 or actual.get('failures')!=0 or actual.get('errors')!=0
            or not actual.get('tests_run',0) or actual.get('test_source_identity')!=source_identity(manager.root,['tests/research_ops_v2'])):
        raise OpsError('test evidence not complete/current/no-update')
    actual_review=read_json(safe_path(manager.root,review['evidence']['path']))
    if actual_review.get('verdict')=='GO':verify_review(registration,actual_review)
    if actual_review.get('source_identity_sha256')!=review['source_identity_sha256'] or actual_review.get('verdict')!=review['status']:
        raise OpsError('review result differs from report')
    return report.get('implementation_status')=='PASS' and review['status']=='GO'


def synchronize(manager):
    manager.verify()
    _,protocol,state,_=manager._load();registration=protocol['registration']
    if state['stage_status'] in ('RUNNING','STOP_REQUESTED'):raise OpsError('stop/recover before sync')
    if git(manager.root,'diff','--cached','--name-only').stdout.strip():raise OpsError('user staged files: refusing publication')
    ready=readiness(manager,registration)
    allowed=registration['sync_files'];baseline=read_json(manager.legacy.local/'history_baseline.json')['protected_files']
    for relative in allowed:
        delivery_path(manager.root,relative)
        if relative in baseline and relative not in registration['approved_changes']:
            raise OpsError('historical file not eligible for V2 publication')
        p=safe_path(manager.root,relative)
        if not p.is_file():raise OpsError('registered delivery file missing: '+relative)
        check_blob(relative,p.read_bytes())
    audit_history(manager,registration)
    pending=state.get('pending_sync')
    if pending:
        if pending['commit']!=manager.legacy.head() or pending['branch']!=manager.legacy.branch():raise OpsError('pending HEAD/branch changed')
        if git(manager.root,'diff','--name-only','HEAD','--',*allowed).stdout.strip():raise OpsError('pending files changed; audit required')
        target=pending['commit'];prior=pending['prior_status']
    else:
        prior=state['stage_status']
        git(manager.root,'add','--',*allowed)
        staged=set(git(manager.root,'diff','--cached','--name-only','-z').stdout.strip('\0').split('\0'))-{''}
        if not staged<=set(allowed):raise OpsError('unexpected index content')
        if staged:
            git(manager.root,'diff','--cached','--check')
            git(manager.root,'commit','-m','Make research operations portable and versioned')
        target=manager.legacy.head();audit_history(manager,registration)
        state.update(stage_status='SYNC_PENDING',phase_complete=False,latest_commit=target,
                     pending_sync={'commit':target,'branch':manager.legacy.branch(),'prior_status':prior,'time':now()})
        manager._save(state)
    try:
        remote=manager.legacy.config['remote'];branch=manager.legacy.branch()
        url=git(manager.root,'remote','get-url',remote).stdout.strip()
        expected=manager.legacy.config.get('expected_remote_url')
        if expected and url!=expected:raise OpsError('remote identity changed')
        if git(manager.root,'remote','get-url','--push','--all',remote).stdout.splitlines()!=[url] or re.search(r'https?://[^\s/]*@',url):
            raise OpsError('unsafe effective push URL')
        result=git(manager.root,'push','-u',remote,'HEAD:refs/heads/'+branch,check=False,timeout=60)
        if result.returncode:raise OpsError('push failed')
        observed=git(manager.root,'ls-remote','--heads',remote,'refs/heads/'+branch,timeout=45).stdout.split()
        if not observed or observed[0]!=target:raise OpsError('remote SHA unverified')
        manager.verify()
    except Exception as error:
        state['sync_error_type']=type(error).__name__;manager._save(state)
        return {'status':'SYNC_PENDING','commit':target,'remote_sha_verified':False}
    state.pop('pending_sync',None);state.pop('sync_error_type',None)
    state.update(stage_status=prior if prior in ('FAIL','INTERRUPTED','BLOCKED','SAFE_STOPPED') else 'PASS',
        phase_complete=ready and prior not in ('FAIL','INTERRUPTED','BLOCKED'),latest_commit=target,
        latest_successful_push={'commit':target,'branch':branch,'verified_at':now()})
    manager._save(state)
    return {'status':'PASS','commit':target,'remote_sha_verified':True,'phase_complete':state['phase_complete']}
