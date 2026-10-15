"""Finite foreground tasks. No command, script, checkpoint or optimizer capability."""
from __future__ import annotations

import subprocess
import os
import signal
import time
import uuid

from research_ops.environment import enforce_resources, runtime_environment
from research_ops.identity import safe_path
from research_ops.storage import OpsError, atomic_json, now, process_identity, read_json, sha256


def group_alive(pgid):
    # Linux /proc excludes dead/zombie children; a live descendant is a blocker.
    from pathlib import Path
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=p.read_text().rsplit(') ',1)[1].split()
            if int(fields[2])==pgid and fields[0]!='Z':return True
        except (OSError,ValueError,IndexError):continue
    return False


def process_live(identity):
    from pathlib import Path
    if not isinstance(identity,dict) or process_identity(identity['pid'])!=identity:return False
    try:return Path('/proc/'+str(identity['pid'])+'/stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
    except OSError:return False


def stop_group(child):
    for sig,grace in ((signal.SIGTERM,1.),(signal.SIGKILL,2.)):
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:pass
        deadline=time.monotonic()+grace
        while time.monotonic()<deadline:
            child.poll()
            if not group_alive(child.pid):break
            time.sleep(.025)
        if not group_alive(child.pid):break
    child.wait(timeout=3)
    if group_alive(child.pid):raise OpsError('task process group still alive; manual audit required')


def verify_runs(manager, state, registration):
    _, _, _, version = manager._load()
    runs = version / 'runs'
    if not runs.exists():
        if state['run_id'] != 'NOT_STARTED': raise OpsError('current attempt missing')
        return
    registered = {t['task_id']: t for t in registration['tasks']}
    for run in runs.iterdir():
        if not run.is_dir() or run.is_symlink(): raise OpsError('invalid run path')
        attempt = read_json(run/'attempt.json')
        if (attempt.get('run_id') != run.name or attempt.get('protocol_sha256') != state['protocol_sha256']
                or attempt.get('task') != registered.get(attempt.get('task_id'))):
            raise OpsError('attempt identity mismatch')
        markers = [run/n for n in ('complete.json', 'stopped.json', 'failure.json') if (run/n).exists()]
        if len(markers) > 1: raise OpsError('contradictory run markers')
        if markers:
            marker = read_json(markers[0])
            if marker.get('run_id') != run.name: raise OpsError('result belongs to another run')
            if markers[0].name != 'failure.json' and marker.get('result_sha256') != sha256(run/'result.json'):
                raise OpsError('completed result changed')
            if markers[0].name=='complete.json' and read_json(run/'result.json').get('task_evidence_sha256')!=sha256(run/'task-evidence.json'):
                raise OpsError('task evidence changed')
        elif run.name != state['run_id'] or state['stage_status'] not in ('RUNNING','STOP_REQUESTED','INTERRUPTED','FAIL'):
            raise OpsError('incomplete old attempt requires audit')
    if state['run_id'] != 'NOT_STARTED' and not (runs/state['run_id']/'attempt.json').is_file():
        raise OpsError('current attempt missing')


def start(manager, task_id):
    _, protocol, state, version = manager._load()
    tasks = {t['task_id']: t for t in protocol['registration']['tasks']}
    if task_id not in tasks: raise OpsError('UNAUTHORIZED task; training budget remains zero')
    if state['stage_status'] not in ('PENDING','PASS','SAFE_STOPPED'):
        raise OpsError('previous interrupted/failed/pending-sync task requires audit')
    manager.verify()
    task = tasks[task_id]
    deadline=time.monotonic()+task['wall_time_s']
    run_id = 'OPS2_'+uuid.uuid4().hex
    run = version/'runs'/run_id
    run.mkdir(parents=True, exist_ok=False)
    state.update(stage_status='RUNNING', run_id=run_id, task_id=task_id,
                 owner=process_identity(), phase_complete=False, last_shutdown_status='RUNNING')
    atomic_json(run/'attempt.json', {'run_id':run_id,'task_id':task_id,'task':task,
        'protocol_sha256':state['protocol_sha256'],'owner':state['owner'],'started_at':now(),
        'deadline_monotonic':deadline}, exclusive=True)
    manager._save(state)
    child = None
    try:
        # The only child entry is a reviewed, fixed capability dispatcher.
        env = runtime_environment(manager.root, manager.legacy.environment_lock)
        atomic_json(run/'launch-intent.json',{'run_id':run_id,'owner':state['owner']},exclusive=True)
        with (run/'stdout.txt').open('xb') as stdout, (run/'stderr.txt').open('xb') as stderr:
            child = subprocess.Popen([manager.legacy.environment_lock['interpreter'], '-m',
                'research_ops_v2.worker', '--run', str(run.relative_to(manager.root))],
                cwd=manager.root,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            # Worker records itself before execution. Release only after that
            # identity has been durably observed by the supervisor.
            while not (run/'child.json').exists():
                if child.poll() is not None or time.monotonic()>=deadline:raise OpsError('worker handshake failed')
                time.sleep(.005)
            record=read_json(run/'child.json')
            if record.get('identity')!=process_identity(child.pid) or record.get('pgid')!=child.pid:
                raise OpsError('worker handshake identity differs')
            atomic_json(run/'launch-permit.json',{'run_id':run_id,'child':record['identity']},exclusive=True)
            stopped = False
            while child.poll() is None:
                enforce_resources(manager.root)
                stop = run/'stop-request.json'
                if stop.exists():
                    if read_json(stop).get('run_id') != run_id: raise OpsError('wrong stop identity')
                    stopped = True
                    stop_group(child)
                    break
                if time.monotonic() >= deadline: raise OpsError('task wall-time budget exceeded')
                if stdout.tell()+stderr.tell() > 4*1024**2: raise OpsError('diagnostic output limit exceeded')
                time.sleep(.025)
            try: child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                stop_group(child)
            if group_alive(child.pid):stop_group(child)
            if not stopped and child.returncode != 0: raise OpsError('fixed task failed; inspect retained logs')
        manager.verify()
        result = {'status':'SAFE_STOPPED' if stopped else 'PASS', 'run_id':run_id,
                  'optimizer_updates':0,'training_timesteps':0,'child_returncode':child.returncode}
        if not stopped:result['task_evidence_sha256']=sha256(run/'task-evidence.json')
        atomic_json(run/'result.json',result,exclusive=True)
        atomic_json(run/('stopped.json' if stopped else 'complete.json'),
                    {'run_id':run_id,'result_sha256':sha256(run/'result.json')},exclusive=True)
        state.update(stage_status=result['status'],last_shutdown_status=result['status'])
        manager._save(state)
        return result
    except BaseException as error:
        if child is not None and (child.poll() is None or group_alive(child.pid)):stop_group(child)
        if not (run/'complete.json').exists() and not (run/'stopped.json').exists():
            atomic_json(run/'failure.json',{'run_id':run_id,'error_type':type(error).__name__,
                        'message':str(error)[:1000],'observed_at':now()},exclusive=True)
        state.update(stage_status='FAIL',last_shutdown_status='FAILED',phase_complete=False)
        manager._save(state)
        raise


def request_stop(manager):
    from research_ops.storage import FileLock,lock_held
    _,_,state,version = manager._load()
    if lock_held(manager.lock_path):
        if state['stage_status'] not in ('RUNNING','STOP_REQUESTED'):
            raise OpsError('writer is not a running task; no takeover')
        run = version/'runs'/state['run_id']
        try: atomic_json(run/'stop-request.json',{'run_id':state['run_id'],'time':now()},exclusive=True)
        except FileExistsError:
            if read_json(run/'stop-request.json').get('run_id')!=state['run_id']: raise OpsError('wrong stop identity')
        return {'status':'STOP_REQUESTED','run_id':state['run_id']}
    with FileLock(manager.lock_path):
        _,_,state,version=manager._load()
        manager.verify()
        if state['stage_status'] in ('RUNNING','STOP_REQUESTED'):
            raise OpsError('unclean exit: recover first')
        if state['stage_status'] not in ('FAIL','INTERRUPTED','BLOCKED','SYNC_PENDING'):
            state.update(stage_status='SAFE_STOPPED',last_shutdown_status='SAFE_STOPPED')
            manager._save(state)
        return {'status':state['stage_status']}
