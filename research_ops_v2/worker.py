"""Private dispatcher: arguments cannot name Python code or shell commands."""
from __future__ import annotations
import argparse
import os
import signal
import time
from pathlib import Path
from research_ops.storage import OpsError,atomic_json,read_json,digest,process_identity
from research_ops.identity import safe_path
from .manager import OperationsV2


def wait_for_supervisor(run,state):
    from .execution import process_live
    identity=process_identity();attempt=read_json(run/'attempt.json')
    atomic_json(run/'child.json',{'identity':identity,'pgid':os.getpgrp()},exclusive=True)
    while not (run/'launch-permit.json').exists():
        if not process_live(state['owner']) or time.monotonic()>=attempt['deadline_monotonic']:
            raise OpsError('supervisor lost before launch permit')
        time.sleep(.005)
    permit=read_json(run/'launch-permit.json')
    if permit!={'run_id':state['run_id'],'child':identity}:raise OpsError('invalid launch permit')
    # Separate Linux process, not a GIL-bound thread: supervisor loss, worker
    # death, resource shortage and absolute wall deadline revoke the whole group.
    watcher=os.fork()
    if watcher==0:
        from research_ops.environment import enforce_resources
        while True:
            try:
                if (not process_live(state['owner']) or not process_live(identity)
                        or time.monotonic()>=attempt['deadline_monotonic']):raise OpsError('task lifetime ended')
                enforce_resources(run)
            except BaseException:
                os.killpg(os.getpgrp(),signal.SIGKILL);os._exit(2)
            time.sleep(.05)
    return watcher


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    manager=OperationsV2(root)
    _,protocol,state,version=manager._load()
    run=safe_path(root,args.run)
    if run!=version/'runs'/state['run_id'] or state['stage_status']!='RUNNING':
        raise OpsError('not an active authorized task')
    task=read_json(run/'attempt.json')['task']
    if task not in protocol['registration']['tasks']:raise OpsError('unregistered task')
    watcher=wait_for_supervisor(run,state)
    try:execute(manager,run,task,protocol,root)
    finally:
        try:os.kill(watcher,signal.SIGTERM)
        except ProcessLookupError:pass
        os.waitpid(watcher,0)


def execute(manager,run,task,protocol,root):
    kind=task['task_type'];parameters=task['parameters']
    if kind=='offline-audit':
        result=manager.verify()
    elif kind=='regression-check':
        if parameters['suite']=='historical':
            from research_regression.run import main as regression
            code=regression(['--suite','historical','--evidence',str(run/'regression.json'),'--child'])
            if code:raise OpsError('historical regression failed')
            result=read_json(run/'regression.json')
        else:
            from .test_runner import run_tests
            result=run_tests(root,protocol['registration'])
            if not result['complete']:raise OpsError('operations regression failed')
    elif kind=='documentation-update':
        from .publication import refresh
        refresh(manager);result={'status':'PASS','scope':'public V2 snapshot only'}
    elif kind=='experiment-analysis':
        relative={'L9':'mujoco/reports/mujoco_lidar_ppo_formal_seed0.json',
                  'L9S':'mujoco/reports/mujoco_lidar_ppo_repaired_integration.json'}[parameters['experiment_id']]
        value=read_json(safe_path(root,relative))
        result={'status':'PASS','source_report':relative,'canonical_sha256':digest(value),
                'scope':'read-only historical report identity; no new performance claim'}
    else:raise OpsError('unsupported capability')
    atomic_json(run/'task-evidence.json',result,exclusive=True)


if __name__=='__main__':main()
