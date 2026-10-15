import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from support_v2 import fixture,proposal,review
from research_ops.storage import OpsError,FileLock,atomic_json,read_json,digest
from research_ops_v2.manager import OperationsV2


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=fixture(Path(self.tmp.name)/'repo')
        self.m=OperationsV2(self.root);self.reg=proposal(self.root)
        self.m.register(self.reg,review(self.reg),migrating=True)
    def tearDown(self):self.tmp.cleanup()

    def test_real_fixed_worker_completes_and_evidence_is_bound(self):
        result=self.m.start('audit');self.assertEqual(result['status'],'PASS')
        self.assertEqual(self.m.verify()['status'],'PASS')
        _,_,state,version=self.m._load();run=version/'runs'/state['run_id']
        atomic_json(run/'task-evidence.json',{'status':'FORCED_TEST_TAMPER'})
        with self.assertRaisesRegex(OpsError,'task evidence'):self.m.verify()

    def test_unregistered_training_does_not_create_attempt(self):
        for task in ('ppo','pilot','formal-seed0','python-script'):
            with self.assertRaisesRegex(OpsError,'UNAUTHORIZED'):self.m.start(task)
        self.assertEqual(self.m.status()['state']['run_id'],'NOT_STARTED')

    def test_lock_duplicate_start_rejected_without_state_mutation(self):
        before=self.m.status()['state']
        with FileLock(self.m.lock_path):
            with self.assertRaisesRegex(OpsError,'flock'):self.m.start('audit')
            self.assertEqual(self.m.recover()['status'],'BLOCKED')
        self.assertEqual(self.m.status()['state'],before)

    def test_stop_without_work_is_safe_and_preserves_v1(self):
        path=self.root/'.research_local/research_state.json';before=path.read_bytes()
        self.assertEqual(self.m.request_stop(sync=False)['status'],'SAFE_STOPPED')
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(self.m.recover()['state']['stage_status'],'SAFE_STOPPED')

    def test_unclean_reboot_is_audited_without_task_reexecution(self):
        _,_,state,version=self.m._load();run=version/'runs'/('OPS2_'+'a'*32)
        run.mkdir(parents=True)
        atomic_json(run/'attempt.json',{'run_id':run.name,'task_id':'audit','task':self.reg['tasks'][0],
                    'protocol_sha256':state['protocol_sha256']})
        state.update(run_id=run.name,task_id='audit',stage_status='RUNNING',owner='NOT_VERIFIED')
        self.m._save(state)
        first=self.m.recover();self.assertEqual(first['state']['stage_status'],'INTERRUPTED')
        self.assertEqual(self.m.recover()['state']['run_id'],run.name)
        self.assertFalse((run/'task-evidence.json').exists())
        with self.assertRaises(OpsError):self.m.start('audit')

    def test_resource_shortage_stops_before_attempt(self):
        with patch('research_ops_v2.manager.enforce_resources',side_effect=OpsError('FORCED resource failure')):
            with self.assertRaises(OpsError):self.m.start('audit')
        self.assertEqual(self.m.status()['state']['run_id'],'NOT_STARTED')

    def test_completion_state_disk_error_never_creates_contradictory_markers(self):
        original=self.m._save;failed=False
        def save(state):
            nonlocal failed
            if state['stage_status']=='PASS' and not failed:
                failed=True;raise OSError('FORCED_TEST_STATE disk failure')
            return original(state)
        with patch.object(self.m,'_save',side_effect=save):
            with self.assertRaises(OSError):self.m.start('audit')
        _,_,state,version=self.m._load();run=version/'runs'/state['run_id']
        self.assertTrue((run/'complete.json').exists());self.assertFalse((run/'failure.json').exists())
        self.assertEqual(self.m.recover()['state']['stage_status'],'FAIL')

    def test_process_group_cleanup_includes_grandchild(self):
        from research_ops_v2.execution import stop_group,group_alive
        from research_ops.storage import process_identity
        import json,time
        run=self.root/'.research_local/group-cleanup-forced-test';run.mkdir()
        atomic_json(run/'attempt.json',{'deadline_monotonic':time.monotonic()+3.})
        script='''import subprocess,time,sys,json
from pathlib import Path
from research_ops.storage import atomic_json,process_identity
from research_ops_v2.worker import wait_for_supervisor
run=Path(sys.argv[1]);state={'run_id':'FORCED_TEST_STATE','owner':json.loads(sys.argv[2])}
atomic_json(run/'launch-permit.json',{'run_id':state['run_id'],'child':process_identity()})
wait_for_supervisor(run,state)
subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
time.sleep(60)
'''
        child=subprocess.Popen([sys.executable,'-c',script,str(run),json.dumps(process_identity())],
            cwd=self.root,start_new_session=True)
        try:
            import time;time.sleep(.15)
            self.assertTrue(group_alive(child.pid));stop_group(child)
            self.assertFalse(group_alive(child.pid))
        finally:
            if child.poll() is None:stop_group(child)

    def test_current_runtime_must_match_unless_explicit_new_version_review(self):
        p=self.root/'research_ops_v2/core.py';p.write_text('# NEW V2 VERSION\n')
        with self.assertRaisesRegex(OpsError,'source changed'):self.m.verify()
        new=proposal(self.root,phase='OPS-03')
        self.m.register(new,review(new))
        self.assertEqual(self.m.verify()['status'],'PASS')

    def test_state_corruption_cannot_grant_training(self):
        _,_,state,version=self.m._load();state['training_budget']=1
        atomic_json(version/'state.json',{'state':state,'sha256':digest(state)})
        with self.assertRaisesRegex(OpsError,'training authority'):self.m.recover()

    def test_live_task_stop_request_and_no_replay_after_restart(self):
        import time
        child=subprocess.Popen([sys.executable,str(self.root/'researchctl-v2'),'start','--task','audit'],
            cwd=self.root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+3
            while self.m.status()['state']['stage_status']!='RUNNING' and child.poll() is None and time.monotonic()<deadline:time.sleep(.001)
            result=self.m.request_stop(sync=False)
            self.assertIn(result['status'],('STOP_REQUESTED','SAFE_STOPPED'))
            out,err=child.communicate(timeout=5);self.assertEqual(child.returncode,0,(out,err))
            state=self.m.recover()['state'];self.assertEqual(state['stage_status'],'SAFE_STOPPED')
            self.assertEqual(len(list((self.m._load()[3]/'runs').iterdir())),1)
        finally:
            if child.poll() is None:child.terminate();child.communicate(timeout=3)

    def test_interrupt_after_attempt_before_running_state_is_recoverable(self):
        with patch.object(self.m,'_save',side_effect=OSError('FORCED_TEST_STATE before RUNNING')):
            with self.assertRaises(OSError):self.m.start('audit')
        result=self.m.recover();self.assertEqual(result['state']['stage_status'],'INTERRUPTED')
        self.assertEqual(result['state']['last_shutdown_status'],'INTERRUPTED_BEFORE_LAUNCH')
        self.assertEqual(self.m.verify()['status'],'PASS')

    def test_supervisor_crash_at_popen_return_cannot_launch_task(self):
        import time
        script='''import os
from research_ops_v2.manager import OperationsV2
from research_ops_v2 import execution
from pathlib import Path
original=execution.subprocess.Popen
def crash(*args,**kwargs):
 child=original(*args,**kwargs)
 if 'research_ops_v2.worker' in args[0]:os._exit(91)
 return child
execution.subprocess.Popen=crash
OperationsV2(Path.cwd()).start('audit')
'''
        child=subprocess.Popen([sys.executable,'-c',script],cwd=self.root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        child.communicate(timeout=5);self.assertEqual(child.returncode,91)
        _,_,state,version=self.m._load();run=version/'runs'/state['run_id']
        deadline=time.monotonic()+3
        while not (run/'child.json').exists() and time.monotonic()<deadline:time.sleep(.01)
        self.assertTrue((run/'child.json').exists(),(run/'stderr.txt').read_text() if (run/'stderr.txt').exists() else 'worker not started')
        from research_ops_v2.execution import group_alive
        record=read_json(run/'child.json')
        while group_alive(record['pgid']) and time.monotonic()<deadline:time.sleep(.01)
        self.assertFalse(group_alive(record['pgid']))
        self.assertFalse((run/'launch-permit.json').exists());self.assertFalse((run/'task-evidence.json').exists())
        self.assertEqual(self.m.recover()['state']['stage_status'],'INTERRUPTED')

    def test_independent_watchdog_enforces_deadline_for_worker_and_descendants(self):
        import time
        from research_ops.storage import process_identity
        from research_ops_v2.execution import group_alive,stop_group
        run=self.root/'.research_local/watchdog-forced-test';run.mkdir()
        atomic_json(run/'attempt.json',{'deadline_monotonic':time.monotonic()+1.})
        script='''from pathlib import Path
import os,sys,subprocess,time,json
from research_ops.storage import atomic_json,process_identity
from research_ops_v2.worker import wait_for_supervisor
run=Path(sys.argv[1]);state={'run_id':'FORCED_TEST_STATE','owner':json.loads(sys.argv[2])}
atomic_json(run/'launch-permit.json',{'run_id':state['run_id'],'child':process_identity()})
wait_for_supervisor(run,state)
subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
time.sleep(60)
'''
        import json
        child=subprocess.Popen([sys.executable,'-c',script,str(run),json.dumps(process_identity())],
            cwd=self.root,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            child.communicate(timeout=4);self.assertLess(child.returncode,0)
            deadline=time.monotonic()+1
            while group_alive(child.pid) and time.monotonic()<deadline:time.sleep(.01)
            self.assertFalse(group_alive(child.pid))
        finally:
            if child.poll() is None or group_alive(child.pid):stop_group(child)


if __name__=='__main__':unittest.main()
