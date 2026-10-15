"""Actual stdlib process/scheduler gates; no PPO updates in these tests."""
import unittest
from research_ops.storage import OpsError


class ExecutionGates(unittest.TestCase):
    def test_independent_guard_kills_group_after_worker_leader_death(self):
        import subprocess,sys,os,signal,time
        from pathlib import Path
        from research_ops_v2.execution import group_alive
        code=('import time,os; from pathlib import Path; '
              'from research_ops.storage import process_identity; '
              'from research_regression.authorized_rl.lifetime import start_guard; '
              'start_guard(process_identity(os.getppid()),time.monotonic()+10,Path.cwd()); '
              'print("READY",flush=True); time.sleep(30)')
        child=subprocess.Popen([sys.executable,'-c',code],start_new_session=True,stdout=subprocess.PIPE,text=True)
        try:
            import select
            self.assertTrue(select.select([child.stdout],[],[],3)[0])
            self.assertEqual(child.stdout.readline().strip(),'READY')
            child.kill();child.wait(timeout=2)
            deadline=time.monotonic()+3
            while group_alive(child.pid) and time.monotonic()<deadline:time.sleep(.025)
            self.assertFalse(group_alive(child.pid))
        finally:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=2);child.stdout.close()

    def test_leader_exit_does_not_leave_live_descendant(self):
        import subprocess,sys
        from research_regression.authorized_rl.runtime import monitor_process
        from research_ops_v2.execution import group_alive
        child=subprocess.Popen([sys.executable,'-c',
            'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])'],
            start_new_session=True)
        try:
            with self.assertRaises(OpsError):monitor_process(child,2,lambda:None)
            self.assertFalse(group_alive(child.pid))
        finally:
            import os,signal
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()

    def test_wall_deadline_stops_actual_isolated_process_group(self):
        import subprocess,sys
        from research_regression.authorized_rl.runtime import monitor_process
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            with self.assertRaises(OpsError):monitor_process(child,.1,lambda:None)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:child.kill();child.wait()

    def test_probe_fault_stops_actual_child_without_retry(self):
        import subprocess,sys
        from research_regression.authorized_rl.runtime import monitor_process
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
        def fault():raise OpsError('forced stop/resource fault')
        try:
            with self.assertRaises(OpsError):monitor_process(child,10,fault)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:child.kill();child.wait()

    def test_unpermitted_worker_never_imports_ml(self):
        from research_regression.authorized_rl.runtime import validate_launch
        with self.assertRaises(OpsError):validate_launch({},'ARL_R1_UNIFORM_SEED0')

    def test_curriculum_threshold_uses_only_registered_empty_then_train(self):
        from research_regression.authorized_rl.curriculum import eligible_rows
        rows=[{'split':'train','layout':'empty','identity_sha256':'a'},
              {'split':'train','layout':'box','identity_sha256':'b'}]
        self.assertEqual([r['identity_sha256'] for r in eligible_rows(rows,'CURRICULUM',32767)],['a'])
        self.assertEqual([r['identity_sha256'] for r in eligible_rows(rows,'CURRICULUM',32768)],['a','b'])
        self.assertEqual(len(eligible_rows(rows,'UNIFORM',0)),2)
        with self.assertRaises(OpsError):eligible_rows([{'split':'test','layout':'empty'}],'UNIFORM',0)

    def test_bad_experiment_id_cannot_escape_runtime_directory(self):
        from research_regression.authorized_rl.runtime import run_directory
        from pathlib import Path
        for value in ('../old','L9','ARL_/tmp'):
            with self.subTest(value=value),self.assertRaises(OpsError):run_directory(Path('/tmp/test'),value)

    def test_stop_and_resource_limits_fail_before_next_step(self):
        from research_regression.authorized_rl.runtime import check_resources
        with self.assertRaises(OpsError):check_resources(1535,3*1024**3)
        with self.assertRaises(OpsError):check_resources(1600,2*1024**3-1)
        check_resources(1536,2*1024**3)


if __name__=='__main__':unittest.main()
