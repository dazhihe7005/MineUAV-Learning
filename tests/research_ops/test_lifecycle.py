import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from support import project, git

SOURCE = Path(__file__).resolve().parents[2]


def api():
    assert importlib.util.find_spec('research_ops.lifecycle'), 'operations lifecycle missing'
    from research_ops.lifecycle import Operations
    return Operations


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = project(Path(self.temp.name)/'repo')

    def tearDown(self): self.temp.cleanup()

    def manager(self):
        manager = api()(self.root)
        manager.initialize()
        return manager

    def test_init_status_verify_and_real_bounded_start(self):
        m = self.manager()
        self.assertEqual(m.status()['state']['stage_status'], 'PENDING')
        self.assertEqual(m.verify()['status'], 'PASS')
        result = m.start('ops-audit', max_seconds=0)
        self.assertEqual(result['state']['stage_status'], 'PASS')
        self.assertEqual(result['state']['last_verified_actual_timesteps'], 0)
        self.assertTrue((m.local/'runs'/result['state']['run_id']/'complete.json').is_file())
        with self.assertRaises(Exception): m.initialize()

    def test_training_and_arbitrary_commands_never_create_attempt(self):
        m = self.manager()
        for task in ('ppo-pilot', 'formal-seed0', 'PX4', '/bin/sh', 'ops-audit;echo x'):
            with self.assertRaises(Exception): m.start(task)
        self.assertFalse((m.local/'runs').exists())
        self.assertEqual(m.status()['state']['budget_remaining']['consumed_training_timesteps'], 0)

    def test_bad_source_state_or_budget_refuses_start(self):
        m = self.manager()
        (self.root/'runtime.py').write_text('# changed')
        with self.assertRaises(Exception): m.start('ops-audit')
        self.assertFalse((m.local/'runs').exists())

    def test_checkpoint_mismatch_is_not_silently_repaired(self):
        from research_ops.storage import sha256
        config_path=self.root/'research_ops/operations_config.json'
        config=json.loads(config_path.read_text())
        p=self.root/'history/model.zip';p.write_bytes(b'checkpoint')
        config['checkpoint']={'path':'history/model.zip','sha256':sha256(p),'resume_class':'WARM_START'}
        config_path.write_text(json.dumps(config))
        m=self.manager();p.write_bytes(b'broken')
        with self.assertRaises(Exception):m.verify()
        self.assertEqual(p.read_bytes(),b'broken')

    def launch(self, seconds=15):
        script='from research_ops.lifecycle import Operations;import sys;Operations(sys.argv[1]).start("ops-audit",max_seconds=float(sys.argv[2]))'
        return subprocess.Popen([sys.executable,'-c',script,str(self.root),str(seconds)],
                                cwd=SOURCE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)

    def wait_running(self,m,p):
        end=time.monotonic()+5
        while time.monotonic()<end:
            if p.poll() is not None:
                self.fail('child exited before RUNNING: '+p.communicate()[1])
            if m.status()['state']['stage_status']=='RUNNING':return
            time.sleep(.02)
        self.fail('child did not start')

    def test_actual_process_stop_and_duplicate_start(self):
        m=self.manager();p=self.launch()
        try:
            self.wait_running(m,p)
            self.assertTrue(m.status()['lock_held'])
            with self.assertRaises(Exception):m.start('ops-audit')
            result=m.request_stop(sync=False,wait_seconds=3)
            p.communicate(timeout=4)
            self.assertEqual(result['state']['stage_status'],'SAFE_STOPPED')
            self.assertEqual(result['state']['last_shutdown_status'],'SAFE_STOPPED')
            self.assertEqual(p.returncode,0)
        finally:
            if p.poll() is None:p.kill();p.communicate()

    def test_sigkill_recover_marks_interrupted_and_preserves_attempt(self):
        m=self.manager();p=self.launch()
        try:
            self.wait_running(m,p); run=m.status()['state']['run_id']
            attempt=m.local/'runs'/run/'attempt.json';before=attempt.read_bytes()
            p.kill();p.communicate(timeout=4)
            result=m.recover()
            self.assertEqual(result['state']['stage_status'],'INTERRUPTED')
            self.assertEqual(attempt.read_bytes(),before)
            self.assertFalse((attempt.parent/'complete.json').exists())
            self.assertEqual(result['state']['checkpoint_resume_class'],'RESTART_REQUIRED')
        finally:
            if p.poll() is None:p.kill();p.communicate()

    def test_public_status_cannot_restore_authority_or_modify_private_state(self):
        m=self.manager()
        (self.root/'docs/RESEARCH_STATUS.json').write_text('{"automatic_training":true}')
        self.assertFalse(m.status()['state']['authorization_scope']['optimizer_updates'])
        with self.assertRaises(Exception):m.start('ppo-pilot')

    def test_corrupt_private_state_is_not_reinitialized_by_recover(self):
        m=self.manager();p=m.local/'research_state.json';p.write_text('{corrupt')
        with self.assertRaises(Exception):m.recover()
        self.assertEqual(p.read_text(),'{corrupt')

    def test_missing_or_tampered_completed_audit_cannot_verify(self):
        m=self.manager();state=m.start()['state']
        p=m.local/'runs'/state['run_id']/'audit.json'
        p.write_text('{"status":"PASS"}')
        with self.assertRaises(Exception):m.verify()

    def test_orphan_attempt_blocks_new_start_instead_of_ignoring_it(self):
        m=self.manager();p=m.local/'runs'/'orphan';p.mkdir(parents=True)
        (p/'attempt.json').write_text('{"run_id":"orphan"}')
        with self.assertRaises(Exception):m.start()

    def test_changed_public_snapshot_is_detected_by_verify(self):
        m=self.manager();p=self.root/'docs/RESEARCH_STATUS.json'
        value=json.loads(p.read_text());value['authorization_scope']['optimizer_updates']=True
        p.write_text(json.dumps(value))
        with self.assertRaises(Exception):m.verify()

    def test_budget_ledger_cannot_reset_or_grant_any_training(self):
        m=self.manager();p=m.local/'budget_ledger.json'
        value=json.loads(p.read_text());value['authorization']['authorized_training_timesteps']=4096
        p.write_text(json.dumps(value))
        with self.assertRaises(Exception):m.start()

    def test_consistent_checksum_with_wrong_experiment_identity_is_rejected(self):
        from research_ops.storage import digest
        m=self.manager();p=m.local/'research_state.json'
        value=json.loads(p.read_text());value['state']['experiment_id']='FORMAL_SEED0'
        value['state_sha256']=digest(value['state']);p.write_text(json.dumps(value))
        with self.assertRaises(Exception):m.verify()

    def test_changed_protocol_is_not_repaired_by_doctor(self):
        m=self.manager();p=m.local/'protocol.json';value=json.loads(p.read_text())
        value['registered_tasks']['ppo']={'level':3};p.write_text(json.dumps(value))
        with self.assertRaises(Exception):m.doctor()
        self.assertIn('ppo',json.loads(p.read_text())['registered_tasks'])

    def test_new_instance_after_restart_keeps_zero_budget_and_previous_attempt(self):
        m=self.manager();state=m.start()['state'];run=state['run_id']
        fresh=api()(self.root)
        self.assertEqual(fresh.recover()['state']['run_id'],run)
        self.assertEqual(fresh.status()['state']['budget_remaining']['authorized_training_timesteps'],0)

    def test_lost_stop_owner_cannot_be_reported_safe(self):
        m=self.manager();p=self.launch()
        self.wait_running(m,p);p.kill();p.communicate(timeout=4)
        with self.assertRaises(Exception):m.request_stop(sync=False)

    def test_stop_after_recovery_preserves_interrupted_experiment_status(self):
        m=self.manager();p=self.launch()
        self.wait_running(m,p);p.kill();p.communicate(timeout=4)
        m.recover();result=m.request_stop(sync=False)
        self.assertEqual(result['state']['stage_status'],'INTERRUPTED')
        self.assertEqual(result['state']['last_shutdown_status'],'UNCLEAN_INTERRUPTION')

    def test_failed_audit_is_preserved_and_cannot_automatically_retry(self):
        m=self.manager()
        from unittest.mock import patch
        from research_ops.storage import OpsError
        with patch('research_ops.lifecycle.enforce_resources',side_effect=OpsError('forced resource failure')):
            with self.assertRaises(OpsError):m.start()
        # Preflight resource failure occurs before any attempt: no run can be falsely completed.
        self.assertFalse((m.local/'runs').exists())

    def test_failed_running_audit_blocks_retry_and_can_be_safely_reported(self):
        m=self.manager();original=m.verify;calls=[]
        def fail_after_start(**kwargs):
            calls.append(1)
            if len(calls)>1:raise RuntimeError('FORCED_TEST_STATE audit failure')
            return original(**kwargs)
        from unittest.mock import patch
        with patch.object(m,'verify',side_effect=fail_after_start):
            with self.assertRaises(RuntimeError):m.start()
        self.assertEqual(m.status()['state']['stage_status'],'FAIL')
        with self.assertRaises(Exception):m.start()
        self.assertEqual(m.request_stop(sync=False)['state']['stage_status'],'FAIL')

    def test_cli_rejects_training_without_importing_ml(self):
        m=self.manager()
        script='import sys;from research_ops.cli import main;code=main(sys.argv[1:]);assert not any(x in sys.modules for x in ("torch","stable_baselines3","mujoco"));raise SystemExit(code)'
        result=subprocess.run([sys.executable,'-c',script,'--root',str(self.root),'start','--task','ppo-pilot'],cwd=SOURCE,capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('UNAUTHORIZED',result.stderr)


if __name__=='__main__':unittest.main()
