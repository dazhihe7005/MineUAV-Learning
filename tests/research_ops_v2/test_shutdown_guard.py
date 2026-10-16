"""R21: real signals in isolated processes; no model, optimizer or real grant writes."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

from research_ops.storage import OpsError, read_json


class ShutdownLatchTests(unittest.TestCase):
    def module(self):
        name = 'research_regression.authorized_rl.shutdown_guard'
        self.assertIsNotNone(importlib.util.find_spec(name), 'missing deferred SIGTERM handling')
        return importlib.import_module(name)

    def test_signal_is_deferred_until_explicit_checkpoint(self):
        m = self.module()
        previous = signal.getsignal(signal.SIGTERM)
        with m.DeferredTermination() as pending:
            os.kill(os.getpid(), signal.SIGTERM)
            self.assertEqual(pending.snapshot()['signal_number'], 15)
            with self.assertRaisesRegex(OpsError, 'SIGTERM'):
                pending.checkpoint()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_repeated_term_preserves_first_observation_and_does_not_interrupt_cleanup(self):
        with self.module().DeferredTermination() as pending:
            os.kill(os.getpid(), signal.SIGTERM)
            first = pending.snapshot()
            os.kill(os.getpid(), signal.SIGTERM)
            self.assertEqual(pending.snapshot(), first)
            self.assertGreater(first['observed_unix_time_ns'], 0)
            self.assertGreater(first['observed_monotonic_ns'], 0)
            self.assertRegex(first['boot_id'], r'^[0-9a-f-]{36}$')

    def test_no_signal_does_not_forge_an_observation(self):
        with self.module().DeferredTermination() as pending:
            pending.checkpoint()
            self.assertEqual(pending.snapshot(), 'NOT_OBSERVED')

    def test_handler_restored_after_ordinary_exception(self):
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with self.module().DeferredTermination():
                raise RuntimeError('fixture')
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)


class SupervisorSignalTests(unittest.TestCase):
    def run_case(self, case):
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.run([sys.executable, '-c',
                               "import runpy;runpy.run_path('tests/research_ops_v2/test_shutdown_guard.py',run_name='__main__')", case, d],
                               cwd=Path(__file__).resolve().parents[2], capture_output=True,
                               text=True, timeout=12)
            self.assertEqual(p.returncode, 0, (p.returncode, p.stdout, p.stderr))
            result = read_json(Path(d)/'fixture_result.json')
            self.assertEqual(result['remaining_timesteps'], 1999872)
            self.assertEqual(result['remaining_wall_seconds'], 86390)
            self.assertEqual(result['events'], ['RESERVE', 'FAIL'] if case != 'complete' else ['RESERVE', 'COMPLETE'])
            self.assertFalse(result['group_alive'])
            return result

    def test_sigterm_during_real_child_monitor_records_failure_and_cleans_group(self):
        r = self.run_case('monitor')
        self.assertEqual(r['signal_number'], 15)
        self.assertEqual(r['exact_final_timesteps'], 'NOT_VERIFIED')

    def test_sigterm_during_reservation_finishes_transaction_then_fails_once(self):
        r = self.run_case('reserve')
        self.assertEqual(r['signal_number'], 15)
        self.assertFalse(r['child_launched'])

    def test_sigterm_before_completion_commit_cannot_publish_success(self):
        r = self.run_case('before_complete')
        self.assertEqual(r['signal_number'], 15)

    def test_sigterm_inside_success_transaction_does_not_append_conflicting_fail(self):
        r = self.run_case('complete')
        self.assertEqual(r['result_status'], 'PASS')

    def test_signal_in_launch_window_may_create_child_but_still_cleans_it(self):
        r = self.run_case('launch_window')
        self.assertTrue(r['child_launched'])
        self.assertEqual(r['signal_number'], 15)


def fixture(case, directory):
    """Only admission and worker launch are doubled; ledger/signals/cleanup stay real."""
    import contextlib
    from unittest.mock import patch
    from research_ops.storage import atomic_json, digest, sha256
    from research_regression.authorized_rl.budget import BudgetStore, experiment, grant
    from research_regression.authorized_rl import budget
    from research_regression.authorized_rl import runtime
    root = Path(directory)
    home = root/'.research_local/authorized_rl_v1'
    store = BudgetStore(home)
    store.initialize(expected_grant_sha256=digest(grant()))
    atomic_json(home/'baseline.json', {'runtime_sources': {}}, exclusive=True)
    atomic_json(root/'research_ops/environment_lock.json', {'interpreter': sys.executable}, exclusive=True)
    reg = {'source_identity': {}}
    p = experiment('UNIFORM', 128, 10, revision='R21_SIGNAL_FIXTURE')
    p.update(registration_sha256=digest(reg), source_identity_sha256=digest({}))
    out = runtime.run_directory(home, p['experiment_id'])
    real_popen = subprocess.Popen
    owned = []
    worker = "import os,signal,time;os.kill(os.getppid(),signal.SIGTERM);time.sleep(5)" if case == 'monitor' else 'pass'

    def launch(*args, **kwargs):
        assert args[0][1:3] == ['-m', 'research_regression.authorized_rl.worker']
        if case == 'launch_window':
            os.kill(os.getpid(), signal.SIGTERM)
        if case != 'monitor':
            summary = {'protocol_sha256': digest(p), 'actual_timesteps': 128,
                       'optimizer_step_count': 4, 'finite_model_and_gradients': True,
                       'runtime_sources': {}}
            atomic_json(out/'summary.json', summary, exclusive=True)
            atomic_json(out/'complete.json', {'summary_sha256': sha256(out/'summary.json')}, exclusive=True)
        child = real_popen([sys.executable, '-c', worker], **kwargs)
        owned.append(child)
        return child

    verify_calls = 0

    def verify(_root):
        nonlocal verify_calls
        verify_calls += 1
        if case == 'before_complete' and verify_calls == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return None, reg

    def journal_write(path, value, **kwargs):
        atomic_json(path, value, **kwargs)
        # Deliver precisely between the REAL event publication and head update.
        # Raising here would leave an orphan event; the handler must only latch.
        if (case == 'reserve' and Path(path) == home/'events/000001.json'
                or case == 'complete' and Path(path) == home/'events/000002.json'):
            os.kill(os.getpid(), signal.SIGTERM)

    class Manager:
        def __init__(self, _root):
            self.lock_path = root/'fixture.lock'

    try:
        with contextlib.ExitStack() as stack:
            for target, value in (
                ('OperationsV2', Manager), ('verify_current', verify),
                ('resource_status', lambda _: {'mem_available_mib': 4096, 'disk_free_bytes': 10*1024**3}),
                ('runtime_environment', lambda *_: os.environ.copy()),
                ('subprocess.Popen', launch)):
                stack.enter_context(patch.object(runtime, target, value) if '.' not in target
                                    else patch.object(runtime.subprocess, 'Popen', value))
            stack.enter_context(patch('research_regression.authorized_rl.baseline.verify_baseline', lambda _: None))
            stack.enter_context(patch.object(budget, 'atomic_json', journal_write))
            result_status = 'NOT_RETURNED'
            try:
                result_status = runtime.supervise(p, root=root)['status']
            except OpsError as error:
                assert 'SIGTERM' in str(error), str(error)
        status = store.status()
        evidence = read_json(out/'supervisor_failure.json') if case != 'complete' else {}
        atomic_json(root/'fixture_result.json', {
            'remaining_timesteps': status['remaining_timesteps'],
            'remaining_wall_seconds': status['remaining_wall_seconds'],
            'events': [read_json(f)['kind'] for f in sorted((home/'events').iterdir())],
            'group_alive': any(runtime.group_alive(c.pid) for c in owned),
            'child_launched': bool(owned), 'result_status': result_status,
            'signal_number': evidence.get('termination_request', {}).get('signal_number'),
            'exact_final_timesteps': evidence.get('exact_final_timesteps')}, exclusive=True)
    finally:
        for child in owned:
            if child.poll() is None or runtime.group_alive(child.pid):
                runtime.stop_group(child)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] in ('monitor', 'reserve', 'before_complete', 'complete', 'launch_window'):
        fixture(sys.argv[1], sys.argv[2])
    else:
        unittest.main()
