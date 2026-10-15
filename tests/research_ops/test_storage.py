import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest


def storage():
    assert importlib.util.find_spec('research_ops') and importlib.util.find_spec('research_ops.storage'), 'atomic operations storage is missing'
    from research_ops import storage as module
    return module


def hold_lock(path, ready, release):
    with storage().FileLock(Path(path)):
        ready.set()
        release.wait(5)


class StorageTests(unittest.TestCase):
    def test_atomic_roundtrip_rejects_nonfinite_without_damaging_previous_state(self):
        m = storage()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'state.json'
            m.atomic_json(path, {'status': 'PASS'})
            with self.assertRaises(ValueError):
                m.atomic_json(path, {'value': float('nan')})
            self.assertEqual(m.read_json(path), {'status': 'PASS'})

    def test_corrupt_duplicate_or_overflow_json_is_rejected(self):
        m = storage()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'state.json'
            for text in ('{', '{"a":1,"a":2}', '{"n":1e309}', '{"n":NaN}'):
                path.write_text(text)
                with self.assertRaises((ValueError, m.OpsError)):
                    m.read_json(path)

    def test_exclusive_publication_never_overwrites_attempt(self):
        m = storage()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'attempt.json'
            m.atomic_json(p, {'id': 'original'}, exclusive=True)
            with self.assertRaises(FileExistsError):
                m.atomic_json(p, {'id': 'replacement'}, exclusive=True)
            self.assertEqual(m.read_json(p)['id'], 'original')

    def test_real_process_flock_excludes_even_with_no_pid_file(self):
        m = storage()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'lock'
            ready, release = multiprocessing.Event(), multiprocessing.Event()
            process = multiprocessing.Process(target=hold_lock, args=(str(path), ready, release))
            process.start()
            try:
                self.assertTrue(ready.wait(3))
                self.assertTrue(m.lock_held(path))
                with self.assertRaises(m.OpsError):
                    with m.FileLock(path):
                        self.fail('second writer entered')
            finally:
                release.set(); process.join(5)
            self.assertFalse(m.lock_held(path))

    def test_state_transition_rejects_false_completion_and_budget_escalation(self):
        m = storage()
        state = m.initial_state('project', 'feature', 'a' * 40, 'b' * 64)
        state = m.transition(state, 'RUNNING')
        self.assertEqual(m.transition(state, 'INTERRUPTED')['stage_status'], 'INTERRUPTED')
        with self.assertRaises(m.OpsError):
            m.transition(state, 'PENDING')
        for key in ('authorized_training_timesteps', 'authorized_training_wall_time'):
            changed = json.loads(json.dumps(state)); changed['budget_remaining'][key] = 1
            with self.assertRaises(m.OpsError):
                m.validate_state(changed)
        changed = dict(state, authorization_scope={'levels': [0, 1, 3]})
        with self.assertRaises(m.OpsError):
            m.validate_state(changed)

    def test_boolean_zero_cannot_substitute_for_typed_budget(self):
        m=storage();state=m.initial_state('project','feature','a'*40,'b'*64)
        state['budget_remaining']['authorized_training_timesteps']=False
        with self.assertRaises(m.OpsError):m.validate_state(state)

    def test_symlink_parent_cannot_redirect_state_output(self):
        m=storage()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'real').mkdir();(root/'link').symlink_to(root/'real')
            with self.assertRaises(m.OpsError):m.atomic_json(root/'link/state.json',{'n':1})
            self.assertFalse((root/'real/state.json').exists())


if __name__ == '__main__':
    unittest.main()
