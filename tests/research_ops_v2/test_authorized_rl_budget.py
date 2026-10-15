"""No optimizer tests: fail if persistence, accounting or authority is bypassed."""
import tempfile
import unittest
from pathlib import Path


class AuthorizedBudgetTests(unittest.TestCase):
    def setUp(self):
        from research_regression.authorized_rl.budget import BudgetStore, grant, experiment
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.home=Path(self.temporary.name)/'budget'
        self.store=BudgetStore(self.home)
        self.grant=grant();self.experiment=experiment

    def initialize(self):
        from research_ops.storage import digest
        self.store.initialize(expected_grant_sha256=digest(self.grant))

    def test_absent_authority_refuses_before_creating_attempt(self):
        with self.assertRaises(Exception):self.store.reserve(self.experiment('UNIFORM',65536,3600))
        self.assertFalse((self.home/'attempts').exists())

    def test_wrong_grant_hash_cannot_initialize(self):
        with self.assertRaises(Exception):self.store.initialize(expected_grant_sha256='0'*64)
        self.assertFalse(self.home.exists())

    def test_reinitialize_never_refreshes_consumed_budget(self):
        self.initialize();self.store.reserve(self.experiment('UNIFORM',65536,3600))
        with self.assertRaises(Exception):self.initialize()
        self.assertEqual(self.store.status()['remaining_timesteps'],1934464)

    def test_cumulative_budget_and_cross_restart(self):
        self.initialize();self.store.reserve(self.experiment('UNIFORM',65536,3600))
        from research_regression.authorized_rl.budget import BudgetStore
        self.assertEqual(BudgetStore(self.home).status()['remaining_wall_seconds'],82800)

    def test_duplicate_experiment_is_permanently_refused(self):
        self.initialize();p=self.experiment('UNIFORM',65536,3600);self.store.reserve(p)
        with self.assertRaises(Exception):self.store.reserve(p)
        self.assertEqual(self.store.status()['reserved_timesteps'],65536)

    def test_unaligned_or_oversized_budget_is_refused(self):
        self.initialize()
        for steps in (129,1000192,2000000):
            with self.subTest(steps=steps),self.assertRaises(Exception):
                self.store.reserve(self.experiment('UNIFORM',steps,3600))

    def test_unknown_schema_or_algorithm_or_test_access_is_refused(self):
        self.initialize()
        for key,value in [('algorithm','SAC'),('workers',2),('test_episode_count',1),('device','cuda')]:
            p=self.experiment('UNIFORM',65536,3600);p[key]=value
            with self.subTest(key=key),self.assertRaises(Exception):self.store.reserve(p)

    def test_unknown_interruption_never_refunds_or_replays(self):
        self.initialize();p=self.experiment('UNIFORM',65536,3600);self.store.reserve(p)
        status=self.store.status()
        self.assertEqual(status['remaining_timesteps'],1934464)
        self.assertEqual(status['remaining_wall_seconds'],82800)
        self.assertEqual(status['attempts'][0]['status'],'RESERVED_OR_INTERRUPTED')

    def test_stop_request_blocks_new_reservations(self):
        self.initialize();self.store.request_stop()
        with self.assertRaises(Exception):self.store.reserve(self.experiment('UNIFORM',65536,3600))
        self.assertTrue(self.store.status()['stop_requested'])

    def test_corrupt_or_deleted_event_is_detected(self):
        self.initialize();self.store.reserve(self.experiment('UNIFORM',65536,3600))
        events=sorted((self.home/'events').glob('*.json'))
        events[0].write_text('{}')
        with self.assertRaises(Exception):self.store.status()

    def test_failure_consumes_reservation_and_does_not_allow_retry(self):
        self.initialize();p=self.experiment('UNIFORM',65536,3600);self.store.reserve(p)
        self.store.fail(p['experiment_id'],'TEST_PROCESS_INTERRUPTION')
        self.assertEqual(self.store.status()['attempts'][0]['status'],'FAIL')
        self.assertEqual(self.store.status()['remaining_wall_seconds'],82800)
        with self.assertRaises(Exception):self.store.reserve(p)

    def test_fake_complete_cannot_refund_or_decrease_spending(self):
        self.initialize();p=self.experiment('UNIFORM',65536,3600);self.store.reserve(p)
        with self.assertRaises(Exception):self.store.complete(p['experiment_id'],{'status':'PASS'})
        self.assertEqual(self.store.status()['remaining_wall_seconds'],82800)

    def test_cumulative_wall_limit_cannot_be_reset_by_new_id(self):
        self.initialize()
        p=self.experiment('UNIFORM',65536,86400);self.store.reserve(p)
        with self.assertRaises(Exception):self.store.reserve(self.experiment('CURRICULUM',65536,1))
        self.assertEqual(self.store.status()['remaining_wall_seconds'],0)

    def test_cumulative_two_million_cap_across_thirty_new_ids(self):
        self.initialize()
        for i in range(30):
            p=self.experiment('UNIFORM',65536,1,revision=f'T{i}')
            self.store.reserve(p);self.store.fail(p['experiment_id'],'SYNTHETIC_NO_OPTIMIZER_TEST')
        self.assertEqual(self.store.status()['remaining_timesteps'],33920)
        with self.assertRaises(Exception):self.store.reserve(self.experiment('UNIFORM',65536,1,revision='OVER'))

    def test_exclusive_budget_lock_rejects_concurrent_writer(self):
        from research_ops.storage import FileLock
        self.initialize()
        with FileLock(self.store.lock):
            with self.assertRaises(Exception):self.store.reserve(self.experiment('UNIFORM',65536,1))
        self.assertEqual(self.store.status()['reserved_timesteps'],0)

    def test_legal_concurrent_readers_and_stop_do_not_fail_spuriously(self):
        from concurrent.futures import ThreadPoolExecutor
        self.initialize();self.store.reserve(self.experiment('UNIFORM',65536,1))
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(self.store.status) for _ in range(80)]
            stopped=pool.submit(self.store.request_stop)
            self.assertTrue(all(f.result()['reserved_timesteps']==65536 for f in futures))
            stopped.result()
        self.assertTrue(self.store.status()['stop_requested'])

    def test_unresolved_attempt_blocks_different_experiment_until_audit(self):
        self.initialize();self.store.reserve(self.experiment('UNIFORM',65536,1))
        with self.assertRaises(Exception):self.store.reserve(self.experiment('CURRICULUM',65536,1))


if __name__=='__main__':unittest.main()
