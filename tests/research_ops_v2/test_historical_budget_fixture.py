"""R34: explicit test-only historical reader, never execution admission."""
import copy
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from research_ops.storage import OpsError, atomic_json, sha256
from research_regression.authorized_rl.budget import BudgetStore

ROOT = Path(__file__).resolve().parents[2]


class HistoricalBudgetFixtureTests(unittest.TestCase):
    def helper(self):
        name = '_historical_budget_fixture'
        self.assertIsNotNone(importlib.util.find_spec(name), 'missing explicit historical test reader')
        return importlib.import_module(name)

    def test_full_frozen_snapshot_not_filtered_current_ledger(self):
        helper = self.helper()
        with helper.r28_budget_reader(ROOT) as historical:
            self.assertEqual(len(historical['attempts']), 16)
            self.assertEqual((historical['remaining_timesteps'], historical['remaining_wall_seconds']), (951424, 28800))
            self.assertNotIn('ARL_R33_EARLY_DISTANCE_MIX_SEED0', [v['experiment_id'] for v in historical['attempts']])
            self.assertEqual(historical, BudgetStore(ROOT/'.research_local/authorized_rl_v1').status())

    def test_copies_are_independent_and_real_reader_restored_after_exception(self):
        helper = self.helper()
        store = BudgetStore(ROOT/'.research_local/authorized_rl_v1')
        actual = store.status()
        original = BudgetStore.status
        with self.assertRaisesRegex(RuntimeError, 'FORCED_TEST_EXCEPTION'):
            with helper.r28_budget_reader(ROOT) as snapshot:
                before = copy.deepcopy(snapshot)
                snapshot['attempts'].clear()
                self.assertEqual(store.status(), before)
                first = store.status()
                first['attempts'].clear()
                self.assertEqual(store.status(), before)
                raise RuntimeError('FORCED_TEST_EXCEPTION')
        self.assertIs(BudgetStore.status, original)
        self.assertEqual(store.status(), actual)

    def test_other_budget_store_is_not_substituted(self):
        helper = self.helper()
        with tempfile.TemporaryDirectory() as directory:
            store = BudgetStore(Path(directory)/'missing')
            with self.assertRaises(FileNotFoundError) as outside:
                store.status()
            with helper.r28_budget_reader(ROOT):
                with self.assertRaises(FileNotFoundError) as inside:
                    store.status()
            self.assertEqual(str(inside.exception), str(outside.exception))

    def test_missing_or_forged_report_rejected_before_reader_change(self):
        helper = self.helper()
        original = BudgetStore.status
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(OpsError):
                with helper.r28_budget_reader(root):
                    self.fail('missing history admitted')
            atomic_json(root/'mujoco/reports/autonomous_rl_held_goal_r28.json', {'budget_after': {}})
            with self.assertRaisesRegex(OpsError, 'historical R28 report identity'):
                with helper.r28_budget_reader(root):
                    self.fail('forged history admitted')
        self.assertIs(BudgetStore.status, original)

    def test_actual_current_failed_attempt_still_blocks_all_three_old_paths(self):
        self.helper()
        from research_regression.authorized_rl.closed_yaw_failure import eligible_after_closed_r24
        from research_regression.authorized_rl.interrupted_yaw_diagnostic import references as r26
        from research_regression.authorized_rl.goal_heading_diagnostic import references as r27
        actual = BudgetStore(ROOT/'.research_local/authorized_rl_v1').status()
        self.assertEqual([v['status'] for v in actual['attempts'] if v['experiment_id'] == 'ARL_R33_EARLY_DISTANCE_MIX_SEED0'], ['FAIL'])
        with self.assertRaisesRegex(OpsError, 'unresolved/unknown'):
            eligible_after_closed_r24(actual, ROOT)
        for reference in (r26, r27):
            with self.assertRaisesRegex(OpsError, 'unknown/pending'):
                reference(ROOT)

    def test_unknown_pending_stop_and_forged_balance_fail_for_intended_branch(self):
        helper = self.helper()
        from research_regression.authorized_rl.closed_yaw_failure import eligible_after_closed_r24
        with helper.r28_budget_reader(ROOT) as snapshot:
            self.assertEqual(eligible_after_closed_r24(snapshot, ROOT)['status'], 'PASS')
            for state in ('FAIL', 'RESERVED_OR_INTERRUPTED', 'UNKNOWN'):
                bad = copy.deepcopy(snapshot)
                bad['attempts'].append({'experiment_id': 'FORCED_TEST_UNKNOWN', 'status': state})
                with self.assertRaisesRegex(OpsError, 'unresolved/unknown'):
                    eligible_after_closed_r24(bad, ROOT)
            for key, value, reason in (('stop_requested', True, 'stop latched'),
                                       ('remaining_timesteps', 2000000, 'actual immutable')):
                bad = copy.deepcopy(snapshot)
                bad[key] = value
                with self.assertRaisesRegex(OpsError, reason):
                    eligible_after_closed_r24(bad, ROOT)

    def test_no_persisted_authority_or_report_writes(self):
        helper = self.helper()
        paths = list((ROOT/'.research_local/authorized_rl_v1').rglob('*'))
        paths.append(ROOT/'mujoco/reports/autonomous_rl_held_goal_r28.json')
        before = {p: sha256(p) for p in paths if p.is_file()}
        original = BudgetStore.status
        with helper.r28_budget_reader(ROOT):
            pass
        self.assertIs(BudgetStore.status, original)
        self.assertEqual({p: sha256(p) for p in before}, before)
