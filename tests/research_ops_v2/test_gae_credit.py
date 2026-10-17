"""R22 profile/closed-failure proof contracts, no ML or optimizer."""
import copy
from contextlib import contextmanager
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_ops.storage import OpsError, digest, read_json, sha256


class GAECreditTests(unittest.TestCase):
    @contextmanager
    def historical_budget_context(self, root):
        """Explicit historical-reader double, never a current training admission.

        Freeze the full published R22 budget, not a hand-edited current ledger.
        Only this unit-process reader is substituted; all real proof files and
        production eligibility checks remain active. Separate integration test
        below proves the untouched current ledger still rejects the later FAIL.
        """
        from research_regression.authorized_rl.budget import BudgetStore
        path = root/'mujoco/reports/autonomous_rl_gae_credit_r22.json'
        self.assertEqual(sha256(path), 'ea1f49e4df91af36b158f318022137e8fa76e0d39ec14ba10b76a219509ba8b5')
        snapshot = read_json(path)['budget_after']
        with patch.object(BudgetStore, 'status', new=lambda _store: copy.deepcopy(snapshot)):
            yield copy.deepcopy(snapshot)

    def module(self, suffix='gae_credit_profile'):
        name = 'research_regression.authorized_rl.' + suffix
        self.assertIsNotNone(importlib.util.find_spec(name), 'missing fixed R22 GAE/provenance interface')
        return importlib.import_module(name)

    def test_new_experiment_only_changes_lambda_not_gamma_or_prior_algorithm(self):
        from research_regression.authorized_rl.empty_rehearsal_profile import rehearsal_experiment
        p = self.module().gae_experiment()
        old = rehearsal_experiment()
        meta = {'schema', 'experiment_id', 'execution_profile', 'control_report_sha256',
                'control_summary_sha256', 'single_variable', 'expected_prefix_policy_sha256', 'evaluation'}
        for key in set(old)-meta:
            self.assertEqual(p[key], old[key], key)
        self.assertEqual((p['gamma'], p['gae_lambda'], p['timesteps'], p['wall_seconds']), (.99, .99, 65536, 3600))
        self.assertNotIn('expected_prefix_policy_sha256', p)

    def test_profile_rejects_unapproved_algorithm_identity_and_budget_changes(self):
        m = self.module()
        for key, value in (('gamma', .995), ('gae_lambda', .95), ('seed', 1), ('timesteps', 131072),
                           ('reward_version', 'OTHER'), ('workers', 2), ('checkpoint_path', 'old.zip')):
            p = m.gae_experiment(); p[key] = value
            with self.assertRaises(OpsError):
                m.validate_gae_profile(p)

    def test_new_schema_uses_existing_grant_without_reinitialization(self):
        from research_regression.authorized_rl.budget import validate_experiment, grant
        validate_experiment(self.module().gae_experiment())
        self.assertEqual(digest(grant()), '1897b889fd3fea3f269abed9533104c123ef4d81fd9843d1cbe0d448db1cb7fc')

    def test_missing_control_or_failure_proof_never_creates_output(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):
                m.verify_gae_proof(m.gae_experiment(), Path(d))
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_closed_failure_requires_real_published_evidence_not_only_failed_status(self):
        m = self.module('closed_failure_proof')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):
                m.verify_r20_closed(Path(d))
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_actual_closed_failure_proof_preserves_failed_status_and_all_files(self):
        from research_ops.storage import sha256
        root = Path(__file__).resolve().parents[2]
        files = {p: sha256(p) for p in (root/'.research_local/authorized_rl_v1/events').iterdir()}
        value = self.module('closed_failure_proof').verify_r20_closed(root)
        self.assertEqual(value['R20_status'], 'PERMANENT_FAIL_NO_RETRY')
        self.assertEqual(value['R20_scientific_result'], 'NOT TESTED')
        self.assertEqual(value['refund'], False)
        for path, expected in files.items():
            self.assertEqual(sha256(path), expected)

    def test_new_gate_accepts_only_real_closed_failures_not_pending_or_unknown(self):
        root = Path(__file__).resolve().parents[2]
        m = self.module('closed_failure_proof')
        with self.historical_budget_context(root) as b:
            m.eligible_after_closed_r20(b, root)
            for status, identity in (('FAIL', 'ARL_UNKNOWN_FAIL_SEED0'),
                                     ('RESERVED_OR_INTERRUPTED', 'ARL_R20_TEMPORAL_CREDIT_SEED0')):
                bad = copy.deepcopy(b)
                bad['attempts'][-1]['status'] = status
                bad['attempts'][-1]['experiment_id'] = identity
                with self.assertRaisesRegex(OpsError, 'unresolved/unknown failed attempt'):
                    m.eligible_after_closed_r20(bad, root)

    def test_stop_and_forged_budget_still_fail_closed(self):
        root = Path(__file__).resolve().parents[2]
        m = self.module('closed_failure_proof')
        with self.historical_budget_context(root) as b:
            for key, value, reason in (('stop_requested', True, 'stop latched'),
                                       ('remaining_timesteps', 2000000, 'caller budget is not the actual immutable grant ledger')):
                bad = copy.deepcopy(b); bad[key] = value
                with self.assertRaisesRegex(OpsError, reason):
                    m.eligible_after_closed_r20(bad, root)

    def test_old_gate_stays_closed_for_r20_and_its_identity_stays_consumed(self):
        from research_regression.authorized_rl.feedback_followup import eligible_budget
        root = Path(__file__).resolve().parents[2]
        with self.historical_budget_context(root) as b:
            self.module('closed_failure_proof').eligible_after_closed_r20(b, root)
            with self.assertRaises(OpsError):
                eligible_budget(b)
            self.assertEqual([a['status'] for a in b['attempts'] if a['experiment_id'] == 'ARL_R20_TEMPORAL_CREDIT_SEED0'], ['FAIL'])

    def test_historical_fixture_is_explicit_bounded_and_restores_actual_ledger_reader(self):
        from research_regression.authorized_rl.budget import BudgetStore
        root = Path(__file__).resolve().parents[2]
        context = getattr(self, 'historical_budget_context', None)
        self.assertTrue(callable(context), 'missing explicit frozen historical budget context')
        original = BudgetStore.status
        actual_before = BudgetStore(root/'.research_local/authorized_rl_v1').status()
        with context(root) as historical:
            self.assertEqual((historical['remaining_timesteps'], historical['remaining_wall_seconds']), (1082496, 36000))
            self.assertEqual(len(historical['attempts']), 14)
            self.assertNotIn('ARL_R24_TRAIN_YAW_COVERAGE_SEED0', [a['experiment_id'] for a in historical['attempts']])
            self.module('closed_failure_proof').eligible_after_closed_r20(historical, root)
        self.assertIs(BudgetStore.status, original)
        self.assertEqual(BudgetStore(root/'.research_local/authorized_rl_v1').status(), actual_before)

    def test_actual_later_failed_attempt_still_blocks_old_gate_outside_fixture(self):
        from research_regression.authorized_rl.budget import BudgetStore
        root = Path(__file__).resolve().parents[2]
        actual = BudgetStore(root/'.research_local/authorized_rl_v1').status()
        self.assertEqual([a['status'] for a in actual['attempts'] if a['experiment_id'] == 'ARL_R24_TRAIN_YAW_COVERAGE_SEED0'], ['FAIL'])
        with self.assertRaisesRegex(OpsError, 'unresolved/unknown failed attempt'):
            self.module('closed_failure_proof').eligible_after_closed_r20(actual, root)
