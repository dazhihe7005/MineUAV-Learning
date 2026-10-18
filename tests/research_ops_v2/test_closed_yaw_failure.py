"""New explicit R24 closure for new experiments; old gate remains fail-closed."""
import copy,importlib,importlib.util,tempfile,unittest
from pathlib import Path
from research_ops.storage import OpsError,digest
from research_regression.authorized_rl.budget import BudgetStore
from _historical_budget_fixture import r28_budget_reader

class ClosedYawFailureTests(unittest.TestCase):
    def module(self):
        name='research_regression.authorized_rl.closed_yaw_failure'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing independent R24 closed-failure gate')
        return importlib.import_module(name)
    def budget(self):return BudgetStore(Path(__file__).resolve().parents[2]/'.research_local/authorized_rl_v1').status()
    def test_actual_failure_is_permanent_no_refund_or_retry_authority(self):
        v=self.module().verify_r24_closed()
        self.assertEqual(v['R24_status'],'PERMANENT_FAIL_NO_RETRY');self.assertFalse(v['refund'])
        self.assertEqual(v['R24_scientific_result'],'NOT TESTED')
    def test_frozen_historical_ledger_accepts_r24_closure_without_current_admission(self):
        m=self.module();actual=self.budget()
        with r28_budget_reader(Path(__file__).resolve().parents[2]) as b:
            before=copy.deepcopy(b)
            self.assertEqual(m.eligible_after_closed_r24(b)['status'],'PASS');self.assertEqual(b,before)
            self.assertEqual(self.budget(),before)
        self.assertEqual(self.budget(),actual)
    def test_missing_proof_rejects_without_creating_files(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):self.module().verify_r24_closed(Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])
    def test_unknown_failed_pending_or_stop_still_refused_for_precise_cause(self):
        m=self.module()
        with r28_budget_reader(Path(__file__).resolve().parents[2]) as snapshot:
            for status in ('FAIL','RESERVED','UNKNOWN'):
                b=copy.deepcopy(snapshot);b['attempts'].append({'experiment_id':'ARL_UNKNOWN_FAIL_SEED0','status':status})
                with self.assertRaisesRegex(OpsError,'unresolved/unknown'):m.eligible_after_closed_r24(b)
            b=copy.deepcopy(snapshot);b['stop_requested']=True
            with self.assertRaisesRegex(OpsError,'stop latched'):m.eligible_after_closed_r24(b)
    def test_forged_balance_cannot_replace_actual_grant_ledger(self):
        with r28_budget_reader(Path(__file__).resolve().parents[2]) as b:
            b['remaining_timesteps']+=65536
            with self.assertRaisesRegex(OpsError,'actual immutable'):self.module().eligible_after_closed_r24(b)
    def test_old_gate_and_failed_id_remain_rejected(self):
        self.module()
        from research_regression.authorized_rl.closed_failure_proof import eligible_after_closed_r20
        with self.assertRaisesRegex(OpsError,'unresolved/unknown'):eligible_after_closed_r20(self.budget())
        self.assertTrue(any(v['experiment_id']=='ARL_R24_TRAIN_YAW_COVERAGE_SEED0' and v['status']=='FAIL' for v in self.budget()['attempts']))

if __name__=='__main__':unittest.main()
