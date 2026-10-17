import contextlib,importlib,importlib.util,io,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from research_ops.storage import OpsError,digest,atomic_json

class HeldGoalEntryTests(unittest.TestCase):
    def cli(self):
        name='research_regression.authorized_rl.held_goal_cli'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing explicitly isolated heldgoal CLI')
        return importlib.import_module(name)
    def test_prepare_rejects_old_phase_without_creating_protocol_or_reservation(self):
        m=self.cli()
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory)
            with patch.object(m,'HOME',home),patch.object(m,'verify_current',return_value=(None,{'phase_id':'AUTORL-R24-TRAIN-YAW-COVERAGE'})),patch('sys.argv',['held','prepare']),contextlib.redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as error:m.main()
            self.assertEqual(error.exception.code,2);self.assertIn('reviewed registration required',out.getvalue());self.assertEqual(list(home.iterdir()),[])
    def test_wrong_explicit_protocol_hash_cannot_reach_supervisor(self):
        m=self.cli()
        def forbidden(*args,**kwargs):raise AssertionError('wrong hash reached supervisor')
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory);atomic_json(home/'protocols/ARL_R28_HOLD_GOAL_YAW_SEED0.json',{'not':'the frozen protocol'},exclusive=True)
            with patch.object(m,'HOME',home),patch.object(m,'verify_current',return_value=(None,{'phase_id':'AUTORL-R28-HELD-GOAL-YAW'})),patch.object(m,'supervise',forbidden),patch('sys.argv',['held','start','--expected-protocol-sha256','a'*64]),contextlib.redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as error:m.main()
            self.assertEqual(error.exception.code,2);self.assertIn('explicit protocol SHA',out.getvalue())
    def test_actual_runtime_new_phase_runs_new_proof_before_any_reservation(self):
        from research_regression.authorized_rl import runtime
        from research_regression.authorized_rl import held_goal_profile
        from research_ops.storage import atomic_bytes
        p=held_goal_profile.held_goal_experiment();reg={'phase_id':'AUTORL-R28-HELD-GOAL-YAW','source_identity':{'test':'only'}}
        p.update(registration_sha256=digest(reg),source_identity_sha256=digest(reg['source_identity']))
        class ProofBoundary(Exception):pass
        def boundary(protocol,root):
            self.assertEqual(protocol,p);raise ProofBoundary()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manager=SimpleNamespace(lock_path=root/'test.lock')
            with patch.object(runtime,'OperationsV2',return_value=manager),patch.object(runtime,'verify_current',return_value=(manager,reg)),patch.object(held_goal_profile,'verify_held_goal_proof',boundary):
                with self.assertRaises(ProofBoundary):runtime._supervise(p,root=root,termination=None)
            self.assertFalse((root/'.research_local/authorized_rl_v1/head.json').exists())
