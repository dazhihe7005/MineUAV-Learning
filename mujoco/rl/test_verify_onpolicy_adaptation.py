"""Corruption tests on actual protocol guards, not report-only assertions."""
import copy,importlib.util,json,unittest
from pathlib import Path
import numpy as np
from onpolicy_adaptation_final_data import require_completed_training

class VerificationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('verify_world_model_onpolicy_adaptation'))
        import verify_world_model_onpolicy_adaptation as api
        return api
    def test_training_budget_and_selection_corruption_rejected(self):
        api=self.api()
        r=dict(final_step=1000,batch_size=16,learning_rate=.0003,parameter_count=74119,best_step=50,
            initial_hashes={'encoder':'a','transition':'b','decoder':'c'},best_hashes={'encoder':'d','transition':'e','decoder':'f'},
            history=[dict(step=s,validation=dict(observation=s/50.)) for s in range(50,1001,50)])
        api.check_training(r)
        for bad in (dict(r,final_step=999),dict(r,best_step=100),dict(r,best_hashes=r['initial_hashes'])):
            with self.assertRaises(AssertionError):api.check_training(bad)
        with self.assertRaises(AssertionError):api.check_training(dict(r,history=r['history'][:1]))

    def archived(self):
        root=Path(__file__).resolve().parents[2];reports=root/'mujoco/reports'
        parts=reports/'world_model_onpolicy_adaptation_seed0_parts'
        if not (parts/'training_mpc_state.json').exists():self.skipTest('local-only experiment archives not available')
        report=json.loads((reports/'world_model_onpolicy_adaptation_seed0.json').read_text())
        data=json.loads((reports/'onpolicy_adaptation_data_manifest_seed0.json').read_text())
        final=json.loads((reports/'onpolicy_adaptation_final_manifest_seed0.json').read_text())
        training={s:json.loads((parts/f'training_{s}.json').read_text()) for s in ('replay','mpc_state')}
        return root,report,data,final,training

    def test_archived_training_conclusion_and_source_corruption_rejected(self):
        api=self.api();root,report,data,final,training=self.archived()
        self.assertTrue(callable(getattr(api,'check_report_sources',None)))
        api.check_report_sources(report,data,final,training)
        bad=copy.deepcopy(report);bad['training']['mpc_state']['best_validation']['observation']=123456
        with self.assertRaises(AssertionError):api.check_report_sources(bad,data,final,training)
        bad=copy.deepcopy(report);bad['conclusion']['nominal_h50_regression_vs_original_percent']['mpc_state']=-99
        with self.assertRaises(AssertionError):api.check_report_sources(bad,data,final,training)
        bad=copy.deepcopy(report);bad['final_source_episodes'][0]['steps']=-1
        with self.assertRaises(AssertionError):api.check_report_sources(bad,data,final,training)
        bad=copy.deepcopy(report);bad['provenance']['training_records']['replay']['sha256']='0'*64
        with self.assertRaises(AssertionError):api.check_report_sources(bad,data,final,training)

    def test_archived_ood_corruption_rejected(self):
        api=self.api();root,report,_,final,_=self.archived()
        self.assertTrue(callable(getattr(api,'check_state_ood',None)))
        ctx=api.collection_context(root)
        geometry=json.loads((root/'mujoco/reports/later_decision_train_distribution_seed0.json').read_text())
        geometry=geometry.get('geometry',geometry)
        state=next(s for s in final['states'] if s['stage']=='S3')
        with np.load(state['path'],allow_pickle=False) as raw:arrays={k:raw[k].copy() for k in raw.files}
        row=next(r for r in report['evaluation_rows'] if r['stage']=='S3' and r['target_id']==state['target_id'])
        api.check_state_ood(row['state_ood'],arrays,ctx,geometry)
        with self.assertRaises(AssertionError):api.check_state_ood(dict(row['state_ood'],latent_norm=123456),arrays,ctx,geometry)
    def test_common_candidate_cost_guard(self):
        api=self.api()
        rows=[dict(target_id='a',stage='S3',model=m,candidate_sha256='a',true_cost_sha256='b',state_array_sha256='c') for m in ('original','replay','mpc_state')]
        api.check_common_states(rows)
        bad=[dict(r) for r in rows];bad[2]['true_cost_sha256']='bad'
        with self.assertRaises(AssertionError):api.check_common_states(bad)
        with self.assertRaises(AssertionError):api.check_common_states(rows[:-1])
    def test_raw_manifest_cannot_assign_final_targets_to_training(self):
        api=self.api();splits=dict(train=[dict(target_id='train')],val=[dict(target_id='val')],final=[dict(target_id='test')])
        data=dict(dataset={s:{split:[dict(target_id=split,split=split,source=s)] for split in ('train','val')} for s in ('replay','mpc_state')})
        api.check_data_labels(data,splits)
        data['dataset']['mpc_state']['train'][0]['target_id']='test'
        with self.assertRaises(AssertionError):api.check_data_labels(data,splits)

if __name__=='__main__':unittest.main()
