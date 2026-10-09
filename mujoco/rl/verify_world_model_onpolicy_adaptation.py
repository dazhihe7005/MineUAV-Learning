"""Independent full-cohort recomputation plus fixed exact-physics repeats."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from run_onpolicy_adaptation_data import collection_context,MANIFEST,TARGETS,CHECKPOINT
from onpolicy_adaptation_data import validate_splits,exclusion_identities,load_dataset
from onpolicy_adaptation_training import selected_step,validation,batch_order
from onpolicy_adaptation_final_data import FINAL_MANIFEST,require_completed_training
from onpolicy_adaptation_evaluation import evaluate_state
from onpolicy_adaptation_analysis import aggregate,ood_correlations
from onpolicy_adaptation_collection import collect_visited
from joint_autonomous_consistency_training import load_autonomous
from joint_latent_world_model import component_hashes,encode_episode
from latent_dynamics_data import file_hash,json_hash
from latent_mpc_core import planning_cost
from latent_mpc_support_evaluation import array_hash
from decision_fidelity_rollouts import true_rollout,scripted_reference
from ppo_pi_env import MineUAVPIEnv
from run_latent_random_shooting_mpc import write_json
from decision_fidelity_metrics import summary
from later_ranking_core import ood_scores

def check_training(r):
    assert r['final_step']==1000 and r['batch_size']==16 and r['learning_rate']==.0003 and r['parameter_count']==74119
    assert r['best_step']==selected_step(r['history'])
    assert all(r['initial_hashes'][k]!=r['best_hashes'][k] for k in r['initial_hashes'])
    assert [h['step'] for h in r['history']]==list(range(50,1001,50))

def check_provenance(value):
    """Visit nested artifacts too, including both full training records."""
    if isinstance(value,dict):
        if {'path','sha256'}<=set(value):assert file_hash(value['path'])==value['sha256']
        for child in value.values():check_provenance(child)
    elif isinstance(value,list):
        for child in value:check_provenance(child)

def check_report_sources(report,data,final,training):
    check_provenance(report['provenance'])
    for source,r in training.items():
        expected={k:v for k,v in r.items() if k!='steps'}
        for output,key in [('gradient_norms','gradient_norms'),('consistency_gradient_ratios','consistency_to_observation_gradient_ratios')]:
            expected[output]={m:summary([s[key][m] for s in r['steps']]) for m in ('encoder','transition','decoder')}
        expected['stability']={k:max(s[k] for s in r['steps']) for k in ('normalized_prediction_abs_max','latent_norm_max')}
        assert expected==report['training'][source]
    assert report['final_source_episodes']==final['episodes']
    assert report['adaptation_source_episodes']==data['source_episodes']
    for stage in ('S2','S3'):
        groups=report['aggregate'][stage]['models']
        replay=groups['replay']['prediction']['whole']['10']['mean'];mpc=groups['mpc_state']['prediction']['whole']['10']['mean']
        assert report['conclusion']['later_prediction_improvement_vs_replay_percent'][stage]==100*(1-mpc/replay)
    for source in ('replay','mpc_state'):
        baseline=report['nominal']['original']['50']['normalized_observation_rmse']
        adapted=report['nominal'][source]['50']['normalized_observation_rmse']
        assert report['conclusion']['nominal_h50_regression_vs_original_percent'][source]==100*(adapted/baseline-1)

@torch.inference_mode()
def check_state_ood(stored,arrays,ctx,geometry):
    previous=np.vstack([np.zeros((1,4),np.float32),arrays['prefix_actions']])
    z=encode_episode(ctx['model'],arrays['prefix_observations'],previous,ctx['stats'])[-1].numpy()
    expected=ood_scores(z,arrays['prefix_observations'][-1],previous[-1],geometry)
    assert stored==expected

def check_common_states(rows):
    groups={}
    for r in rows:groups.setdefault((r['target_id'],r['stage']),[]).append(r)
    for g in groups.values():
        assert len(g)==3 and {r['model'] for r in g}=={'original','replay','mpc_state'}
        assert len({(r['candidate_sha256'],r['true_cost_sha256'],r['state_array_sha256']) for r in g})==1

def check_data_labels(data,splits):
    for source,sets in data['dataset'].items():
        assert source in ('replay','mpc_state') and set(sets)=={'train','val'}
        for split,states in sets.items():
            ids={r['target_id'] for r in splits[split]}
            assert all(r['target_id'] in ids and r['source']==source and r['split']==split for r in states)

def verify(root):
    ctx=collection_context(root);root=ctx['root'];reports=root/'mujoco/reports';parts=reports/'world_model_onpolicy_adaptation_seed0_parts'
    report=json.loads((reports/'world_model_onpolicy_adaptation_seed0.json').read_text());data=json.loads((reports/MANIFEST).read_text())
    final=json.loads((reports/FINAL_MANIFEST).read_text());target=json.loads((reports/TARGETS).read_text())
    old,seeds=exclusion_identities(ctx);validate_splits(target['splits'],old,seeds)
    check_data_labels(data,target['splits'])
    assert data['identity']==ctx['identity']==final['identity']['collection']
    check_provenance(report['provenance'])
    for name,h in report['provenance']['source_sha256'].items():assert file_hash(Path(__file__).with_name(name))==h
    training={s:json.loads((parts/f'training_{s}.json').read_text()) for s in ('replay','mpc_state')}
    require_completed_training(training,{s:file_hash(r['checkpoint']['path'])==r['checkpoint']['sha256'] for s,r in training.items()})
    check_report_sources(report,data,final,training)
    geometry_meta=report['provenance']['train_distribution']
    previous_audit=json.loads((reports/'world_model_later_decision_ranking_audit_seed0.json').read_text())
    assert geometry_meta==previous_audit['geometry']
    geometry=json.loads(Path(geometry_meta['path']).read_text());geometry=geometry.get('geometry',geometry)
    original_hashes=component_hashes(ctx['model']);scale=load_autonomous(ctx['checkpoint'])[2]['initial_latent_std']
    for source,r in training.items():
        check_training(r);assert r['initial_hashes']==original_hashes
        assert len(r['steps'])==1000 and [x['step'] for x in r['steps']]==list(range(1,1001))
        assert r['normalization_sha256']==json_hash(ctx['stats']) and r['scale_sha256']==json_hash(scale)
        assert r['provenance']['data_manifest_sha256']==file_hash(reports/MANIFEST)
        assert r['provenance']['training_source_sha256']==file_hash(Path(__file__).with_name('onpolicy_adaptation_training.py'))
        order_hash=hashlib.sha256()
        for ids in batch_order(6400,16,1000,0):order_hash.update(ids.astype('<i8').tobytes())
        assert order_hash.hexdigest()==r['order_sha256']
        model,stats,meta=load_autonomous(r['checkpoint']['path']);model.requires_grad_(False)
        assert component_hashes(model)==r['best_hashes'] and meta['best_step']==r['best_step']
        datasets={s:load_dataset(data['dataset'][source][s]) for s in ('train','val')}
        assert len(datasets['train'])==6400 and len(datasets['val'])==1600
        valid=validation(model,datasets['val'],stats,scale)
        np.testing.assert_allclose([valid[k] for k in valid],[r['best_validation'][k] for k in valid],atol=1e-12,rtol=0)
    assert len(final['episodes'])==100 and {e['target_id'] for e in final['episodes']}=={e['target_id'] for e in target['splits']['final']}
    rows=report['evaluation_rows'];check_common_states(rows)
    lookup={(r['target_id'],r['stage'],r['model']):r for r in rows};state_keys={(s['target_id'],s['stage']) for s in final['states']}
    assert set(lookup)=={(*s,m) for s in state_keys for m in report['models']}
    for name,model_meta in report['models'].items():
        assert file_hash(model_meta['path'])==model_meta['sha256']
        model,stats,_=load_autonomous(model_meta['path']);model.eval().requires_grad_(False);before=component_hashes(model)
        for s in final['states']:
            assert file_hash(s['path'])==s['sha256'] and s['repeat_exact']
            with np.load(s['path'],allow_pickle=False) as raw:a={k:raw[k].copy() for k in raw.files}
            previous=a['prefix_actions'][-1] if len(a['prefix_actions']) else np.zeros(4,np.float32)
            truecost,_=planning_cost(a['truth'][:,-1],a['candidates'],previous)
            np.testing.assert_array_equal(truecost,a['true_costs'])
            assert array_hash(a['candidates'])==s['candidate_sha256'] and array_hash(truecost)==s['true_cost_sha256']
            metrics,cost=evaluate_state(model,stats,a,s);stored=lookup[(s['target_id'],s['stage'],name)]
            assert metrics==stored['metrics'] and array_hash(cost)==stored['predicted_cost_sha256']
            check_state_ood(stored['state_ood'],a,ctx,geometry)
            if name=='original':assert metrics['pred_best_index']==s['original_selected_index']
        assert component_hashes(model)==before==model_meta['parameter_hashes']
    assert aggregate(rows)==report['aggregate'] and ood_correlations(rows)==report['state_ood_correlations']
    independently_run_nominal=json.loads((parts/'nominal_results.json').read_text())
    for name,value in independently_run_nominal['models'].items():
        assert value['checkpoint_sha256']==report['models'][name]['sha256']
        assert value['prediction']==report['nominal'][name] and value['latent']==report['latent_diagnostics'][name]
    assert independently_run_nominal['test_manifest']==report['provenance']['nominal_manifest']
    # Fixed identity, not outcome chosen: first fresh Final target, all3 stages.
    env=MineUAVPIEnv(reward_version='v2')
    try:
        first=target['splits']['final'][0];trajectory=collect_visited(ctx,env,'mpc_state',first,keep_all_candidates=True)
        archived=next(e for e in final['episodes'] if e['target_id']==first['target_id'])
        assert trajectory['episode']==archived
        for s in final['states']:
            if s['target_id']!=first['target_id']:continue
            d=trajectory['decisions'][s['decision_index']];assert d['snapshot_sha256']==s['snapshot_sha256']
            with np.load(s['path'],allow_pickle=False) as raw:
                np.testing.assert_array_equal(d['candidates'],raw['candidates'])
                one=true_rollout(env,d['snapshot'],d['candidates'][0]);two=true_rollout(env,d['snapshot'],d['candidates'][0])
                np.testing.assert_array_equal(one['observations'],two['observations']);np.testing.assert_array_equal(one['observations'][1:],raw['truth'][0])
                assert one['final_snapshot_sha256']==two['final_snapshot_sha256']
                script=scripted_reference(env,d['snapshot']);np.testing.assert_array_equal(script['actions'],raw['script_actions'])
                np.testing.assert_array_equal(script['observations'][1:],raw['script_truth'])
    finally:env.close()
    assert file_hash(ctx['checkpoint'])==CHECKPOINT and component_hashes(ctx['model'])==original_hashes
    result=dict(passed=True,target_groups=180,source_episodes=160,final_episodes=100,shared_states=len(final['states']),
        candidate_rollouts=len(final['states'])*512,all_candidate_costs_recomputed=True,all_three_model_metrics_recomputed=True,
        checkpoint_hash_unchanged=True,initial_hashes_paired=True,training_budget_paired=True,selection_validation_only=True,
        exact_repeat='collector every accepted training snapshot and every Finalstate candidate0; verifier firstFinaltarget all3states candidate0 +script',
        best_validation_recomputed=True,normalization_unchanged=True,complete_validation_schedule=True,
        nested_provenance_verified=True,training_report_recomputed=True,all_state_ood_recomputed=True,
        numeric_conclusions_recomputed=True,source_episode_summaries_verified=True)
    write_json(parts/'verification.json',result);print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args();verify(a.root)
