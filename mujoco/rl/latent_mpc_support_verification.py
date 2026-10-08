"""Read-only replay of cohort/support/noise/hash/prefix evidence, not planning tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from latent_dynamics_data import file_hash,json_hash
from joint_latent_world_model import component_hashes
from joint_autonomous_consistency_training import load_autonomous
from latent_mpc_evaluation import load_targets,summarize_episodes,realized_prefix_check,timing_summary
from latent_mpc_verification import close,verify_episode_cohort
from latent_mpc_support_evaluation import support_metrics,window_manifest,verify_pairing,prediction_metrics,array_hash
from run_latent_mpc_action_support import load_arrays,paired_rows,REPORT


def verify_shape(results):
    if set(results)!={'unconstrained','train_supported'}: raise AssertionError('exactly two control conditions required')
    total=0
    for group in results.values():
        if set(group)!={'benchmark','holdout'}: raise AssertionError('missing/extra target split')
        for result in group.values():
            if len(result['episodes'])!=100: raise AssertionError('100targets per condition/split required')
            total+=len(result['episodes'])
    return total


def first_epsilon_hash(split,index,std):
    seed=int(np.random.SeedSequence([0,('benchmark','holdout').index(split),index]).generate_state(1)[0])
    epsilon=np.random.default_rng(seed).normal(size=(510,10,4))*(.5*np.asarray(std,np.float64))
    return hashlib.sha256(epsilon.astype('<f8').tobytes()).hexdigest()


def verify(root):
    root=Path(root); report=json.loads((root/'mujoco/reports'/REPORT).read_text()); count=verify_shape(report['results'])
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model,stats,_=load_autonomous(report['model']['checkpoint']); model.eval().requires_grad_(False)
    before=component_hashes(model)
    if before!=report['model']['components_before'] or before!=report['model']['components_after']: raise AssertionError('component parameters mutated')
    if file_hash(report['model']['checkpoint'])!=report['model']['checkpoint_sha256']: raise AssertionError('checkpoint changed')
    if json_hash(stats)!=report['normalization_sha256']: raise AssertionError('normalization changed')
    for p,h in report['immutable_artifacts'].items():
        if file_hash(p)!=h: raise AssertionError(f'execution source changed: {p}')
    support=report['train_support']
    if support['split']!='train' or Path(support['source']).name!='train.npz': raise AssertionError('support not Train-only')
    if file_hash(support['source'])!=support['source_sha256']: raise AssertionError('support Train source mutated')
    with np.load(support['source'],allow_pickle=False) as raw: commands=np.asarray(raw['actions'],np.float64)
    np.testing.assert_array_equal(np.quantile(commands,.025,axis=0),support['lower'])
    np.testing.assert_array_equal(np.quantile(commands,.975,axis=0),support['upper'])
    if len(commands)!=support['count']: raise AssertionError('support omitted executed actions')
    if json_hash({k:v for k,v in support.items() if k!='sha256'})!=support['sha256']: raise AssertionError('support metadata hash changed')
    close(support,json.loads((root/'mujoco/reports/train_action_support_central95.json').read_text()))
    targets,_=load_targets(root); arrays={}; checks=0; future_invariant=True
    for condition,group in report['results'].items():
        arrays[condition]={}
        for split,value in group.items():
            verify_episode_cohort(value['episodes'],targets[split]); close(summarize_episodes(value['episodes']),value['summary'])
            a=load_arrays(value); arrays[condition][split]=a
            n=sum(r['episode_steps'] for r in value['episodes'])
            if len(a['actions'])!=n or len(a['epsilon_hashes'])!=n or np.abs(a['actions']).max()>1: raise AssertionError('action/noise/bounds count mismatch')
            offsets=np.r_[0,np.cumsum([r['episode_steps'] for r in value['episodes']])]
            np.testing.assert_array_equal(a['offsets'],offsets)
            close(support_metrics(a['actions'],report['noise_normalization'],support),value['action_support'])
            if condition=='train_supported' and value['action_support']['outside_train_95_any_dimension_fraction']!=0:
                raise AssertionError('supported selected action outside box')
            for raw,key in [('planning_seconds','planning_time'),('decision_seconds','total_decision_time')]:
                if len(a[raw])!=n or not np.isfinite(a[raw]).all() or (a[raw]<=0).any(): raise AssertionError('missing/nonfinite timing')
                close(timing_summary(a[raw]),value[key])
            if (a['costs'][:,0]>a['costs'][:,1]+1e-9).any() or (a['costs'][:,0]>a['costs'][:,2]+1e-9).any(): raise AssertionError('cost selection not minimum')
            for i,row in enumerate(value['episodes']):
                if row['first_epsilon_sha256']!=first_epsilon_hash(split,i,stats['action']['std']): raise AssertionError('firstdecision raw epsilon differs from original sampler')
                np.testing.assert_array_equal(row['first_action'],a['actions'][offsets[i]])
                if row['first_selected_cost']!=a['costs'][offsets[i],0]: raise AssertionError('first cost differs')
            pred=[]; actual=[]; manifest=[]
            if [v['episode_index'] for v in value['representative_local_traces']]!=[0,1,2]: raise AssertionError('prediction episodes outcome-selected')
            for record in value['representative_local_traces']:
                if file_hash(record['path'])!=record['sha256']: raise AssertionError('prefix trace changed')
                with np.load(record['path'],allow_pickle=False) as raw: trace={k:raw[k] for k in raw.files}
                if len(trace['observations'])!=len(trace['actions'])+1: raise AssertionError('observation/action indexing')
                np.testing.assert_array_equal(trace['previous_actions'][0],np.zeros(4))
                np.testing.assert_array_equal(trace['previous_actions'][1:],trace['actions'][:-1])
                result=realized_prefix_check(model,stats,trace)
                changed=realized_prefix_check(model,stats,dict(trace,observations=np.zeros_like(trace['observations'])))
                np.testing.assert_array_equal(result['predicted'],changed['predicted'])
                np.testing.assert_array_equal(result['predicted'],trace['check_predicted']); np.testing.assert_array_equal(result['actual'],trace['check_actual'])
                windows=window_manifest(split,record['episode_index'],len(trace['actions']))
                if result['starts'].tolist()!=[w['decision_index'] for w in windows]: raise AssertionError('illegal/changed window')
                manifest.extend(windows); pred.append(result['predicted']); actual.append(result['actual']); checks+=len(result['starts'])
                ep=value['episodes'][record['episode_index']]
                if ep['first_latent_sha256']!=array_hash(trace['latents'][0]) or ep['first_observation_sha256']!=array_hash(trace['observations'][0]): raise AssertionError('initial stored hash mismatch')
            if manifest!=value['window_manifest'] or json_hash(manifest)!=value['window_manifest_sha256']: raise AssertionError('prefix manifest/hash changed')
            close(prediction_metrics(np.concatenate(pred),np.concatenate(actual),stats),value['prediction_sanity'])
    matched=0
    for split in targets:
        u=report['results']['unconstrained'][split]; s=report['results']['train_supported'][split]
        paired=verify_pairing(paired_rows(u,arrays['unconstrained'][split]),paired_rows(s,arrays['train_supported'][split]))
        close(paired,report['paired_rng_and_initial_state'][split]); matched+=paired['matched_noise_decisions']
    for condition in arrays:
        actions=np.concatenate([a['actions'] for a in arrays[condition].values()])
        close(support_metrics(actions,report['noise_normalization'],support),report['overall'][condition]['action_support'])
        for raw,key in [('planning_seconds','planning_time'),('decision_seconds','total_decision_time')]:
            close(timing_summary(np.concatenate([a[raw] for a in arrays[condition].values()])),report['overall'][condition][key])
    if component_hashes(model)!=before: raise AssertionError('read-only verifier mutated model')
    return dict(verified=True,episodes=count,matched_raw_epsilon_decisions=matched,
        realized_prefix_windows_recomputed=checks,all_future_observation_perturbation_predictions_bit_identical=future_invariant,
        checkpoint_and_all_three_parameter_hashes_unchanged=True,full_train_support_quantiles_reproduced=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); args=parser.parse_args()
    print(json.dumps(verify(args.root),indent=2))
