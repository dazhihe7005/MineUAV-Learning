"""Recompute labels, indices, splits, scaling, selection, metrics and real replay."""
import json
from pathlib import Path
import numpy as np
import torch
from latent_dynamics_data import file_hash,json_hash
from test_env_scripted_policy import scripted_action
from uav_bc_data import PARTS,MANIFEST,SPLITS,identity,split_targets,validate_record,load_split,collect_episode
from uav_bc_policy import load,fit_stats,parameter_hash
from uav_bc_training import MODEL,select_epoch
from uav_bc_evaluation import EVALUATION,CANONICAL,action_metrics,summarize,closed_loop_identity
from uav_bc_safety import atomic_json

def verify_arrays(trace,row,expert=False):
    n=row['steps'];x=trace['observations'];a=trace['actions']
    assert x.shape==(n+1,7) and a.shape==(n,4)
    assert np.isfinite(x).all() and np.isfinite(a).all() and (np.abs(a)<=1).all()
    np.testing.assert_array_equal(trace['step_index'],np.arange(n))
    np.testing.assert_array_equal(trace['previous_actions'][0],np.zeros(4))
    np.testing.assert_array_equal(trace['previous_actions'][1:],a[:-1])
    for key,value in [('target_id',row['target_id']),('episode_id',row['episode_id']),('success',row['success']),('failure',row['failure'])]:
        np.testing.assert_array_equal(trace[key],np.full(n,value))
    if expert:np.testing.assert_array_equal(a,np.asarray([scripted_action(o) for o in x[:-1]]))

def mse_contract(prediction,target,std):return float(np.mean(((np.asarray(prediction,float)-target)/std)**2))

def verify(root):
    from ppo_pi_env import MineUAVPIEnv
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS;torch.set_num_threads(1)
    manifest=json.loads((reports/MANIFEST).read_text());groups,targets,_=split_targets(root)
    assert manifest['identity']==identity(root)
    split=json.loads((reports/SPLITS).read_text());assert split['groups']==groups and split['sha256']==json_hash(groups)
    assert file_hash(reports/SPLITS)==manifest['split_manifest']['sha256']
    counts={};seen=set()
    for name,rows in manifest['records'].items():
        expected={t['target_id']:t for t in groups[name]}
        assert len(rows)==len(expected)
        for row in rows:
            assert row['target_id'] not in seen;seen.add(row['target_id'])
            assert row['env_seed']==expected[row['target_id']]['env_seed'] and row['target']==expected[row['target_id']]['target']
            validate_record(row,manifest['identity'],row['target_id'])
            with np.load(row['path'],allow_pickle=False) as trace:verify_arrays(trace,row,expert=True)
        counts[name]=dict(episodes=len(rows),steps=sum(r['steps'] for r in rows),successes=sum(r['success'] for r in rows))
    assert counts==manifest['counts'] and len(seen)==120
    training=json.loads((parts/'training.json').read_text());policy,meta=load(root/'mujoco/rl/models'/MODEL)
    train=load_split(manifest,'train');val=load_split(manifest,'val');test=load_split(manifest,'test')
    assert policy.stats==fit_stats(train['observations'],train['actions'])==training['stats']
    assert training['epochs_completed']==100 and len(training['history'])==100
    assert meta['max_epochs']==100 and meta['batch_size']==512 and meta['learning_rate']==.001 and meta['seed']==0
    assert training['best_epoch']==meta['best_epoch']==select_epoch(training['history'])
    assert meta['provenance']['dataset_manifest_sha256']==file_hash(reports/MANIFEST)
    assert meta['provenance']['training_source_sha256']==file_hash(root/'mujoco/rl/uav_bc_training.py')
    assert meta['provenance']['policy_source_sha256']==file_hash(root/'mujoco/rl/uav_bc_policy.py')
    assert parameter_hash(policy.actor)==training['best_parameter_hash']
    for name,values,key in [('train',train,'best_train_loss'),('validation',val,'best_validation_loss')]:
        raw=policy.raw_actions(values['observations'])
        np.testing.assert_allclose(mse_contract(raw,values['actions'],policy.stats['action']['std']),training[key],rtol=2e-5,atol=1e-9)
    evaluation=json.loads((parts/'evaluation.json').read_text());eval_manifest=json.loads((reports/EVALUATION).read_text())
    assert eval_manifest['records']==evaluation['records']
    checkpoint=root/'mujoco/rl/models'/MODEL;sha=file_hash(checkpoint);ident=closed_loop_identity(root,sha)
    assert eval_manifest['identity']==ident
    for condition,tasks in evaluation['records'].items():
        for task,rows in tasks.items():
            assert len(rows)==100 and [r['target_id'] for r in rows]==list(range(100))
            for row,(seed,target) in zip(rows,targets[task]):
                assert row['env_seed']==seed and row['target']==target and row['condition']==condition and row['task']==task
                validate_record(row,ident,row['target_id'])
                with np.load(row['path'],allow_pickle=False) as trace:
                    verify_arrays(trace,row,expert=condition=='scripted')
                    if condition=='bc':np.testing.assert_allclose(trace['actions'],np.clip(policy.raw_actions(trace['observations'][:-1]),-1,1),atol=2e-7,rtol=2e-6)
            summary=summarize(rows)
            assert all(evaluation['closed_loop'][condition][task][k]==v for k,v in summary.items())
    metrics=action_metrics(policy.raw_actions(test['observations']),test['actions'],policy.stats['action']['std'])
    assert evaluation['offline']['raw']==metrics
    before=parameter_hash(policy.actor);env=MineUAVPIEnv(reward_version='v2')
    try:
        for condition,controller in [('scripted',scripted_action),('bc',policy.predict)]:
            r=evaluation['records'][condition]['benchmark'][0]
            row,trace=collect_episode(env,dict(target_id=0,env_seed=r['env_seed'],target=r['target']),controller)
            assert row['termination_reason']==r['termination_reason'] and row['steps']==r['steps']
            with np.load(r['path'],allow_pickle=False) as original:
                for k,v in trace.items():np.testing.assert_array_equal(v,original[k])
    finally:env.close()
    assert sha==file_hash(checkpoint) and before==parameter_hash(policy.actor)
    assert file_hash(root/'mujoco/rl/models/joint_latent_world_model_v3_autonomous_consistency.pt')==CANONICAL
    result=dict(passed=True,expert_episodes_verified=120,closed_loop_episodes_verified=400,
        deterministic_exact_episode_replays=2,expert_labels_bitwise_equal=True,
        all_record_indices_and_outcomes_verified=True,train_only_statistics_verified=True,
        no_split_overlap=True,validation_selection_verified=True,offline_metrics_recomputed=True,
        independent_bc_actions_verified=True,bc_checkpoint_unchanged=True,canonical_world_model_unchanged=True)
    atomic_json(parts/'verification.json',result);print(json.dumps(result,indent=2));return result

if __name__=='__main__':verify(Path(__file__).resolve().parents[2])
