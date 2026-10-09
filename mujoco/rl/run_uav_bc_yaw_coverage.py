"""Independent read-only integrity gate and scalar report publication."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from latent_dynamics_data import file_hash,json_hash
from uav_bc_safety import atomic_json
from uav_bc_policy import load,parameter_hash
from uav_bc_yaw_data import PARTS,MANIFEST,MODEL_NAMES,MODEL_SHA,check_record,load_selected,phase_labels,validate_expert_trace
from uav_bc_robustness import validate_record,validate_cohort
from uav_bc_yaw_evaluation import CONTROLLERS,summaries,extra_metrics

REPORT='uav_bc_yaw_coverage_seed0.json'

def verify_rollout_metrics(r,z):
    valid=z['physical_state_valid'];o=z['observations'];actions=z['actions']
    d=np.linalg.norm(o[:,:3],axis=1);v=np.linalg.norm(o[:,3:6],axis=1);yaw=abs(o[:,6]);n=len(actions)
    if r['steps']!=n or r['success']!=(r['termination_reason']=='success') or r['timeout']!=(r['termination_reason']=='time_limit'):
        raise ValueError('outcome/step labels corrupted')
    if r['completion_time_s']!=(r['simulated_seconds'] if r['success'] else None):raise ValueError('failed completion not null')
    physical_failure=r['termination_reason'] not in ('success','time_limit')
    if r['physical_failure']!=physical_failure:raise ValueError('physical-failure label inconsistent')
    elapsed=r['simulated_seconds']
    if physical_failure:
        # The last action may terminate before all twenty500Hz physics ticks;
        # all preceding actions must still account for their full40ms.
        if not np.isfinite(elapsed) or not (max(0,n-1)*.04-1e-8<=elapsed<=n*.04+1e-8):
            raise ValueError('physical-failure duration outside final policy interval')
        np.testing.assert_allclose(elapsed,round(elapsed/.002)*.002,atol=1e-8,rtol=0)
    else:np.testing.assert_allclose(elapsed,n*.04,atol=1e-8,rtol=0)
    np.testing.assert_allclose(r['maximum_speed_m_s'],v[valid].max(),atol=1e-7,rtol=0)
    np.testing.assert_allclose(r['max_position_excursion_m'],np.linalg.norm(z['positions'][valid]-z['positions'][0],axis=1).max(),atol=1e-7,rtol=0)
    np.testing.assert_allclose(r['minimum_distance_m'],d[valid].min(),atol=1e-7,rtol=0)
    near=d[:-1]<.1
    if r['near_steps']!=int(near.sum()) or r['action_saturation_elements']!=int((abs(actions)>=.95).sum()):raise ValueError('near/saturation counts inconsistent')
    np.testing.assert_allclose(r['near_actual_sum'],float(v[:-1][near].sum()),atol=1e-7,rtol=0)
    np.testing.assert_allclose(r['near_command_sum'],float(np.linalg.norm(actions[near,:3]*[1.5,1.5,1.],axis=1).sum()),atol=1e-7,rtol=0)
    if valid[-1]:
        np.testing.assert_allclose(r['final_distance_m'],d[-1],atol=1e-6,rtol=0)
        np.testing.assert_allclose(r['final_speed_m_s'],v[-1],atol=1e-6,rtol=0)
        np.testing.assert_allclose(r['final_abs_yaw_error_rad'],yaw[-1],atol=1e-7,rtol=0)
    elif r['final_distance_m'] is not None or r['final_speed_m_s'] is not None or r['final_abs_yaw_error_rad'] is not None:
        raise ValueError('terminal placeholders treated as measurements')

def verify_selected_arrays(a,records,source,expected_ids):
    if len(set(a['sample_key']))!=len(a['sample_key']) or set(a['target_id'])!=set(expected_ids):raise ValueError('duplicate sample/target omission/leakage')
    cache={}
    for i,key in enumerate(a['sample_key']):
        episode,j=str(key).rsplit('/',1);j=int(j);r=records[episode]
        if r['source']!=source:raise ValueError('cross-source label leakage')
        if episode not in cache:
            if file_hash(r['trace_path'])!=r['trace_sha256']:raise ValueError('raw expert hash mismatch')
            with np.load(r['trace_path'],allow_pickle=False) as z:cache[episode]={k:z[k].copy() for k in z.files}
        z=cache[episode]
        if not z['physical_state_valid'][j] or j>=len(z['actions']):raise ValueError('invalid/preterminal sample indexing')
        np.testing.assert_array_equal(a['observations'][i],z['observations'][j]);np.testing.assert_array_equal(a['actions'][i],z['actions'][j])
        if a['step_index'][i]!=j or a['target_id'][i]!=r['target_id'] or a['yaw_group'][i]!=r['yaw_group'] or a['initial_yaw_degrees'][i]!=r['initial_yaw_degrees']:
            raise ValueError('selected sample labels mismatched')
        if a['phase'][i]!=phase_labels(z['observations'][j:j+1],np.array([j]))[0]:raise ValueError('state/time stratum mismatch')

def interpret(metrics):
    def delta(c):return metrics[c]['yaw_augmented']['successes']-metrics[c]['nominal_expanded']['successes']
    nominal=metrics['nominal']['yaw_augmented']['successes'];yaw=metrics['yaw_large']['yaw_augmented']['successes']
    return dict(engineering_targets_met=nominal>=98 and yaw>=90,nominal_regression=nominal<98,
        control_comparison_yaw30_delta=delta('yaw_large'),control_comparison_yaw10_delta=delta('yaw_small'),
        position_velocity_regression_gt5pp=[c for c in ['position_small','position_large','velocity_small','velocity_large'] if delta(c)<-5],
        support='paired yaw-coverage evidence' if delta('yaw_large')>0 and delta('yaw_small')>=0 else 'yaw coverage not consistently beneficial',
        next_step='A single fixed-policy persistent external-disturbance recovery evaluation; not executed.'
            if nominal>=98 and yaw>=90 else 'A single strictly isolated DAgger coverage-control study for remaining off-expert-state failures; not executed.',
        inference='Only this fixed-seed data intervention is tested; no unique causal proof, population guarantee, or training-budget equivalence with Original BC.')

def verify(root):
    import torch
    from uav_bc_yaw_data import exclusion_targets,assert_isolated
    from uav_bc_yaw_training import select_best,array_hash
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS;models=root/'mujoco/rl/models'
    m=json.loads((reports/MANIFEST).read_text());d=json.loads((parts/'dataset.json').read_text());training=json.loads((parts/'training.json').read_text())
    ev=json.loads((parts/'evaluation.json').read_text());em=json.loads((parts/'evaluation_manifest.json').read_text())
    if json_hash({k:v for k,v in m.items() if k!='sha256'})!=m['sha256'] or d['manifest_sha256']!=m['sha256']:raise ValueError('immutable dataset design/hash changed')
    for n,h in m['identity']['sources'].items():
        if file_hash(root/'mujoco/rl'/n)!=h:raise ValueError('data generation source changed')
    for n,h in m['identity']['environment']['source_sha256'].items():
        if file_hash(root/'mujoco'/n)!=h:raise ValueError('environment/expert source changed')
    for n,h in m['target_sources'].items():
        if file_hash(reports/n)!=h:raise ValueError('original target sources changed')
    bc=json.loads((reports/'uav_bc_target_splits_seed0.json').read_text());adapt=json.loads((reports/'onpolicy_adaptation_target_splits_seed0.json').read_text())
    assert_isolated(m['final_targets'],exclusion_targets(bc,adapt))
    if set(t['target_id'] for t in m['train_target_groups'])&set(t['target_id'] for t in m['val_target_groups']):raise ValueError('Train/Val overlap')
    rmap={r['key']:r for r in d['records']};states={r['key']:r for r in m['expert_states']}
    if len(rmap)!=1860 or rmap.keys()!=states.keys():raise ValueError('expert episode missing/duplicate')
    for key,r in rmap.items():
        check_record(r,m['sha256'],key);s=states[key]
        for k in ['source','split','target_id','env_seed','yaw_group','yaw_sign','initial_yaw_degrees','perturbation']:
            if r[k]!=s[k]:raise ValueError('expert cache relabeled')
        if r['initial_snapshot_sha256']!=s['initial_state']['snapshot_sha256']:raise ValueError('expert initial state changed')
        with np.load(r['trace_path'],allow_pickle=False) as z:validate_expert_trace(z)
    for n,count in [('nominal_expanded',20000),('yaw_augmented',20000),('shared_validation',4000)]:
        a=load_selected(root,n);ids={t['target_id'] for t in m['val_target_groups'] if n=='shared_validation'} if n=='shared_validation' else {t['target_id'] for t in m['train_target_groups']}
        verify_selected_arrays(a,rmap,n,ids)
        if len(a['actions'])!=count or json_hash(a['sample_key'].tolist())!=d['selected'][n]['sample_key_sha256']:raise ValueError('selection size/hash changed')
        if dict(Counter(a['phase']))!=d['selected'][n]['phases'] or dict(Counter(str(g) for g in a['yaw_group']))!=d['selected'][n]['yaw_groups']:raise ValueError('selection quotas misreported')
    a,b=training.values();torch.manual_seed(0)
    from uav_bc_policy import BCActor
    initial=parameter_hash(BCActor())
    if any(v['initial_parameter_hash']!=initial or v['optimizer_updates']!=4000 or len(v['history'])!=100 or select_best(v['history'])!=v['best_epoch'] for v in training.values()):
        raise ValueError('random initialization/budget/Val selector invalid')
    if [r['order_sha256'] for r in a['history']]!=[r['order_sha256'] for r in b['history']]:raise ValueError('paired schedule differs')
    normalization=m['preflight']['normalization_sha256']
    paths={'original':models/'uav_bc_mlp_seed0.pt',**{n:models/f for n,f in MODEL_NAMES.items()}}
    for n,path in paths.items():
        p,meta=load(path)
        if file_hash(path)!=ev['frozen_hashes_before'][n] or parameter_hash(p.actor)!=ev['parameter_hashes_before'][n] or json_hash(p.stats)!=normalization:
            raise ValueError('model/hash/normalization changed')
        if n in MODEL_NAMES and (meta['best_epoch']!=training[n]['best_epoch'] or file_hash(path)!=training[n]['checkpoint_sha256']):raise ValueError('best checkpoint mismatch')
    if file_hash(paths['original'])!=MODEL_SHA or ev['frozen_hashes_before']!=ev['frozen_hashes_after'] or ev['parameter_hashes_before']!=ev['parameter_hashes_after']:raise ValueError('frozen reference changed')
    if json_hash({k:v for k,v in em.items() if k!='sha256'})!=em['sha256'] or em['sha256']!=ev['evaluation_manifest_sha256']:raise ValueError('final manifest modified')
    for n,h in ev['identity']['source'].items():
        if file_hash(root/'mujoco/rl'/n)!=h:raise ValueError('evaluation source changed')
    rows=ev['rows'];smap={s['key']:s for s in em['states']};validate_cohort(rows,[k+'/'+n for k in smap for n in CONTROLLERS]);pairs={}
    for r in rows:
        s=smap[r['key']];validate_record(r,ev['identity'],s,r['controller']);pairs.setdefault(r['key'],[]).append(r['initial_snapshot_sha256'])
        with np.load(r['trace_path'],allow_pickle=False) as z:
            verify_rollout_metrics(r,z)
            np.testing.assert_allclose(np.asarray(r['target'])-z['positions'][z['physical_state_valid']],z['observations'][z['physical_state_valid'],:3],atol=1e-6,rtol=0)
            if r['controller']=='scripted':validate_expert_trace(z)
            if any(r[k]!=v for k,v in extra_metrics(z).items()):raise ValueError('extra metrics mismatch')
            if r['success'] and (z['success_streak'][-1]!=5 or not np.all(np.linalg.norm(z['observations'][-5:,:3],axis=1)<.1) or not np.all(np.linalg.norm(z['observations'][-5:,3:6],axis=1)<.15)):
                raise ValueError('changed success hold')
    if len(pairs)!=900 or any(len(p)!=4 or len(set(p))!=1 for p in pairs.values()):raise ValueError('unequal/omitted paired snapshots')
    if summaries(rows)!=ev['metrics']:raise ValueError('reported summaries inconsistent')
    value=dict(passed=True,expert_episodes=1860,unique_train_B=20000,unique_train_C=20000,common_validation=4000,
        expert_label_indexing_verified=True,all_100_final_targets_isolated=True,paired_final_states=900,controller_episodes=3600,
        all_cache_and_checkpoint_hashes_verified=True,no_duplicate_or_missing_tasks=True,initial_hashes_identical=True,optimizer_updates_each=4000,
        batch_schedule_identical=True,normalization_unchanged=True,original_checkpoint_unchanged=True)
    atomic_json(parts/'verification.json',value);return value

def publish(root):
    from uav_bc_yaw_plots import plots
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS
    verified=verify(root);m=json.loads((reports/MANIFEST).read_text());d=json.loads((parts/'dataset.json').read_text())
    training=json.loads((parts/'training.json').read_text());ev=json.loads((parts/'evaluation.json').read_text());em=json.loads((parts/'evaluation_manifest.json').read_text())
    figures=plots(root,m,d,training,ev);resources={p.stem.removeprefix('resource_'):json.loads(p.read_text()) for p in parts.glob('resource_*.json') if p.stem!='resource_publish'}
    for phase in ['collection','training','smoke','evaluation','resume','tests']:
        r=resources[phase]
        if r['exit_code'] or r['failure'] or r['oom_count_delta']:raise ValueError('unsafe/incomplete phase')
    report=dict(experiment='CONTROLLED YAW-COVERAGE BEHAVIOR CLONING',date='2026-10-09',seed=0,
        branch='feat/uav-bc-yaw-coverage',base_commit=m['base_commit'],preflight=m['preflight'],config=m['config'],
        architecture='7→128ReLU→128ReLU→4linear;18052params;unchanged normalized action limits[-1,1]',
        original_model_reference=dict(sha256=MODEL_SHA,frozen=True,original_train_samples=m['preflight']['original_train_samples'],
            training_budget_note='Original8974samples/100epochs≈1800updates; B/C both20000samples/100epochs/4000updates, Original is historical reference not the causal budget-matched control'),
        data=dict(selected=d['selected'],expert_episode_outcomes={n:dict(Counter(r['termination_reason'] for r in d['records'] if r['source']==n)) for n in d['selected']},
            expert_yaw_episode_outcomes={n:{str(g):dict(Counter(r['termination_reason'] for r in d['records'] if r['source']==n and r['yaw_group']==g)) for g in (0,10,30)} for n in ('yaw_augmented','shared_validation')},
            all_failures_retained=True,paired_base_states=True,subsequent_trajectories_not_assumed_identical=True,
            note='B actualyaw0; shadow yaw_group is only a matched sampling stratum, NOT real yaw coverage'),
        training=training,offline_test=ev['offline_test'],metrics=ev['metrics'],interpretation=interpret(ev['metrics']),
        frozen_hashes_before=ev['frozen_hashes_before'],frozen_hashes_after=ev['frozen_hashes_after'],
        parameter_hashes_before=ev['parameter_hashes_before'],parameter_hashes_after=ev['parameter_hashes_after'],
        verification=verified,manifest=dict(path='mujoco/reports/'+MANIFEST,file_sha256=file_hash(reports/MANIFEST),semantic_sha256=m['sha256'],
            final_decision_manifest_sha256=em['sha256']),final_states=em['states'],
        episodes=[{k:v for k,v in r.items() if k not in ('identity','trace_path')} for r in ev['rows']],
        resource_safety=dict(workers=1,oom_count_delta=sum(r['oom_count_delta'] for r in resources.values()),
            peak_sampled_process_tree_rss_bytes=max(r['process_tree_peak_sampled_rss_bytes'] for r in resources.values()),
            peak_process_tree_hwm_bytes=max(r['process_tree_peak_hwm_bytes'] for r in resources.values()),
            minimum_available_bytes=min(r['minimum_available_bytes'] for r in resources.values()),swap_modified=False,phases=resources,
            note='sampled RSS and summed per-process HWM, not hard OS guarantee'),tests=json.loads((parts/'test_evidence.json').read_text()),
        figures={name:file_hash(reports/name) for name in figures},local_only=dict(directory='mujoco/reports/'+PARTS,
            raw_datasets_and_trajectories=True,resume_optimizer_checkpoints=True,temporary_logs=True,
            expert_records_sha256=json_hash([r['record_sha256'] for r in d['records']]),final_records_sha256=json_hash([r['record_sha256'] for r in ev['rows']])),
        limitations=['one training seed and100newfixedtargets','shared small position/velocity variations in B/C expert collection',
            'phase-stratified training frequency differs from natural visitation','same moving low-level PI/task, nominal MuJoCo only',
            'all B/C normalization fixed to small-yaw original Train, large normalized scales explicitly retained',
            'same-observation expert sufficiency is not full physical/controller Markov sufficiency','no DAgger/PPO/SAC/planning or follow-on experiment'])
    atomic_json(reports/REPORT,report);return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[2]
    print(json.dumps(verify(root) if a.verify_only else publish(root)['interpretation'],indent=2))
