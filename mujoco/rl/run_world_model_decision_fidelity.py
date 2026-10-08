"""Offline fixed-candidate audit. No optimizer, new controller or cost tuning."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
import torch
from decision_fidelity_candidates import reconstruct
from decision_fidelity_rollouts import true_rollout,scripted_reference,frozen_prediction
from decision_fidelity_metrics import candidate_metrics,summary,correlation
from latent_dynamics_data import file_hash,json_hash
from latent_mpc_core import planning_cost
from joint_latent_world_model import component_hashes
from run_latent_random_shooting_mpc import context,execution_hardware,write_json
from ppo_pi_env import MineUAVPIEnv

BASE='d565cb8f3f85dccef1da4d5a9f713cf2f4142433'
BRANCH='feat/world-model-decision-cost-fidelity'
CHECKPOINT='42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9'
CONDITIONS=('unconstrained','train_supported')
SPLITS=('benchmark','holdout')
REPORT='world_model_decision_cost_fidelity_seed0.json'
MANIFEST='decision_cost_fidelity_manifest_seed0.json'


def audit_context(root):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    ctx=context(root); directory=ctx['root']/'mujoco/reports'
    ctx['previous_report']=json.loads((directory/'latent_mpc_train_supported_action_control_seed0.json').read_text())
    ctx['support']=json.loads((directory/'train_action_support_central95.json').read_text())
    if file_hash(ctx['checkpoint'])!=CHECKPOINT or ctx['support']!=ctx['previous_report']['train_support']:
        raise AssertionError('canonical checkpoint/support changed')
    for path,sha in ctx['previous_report']['immutable_artifacts'].items():
        if file_hash(path)!=sha: raise AssertionError(f'previous execution source changed: {path}')
    immutable=dict(ctx['previous_report']['immutable_artifacts'])
    for name in ('decision_fidelity_snapshot.py','decision_fidelity_candidates.py','decision_fidelity_rollouts.py',
                 'decision_fidelity_metrics.py',Path(__file__).name):
        path=Path(__file__).with_name(name); immutable[str(path)]=file_hash(path)
    for name in ('latent_mpc_train_supported_action_control_seed0.json','train_action_support_central95.json'):
        path=directory/name; immutable[str(path)]=file_hash(path)
    identity=dict(checkpoint_sha256=CHECKPOINT,components=ctx['before'],normalization_sha256=ctx['normalization_sha256'],
        dataset_manifest_sha256=ctx['dataset_manifest_sha256'],support_sha256=ctx['support']['sha256'],
        targets={s:ctx['target_meta'][s]['target_sha256'] for s in SPLITS},sources=immutable,
        hardware=execution_hardware(),N=512,H=10,seed_rule='original episode seed; first draw only',
        protocol='all200 first-decision resets; full MjData/Python snapshot; terminal absorbing after original done; raw original cost no new penalty')
    ctx['identity']=identity
    return ctx


def validate_cache(saved,identity,split,index):
    if saved['identity']!=identity: raise ValueError('cached executable/model/hardware identity differs')
    if (saved['row']['split'],saved['row']['index'])!=(split,index): raise ValueError('cached initial-state identity differs')


def verify_cohort(rows):
    required={(s,i) for s in SPLITS for i in range(100)}
    keys=[(r['split'],r['index']) for r in rows]
    if len(keys)!=200 or set(keys)!=required: raise AssertionError('all200 unique initial states required')
    for r in rows:
        if set(r['conditions'])!=set(CONDITIONS) or any(r['conditions'][c]['candidate_count']!=512 for c in CONDITIONS):
            raise AssertionError('both complete original512 candidate sets required')
    return 204800


def finite(value):
    if isinstance(value,dict): return {k:finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [finite(v) for v in value]
    if isinstance(value,np.ndarray): return finite(value.tolist())
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
    if isinstance(value,np.generic): return value.item()
    return value


def audit_state(ctx,env,split,index,parts):
    started=time.monotonic(); first=reconstruct(ctx,env,split,index); snapshot=first['snapshot']
    scripted=scripted_reference(env,snapshot)
    replay=true_rollout(env,snapshot,scripted['actions'])
    np.testing.assert_array_equal(scripted['observations'],replay['observations'])
    if scripted['final_snapshot_sha256']!=replay['final_snapshot_sha256']: raise AssertionError('scripted replay differs')
    script_terminal=frozen_prediction(ctx['model'],ctx['stats'],first['z'],scripted['actions'][None])
    script_cost=float(planning_cost(script_terminal,scripted['actions'][None],first['previous_action'])[0][0])
    d0=float(snapshot.environment['previous_distance']); arrays={}; conditions={}
    for c in CONDITIONS:
        actions=first['conditions'][c]['candidates']
        predicted=frozen_prediction(ctx['model'],ctx['stats'],first['z'],actions)
        pred_cost=planning_cost(predicted,actions,first['previous_action'])[0]
        np.testing.assert_array_equal(pred_cost,first['conditions'][c]['costs'])
        costs=[]; terminal=[]; distances=[]; speeds=[]; failures=[]; invalid=[]; steps=[]; reasons=[]; end_hashes=[]
        for a in actions:
            result=true_rollout(env,snapshot,a)
            costs.append(result['cost']); terminal.append(result['observations'][-1]); distances.append(result['terminal_distance'])
            speeds.append(result['terminal_speed']); failures.append(result['physical_failure']); invalid.append(result['invalid_state'])
            steps.append(result['executed_steps']); reasons.append(result['termination_reason'] or 'none'); end_hashes.append(result['final_snapshot_sha256'])
        values=dict(pred_cost=np.asarray(pred_cost),true_cost=np.asarray(costs),terminal=np.asarray(terminal),pred_terminal=predicted,
            distance=np.asarray(distances),speed=np.asarray(speeds),failure=np.asarray(failures),invalid=np.asarray(invalid),
            executed_steps=np.asarray(steps),termination_reason=np.asarray(reasons,dtype='U32'),end_hash=np.asarray(end_hashes,dtype='U64'))
        metrics=candidate_metrics(values['pred_cost'],values['true_cost'],values['terminal'],predicted,values['distance'],
            values['speed'],values['failure'],d0,scripted,script_cost,ctx['stats'])
        metrics.update(candidate_count=512,invalid_count=int(np.sum(invalid)),early_termination_count=int(np.sum(np.asarray(steps)<10)),
            termination_reason_counts={r:reasons.count(r) for r in sorted(set(reasons))})
        # Independent restore/repeat of model-selected and true-oracle sequences.
        for j in {metrics['pred_best_index'],metrics['true_best_index']}:
            repeated=true_rollout(env,snapshot,actions[j])
            if repeated['cost']!=costs[j] or repeated['final_snapshot_sha256']!=end_hashes[j]: raise AssertionError('candidate restore determinism failed')
            np.testing.assert_array_equal(repeated['observations'][-1],terminal[j])
        conditions[c]=metrics; arrays.update({f'{c}_{k}':v for k,v in values.items()})
    arrays.update(scripted_actions=scripted['actions'],scripted_observations=scripted['observations'],scripted_pred_terminal=script_terminal)
    path=parts/f'{split}_{index:03d}.npz'; np.savez_compressed(path,**arrays)
    row=dict(split=split,index=index,env_seed=first['env_seed'],target=first['target'],initial_observation=first['observation'].tolist(),
        initial_distance=d0,snapshot_sha256=first['snapshot_sha256'],reconstruction=first['evidence'],conditions=conditions,
        scripted={k:v for k,v in scripted.items() if k not in ('actions','observations','executed_actions')},
        scripted_pred_cost=script_cost,scripted_h10_nrmse=float(np.sqrt(np.mean(((script_terminal[0]-scripted['observations'][-1])/ctx['stats']['obs']['std'])**2))),
        local_arrays=dict(path=str(path),sha256=file_hash(path)),elapsed_seconds=time.monotonic()-started,
        deterministic_replays_verified=True)
    if component_hashes(ctx['model'])!=ctx['before']: raise AssertionError('frozen parameters changed')
    return finite(row)


def worker_init(root,parts,identity):
    global WORKER_CONTEXT,WORKER_ENV,WORKER_PARTS
    WORKER_CONTEXT=audit_context(root); WORKER_ENV=MineUAVPIEnv(reward_version='v2'); WORKER_PARTS=Path(parts)
    if WORKER_CONTEXT['identity']!=identity: raise AssertionError('worker identity mismatch')


def worker_run(key):
    split,index=key
    row=audit_state(WORKER_CONTEXT,WORKER_ENV,split,index,WORKER_PARTS)
    write_json(WORKER_PARTS/f'{split}_{index:03d}.json',dict(identity=WORKER_CONTEXT['identity'],row=row))
    return row


def aggregate(rows):
    output={}
    for c in CONDITIONS:
        output[c]={}
        for split in (*SPLITS,'all'):
            selected=[r for r in rows if split=='all' or r['split']==split]; metrics=[r['conditions'][c] for r in selected]
            keys=('spearman','kendall','regret','normalized_regret','random_expected_regret','random_normalized_regret',
                  'model_selected_true_rank_percentile','true_best_margin','pred_best_margin','true_cost_spread',
                  'calibration_pearson','calibration_mae','calibration_rmse')
            out={k:summary([m[k] for m in metrics]) for k in keys}
            for key in ('top1_exact','pred_best_in_true_top5','true_best_in_pred_top5','model_better_than_random'):
                out[key]=dict(count=sum(m[key] for m in metrics),states=len(metrics),fraction=float(np.mean([m[key] for m in metrics])))
            for key in ('oracle','model_selected'):
                out[key]={k:summary([m[key][k] for m in metrics]) for k in ('true_cost','terminal_distance','terminal_speed','distance_reduction')}
                out[key].update(distance_improved_fraction=float(np.mean([m[key]['distance_improved'] for m in metrics])),
                    failure_fraction=float(np.mean([m[key]['physical_failure'] for m in metrics])),
                    speed_below_success_threshold_fraction=float(np.mean([m[key]['terminal_speed_below_success_threshold'] for m in metrics])))
            out['scripted']={k:summary([r['scripted'][k] for r in selected]) for k in ('cost','terminal_distance','terminal_speed','distance_reduction')}
            out['scripted']['physical_failure_fraction']=float(np.mean([r['scripted']['physical_failure'] for r in selected]))
            out['scripted']['distance_improved_fraction']=float(np.mean([r['scripted']['distance_reduction']>1e-6 for r in selected]))
            out['scripted']['prediction_h10_nrmse']=summary([r['scripted_h10_nrmse'] for r in selected])
            for key in ('script_true_percentile','script_pred_percentile'):
                out[key]={k:summary([m[key][k] for m in metrics]) for k in ('cost_percentile','strictly_better_than_fraction')}
                out[key]['beats_every_candidate_fraction']=float(np.mean([m[key]['beats_every_candidate'] for m in metrics]))
            out['cost_task_alignment']={k:summary([m['cost_task_alignment'][k] for m in metrics]) for k in ('distance','distance_reduction','speed')}
            out['prediction_error']={k:summary([m['prediction_error'][k] for m in metrics]) for k in ('whole_set_h10_nrmse','selected_h10_nrmse')}
            out['candidate_failure_count']=sum(round(m['candidate_failure_fraction']*512) for m in metrics)
            out['invalid_count']=sum(m['invalid_count'] for m in metrics)
            out['early_termination_count']=sum(m['early_termination_count'] for m in metrics)
            out['state_count']=len(selected); out['candidate_count']=512*len(selected)
            relationships={}
            for error in ('whole_set_h10_nrmse','selected_h10_nrmse'):
                for metric in ('spearman','normalized_regret'):
                    pairs=[(m['prediction_error'][error],m[metric]) for m in metrics if m[metric] is not None]
                    x,y=zip(*pairs) if pairs else ([],[])
                    relationships[f'{error}_vs_{metric}']={method:correlation(x,y,method) for method in ('pearson','spearman')}
            out['prediction_vs_ranking']=relationships
            out['margin_vs_ranking']={k:{method:correlation([m[k] for m in metrics],[m['normalized_regret'] for m in metrics],method)
                for method in ('pearson','spearman')} for k in ('true_best_margin','true_cost_spread')}
            order=sorted(metrics,key=lambda m:m['true_cost_spread']); out['true_spread_quartiles']=[]
            for group in np.array_split(np.arange(len(order)),4):
                subset=[order[i] for i in group]
                out['true_spread_quartiles'].append(dict(count=len(subset),spread=summary([m['true_cost_spread'] for m in subset]),
                    spearman=summary([m['spearman'] for m in subset]),normalized_regret=summary([m['normalized_regret'] for m in subset])))
            output[c][split]=out
    return output


def finalize(ctx,rows,workers,elapsed):
    total=verify_cohort(rows); directory=ctx['root']/'mujoco/reports'; identity=ctx['identity']
    if component_hashes(ctx['model'])!=ctx['before'] or any(file_hash(p)!=sha for p,sha in identity['sources'].items()):
        raise AssertionError('immutable artifact/parameter changed during audit')
    states=[{k:r[k] for k in ('split','index','env_seed','target','snapshot_sha256','reconstruction','local_arrays')} for r in rows]
    manifest=dict(base_commit=BASE,identity_sha256=json_hash(identity),state_count=200,candidate_rollout_count=total,
        states=states,states_sha256=json_hash(states),reconstruction='old raw arrays/snapshots were not archived; original deterministic reset/sampler/source replay, exact saved obs/latent/epsilon/action/selected index/min/median/mean cost checked for all200; new candidate hashes only')
    write_json(directory/MANIFEST,manifest)
    report=dict(experiment='Offline World Model Decision-Cost Fidelity Audit',branch=BRANCH,base_branch='feat/latent-mpc-action-support',base_commit=BASE,
        model=dict(checkpoint=str(ctx['checkpoint']),checkpoint_sha256=CHECKPOINT,components_before=ctx['before'],
            components_after=component_hashes(ctx['model']),frozen=True,no_optimizer=True),
        identity=identity,normalization=ctx['stats'],normalization_sha256=ctx['normalization_sha256'],targets=ctx['target_meta'],
        configuration=dict(N=512,H=10,physics_hz=500,control_hz=100,policy_hz=25,
            cost=ctx['previous_report']['configuration']['cost'],sampling=ctx['previous_report']['configuration']['sampling'],
            support=ctx['support'],failure_penalty='NONE: original terminal observation cost retained with failure flags; terminal observation absorbing after original done; all512 retained',
            statistics='Pearson; Spearman via average tied ranks; Kendall tau-b; stable argmin/Top5; analytical mean random true cost; perstate then aggregate',
            rank_percentiles='model selected average rank:0best1worst; scripted empirical mid-CDF:0best1worst; separate strictly-better fraction',
            snapshot='full independent MjData copy/mj_copyData plus all mutable env/task/RNG fields, full PI/nested attitude/yaw controller and allocator dicts, previous executed action; compiled model immutable'),
        manifest=dict(path=str(directory/MANIFEST),sha256=file_hash(directory/MANIFEST)),results=aggregate(rows),states=rows,
        execution=dict(workers=workers,wall_seconds=elapsed,per_state_seconds=summary([r['elapsed_seconds'] for r in rows])),
        verification=dict(all200_states=True,all204800_candidates=True,all_first_decisions_exactly_reconstructed=True,
            all_selected_oracle_scripted_replays_exact=True,no_parameter_mutation=True),
        limitations=['First-decision nominal reset cohort only; no new closed-loop controller run.',
            'Finite horizon10 and original candidate sets/cost fixed; no causal uniqueness claim.',
            'Scripted future commands use true feedback to generate reference sequence, then same sequence supplied open loop to model.',
            'Original raw candidate/snapshot arrays not archived; documented deterministic reconstruction, not invented historical raw hashes.',
            'No failure penalty existed; original raw terminal cost and physical flags reported separately. Invalid original observations sanitized by environment, never silently exclude candidates.',
            'Candidate cost/endpoints local-only with hashes; no large trajectory arrays committed.'])
    write_json(directory/REPORT,report)
    return report


def run(root,workers=8):
    ctx=audit_context(root); parts=ctx['root']/'mujoco/reports/world_model_decision_cost_fidelity_seed0_parts'
    parts.mkdir(parents=True,exist_ok=True); started=time.monotonic(); rows=[]; pending=[]
    for split in SPLITS:
        for index in range(100):
            path=parts/f'{split}_{index:03d}.json'
            if path.exists():
                saved=json.loads(path.read_text()); validate_cache(saved,ctx['identity'],split,index)
                row=saved['row']; artifact=row['local_arrays']
                if file_hash(artifact['path'])!=artifact['sha256']: raise ValueError('local arrays changed')
                rows.append(row)
            else: pending.append((split,index))
    print(f'Reuse {len(rows)}/200; evaluate {len(pending)} states with {workers} independent workers',flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=worker_init,initargs=(str(ctx['root']),str(parts),ctx['identity'])) as pool:
        futures={pool.submit(worker_run,key):key for key in pending}
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows)%10==0: print(f'Completed {len(rows)}/200; elapsed {time.monotonic()-started:.1f}s',flush=True)
    rows.sort(key=lambda r:(SPLITS.index(r['split']),r['index']))
    report=finalize(ctx,rows,workers,time.monotonic()-started)
    print(f'Saved {REPORT}',flush=True); return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args(); run(args.root,args.workers)
