"""Read-only later-decision exact-state audit of the archived MPC policy."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
import torch
from run_world_model_decision_fidelity import audit_context,finite,CHECKPOINT
from run_latent_random_shooting_mpc import write_json
from latent_dynamics_data import file_hash,json_hash,normalize
from joint_latent_world_model import component_hashes
from decision_fidelity_rollouts import true_rollout,scripted_reference
from decision_fidelity_metrics import candidate_metrics
from latent_mpc_core import planning_cost
from later_ranking_core import fit_geometry,ood_scores,prediction_trajectory
from later_ranking_replay import replay_episode
from later_ranking_analysis import CONDITIONS,SPLITS,aggregate,validate_cohort
from ppo_pi_env import MineUAVPIEnv

BASE='e21e7c0cff7e18ce1e4e8074bd9a323e387b30e5'
BRANCH='feat/world-model-later-decision-ranking'
REPORT='world_model_later_decision_ranking_audit_seed0.json'
MANIFEST='later_decision_ranking_manifest_seed0.json'
GEOMETRY='later_decision_train_distribution_seed0.json'

def task_distance(snapshot):
    return float(snapshot.environment['previous_distance'])

@torch.inference_mode()
def build_train_geometry(ctx):
    source=next(Path(p) for p in ctx['identity']['sources'] if p.endswith('/train.npz'))
    with np.load(source,allow_pickle=False) as raw:
        inputs=raw['inputs'].copy(); actions=raw['actions'].copy(); offsets=raw['offsets'].copy()
    latents=[]
    for lo,hi in zip(offsets[:-1],offsets[1:]):
        x=np.concatenate([normalize(inputs[lo:hi,:7],ctx['stats'],'obs'),normalize(inputs[lo:hi,7:],ctx['stats'],'previous_action')],1)
        z,_=ctx['model'].encoder(torch.from_numpy(x)[None],None); latents.append(z[0].numpy())
    geometry=fit_geometry(np.concatenate(latents),inputs[:,:7],actions,'train')
    geometry['provenance'].update(source=str(source),source_sha256=file_hash(source),encoder_sha256=ctx['before']['encoder'],
        checkpoint_sha256=CHECKPOINT,episodes=len(offsets)-1,encoding='complete real Train episodes, reset hidden each episode, frozen GRU')
    return geometry

def later_context(root,geometry=None):
    ctx=audit_context(root)
    if geometry is None:
        geometry=build_train_geometry(ctx)
        write_json(ctx['root']/'mujoco/reports'/GEOMETRY,geometry)
    ctx['geometry']=geometry
    sources=dict(ctx['identity']['sources'])
    for name in ('later_ranking_core.py','later_ranking_replay.py','later_ranking_analysis.py',Path(__file__).name):
        path=Path(__file__).with_name(name); sources[str(path)]=file_hash(path)
    for result in ctx['previous_report']['results'].values():
        for part in result.values():
            for art in [part['aggregate'],*part['representative_local_traces']]:
                if file_hash(art['path'])!=art['sha256']: raise AssertionError('original trajectory artifact changed')
                sources[art['path']]=art['sha256']
    old=ctx['root']/'mujoco/reports/world_model_decision_cost_fidelity_seed0.json'
    sources[str(old)]=file_hash(old)
    ctx['identity'].update(sources=sources,geometry_sha256=json_hash(geometry),
        seed_rule='original full episode RNG stream; every original decision reproduced',
        protocol='original400 episodes; distinct nearest0/25/50/75% decisions; all512 original candidates per state, same H10/cost, no new policy')
    return ctx

def audit_decision(ctx,env,d,episode,condition,split,index,parts):
    from later_ranking_core import tail_metrics
    snapshot=d['snapshot']; actions=d['plan']['candidates']; z=d['z']; previous=d['previous_action']
    predicted=prediction_trajectory(ctx['model'],ctx['stats'],z,actions)
    pred_cost=planning_cost(predicted[:,-1],actions,previous)[0]
    np.testing.assert_array_equal(pred_cost,d['plan']['costs'])
    script=scripted_reference(env,snapshot)
    spred=prediction_trajectory(ctx['model'],ctx['stats'],z,script['actions'][None])[0]
    scost=float(planning_cost(spred[-1:],script['actions'][None],previous)[0][0])
    true_cost=[]; terminals=[]; dist=[]; speed=[]; failure=[]; invalid=[]; reason=[]; steps=[]; hashes=[]; trajectory_errors=[]
    for i,a in enumerate(actions):
        out=true_rollout(env,snapshot,a)
        true_cost.append(out['cost']); terminals.append(out['observations'][-1]); dist.append(out['terminal_distance']); speed.append(out['terminal_speed'])
        failure.append(out['physical_failure']); invalid.append(out['invalid_state']); reason.append(out['termination_reason'] or 'none')
        steps.append(out['executed_steps']); hashes.append(out['final_snapshot_sha256'])
        trajectory_errors.append(float(np.sqrt(np.mean(((predicted[i]-out['observations'][1:])/ctx['stats']['obs']['std'])**2))))
    values=dict(pred_cost=pred_cost,true_cost=np.asarray(true_cost),terminal=np.asarray(terminals),pred_terminal=predicted[:,-1],
        distance=np.asarray(dist),speed=np.asarray(speed),failure=np.asarray(failure),invalid=np.asarray(invalid),
        executed_steps=np.asarray(steps),termination_reason=np.asarray(reason,dtype='U32'),end_hash=np.asarray(hashes,dtype='U64'),
        trajectory_nrmse=np.asarray(trajectory_errors))
    distance=task_distance(snapshot)
    m=candidate_metrics(pred_cost,values['true_cost'],values['terminal'],predicted[:,-1],values['distance'],values['speed'],values['failure'],
        distance,script,scost,ctx['stats'])
    m.update(candidate_count=512,invalid_count=int(sum(invalid)),early_termination_count=int(np.sum(np.asarray(steps)<10)),
        termination_reason_counts={r:reason.count(r) for r in sorted(set(reason))})
    ip=m['pred_best_index']; it=m['true_best_index']
    if ip!=d['plan']['index']: raise AssertionError('original selected candidate changed')
    error=(predicted[:,-1]-values['terminal'])/ctx['stats']['obs']['std']
    prediction=dict(m['prediction_error'],candidate_mean_h10_nrmse=float(np.sqrt(np.mean(error**2,1)).mean()),
        true_best_h10_nrmse=float(np.sqrt(np.mean(error[it]**2))),scripted_h10_nrmse=float(np.sqrt(np.mean(((spred[-1]-script['observations'][-1])/ctx['stats']['obs']['std'])**2))),
        whole_set_trajectory_nrmse=float(np.sqrt(np.mean(values['trajectory_nrmse']**2))),
        selected_trajectory_nrmse=trajectory_errors[ip],true_best_trajectory_nrmse=trajectory_errors[it],
        scripted_trajectory_nrmse=float(np.sqrt(np.mean(((spred-script['observations'][1:])/ctx['stats']['obs']['std'])**2))))
    for j in {ip,it}:
        repeat=true_rollout(env,snapshot,actions[j])
        if repeat['cost']!=true_cost[j] or repeat['final_snapshot_sha256']!=hashes[j]: raise AssertionError('exact candidate restore failed')
        np.testing.assert_array_equal(repeat['observations'][-1],terminals[j])
    repeat=true_rollout(env,snapshot,script['actions'])
    np.testing.assert_array_equal(repeat['observations'],script['observations'])
    if repeat['final_snapshot_sha256']!=script['final_snapshot_sha256']: raise AssertionError('scripted exact-state repeat failed')
    path=parts/f'{condition}_{split}_{index:03d}_{d["stage"]}.npz'
    np.savez_compressed(path,**values,scripted_actions=script['actions'],scripted_observations=script['observations'],scripted_predicted=spred)
    row=dict(condition=condition,split=split,index=index,stage=d['stage'],decision_index=d['decision_index'],
        realized_progress=d['decision_index']/max(1,episode['episode_steps']-1),episode_steps=episode['episode_steps'],
        termination_reason=episode['termination_reason'],env_seed=episode['env_seed'],target=episode['target'],
        initial_distance=episode['initial_distance'],observation=d['observation'].tolist(),previous_action=previous.tolist(),
        selected_action=d['plan']['action'].tolist(),latent=z[0].tolist(),distance=distance,speed=float(np.linalg.norm(d['observation'][3:6])),
        pi_magnitude=d['pi_magnitude'],snapshot_sha256=d['snapshot_sha256'],candidate_sha256=d['candidate_sha256'],epsilon_sha256=d['epsilon_sha256'],
        planner_rng_before_sha256=d['planner_rng_before_sha256'],planner_rng_after_sha256=d['planner_rng_after_sha256'],
        metrics=m,best_tail=tail_metrics(pred_cost,values['true_cost']),prediction=prediction,
        ood=ood_scores(z,d['observation'],d['plan']['action'],ctx['geometry']),
        previous_action_standardized_rms=float(np.sqrt(np.mean(((previous-ctx['stats']['action']['mean'])/ctx['stats']['action']['std'])**2))),
        scripted={k:v for k,v in script.items() if k not in ('actions','observations','executed_actions')},scripted_pred_cost=scost,
        candidate_invalid_count=int(sum(invalid)),candidate_physical_failure_count=int(sum(failure)),candidate_early_termination_count=int(np.sum(np.asarray(steps)<10)),
        local_arrays=dict(path=str(path),sha256=file_hash(path)),repeat_verified=True)
    return finite(row)

def worker_init(root,parts,identity,geometry):
    global CTX,ENV,PARTS
    CTX=later_context(root,geometry); ENV=MineUAVPIEnv(reward_version='v2'); PARTS=Path(parts)
    if CTX['identity']!=identity: raise AssertionError('worker scientific execution identity mismatch')

def worker_run(key):
    condition,split,index=key; start=time.monotonic()
    replay=replay_episode(CTX,ENV,condition,split,index)
    episode=dict(replay['episode'],condition=condition,split=split,index=index,
        initial_distance=float(np.linalg.norm(replay['decisions']['S0']['observation'][:3])),replay_verification=replay['verification'])
    records=[audit_decision(CTX,ENV,d,episode,condition,split,index,PARTS) for d in replay['decisions'].values()]
    if component_hashes(CTX['model'])!=CTX['before']: raise AssertionError('frozen model changed')
    result=dict(identity=CTX['identity'],episode=episode,records=records,wall_seconds=time.monotonic()-start)
    write_json(PARTS/f'{condition}_{split}_{index:03d}.json',result)
    return result

def run(root,workers=8,only=None):
    ctx=later_context(root); parts=ctx['root']/'mujoco/reports/world_model_later_decision_ranking_seed0_parts'
    parts.mkdir(parents=True,exist_ok=True); rows=[]; episodes=[]; pending=[]; start=time.monotonic()
    keys=[(c,s,i) for c in CONDITIONS for s in SPLITS for i in range(100)] if only is None else [tuple(only)]
    for key in keys:
        path=parts/f'{key[0]}_{key[1]}_{int(key[2]):03d}.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['identity']!=ctx['identity']: raise ValueError('cached execution identity changed, do not relabel stale results')
            for r in saved['records']:
                if file_hash(r['local_arrays']['path'])!=r['local_arrays']['sha256']: raise ValueError('local candidate costs changed')
            episodes.append(saved['episode']); rows.extend(saved['records'])
        else: pending.append((key[0],key[1],int(key[2])))
    print(f'Reuse{len(episodes)}/{len(keys)} episodes; pending{len(pending)}; independent workers={workers}',flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,
            initargs=(str(ctx['root']),str(parts),ctx['identity'],ctx['geometry'])) as pool:
        futures={pool.submit(worker_run,k):k for k in pending}
        for f in as_completed(futures):
            result=f.result(); episodes.append(result['episode']); rows.extend(result['records'])
            if len(episodes)%5==0 or only: print(f'Complete{len(episodes)}/{len(keys)} episodes,{len(rows)} states; {time.monotonic()-start:.1f}s',flush=True)
    if only: return rows
    rows.sort(key=lambda r:(CONDITIONS.index(r['condition']),SPLITS.index(r['split']),r['index'],r['stage']))
    episodes.sort(key=lambda e:(CONDITIONS.index(e['condition']),SPLITS.index(e['split']),e['index']))
    count=validate_cohort(rows,episodes)
    if component_hashes(ctx['model'])!=ctx['before'] or any(file_hash(p)!=sha for p,sha in ctx['identity']['sources'].items()):
        raise AssertionError('model/source/artifact mutation during audit')
    old=json.loads((ctx['root']/'mujoco/reports/world_model_decision_cost_fidelity_seed0.json').read_text())
    for r in rows:
        if r['stage']!='S0': continue
        previous=next(s['conditions'][r['condition']] for s in old['states'] if s['split']==r['split'] and s['index']==r['index'])
        if r['metrics']!=previous: raise AssertionError('S0 audit fails exact previous metric reproduction')
    manifest=dict(base_commit=BASE,identity_sha256=json_hash(ctx['identity']),episode_count=400,state_count=count,candidate_rollout_count=count*512,
        stage_rule='nearest floor(progress*(actual decision length-1)+0.5); duplicate indices excluded, earlier stratum retained',
        episodes=[{k:e[k] for k in ('condition','split','index','episode_steps','termination_reason','replay_verification')} for e in episodes],
        states=[{k:r[k] for k in ('condition','split','index','stage','decision_index','snapshot_sha256','candidate_sha256','epsilon_sha256',
            'planner_rng_before_sha256','planner_rng_after_sha256','local_arrays')} for r in rows])
    manifest['states_sha256']=json_hash(manifest['states']); directory=ctx['root']/'mujoco/reports'; write_json(directory/MANIFEST,manifest)
    report=dict(experiment='Later-decision exact-state ranking audit',branch=BRANCH,base_branch='feat/world-model-decision-cost-fidelity',base_commit=BASE,
        model=dict(checkpoint=str(ctx['checkpoint']),checkpoint_sha256=CHECKPOINT,components_before=ctx['before'],components_after=component_hashes(ctx['model']),frozen=True),
        identity=ctx['identity'],normalization=ctx['stats'],configuration=old['configuration'],targets=ctx['target_meta'],
        geometry=dict(path=str(directory/GEOMETRY),sha256=file_hash(directory/GEOMETRY)),manifest=dict(path=str(directory/MANIFEST),sha256=file_hash(directory/MANIFEST)),
        results=aggregate(rows),states=rows,episodes=episodes,
        verification=dict(all400_original_episodes_replayed=True,all_scientific_actions_costs_noise_exact=True,all_archived_representative_traces_exact=True,
            S0_exact_previous_metrics=True,all_selected_oracle_scripted_repeats_exact=True,model_hash_unchanged=True,
            no_future_observation_input=True,no_new_policy=True),
        execution=dict(workers=workers,wall_seconds=time.monotonic()-start),
        limitations=['Original trajectories replayed, not modified controllers; wall-clock timing excluded from exact comparison.',
            'All original episodes including one successful Holdout retained; failed-only paired sensitivity separately reported.',
            'Relative stages depend on realized episode lengths and selected states; paired comparisons are not a causal time intervention.',
            'No new failure penalty; original absorbing terminal observation and all512 candidate costs retained.',
            'OOD distances are Train-only descriptive proxies, not strict manifold membership.',
            'True best-tail costs can be nearly tied; pairwise accuracy counts prediction ties as half credit, excludes true ties.',
            'First-decision S0 exactly reproduces old metrics; later raw candidates not historically persisted, reproduced from verified original RNG/action/cost streams.',
            'Only fixed canonical model, nominal physics, original H10/N512 proposal/cost; no model training or MPC changes.'])
    write_json(directory/REPORT,report); print(f'Saved {REPORT}',flush=True); return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); p.add_argument('--workers',type=int,default=8)
    p.add_argument('--episode',nargs=3,metavar=('CONDITION','SPLIT','INDEX'))
    a=p.parse_args(); run(a.root,a.workers,a.episode)
