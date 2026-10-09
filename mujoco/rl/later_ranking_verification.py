"""Independent replay/candidate/cost/derived-statistic verification."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import numpy as np
from decision_fidelity_metrics import ranking,candidate_metrics
from decision_fidelity_rollouts import true_rollout,scripted_reference
from decision_fidelity_snapshot import fingerprint
from latent_dynamics_data import file_hash,json_hash
from latent_mpc_core import planning_cost
from later_ranking_core import tail_metrics,prediction_trajectory,ood_scores
from later_ranking_replay import replay_episode
from later_ranking_analysis import validate_cohort
from later_ranking_reporting import aggregate_for_report as aggregate
from run_later_decision_ranking import later_context,build_train_geometry,task_distance,REPORT,MANIFEST,GEOMETRY
from joint_latent_world_model import component_hashes
from ppo_pi_env import MineUAVPIEnv

def verify_ranking(saved,pred,true):
    for key,value in ranking(pred,true).items():
        if saved.get(key)!=value: raise AssertionError(f'ranking metric changed: {key}')

def verify_decision_identity(saved,replayed):
    for key in ('candidate_sha256','snapshot_sha256','epsilon_sha256','decision_index','planner_rng_before_sha256','planner_rng_after_sha256'):
        if saved.get(key)!=replayed.get(key): raise AssertionError(f'exact decision identity changed: {key}')

def verify_cache_identity(saved,identity,key):
    if saved['identity']!=identity: raise ValueError('execution identity changed')
    e=saved['episode']
    if (e['condition'],e['split'],e['index'])!=tuple(key): raise ValueError('cached episode identity changed')

def verify_prior_S0(row,previous):
    c=row['condition']
    if row['candidate_sha256']!=previous['reconstruction']['candidate_sha256'][c] or row['snapshot_sha256']!=previous['snapshot_sha256']:
        raise AssertionError('S0 original exact candidate/snapshot identity changed')
    if row['metrics']!=previous['conditions'][c]: raise AssertionError('S0 prior metrics changed')

def init_worker(root,geometry,identity):
    global CTX,ENV
    CTX=later_context(root,geometry); ENV=MineUAVPIEnv(reward_version='v2')
    if CTX['identity']!=identity: raise AssertionError('verification source identity changed')

def verify_episode(payload):
    episode,records=payload; c,s,i=(episode[k] for k in ('condition','split','index'))
    replay=replay_episode(CTX,ENV,c,s,i)
    if episode['replay_verification']!=replay['verification']: raise AssertionError('original episode replay proof changed')
    expected_episode=dict(replay['episode'],condition=c,split=s,index=i,
        initial_distance=float(np.linalg.norm(replay['decisions']['S0']['observation'][:3])))
    for key,value in expected_episode.items():
        if key not in ('planning_time','encoding_time','total_decision_time') and episode.get(key)!=value:
            raise AssertionError(f'episode outcome metadata changed: {key}')
    repeats=0
    for r in records:
        d=replay['decisions'][r['stage']]; verify_decision_identity(r,d)
        for key in ('episode_steps','termination_reason','env_seed','target','initial_distance'):
            if r.get(key)!=expected_episode[key]: raise AssertionError(f'state episode metadata changed: {key}')
        if fingerprint(d['snapshot'])!=r['snapshot_sha256']: raise AssertionError('snapshot changed')
        if r['observation']!=d['observation'].tolist() or r['latent']!=d['z'][0].tolist(): raise AssertionError('reference state changed')
        art=r['local_arrays']
        if file_hash(art['path'])!=art['sha256']: raise AssertionError('local candidate file changed')
        with np.load(art['path'],allow_pickle=False) as raw: v={k:raw[k] for k in raw.files}
        a=d['plan']['candidates']; prev=d['previous_action']; pred=prediction_trajectory(CTX['model'],CTX['stats'],d['z'],a)
        for key,shape in dict(true_cost=(512,),pred_cost=(512,),terminal=(512,7),distance=(512,),speed=(512,),failure=(512,),
                invalid=(512,),executed_steps=(512,),termination_reason=(512,),trajectory_nrmse=(512,)).items():
            if v[key].shape!=shape: raise AssertionError('all512 candidates required')
        np.testing.assert_array_equal(v['pred_terminal'],pred[:,-1])
        np.testing.assert_array_equal(v['pred_cost'],d['plan']['costs'])
        np.testing.assert_array_equal(v['true_cost'],planning_cost(v['terminal'],a,prev)[0])
        np.testing.assert_array_equal(v['pred_cost'],planning_cost(pred[:,-1],a,prev)[0])
        verify_ranking(r['metrics'],v['pred_cost'],v['true_cost'])
        if r['best_tail']!=tail_metrics(v['pred_cost'],v['true_cost']): raise AssertionError('tail metric changed')
        # Re-run the actual scripted feedback reference at EVERY audited state;
        # task utility and termination metadata must not be trusted saved inputs.
        script=scripted_reference(ENV,d['snapshot']); repeats+=1
        np.testing.assert_array_equal(script['actions'],v['scripted_actions'])
        np.testing.assert_array_equal(script['observations'],v['scripted_observations'])
        script_summary={k:value for k,value in script.items() if k not in ('actions','observations','executed_actions')}
        if r['scripted']!=script_summary: raise AssertionError('scripted outcome/task utility changed')
        spred=prediction_trajectory(CTX['model'],CTX['stats'],d['z'],v['scripted_actions'][None])[0]
        np.testing.assert_array_equal(spred,v['scripted_predicted'])
        scost=float(planning_cost(spred[-1:],v['scripted_actions'][None],prev)[0][0])
        stcost=float(planning_cost(v['scripted_observations'][-1:],v['scripted_actions'][None],prev)[0][0])
        if scost!=r['scripted_pred_cost'] or stcost!=r['scripted']['cost']: raise AssertionError('scripted cost changed')
        m=candidate_metrics(v['pred_cost'],v['true_cost'],v['terminal'],pred[:,-1],v['distance'],v['speed'],v['failure'],
            task_distance(d['snapshot']),script,scost,CTX['stats'])
        for key,value in m.items():
            if r['metrics'].get(key)!=value: raise AssertionError(f'derived metric changed: {key}')
        ip=m['pred_best_index']; it=m['true_best_index']; error=(pred[:,-1]-v['terminal'])/CTX['stats']['obs']['std']
        expected=dict(m['prediction_error'],candidate_mean_h10_nrmse=float(np.sqrt(np.mean(error**2,1)).mean()),
            true_best_h10_nrmse=float(np.sqrt(np.mean(error[it]**2))),
            scripted_h10_nrmse=float(np.sqrt(np.mean(((spred[-1]-v['scripted_observations'][-1])/CTX['stats']['obs']['std'])**2))),
            whole_set_trajectory_nrmse=float(np.sqrt(np.mean(v['trajectory_nrmse']**2))),selected_trajectory_nrmse=float(v['trajectory_nrmse'][ip]),
            true_best_trajectory_nrmse=float(v['trajectory_nrmse'][it]),
            scripted_trajectory_nrmse=float(np.sqrt(np.mean(((spred-v['scripted_observations'][1:])/CTX['stats']['obs']['std'])**2))))
        if r['prediction']!=expected or r['ood']!=ood_scores(d['z'],d['observation'],d['plan']['action'],CTX['geometry']):
            raise AssertionError('prediction or Train distribution metric changed')
        valid=~v['invalid']
        np.testing.assert_allclose(v['distance'][valid],np.linalg.norm(v['terminal'][valid,:3].astype(float),axis=1),rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(v['speed'][valid],np.linalg.norm(v['terminal'][valid,3:6].astype(float),axis=1),rtol=1e-6,atol=1e-8)
        if not np.array_equal(v['failure'],~np.isin(v['termination_reason'],['none','success','time_limit'])): raise AssertionError('failure flag changed')
        if not np.array_equal(v['invalid'],v['termination_reason']=='nonfinite_state'): raise AssertionError('invalid flag changed')
        reasons,counts=np.unique(v['termination_reason'],return_counts=True)
        outcomes=dict(candidate_count=len(v['true_cost']),invalid_count=int(v['invalid'].sum()),
            early_termination_count=int(np.sum(v['executed_steps']<10)),
            termination_reason_counts={str(reason):int(count) for reason,count in zip(reasons,counts)})
        for key,value in outcomes.items():
            if r['metrics'].get(key)!=value: raise AssertionError(f'candidate outcome metric changed: {key}')
        if r['candidate_early_termination_count']!=outcomes['early_termination_count']:
            raise AssertionError('candidate early termination count changed')
        if r['candidate_physical_failure_count']!=int(v['failure'].sum()) or r['candidate_invalid_count']!=int(v['invalid'].sum()): raise AssertionError('failure count changed')
        # Fixed first3 episode identities in EACH condition/split, all stages.
        if i<3:
            for j in {ip,it}:
                out=true_rollout(ENV,d['snapshot'],a[j]); repeats+=1
                if out['cost']!=v['true_cost'][j] or out['final_snapshot_sha256']!=v['end_hash'][j]: raise AssertionError('physics repeat changed')
                np.testing.assert_array_equal(out['observations'][-1],v['terminal'][j])
                nrmse=float(np.sqrt(np.mean(((pred[j]-out['observations'][1:])/CTX['stats']['obs']['std'])**2)))
                if nrmse!=v['trajectory_nrmse'][j]: raise AssertionError('trajectory prediction error changed')
    if component_hashes(CTX['model'])!=CTX['before']: raise AssertionError('parameters mutated')
    return len(records),repeats

def verify_report(root,workers=8):
    directory=Path(root)/'mujoco/reports'; report=json.loads((directory/REPORT).read_text())
    geometry=json.loads((directory/GEOMETRY).read_text()); ctx=later_context(root,geometry)
    if ctx['identity']!=report['identity']: raise AssertionError('report execution identity changed')
    if build_train_geometry(ctx)!=geometry: raise AssertionError('Train-only statistics not exactly reproduced')
    manifest=json.loads((directory/MANIFEST).read_text()); count=validate_cohort(report['states'],report['episodes'])
    if file_hash(directory/MANIFEST)!=report['manifest']['sha256'] or file_hash(directory/GEOMETRY)!=report['geometry']['sha256']:
        raise AssertionError('manifest/statistics hash changed')
    if manifest['identity_sha256']!=json_hash(ctx['identity']) or json_hash(manifest['states'])!=manifest['states_sha256']:
        raise AssertionError('manifest scientific identity changed')
    fields=tuple(manifest['states'][0]); expected=[{k:r[k] for k in fields} for r in report['states']]
    if manifest['states']!=expected or manifest['state_count']!=count or manifest['candidate_rollout_count']!=count*512:
        raise AssertionError('manifest stage/candidate mapping changed')
    fields=tuple(manifest['episodes'][0])
    if manifest['episodes']!=[{k:e[k] for k in fields} for e in report['episodes']]:
        raise AssertionError('manifest episode outcome metadata changed')
    prior={(r['split'],r['index']):r for r in json.loads((directory/'world_model_decision_cost_fidelity_seed0.json').read_text())['states']}
    for r in report['states']:
        if r['stage']=='S0': verify_prior_S0(r,prior[(r['split'],r['index'])])
    payloads=[(e,[r for r in report['states'] if (r['condition'],r['split'],r['index'])==(e['condition'],e['split'],e['index'])]) for e in report['episodes']]
    states=0; repeats=0; completed=0
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,
            initargs=(str(Path(root).resolve()),geometry,ctx['identity'])) as pool:
        for f in as_completed([pool.submit(verify_episode,p) for p in payloads]):
            n,reps=f.result(); states+=n; repeats+=reps; completed+=1
            if completed%25==0: print(f'Verified{completed}/400 original episodes',flush=True)
    if report['results']!=aggregate(report['states']): raise AssertionError('cohort/paired/split statistics changed')
    if report['model']['components_before']!=ctx['before'] or report['model']['components_after']!=ctx['before']:
        raise AssertionError('frozen model report hash changed')
    # Unmodified tracked manifold primitive is bound to the requested base.
    import subprocess,hashlib
    relative='mujoco/rl/joint_composition_core.py'
    baseline=subprocess.check_output(['git','show',report['base_commit']+':'+relative],cwd=root)
    if file_hash(Path(root)/relative)!=hashlib.sha256(baseline).hexdigest(): raise AssertionError('base manifold primitive changed')
    for path,sha in report.get('postprocessing_sources',{}).items():
        if file_hash(path)!=sha: raise AssertionError('postprocessing source changed')
    for fig in report.get('figures',[]):
        if file_hash(fig['path'])!=fig['sha256']: raise AssertionError('figure changed')
    if 'analysis' in report:
        from finalize_later_ranking import interpretation
        if report['analysis']!=interpretation(report): raise AssertionError('interpretation evidence changed')
    if report['model'].get('checkpoint_sha256_after',report['model']['checkpoint_sha256'])!=file_hash(report['model']['checkpoint']):
        raise AssertionError('final checkpoint hash changed')
    return dict(episodes=400,states=states,candidates=states*512,all_trajectory_replays_exact=True,all_candidate_sets_reconstructed=True,
        all_endpoint_costs_recomputed=True,all_derived_statistics_recomputed=True,Train_geometry_recomputed=True,
        additional_fixed_exact_physics_replays=repeats,frozen_hashes_unchanged=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); p.add_argument('--workers',type=int,default=8)
    a=p.parse_args(); print(json.dumps(verify_report(a.root,a.workers),indent=2))
