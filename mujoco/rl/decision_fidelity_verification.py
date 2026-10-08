"""Independent local-file/source/cohort and full derived-statistic verification."""
import argparse
import json
from pathlib import Path
import numpy as np
from decision_fidelity_candidates import reconstruct
from decision_fidelity_rollouts import frozen_prediction,true_rollout
from decision_fidelity_metrics import candidate_metrics
from run_world_model_decision_fidelity import audit_context,verify_cohort,aggregate,CONDITIONS,REPORT,MANIFEST,finite
from latent_dynamics_data import file_hash,json_hash
from latent_mpc_core import planning_cost
from joint_latent_world_model import component_hashes
from ppo_pi_env import MineUAVPIEnv


def verify_derived(saved,*args):
    expected=finite(candidate_metrics(*args))
    for key,value in expected.items():
        if saved.get(key)!=value: raise AssertionError(f'derived metric changed: {key}')


def require_shapes(values):
    for key,shape in dict(pred_cost=(512,),true_cost=(512,),terminal=(512,7),pred_terminal=(512,7),distance=(512,),
        speed=(512,),failure=(512,),invalid=(512,),executed_steps=(512,),termination_reason=(512,),end_hash=(512,)).items():
        if key not in values or values[key].shape!=shape: raise AssertionError(f'complete512 shape required: {key}')


def verify_postprocessing(report):
    for path,sha in report.get('postprocessing_sources',{}).items():
        if file_hash(path)!=sha: raise AssertionError('postprocessing source changed')
    for item in report.get('figures',[]):
        if file_hash(item['path'])!=item['sha256']: raise AssertionError('figure changed')
    if 'analysis' in report:
        from finalize_decision_fidelity import supplement
        if report['analysis']!=supplement(report): raise AssertionError('postprocessing statistics changed')


def verify_report(root,replay=True,check_postprocessing=True):
    ctx=audit_context(root); directory=ctx['root']/'mujoco/reports'
    report=json.loads((directory/REPORT).read_text()); manifest=json.loads((directory/MANIFEST).read_text())
    if report['identity']!=ctx['identity']: raise AssertionError('report/source runtime identity changed')
    if report['model']['components_before']!=ctx['before'] or report['model']['components_after']!=ctx['before']:
        raise AssertionError('report frozen hashes changed')
    if file_hash(directory/MANIFEST)!=report['manifest']['sha256']: raise AssertionError('manifest hash changed')
    rows=report['states']; verify_cohort(rows)
    if json_hash(manifest['states'])!=manifest['states_sha256'] or manifest['identity_sha256']!=json_hash(ctx['identity']):
        raise AssertionError('state manifest identity changed')
    expected_states=[{k:r[k] for k in ('split','index','env_seed','target','snapshot_sha256','reconstruction','local_arrays')} for r in rows]
    if expected_states!=manifest['states']: raise AssertionError('manifest/report state mapping changed')
    env=MineUAVPIEnv(reward_version='v2'); replays=0
    try:
        for row in rows:
            artifact=row['local_arrays']
            if file_hash(artifact['path'])!=artifact['sha256']: raise AssertionError('candidate endpoint/cost artifact changed')
            with np.load(artifact['path'],allow_pickle=False) as raw: arrays={k:raw[k] for k in raw.files}
            first=reconstruct(ctx,env,row['split'],row['index'])
            if first['evidence']!=row['reconstruction'] or first['snapshot_sha256']!=row['snapshot_sha256']:
                raise AssertionError('first-decision reconstruction mismatch')
            if first['env_seed']!=row['env_seed'] or first['target']!=row['target'] or first['observation'].tolist()!=row['initial_observation']:
                raise AssertionError('initial state metadata changed')
            d0=float(first['snapshot'].environment['previous_distance'])
            if d0!=row['initial_distance']: raise AssertionError('initial distance changed')
            script=row['scripted']; script_actions=arrays['scripted_actions']
            if script_actions.shape!=(10,4) or arrays['scripted_observations'].shape!=(11,7): raise AssertionError('scripted shape changed')
            actual_script_cost=float(planning_cost(arrays['scripted_observations'][-1:],script_actions[None],first['previous_action'])[0][0])
            predicted_script=frozen_prediction(ctx['model'],ctx['stats'],first['z'],script_actions[None])
            np.testing.assert_array_equal(predicted_script,arrays['scripted_pred_terminal'])
            if actual_script_cost!=script['cost']: raise AssertionError('scripted true cost changed')
            predicted_script_cost=float(planning_cost(predicted_script,script_actions[None],first['previous_action'])[0][0])
            script_error=float(np.sqrt(np.mean(((predicted_script[0]-arrays['scripted_observations'][-1])/ctx['stats']['obs']['std'])**2)))
            if predicted_script_cost!=row['scripted_pred_cost'] or script_error!=row['scripted_h10_nrmse']: raise AssertionError('scripted prediction metric changed')
            if not script['invalid_state']:
                np.testing.assert_allclose(script['terminal_distance'],np.linalg.norm(arrays['scripted_observations'][-1,:3]),rtol=1e-6,atol=1e-8)
                np.testing.assert_allclose(script['terminal_speed'],np.linalg.norm(arrays['scripted_observations'][-1,3:6]),rtol=1e-6,atol=1e-8)
                if script['distance_reduction']!=d0-script['terminal_distance']: raise AssertionError('scripted progress changed')
            for c in CONDITIONS:
                v={k.removeprefix(c+'_'):a for k,a in arrays.items() if k.startswith(c+'_')}; require_shapes(v)
                actions=first['conditions'][c]['candidates']
                np.testing.assert_array_equal(v['true_cost'],planning_cost(v['terminal'],actions,first['previous_action'])[0])
                predicted=frozen_prediction(ctx['model'],ctx['stats'],first['z'],actions)
                np.testing.assert_array_equal(predicted,v['pred_terminal'])
                np.testing.assert_array_equal(v['pred_cost'],planning_cost(predicted,actions,first['previous_action'])[0])
                m=row['conditions'][c]
                verify_derived(m,v['pred_cost'],v['true_cost'],v['terminal'],predicted,v['distance'],v['speed'],v['failure'],
                    d0,script,predicted_script_cost,ctx['stats'])
                reasons=v['termination_reason'].tolist()
                counts={r:reasons.count(r) for r in sorted(set(reasons))}
                if m['invalid_count']!=int(v['invalid'].sum()) or m['early_termination_count']!=int(np.sum(v['executed_steps']<10)) or m['termination_reason_counts']!=counts:
                    raise AssertionError('failure/termination aggregate changed')
                valid=~v['invalid']
                np.testing.assert_allclose(v['distance'][valid],np.linalg.norm(v['terminal'][valid,:3].astype(float),axis=1),rtol=1e-6,atol=1e-8)
                np.testing.assert_allclose(v['speed'][valid],np.linalg.norm(v['terminal'][valid,3:6].astype(float),axis=1),rtol=1e-6,atol=1e-8)
                if not np.array_equal(v['failure'],~np.isin(v['termination_reason'],['none','success','time_limit'])):
                    raise AssertionError('physical failure flags changed')
                if replay and row['index']==0:
                    for j in {m['true_best_index'],m['pred_best_index']}:
                        result=true_rollout(env,first['snapshot'],actions[j]); replays+=1
                        if result['cost']!=v['true_cost'][j] or result['final_snapshot_sha256']!=v['end_hash'][j]: raise AssertionError('exact physics replay differs')
                        np.testing.assert_array_equal(result['observations'][-1],v['terminal'][j])
            if replay and row['index']==0:
                result=true_rollout(env,first['snapshot'],script_actions); replays+=1
                np.testing.assert_array_equal(result['observations'],arrays['scripted_observations'])
                if result['final_snapshot_sha256']!=script['final_snapshot_sha256']: raise AssertionError('scripted physics replay differs')
    finally: env.close()
    if report['results']!=aggregate(rows): raise AssertionError('derived split/all-state aggregate changed')
    if check_postprocessing: verify_postprocessing(report)
    if component_hashes(ctx['model'])!=ctx['before']: raise AssertionError('verification mutated frozen model')
    return dict(states=200,candidates=204800,scripted_references=200,derived_metrics_recomputed=True,
        candidate_sets_and_model_outputs_reconstructed=True,all_endpoint_costs_recomputed=True,exact_fixed_physics_replays=replays,
        frozen_hashes_unchanged=True,source_identity_verified=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args(); print(json.dumps(verify_report(args.root),indent=2))
