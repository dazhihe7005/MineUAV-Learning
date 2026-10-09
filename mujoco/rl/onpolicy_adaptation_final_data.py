"""Collect Final only AFTER both training budgets/checkpoints are complete."""
import argparse,json,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from onpolicy_adaptation_collection import collect_visited
from onpolicy_adaptation_evaluation import final_stages
from run_onpolicy_adaptation_data import collection_context,MANIFEST
from decision_fidelity_rollouts import true_rollout,scripted_reference
from decision_fidelity_snapshot import fingerprint
from latent_mpc_support_evaluation import array_hash
from latent_dynamics_data import file_hash,json_hash
from run_latent_random_shooting_mpc import write_json
from ppo_pi_env import MineUAVPIEnv
from joint_latent_world_model import component_hashes

FINAL_MANIFEST='onpolicy_adaptation_final_manifest_seed0.json'

def require_completed_training(rows,checkpoints):
    if set(rows)!=set(('replay','mpc_state')) or not all(checkpoints.values()):raise ValueError('both completed training runs required before Final')
    for r in rows.values():
        if (r['final_step'],r['train_windows'],r['validation_windows'])!=(1000,6400,1600):raise ValueError('completed matched budgets required')
    a,b=rows['replay'],rows['mpc_state']
    if a['order_sha256']!=b['order_sha256'] or a['initial_hashes']!=b['initial_hashes']:raise ValueError('paired order/initial hashes changed')

def init_worker(root,identity):
    global CTX,ENV,IDENTITY
    CTX=collection_context(root);ENV=MineUAVPIEnv(reward_version='v2');IDENTITY=identity
    if CTX['identity']!=identity['collection']:raise AssertionError('Final collection identity changed')

def collect_final_target(ctx,env,target,directory):
    trajectory=collect_visited(ctx,env,'mpc_state',target,keep_all_candidates=True)
    rows=[]
    for stage,t in final_stages(len(trajectory['actions'])).items():
        d=trajectory['decisions'][t];a=d['candidates']
        if a.shape!=(512,10,4) or array_hash(a)!=d['candidate_set_sha256']:raise AssertionError('candidate set changed')
        outs=[true_rollout(env,d['snapshot'],actions) for actions in a]
        repeat=true_rollout(env,d['snapshot'],a[0])
        np.testing.assert_array_equal(repeat['observations'],outs[0]['observations'])
        if repeat['final_snapshot_sha256']!=outs[0]['final_snapshot_sha256']:raise AssertionError('Final exact restore failed')
        script=scripted_reference(env,d['snapshot'])
        path=Path(directory)/f'final_{target["global_index"]:03d}_{stage}.npz'
        np.savez_compressed(path,prefix_observations=trajectory['observations'][:t+1],prefix_actions=trajectory['actions'][:t],
            candidates=a,truth=np.stack([o['observations'][1:] for o in outs]),true_costs=np.asarray([o['cost'] for o in outs]),
            distances=np.asarray([o['terminal_distance'] for o in outs]),speeds=np.asarray([o['terminal_speed'] for o in outs]),
            failures=np.asarray([o['physical_failure'] for o in outs]),executed_steps=np.asarray([o['executed_steps'] for o in outs]),
            script_actions=script['actions'],script_truth=script['observations'][1:])
        rows.append(dict(target_id=target['target_id'],global_index=target['global_index'],stage=stage,decision_index=t,
            episode_steps=len(trajectory['actions']),snapshot_sha256=d['snapshot_sha256'],candidate_sha256=array_hash(a),
            true_cost_sha256=array_hash(np.asarray([o['cost'] for o in outs])),original_selected_index=d['selected_index'],
            original_selected_action=a[d['selected_index'],0].tolist(),original_predicted_cost_sha256=array_hash(d['predicted_costs']),
            initial_distance=float(d['snapshot'].environment['previous_distance']),
            pi_magnitude=float(np.linalg.norm(d['snapshot'].controller['_integral_error']*
                np.array([d['snapshot'].controller['ki_xy'],d['snapshot'].controller['ki_xy'],d['snapshot'].controller['ki_z']]))),
            script={k:script[k] for k in ('cost','terminal_distance','terminal_speed','distance_reduction','physical_failure','executed_steps','termination_reason')},
            path=str(path),sha256=file_hash(path),repeat_exact=True))
    if component_hashes(ctx['model'])!=ctx['before']:raise AssertionError('frozen Original changed')
    return dict(episode=trajectory['episode'],states=rows)

def worker(payload):
    target,parts=payload;result=collect_final_target(CTX,ENV,target,parts)
    write_json(Path(parts)/f'final_{target["global_index"]:03d}.json',dict(identity=IDENTITY,result=result));return result

def run(root,workers=8):
    c=collection_context(root);reports=c['root']/'mujoco/reports';parts=reports/'world_model_onpolicy_adaptation_seed0_parts'
    training={s:json.loads((parts/f'training_{s}.json').read_text()) for s in ('replay','mpc_state')}
    require_completed_training(training,{s:file_hash(r['checkpoint']['path'])==r['checkpoint']['sha256'] for s,r in training.items()})
    identity=dict(collection=c['identity'],final_source_sha256=file_hash(__file__),training_checkpoints={s:r['checkpoint']['sha256'] for s,r in training.items()},
        stages=['S0','S2','S3'],N=512,H=10,controller='original canonical frozen unconstrained MPC only')
    results=[];pending=[]
    for target in c['splits']['final']:
        path=parts/f'final_{target["global_index"]:03d}.json'
        if path.exists():
            cache=json.loads(path.read_text())
            if cache['identity']!=identity:raise ValueError('stale Final identity')
            result=cache['result']
            if any(file_hash(r['path'])!=r['sha256'] for r in result['states']):raise ValueError('Final truth arrays changed')
            results.append(result)
        else:pending.append((target,str(parts)))
    print(f'Final: reuse{len(results)}/100 trajectories; pending{len(pending)}',flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(str(c['root']),identity)) as pool:
        for f in as_completed([pool.submit(worker,x) for x in pending]):
            results.append(f.result())
            if len(results)%10==0:print(f'Final complete{len(results)}/100',flush=True)
    results.sort(key=lambda r:r['episode']['global_index'])
    manifest=dict(identity=identity,episodes=[r['episode'] for r in results],states=[s for r in results for s in r['states']],
        training_completed_before_final=True,termination='original finite absorbing terminal afterdone, all512 retained, no new penalty')
    write_json(reports/FINAL_MANIFEST,manifest);return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();run(a.root,a.workers)
