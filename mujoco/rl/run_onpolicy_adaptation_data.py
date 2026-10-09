"""Matched collection runner with immutable cache identities and sealed Final."""
import argparse,json,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import torch
from run_latent_random_shooting_mpc import context,write_json
from ppo_pi_env import MineUAVPIEnv
from latent_dynamics_data import file_hash,json_hash
from joint_latent_world_model import component_hashes
from onpolicy_adaptation_data import make_targets,exclusion_identities,quotas,SOURCES
from onpolicy_adaptation_collection import collect_branches

BASE='db532ac9b40d9fc49a1c2e18b55090aba4f6b19a'
CHECKPOINT='42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9'
MANIFEST='onpolicy_adaptation_data_manifest_seed0.json'
TARGETS='onpolicy_adaptation_target_splits_seed0.json'

def collection_context(root):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    c=context(root)
    if file_hash(c['checkpoint'])!=CHECKPOINT: raise AssertionError('canonical checkpoint changed')
    old,seeds=exclusion_identities(c); splits=make_targets(old,seeds)
    sources=dict(c['immutable'])
    for name in ('onpolicy_adaptation_data.py','onpolicy_adaptation_collection.py',Path(__file__).name,
            'latent_mpc_core.py','latent_mpc_evaluation.py','decision_fidelity_snapshot.py','decision_fidelity_rollouts.py'):
        p=Path(__file__).with_name(name); sources[str(p)]=file_hash(p)
    c['splits']=splits;c['identity']=dict(checkpoint=CHECKPOINT,normalization=json_hash(c['stats']),target_splits=json_hash(splits),
        source_sha256=sources,train_snapshots=800,val_snapshots=200,branches_per_snapshot=8,horizon=10,
        invalid_rule='only10 actual finite steps; never pad earlydone for training',counterfactual=True)
    return c

def init_worker(root,identity):
    global CTX,ENV
    CTX=collection_context(root); ENV=MineUAVPIEnv(reward_version='v2')
    if CTX['identity']!=identity: raise AssertionError('data source identity changed')

def worker(payload):
    source,target,quota,parts=payload
    result=collect_branches(CTX,ENV,source,target,quota,parts)
    if component_hashes(CTX['model'])!=CTX['before']: raise AssertionError('collection model mutated')
    name=f'{source}_{target["split"]}_{target["global_index"]:03d}.json'
    write_json(Path(parts)/name,dict(identity=CTX['identity'],result=result));return result

def run(root,workers=8):
    c=collection_context(root); out=c['root']/'mujoco/reports';parts=out/'world_model_onpolicy_adaptation_seed0_parts';parts.mkdir(parents=True,exist_ok=True)
    target_meta=dict(splits=c['splits'],generation='fresh202710091 full nominal target distribution; reset710090000+global index',
        exclusion='all120 original dataset targets/episode seeds plus original200 benchmark/holdout targets/seeds; tolerance1e-8',
        semantic_sha256=json_hash(c['splits']))
    write_json(out/TARGETS,target_meta); results=[]; pending=[]
    for source in SOURCES:
        for split,total in [('train',800),('val',200)]:
            for target,quota in zip(c['splits'][split],quotas(total,len(c['splits'][split]))):
                p=parts/f'{source}_{split}_{target["global_index"]:03d}.json'
                if p.exists():
                    cached=json.loads(p.read_text())
                    if cached['identity']!=c['identity']: raise ValueError('stale collection identity, never relabel')
                    result=cached['result']
                    if result['episode']['target_id']!=target['target_id'] or result['requested_snapshot_count']!=quota:
                        raise AssertionError('cached target/budget changed')
                    if any(file_hash(r['path'])!=r['sha256'] for r in result['states']): raise ValueError('branch arrays changed')
                    results.append(result)
                else: pending.append((source,target,quota,str(parts)))
    print(f'Adaptation data: reuse{len(results)}/160 source episodes, pending{len(pending)}',flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,
            initargs=(str(c['root']),c['identity'])) as pool:
        for f in as_completed([pool.submit(worker,p) for p in pending]):
            results.append(f.result())
            if len(results)%10==0: print(f'Collected{len(results)}/160 source episodes',flush=True)
    results.sort(key=lambda r:(r['episode']['source'],r['episode']['global_index']))
    dataset={source:{split:[s for r in results if r['episode']['source']==source and r['episode']['split']==split for s in r['states']]
        for split in ('train','val')} for source in SOURCES}
    for source in SOURCES:
        if len(dataset[source]['train'])!=800 or len(dataset[source]['val'])!=200: raise AssertionError('unmatched dataset budget')
    m=dict(base_commit=BASE,identity=c['identity'],targets=dict(path=str(out/TARGETS),sha256=file_hash(out/TARGETS)),
        dataset=dataset,source_episodes=[r['episode'] for r in results],collection_diagnostics=results,
        window_counts={s:dict(train=6400,val=1600) for s in SOURCES},
        selection='all sources/targets retained; fixed balanced quota; predetermined decision permutation, all8 branches must have10 actual steps')
    if component_hashes(c['model'])!=c['before'] or any(file_hash(p)!=h for p,h in c['identity']['source_sha256'].items()):
        raise AssertionError('immutable model/source changed during collection')
    write_json(out/MANIFEST,m);print(f'Saved {MANIFEST}',flush=True);return m

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();run(a.root,a.workers)
