"""Fresh split identities and exact-prefix counterfactual branch windows."""
from pathlib import Path
import json
import numpy as np
from latent_dynamics_data import file_hash,json_hash

GENERATION_SEED=202710091
COUNTS=dict(train=60,val=20,final=100)
SOURCES=('replay','mpc_state')

def validate_splits(splits,old_targets=(),old_seeds=()):
    if set(splits)!=set(COUNTS) or any(len(splits[k])!=n for k,n in COUNTS.items()):
        raise ValueError('fixed isolated60/20/100 target groups required')
    rows=[r for v in splits.values() for r in v]; ids=[r['target_id'] for r in rows]; seeds=[r['env_seed'] for r in rows]
    if len(set(ids))!=180 or len(set(seeds))!=180 or set(seeds)&set(old_seeds): raise ValueError('target/reset identity leakage')
    points=np.asarray(old_targets,float).reshape(-1,3).tolist()
    for r in rows:
        t=np.asarray(r['target'],float)
        if t.shape!=(3,) or not np.isfinite(t).all() or (np.abs(t[:2])>2).any() or not .7<=t[2]<=1.5:
            raise ValueError('invalid unchanged nominal target bounds')
        if points and np.min(np.linalg.norm(np.asarray(points)-t,axis=1))<1e-8: raise ValueError('target coordinate overlap')
        points.append(t.tolist())
    return True

def make_targets(old_targets=(),old_seeds=()):
    rng=np.random.default_rng(GENERATION_SEED); result={}; index=0
    for split,n in COUNTS.items():
        rows=[]
        for j in range(n):
            target=np.r_[rng.uniform(-2,2,2),rng.uniform(.7,1.5)].tolist()
            rows.append(dict(target_id=f'adapt-{GENERATION_SEED}-{split}-{j:03d}',global_index=index,
                env_seed=710090000+index,target=target,split=split)); index+=1
        result[split]=rows
    validate_splits(result,old_targets,old_seeds); return result

def quotas(total,groups):
    if type(total) is not int or type(groups) is not int or total<groups or groups<1: raise ValueError('invalid balanced budget')
    q,r=divmod(total,groups); return [q+(i<r) for i in range(groups)]

def branch_indices(target_index,decision_index):
    rng=np.random.default_rng(np.random.SeedSequence([GENERATION_SEED,17,target_index,decision_index]))
    return np.r_[0,1,rng.choice(np.arange(2,512),6,replace=False)].astype(int)

def valid_branch(out):
    obs=np.asarray(out['observations'])
    return bool(out['executed_steps']==10 and not out['invalid_state'] and obs.shape==(11,7) and np.isfinite(obs).all())

def branch_episode(state,branch):
    prefix=np.asarray(state['prefix_observations'],np.float32); pa=np.asarray(state['prefix_actions'],np.float32)
    action=np.asarray(state['branch_actions'][branch],np.float32); future=np.asarray(state['branch_observations'][branch],np.float32)
    if prefix.ndim!=2 or prefix.shape[1]!=7 or len(prefix)!=len(pa)+1 or action.shape!=(10,4) or future.shape!=(11,7):
        raise ValueError('exact prefix and ten real branch steps required')
    if not np.array_equal(prefix[-1],future[0]): raise ValueError('branch does not start at visited observation')
    obs=np.vstack([prefix,future[1:]]); commands=np.vstack([pa,action]); start=len(pa)
    previous=np.vstack([np.zeros((1,4),np.float32),commands[:-1]])
    return dict(obs=obs[:-1],next_obs=obs[1:],action=commands,previous_action=previous),start

class BranchDataset:
    def __init__(self,states):
        self.states=states
        if not states or any(np.asarray(s['branch_actions']).shape!=(8,10,4) for s in states):
            raise ValueError('exactly8 full branches per valid snapshot required')
    def __len__(self): return len(self.states)*8
    def batch(self,indices):
        pairs=[branch_episode(self.states[int(i)//8],int(i)%8) for i in indices]
        return [p[0] for p in pairs],[np.array([p[1]],dtype=int) for p in pairs]

def load_dataset(metadata):
    states=[]
    for row in metadata:
        if file_hash(row['path'])!=row['sha256']: raise ValueError('branch dataset changed')
        with np.load(row['path'],allow_pickle=False) as raw: state={k:raw[k].copy() for k in raw.files}
        if not np.all(state['executed_steps']==10) or state['invalid_state'].any(): raise ValueError('invalid training branch admitted')
        states.append(state)
    return BranchDataset(states)

def exclusion_identities(ctx):
    source=ctx['root']/'mujoco/reports/joint_latent_autonomous_consistency_v3_seed0.json'
    previous=json.loads(source.read_text()); directory=Path(previous['dataset']['source_directory'])
    original=json.loads((directory/'manifest.json').read_text()); targets=list(original['targets']); seeds=set()
    for split in ('train','val','test'):
        with np.load(directory/f'{split}.npz',allow_pickle=False) as raw:
            seeds.update(r['seed'] for r in json.loads(str(raw['metadata'])))
    for rows in ctx['targets'].values():
        for seed,target in rows: seeds.add(seed); targets.append(target)
    return targets,seeds
