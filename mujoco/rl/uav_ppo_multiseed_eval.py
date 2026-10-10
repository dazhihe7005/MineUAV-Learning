"""Same exact fresh targets for five final PPOs and three frozen baselines."""
import gc
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from ppo_pi_env import MineUAVPIEnv
from uav_ppo_multiseed import ROOT,PARTS,read_manifest,verify_completion,seal
from uav_ppo_bc_nominal import frozen_controller,initial_snapshot,compatibility
from uav_ppo_bc_nominal_run import evaluate,repeat_check
from uav_bc_policy import parameter_hash
from uav_bc_robustness import validate_cohort
from uav_bc_safety import atomic_json
from latent_dynamics_data import file_hash

def run(root=ROOT):
    root=Path(root);m=read_manifest(root);parts=root/'mujoco/reports'/PARTS;records=[];identities={}
    completions={s:json.loads((parts/f'seed{s}.json').read_text()) for s in range(5)}
    for s,r in completions.items():verify_completion(r,m,s)
    if len({r['initial_parameter_sha256'] for r in completions.values()})!=5:raise ValueError('non-independent random initialization')
    torch.set_num_threads(1);env=MineUAVPIEnv(reward_version='v2');new=0;repeats={};frozen={}
    try:
        for name in ['scripted','original_bc','yaw_bc']+[f'ppo_seed{s}' for s in range(5)]:
            if name.startswith('ppo_'):
                s=int(name[-1]);r=completions[s];model=PPO.load(r['checkpoint_path'],device='cpu')
                compatibility(env,model.policy);module=model.policy;module.set_training_mode(False);module.requires_grad_(False)
                def predict(o):return model.predict(np.asarray(o,np.float32),deterministic=True)[0].astype(np.float32)
                meta=dict(checkpoint_sha256=r['checkpoint_sha256'],parameter_sha256=parameter_hash(module))
            else:predict,module,meta=frozen_controller(root,name)
            identities[name]=meta;before=parameter_hash(module) if module is not None else None
            identity=dict(manifest_sha256=m['sha256'],controller=name,model_identity=meta)
            cohort=[]
            for i,state in enumerate(m['states']):
                snap,initial=initial_snapshot(env,state)
                if initial!=state['initial_state']:raise ValueError('paired initial snapshot mismatch')
                row,is_new=evaluate(env,snap,predict,state,name,parts/f'eval_{i:03d}_{name}.json',identity)
                records.append(row);cohort.append(row);new+=int(is_new)
                if i==0:
                    repeat,added=repeat_check(env,snap,predict,row,parts,name,identity);repeats[name]=repeat;new+=int(added)
                if (i+1)%50==0:print('Final',name,i+1,'/200 successes',sum(x['success'] for x in cohort),flush=True)
            after=parameter_hash(module) if module is not None else None
            if before!=after:raise ValueError('evaluation policy mutation')
            frozen[name]=dict(before=before,after=after,eval=True,requires_grad_false=True)
            del predict,module
            if name.startswith('ppo_'):del model
            gc.collect()
    finally:env.close()
    expected=[s['key']+'/'+n for s in m['states'] for n in identities]
    validate_cohort(records,expected)
    value=seal(dict(manifest_sha256=m['sha256'],records=records,identities=identities,frozen=frozen,
        repeats=repeats,formal_episodes=len(records),new_executions=new,unique_executions=len(records)+len(repeats)))
    atomic_json(parts/'evaluation.json',value);print('EVAL COMPLETE',len(records),'new executions',new,flush=True)
    return value

if __name__=='__main__':run()
