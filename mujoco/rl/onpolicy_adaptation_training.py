"""Fixed optimizer-step budget, original v3 objective, explicit branch starts."""
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from joint_latent_world_model import component_hashes
from joint_latent_training import diagnostics
from joint_autonomous_consistency_training import loss_batch,save_autonomous,load_autonomous
from latent_dynamics_data import file_hash,json_hash
from onpolicy_adaptation_data import load_dataset
from run_onpolicy_adaptation_data import CHECKPOINT,MANIFEST
from run_latent_random_shooting_mpc import write_json

def batch_order(n,batch_size,updates,seed):
    if n<batch_size or n%batch_size or updates<1: raise ValueError('equal full batches required')
    rng=np.random.default_rng(seed);done=0
    while done<updates:
        order=rng.permutation(n)
        for begin in range(0,n,batch_size):
            if done==updates:return
            yield order[begin:begin+batch_size];done+=1

def selected_step(history):
    return min(history,key=lambda r:r['validation']['observation'])['step']

@torch.no_grad()
def validation(model,data,stats,scale):
    model.eval();total=np.zeros(3);n=0
    for begin in range(0,len(data),16):
        episodes,starts=data.batch(np.arange(begin,min(begin+16,len(data))))
        row=loss_batch(model,episodes,stats,scale,10,starts);diagnostics(row)
        values=np.array([row[k].item() for k in ('observation_loss','consistency_loss','loss')])
        if not np.isfinite(values).all():raise FloatingPointError('invalid adaptation validation loss')
        total+=values*row['window_count'];n+=row['window_count']
    return dict(zip(('observation','consistency','total'),(total/n).tolist()))

def train_updates(model,train,val,stats,scale,path,updates=1000,validation_interval=50,provenance=None):
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model.requires_grad_(True);before=component_hashes(model);started=time.monotonic()
    initial=validation(model,val,stats,scale);opt=torch.optim.Adam(model.parameters(),lr=.0003)
    order_hash=hashlib.sha256();history=[];step_rows=[];best=float('inf')
    parameters=list(model.parameters());slices={};offset=0
    for key in before:
        length=len(list(getattr(model,key).parameters()));slices[key]=slice(offset,offset+length);offset+=length
    for step,ids in enumerate(batch_order(len(train),16,updates,0),1):
        model.train();episodes,starts=train.batch(ids);order_hash.update(ids.astype('<i8').tobytes())
        opt.zero_grad(set_to_none=True);row=loss_batch(model,episodes,stats,scale,10,starts);maxima=diagnostics(row)
        if not torch.isfinite(row['loss']):raise FloatingPointError('nonfinite adaptation loss; no clipping')
        cg=torch.autograd.grad(.1*row['consistency_loss'],parameters,retain_graph=True,allow_unused=True)
        row['loss'].backward();norms={};ratios={}
        for key,sl in slices.items():
            ps=parameters[sl];cs=cg[sl]
            if any(p.grad is None for p in ps):raise AssertionError(f'{key} missing gradient')
            total=torch.sqrt(sum(p.grad.square().sum() for p in ps))
            cons=torch.sqrt(sum(c.square().sum() for c in cs if c is not None)) if any(c is not None for c in cs) else torch.tensor(0.)
            obs=torch.sqrt(sum((p.grad-(c if c is not None else 0)).square().sum() for p,c in zip(ps,cs)))
            if not torch.isfinite(torch.stack([total,cons,obs])).all():raise FloatingPointError('invalid adaptation gradient; stop')
            norms[key]=total.item();ratios[key]=cons.item()/max(obs.item(),1e-12)
        opt.step()
        step_rows.append(dict(step=step,train={k:row[f'{k}_loss'].item() for k in ('observation','consistency')},
            total=row['loss'].item(),gradient_norms=norms,consistency_to_observation_gradient_ratios=ratios,**maxima))
        if step%validation_interval==0 or step==updates:
            valid=validation(model,val,stats,scale);history.append(dict(step=step,validation=valid,
                train={k:float(np.mean([r['train'][k] for r in step_rows[-validation_interval:]])) for k in ('observation','consistency')}))
            if valid['observation']<best:
                best=valid['observation'];save_autonomous(path,model,stats,dict(best_step=step,
                    initial_latent_std=list(scale),initial_hashes=before,provenance=provenance or {},
                    selection_metric='own independent Validation L_obs only; post-update checkpoints every50 steps',
                    config=dict(seed=0,updates=updates,batch_size=16,learning_rate=.0003,horizon=10,lambda_auto_cons=.1),
                    validation=valid,local_consistency_enabled=False))
            print(f'Adaptation step{step}/{updates} val_obs={valid["observation"]:.7f} val_auto={valid["consistency"]:.7f}',flush=True)
    best_model,_,meta=load_autonomous(path)
    return dict(initial_hashes=before,best_hashes=component_hashes(best_model),final_hashes=component_hashes(model),
        initial_validation=initial,best_step=meta['best_step'],final_step=updates,history=history,steps=step_rows,
        best_validation=validation(best_model,val,stats,scale),best_train=validation(best_model,train,stats,scale),
        final_validation=history[-1]['validation'],order_sha256=order_hash.hexdigest(),
        train_windows=len(train),validation_windows=len(val),batch_size=16,learning_rate=.0003,
        scale_sha256=json_hash(list(scale)),normalization_sha256=json_hash(stats),
        elapsed_seconds=time.monotonic()-started,optimizer='Adam defaults; no clipping',
        checkpoint=dict(path=str(path),sha256=file_hash(path)),parameter_count=sum(p.numel() for p in model.parameters()))

def run(root,source):
    root=Path(root);reports=root/'mujoco/reports';m=json.loads((reports/MANIFEST).read_text())
    original=root/'mujoco/rl/models/joint_latent_world_model_v3_autonomous_consistency.pt'
    if file_hash(original)!=CHECKPOINT:raise AssertionError('original changed')
    model,stats,meta=load_autonomous(original)
    if json_hash(stats)!=m['identity']['normalization']:raise AssertionError('normalization changed')
    data={s:load_dataset(m['dataset'][source][s]) for s in ('train','val')}
    if len(data['train'])!=6400 or len(data['val'])!=1600:raise AssertionError('paired budget changed')
    path=root/'mujoco/rl/models'/('world_model_replay_adapted.pt' if source=='replay' else 'world_model_mpc_state_adapted.pt')
    provenance=dict(source=source,original_checkpoint=CHECKPOINT,data_manifest_sha256=file_hash(reports/MANIFEST),
        training_source_sha256=file_hash(__file__),loss_source_sha256=file_hash(Path(__file__).with_name('joint_autonomous_consistency_training.py')))
    result=train_updates(model,data['train'],data['val'],stats,meta['initial_latent_std'],path,provenance=provenance)
    if file_hash(original)!=CHECKPOINT:raise AssertionError('original checkpoint mutated')
    result['provenance']=provenance
    write_json(reports/'world_model_onpolicy_adaptation_seed0_parts'/f'training_{source}.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--source',choices=('replay','mpc_state'),required=True)
    a=p.parse_args();run(a.root,a.source)
