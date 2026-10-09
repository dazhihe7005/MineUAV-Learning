"""Fixed100-epoch BC; only Train statistics and Val-loss checkpoint selection."""
import json,random,time
from pathlib import Path
import numpy as np
import torch
from uav_bc_policy import BCActor,fit_stats,normalize,normalized_loss,save,load,parameter_hash
from uav_bc_safety import atomic_json,available_memory,check_running
from uav_bc_data import MANIFEST,PARTS,load_split
from latent_dynamics_data import file_hash,json_hash

MODEL='uav_bc_mlp_seed0.pt'

def select_epoch(history):return min(history,key=lambda r:r['validation_loss'])['epoch']

@torch.no_grad()
def evaluate_loss(actor,x,y):
    actor.eval();total=0.
    for begin in range(0,len(x),512):total+=normalized_loss(actor(x[begin:begin+512]),y[begin:begin+512]).item()*len(x[begin:begin+512])
    return total/len(x)

def train(train_x,train_y,val_x,val_y,path,epochs=100,batch_size=512,provenance=None):
    if min(epochs,batch_size,len(train_x),len(val_x))<1:raise ValueError('positive training budget')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    stats=fit_stats(train_x,train_y);actor=BCActor();initial=parameter_hash(actor);optimizer=torch.optim.Adam(actor.parameters(),lr=.001)
    tx,ty,vx,vy=[torch.from_numpy(a) for a in (normalize(train_x,stats,'obs'),normalize(train_y,stats,'action'),
        normalize(val_x,stats,'obs'),normalize(val_y,stats,'action'))]
    best=float('inf');history=[];rng=np.random.default_rng(0);started=time.monotonic()
    for epoch in range(1,epochs+1):
        check_running(available_memory(),0,0);actor.train();grad=[]
        for begin in range(0,len(tx),batch_size):
            if begin==0:order=rng.permutation(len(tx))
            ids=order[begin:begin+batch_size];optimizer.zero_grad(set_to_none=True)
            loss=normalized_loss(actor(tx[ids]),ty[ids]);loss.backward()
            norm=torch.sqrt(sum(p.grad.square().sum() for p in actor.parameters()))
            if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite BC loss/gradient; stop, no clipping')
            grad.append(norm.item());optimizer.step()
        train_loss=evaluate_loss(actor,tx,ty);validation_loss=evaluate_loss(actor,vx,vy)
        if not np.isfinite([train_loss,validation_loss]).all():raise FloatingPointError('invalid supervised metrics')
        # Physical action-error metrics evaluated without deployment clipping.
        with torch.no_grad():
            error=(actor(vx)-vy).numpy()*np.asarray(stats['action']['std'])
        row=dict(epoch=epoch,train_loss=train_loss,validation_loss=validation_loss,
            gradient_norm_mean=float(np.mean(grad)),gradient_norm_max=max(grad),
            validation_action_rmse=float(np.sqrt(np.mean(error**2))),validation_action_mae=float(np.mean(np.abs(error))))
        history.append(row)
        if validation_loss<best:
            best=validation_loss;save(path,actor,stats,dict(best_epoch=epoch,selection='Validation normalized action MSE only',
                seed=0,max_epochs=epochs,batch_size=batch_size,learning_rate=.001,initial_parameter_hash=initial,provenance=provenance or {}))
        if epoch==1 or epoch%10==0:print(f'BC epoch{epoch}/{epochs} Train={train_loss:.8f} Val={validation_loss:.8f}',flush=True)
    policy,meta=load(path)
    return dict(history=history,best_epoch=meta['best_epoch'],best_validation_loss=best,
        best_train_loss=evaluate_loss(policy.actor,tx,ty),initial_hash=initial,best_parameter_hash=parameter_hash(policy.actor),
        final_parameter_hash=parameter_hash(actor),stats=stats,normalization_sha256=json_hash(stats),
        epochs_completed=epochs,elapsed_seconds=time.monotonic()-started,checkpoint=dict(path=str(path),sha256=file_hash(path)))

def run(root):
    root=Path(root);reports=root/'mujoco/reports';manifest_path=reports/MANIFEST;data=json.loads(manifest_path.read_text())
    # Deliberately never load the Test samples in this phase.
    training=load_split(data,'train');validation=load_split(data,'val')
    provenance=dict(dataset_manifest_sha256=file_hash(manifest_path),training_source_sha256=file_hash(__file__),
        policy_source_sha256=file_hash(Path(__file__).with_name('uav_bc_policy.py')))
    result=train(training['observations'],training['actions'],validation['observations'],validation['actions'],
        root/'mujoco/rl/models'/MODEL,provenance=provenance)
    result['provenance']=provenance;atomic_json(reports/PARTS/'training.json',result);return result

if __name__=='__main__':run(Path(__file__).resolve().parents[2])
