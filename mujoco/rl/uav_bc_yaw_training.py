"""Matched fresh BC fits: fixed old stats, identical shuffle, common Val-only selection."""
import copy,json,math,random
from pathlib import Path
import numpy as np
import torch
from uav_bc_policy import BCActor,normalize,normalized_loss,parameter_hash,save,load
from uav_bc_training import evaluate_loss
from uav_bc_safety import atomic_bytes,atomic_json,check_running,available_memory
from latent_dynamics_data import file_hash,json_hash
from uav_bc_yaw_data import PARTS,MANIFEST,MODEL_NAMES,load_selected,MODEL_SHA

def select_best(rows):return min(rows,key=lambda r:r['validation_loss'])['epoch']

def array_hash(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def train_one(training,validation,stats,path,resume_path,epochs=100,batch_size=512,on_epoch=None):
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    x,y=training['observations'],training['actions'];vx,vy=validation['observations'],validation['actions']
    if not len(x) or not len(vx) or epochs<1 or batch_size<1:raise ValueError('positive fixed budget required')
    arrays=[x,y,vx,vy]
    if any(not np.isfinite(a).all() for a in arrays):raise ValueError('invalid supervised data')
    ident=dict(arrays=[array_hash(a) for a in arrays],stats_sha256=json_hash(stats),seed=0,epochs=epochs,batch_size=batch_size,
        lr=.001,training_source_sha256=file_hash(__file__),policy_source_sha256=file_hash(Path(__file__).with_name('uav_bc_policy.py')))
    actor=BCActor();initial=parameter_hash(actor);optimizer=torch.optim.Adam(actor.parameters(),lr=.001)
    tx,ty,rx,ry=[torch.from_numpy(a) for a in (normalize(x,stats,'obs'),normalize(y,stats,'action'),normalize(vx,stats,'obs'),normalize(vy,stats,'action'))]
    rng=np.random.default_rng(0);history=[];updates=0;best=float('inf');best_state=None;resume_path=Path(resume_path)
    if resume_path.exists():
        state=torch.load(resume_path,map_location='cpu',weights_only=False)
        if state['identity']!=ident or state['initial_parameter_hash']!=initial:raise ValueError('resume identity/budget mismatch')
        actor.load_state_dict(state['actor']);optimizer.load_state_dict(state['optimizer']);rng.bit_generator.state=state['rng']
        history=state['history'];updates=state['updates'];best=state['best'];best_state=state['best_state']
        if updates!=len(history)*math.ceil(len(x)/batch_size) or [r['epoch'] for r in history]!=list(range(1,len(history)+1)):
            raise ValueError('duplicate/missing optimizer updates')
    for epoch in range(len(history)+1,epochs+1):
        check_running(available_memory(),0,0);actor.train();order=rng.permutation(len(tx));grad=[]
        for start in range(0,len(tx),batch_size):
            ids=order[start:start+batch_size];optimizer.zero_grad(set_to_none=True)
            loss=normalized_loss(actor(tx[ids]),ty[ids]);loss.backward()
            norm=torch.sqrt(sum(p.grad.square().sum() for p in actor.parameters()))
            if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite training, no automatic clipping/tuning')
            grad.append(norm.item());optimizer.step();updates+=1
        tr=evaluate_loss(actor,tx,ty);va=evaluate_loss(actor,rx,ry)
        groups={str(g):evaluate_loss(actor,rx[validation['yaw_group']==g],ry[validation['yaw_group']==g])
            for g in np.unique(validation['yaw_group'])}
        with torch.no_grad():error=(actor(rx)-ry).numpy()*np.asarray(stats['action']['std'])
        row=dict(epoch=epoch,train_loss=tr,validation_loss=va,validation_by_initial_yaw=groups,
            gradient_norm_mean=float(np.mean(grad)),gradient_norm_max=float(max(grad)),order_sha256=array_hash(order),
            validation_action_rmse=float(np.sqrt(np.mean(error**2))),validation_action_mae=float(np.mean(abs(error))))
        if not np.isfinite([tr,va,row['gradient_norm_max']]).all():raise FloatingPointError('nonfinite metrics')
        history.append(row)
        if va<best:best=va;best_state=copy.deepcopy(actor.state_dict())
        state=dict(identity=ident,initial_parameter_hash=initial,actor=actor.state_dict(),optimizer=optimizer.state_dict(),
            rng=rng.bit_generator.state,history=history,updates=updates,best=best,best_state=best_state)
        atomic_bytes(resume_path,lambda f:torch.save(state,f))
        if epoch==1 or epoch%10==0:print(f'Yaw BC {Path(path).name} epoch{epoch} Train={tr:.7f} Val={va:.7f}',flush=True)
        if on_epoch:on_epoch(row)
    expected=epochs*math.ceil(len(tx)/batch_size)
    if updates!=expected or len(history)!=epochs:raise ValueError('fixed optimizer budget not completed')
    final_hash=parameter_hash(actor);actor.load_state_dict(best_state)
    meta=dict(best_epoch=select_best(history),selection='common Validation normalized action MSE only',seed=0,max_epochs=epochs,
        batch_size=batch_size,learning_rate=.001,initial_parameter_hash=initial,optimizer_updates=updates,identity=ident,
        experiment='controlled yaw coverage; random initialization, NOT Original BC fine-tuning')
    save(path,actor,stats,meta);policy,_=load(path)
    return dict(history=history,best_epoch=meta['best_epoch'],best_train_loss=evaluate_loss(actor,tx,ty),best_validation_loss=best,
        best_validation_by_yaw=history[meta['best_epoch']-1]['validation_by_initial_yaw'],initial_parameter_hash=initial,
        best_parameter_hash=parameter_hash(policy.actor),final_parameter_hash=final_hash,optimizer_updates=updates,
        normalization=stats,normalization_sha256=json_hash(stats),checkpoint_sha256=file_hash(path),identity=ident,
        complete=True,epochs_completed=epochs,selected_checkpoint='validation best, not final epoch')

def run(root):
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS;models=root/'mujoco/rl/models'
    original=models/'uav_bc_mlp_seed0.pt'
    if file_hash(original)!=MODEL_SHA:raise ValueError('original model changed')
    stats=load(original)[0].stats;validation=load_selected(root,'shared_validation');result={}
    # No Test dataset or final trajectories are read by this function.
    for name,model in MODEL_NAMES.items():
        result[name]=train_one(load_selected(root,name),validation,stats,models/model,parts/(name+'_resume.pt'))
    a,b=result.values()
    if a['initial_parameter_hash']!=b['initial_parameter_hash'] or a['optimizer_updates']!=b['optimizer_updates'] or a['optimizer_updates']!=4000:
        raise ValueError('unmatched initialization/optimizer budget')
    if [r['order_sha256'] for r in a['history']]!=[r['order_sha256'] for r in b['history']]:raise ValueError('paired batch schedule mismatch')
    if file_hash(original)!=MODEL_SHA:raise ValueError('Original frozen reference modified')
    atomic_json(parts/'training.json',result);return result

if __name__=='__main__':run(Path(__file__).resolve().parents[2])
