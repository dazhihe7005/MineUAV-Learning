"""Standalone7D supervised actor; no environment, expert or world-model import."""
import hashlib
import numpy as np
import torch
from torch import nn
from uav_bc_safety import atomic_bytes

class BCActor(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(7,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,4))
    def forward(self,x):return self.net(x)

def fit_stats(observations,actions):
    result={}
    for name,value,dim in [('obs',observations,7),('action',actions,4)]:
        x=np.asarray(value,np.float64)
        if x.ndim!=2 or x.shape[1]!=dim or not len(x) or not np.isfinite(x).all():raise ValueError('invalid Train samples')
        result[name]=dict(mean=x.mean(0).tolist(),std=np.maximum(x.std(0),1e-6).tolist(),raw_std=x.std(0).tolist())
    result['provenance']='train only';return result

def normalize(value,stats,key):
    return ((np.asarray(value)-stats[key]['mean'])/stats[key]['std']).astype(np.float32)

def normalized_loss(prediction,target):return (prediction-target).square().mean()

def parameter_hash(actor):
    h=hashlib.sha256()
    for name,value in actor.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

class BCPolicy:
    def __init__(self,actor,stats):self.actor=actor.cpu().eval().requires_grad_(False);self.stats=stats
    @torch.inference_mode()
    def raw_actions(self,observations):
        x=np.asarray(observations,np.float32)
        if x.ndim!=2 or x.shape[1]!=7 or not np.isfinite(x).all():raise ValueError('only finite7D observations accepted')
        out=[]
        for begin in range(0,len(x),512):
            z=self.actor(torch.from_numpy(normalize(x[begin:begin+512],self.stats,'obs'))).numpy()
            out.append(z*np.asarray(self.stats['action']['std'])+self.stats['action']['mean'])
        return np.concatenate(out).astype(np.float32)
    def predict(self,observation):
        x=np.asarray(observation,np.float32)
        if x.shape!=(7,):raise ValueError('no history/teacher/hiddenstate input')
        return np.clip(self.raw_actions(x[None])[0],-1,1).astype(np.float32)

def save(path,actor,stats,metadata):
    atomic_bytes(path,lambda f:torch.save(dict(kind='uav_bc_mlp_seed0',state=actor.state_dict(),stats=stats,metadata=metadata),f))

def load(path):
    c=torch.load(path,map_location='cpu',weights_only=False)
    if c['kind']!='uav_bc_mlp_seed0':raise ValueError('wrong checkpoint type')
    actor=BCActor();actor.load_state_dict(c['state']);return BCPolicy(actor,c['stats']),c['metadata']
