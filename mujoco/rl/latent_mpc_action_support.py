"""Train-only support proxy and a candidate-only frozen-MPC intervention."""
import hashlib
from pathlib import Path
import numpy as np
from latent_dynamics_data import file_hash,json_hash
from latent_mpc_core import RandomShootingMPC,sample_candidates,N_CANDIDATES,HORIZON

TOLERANCE=1e-7


def compute_support(train_path,expected_sha256):
    path=Path(train_path)
    if path.name!='train.npz' or file_hash(path)!=expected_sha256:
        raise ValueError('verified Train archive only for support statistics')
    with np.load(path,allow_pickle=False) as data:
        actions=np.asarray(data['actions'],np.float64)
    if actions.ndim!=2 or actions.shape[1]!=4 or not len(actions) or not np.isfinite(actions).all() or (np.abs(actions)>1).any():
        raise ValueError('finite legal executed Train actions required')
    lower=np.quantile(actions,.025,axis=0); upper=np.quantile(actions,.975,axis=0)
    if (lower>=upper).any(): raise ValueError('degenerate Train support box')
    result=dict(name='Train central-95% support proxy',split='train',count=len(actions),
        lower=lower.tolist(),upper=upper.tolist(),quantiles=[.025,.975],quantile_method='NumPy linear',
        source=str(path),source_sha256=expected_sha256,
        actions_sha256=hashlib.sha256(actions.astype('<f4').tobytes()).hexdigest(),
        sample_rule='ALL executed Train commands including terminal unpaired commands; no Val/Test/benchmark/holdout',
        units='normalized environment command',tolerance=TOLERANCE,
        caveat='per-dimension central95% box; NOT complete in-distribution manifold or strict OOD boundary')
    return dict(result,sha256=json_hash(result))


class NoiseRecorder:
    """One unmodified RNG draw; records paired epsilon without consuming more RNG."""
    def __init__(self,rng): self.rng=rng; self.draw=None
    def normal(self,*,size):
        if self.draw is not None: raise AssertionError('one fixed draw per decision required')
        self.draw=self.rng.normal(size=size)
        return self.draw


def sample_projected_candidates(previous,train_std,low,high,support_low,support_high,rng):
    previous,train_std,low,high,lower,upper=[np.asarray(x,np.float64) for x in
        (previous,train_std,low,high,support_low,support_high)]
    if any(x.shape!=(4,) or not np.isfinite(x).all() for x in (previous,train_std,low,high,lower,upper)):
        raise ValueError('finite four-dimensional statistics required')
    if (lower>=upper).any() or (lower<low).any() or (upper>high).any() or (train_std<=0).any() or (low>=high).any() or (low>0).any() or (high<0).any() or (previous<low).any() or (previous>high).any():
        raise ValueError('support must be valid subset of legal environment box')
    noise=rng.normal(size=(N_CANDIDATES-2,HORIZON,4))*(.5*train_std)
    candidates=np.empty((N_CANDIDATES,HORIZON,4),np.float32)
    displacement=np.zeros((N_CANDIDATES,HORIZON,4),np.float64)
    for j,anchor in enumerate((np.zeros(4),previous)):
        candidates[j]=np.clip(anchor,lower,upper)
        displacement[j]=np.clip(anchor,lower,upper)-anchor
    current=np.broadcast_to(previous,(N_CANDIDATES-2,4)).copy()
    for k in range(HORIZON):
        legal=np.clip(current+noise[:,k],low,high)
        current=np.clip(legal,lower,upper)
        candidates[2:,k]=current
        displacement[2:,k]=current-legal
    return candidates,dict(projection_magnitudes=np.linalg.norm(displacement,axis=2),
        projected_elements=np.abs(displacement)>TOLERANCE,
        anchor_projected_candidates=int(np.any(np.abs(displacement[:2])>TOLERANCE,axis=(1,2)).sum()))


def audited_candidates(previous,train_std,low,high,rng,support):
    recorder=NoiseRecorder(rng)
    if support is None:
        candidates=sample_candidates(previous,train_std,low,high,recorder)
        audit=dict(projection_magnitudes=np.zeros((512,10)),projected_elements=np.zeros((512,10,4),bool),anchor_projected_candidates=0)
    else:
        candidates,audit=sample_projected_candidates(previous,train_std,low,high,*support,recorder)
    epsilon=recorder.draw*(.5*np.asarray(train_std,np.float64))
    if epsilon.shape!=(510,10,4): raise AssertionError('paired draw shape changed')
    audit['epsilon_sha256']=hashlib.sha256(epsilon.astype('<f8').tobytes()).hexdigest()
    return candidates,audit


class ActionSupportMPC(RandomShootingMPC):
    """Same latent/cost/selection/timing path, only candidate generator differs."""
    def __init__(self,model,statistics,low,high,seed=0,support=None):
        self.support=None if support is None else tuple(np.asarray(x,np.float64) for x in support)
        self.candidate_audit=None
        super().__init__(model,statistics,low,high,seed)

    def make_candidates(self,previous_action):
        candidates,self.candidate_audit=audited_candidates(previous_action,self.statistics['action']['std'],self.low,self.high,self.rng,self.support)
        return candidates
