"""Finite, deterministic counterfactual inference; never controls an environment."""
import copy
import time
import hashlib
from .feedback_spec import validate_case


def observation_identity(obs):
    import numpy as np
    from research_ops.storage import digest
    return digest({k:{'dtype':np.asarray(v).dtype.str,'shape':list(np.asarray(v).shape),
        'bytes_sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k,v in sorted(obs.items())})


def seal_snapshot(directory,obs):
    import numpy as np
    from research_ops.storage import OpsError,sha256
    with np.load(directory/'base_observation.npz',allow_pickle=False) as saved:
        if set(saved.files)!=set(obs) or observation_identity({k:saved[k] for k in saved.files})!=observation_identity(obs):
            raise OpsError('saved snapshot does not match actual inference input')
    return {'observation_identity_sha256':observation_identity(obs),'files':{
        name:{'sha256':sha256(directory/name),'bytes':(directory/name).stat().st_size}
        for name in ('sensor_frame.npz','base_observation.npz','snapshot.json')}}


def verify_snapshot(directory,obs,sealed):
    from research_ops.storage import OpsError
    if seal_snapshot(directory,obs)!=sealed:raise OpsError('runtime snapshot changed before audit completion')


def infer_panel(model,base,cases):
    import numpy as np
    import torch
    from lidar_integration.ppo_training_runner import _weights_sha
    from lidar_integration.ppo_formal_validation import _optimizer_states_equal
    if not isinstance(base,dict) or set(base)!={'lidar','state','heading','lidar_age'}:
        raise ValueError('fixed policy observation keys required')
    shapes={'lidar':(3,12,24),'state':(7,),'heading':(2,),'lidar_age':(1,)}
    for k,shape in shapes.items():
        a=np.asarray(base[k])
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('invalid fixed sensor/state observation')
    if (np.any((base['lidar']<0)|(base['lidar']>1))
        or not np.isin(base['lidar'][1:],(0.,1.)).all()
        or np.any(base['lidar'][2]>base['lidar'][1])
        or np.any(base['lidar'][0][base['lidar'][2]==0]!=1)
        or not np.isclose(np.linalg.norm(base['heading']),1,atol=1e-6)
        or np.any(np.abs(base['state'][:6])>100) or abs(float(base['state'][6]))>np.pi
        or not 0<=float(base['lidar_age'][0])<=.15):raise ValueError('invalid sensor range/heading/age')
    if not isinstance(cases,list) or not cases:raise ValueError('explicit counterfactual cases required')
    for c in cases:validate_case(c)
    observations={k:np.repeat(np.asarray(a,dtype=np.float32)[None,...],len(cases),axis=0) for k,a in base.items()}
    observations['state']=np.asarray([c['state'] for c in cases],dtype=np.float32)
    before=_weights_sha(model.policy);optimizer=copy.deepcopy(model.policy.optimizer.state_dict())
    rng=torch.get_rng_state().clone();begun=time.perf_counter()
    with torch.inference_mode():
        model.policy.set_training_mode(False)
        tensors,_=model.policy.obs_to_tensor(observations)
        distribution=model.policy.get_distribution(tensors).distribution
        means=distribution.mean.cpu().numpy().copy()
        std=distribution.stddev.cpu().numpy().copy()
        values=model.policy.predict_values(tensors).cpu().numpy().reshape(-1).copy()
        actions=model.predict(observations,deterministic=True)[0]
    elapsed=(time.perf_counter()-begun)*1000
    if (means.shape!=(len(cases),4) or std.shape!=means.shape or values.shape!=(len(cases),)
        or not all(np.isfinite(a).all() for a in (means,std,values,actions))
        or not np.array_equal(np.clip(means,-1,1),actions)):
        raise ValueError('invalid deterministic mean/std/value/clip contract')
    weights_ok=_weights_sha(model.policy)==before
    optimizer_ok=_optimizer_states_equal(optimizer,model.policy.optimizer.state_dict())
    rng_ok=bool(torch.equal(rng,torch.get_rng_state()))
    if not (weights_ok and optimizer_ok and rng_ok):raise ValueError('read-only probe changed model/optimizer/RNG')
    rows=[{**c,'raw_mean':means[i].tolist(),'std':std[i].tolist(),
        'normalized_action':actions[i].tolist(),'declared_ENU_velocity_yaw_rate_command':(actions[i]*np.array([1.5,1.5,1.,1.])).tolist(),
        'critic_value':float(values[i])} for i,c in enumerate(cases)]
    return {'rows':rows,'policy_sha256':before,'weights_unchanged':weights_ok,
        'optimizer_state_unchanged':optimizer_ok,'torch_rng_unchanged':rng_ok,'optimizer_updates':0,
        'batch_inference_ms':elapsed,'claim_limit':'counterfactual deterministic responses, not physical motion or success rate'}
