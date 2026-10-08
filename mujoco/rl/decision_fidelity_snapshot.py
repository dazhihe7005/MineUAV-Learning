"""Full nominal MuJoCo/Python-controller snapshot, never a 7D reconstruction."""
import copy
import hashlib
from dataclasses import dataclass
import mujoco
import numpy as np
from latent_dynamics_data import json_hash

RESOURCES={'model','data','allocator','velocity_controller','observation_space','action_space','_viewer'}


@dataclass
class Snapshot:
    model: object
    data: object
    environment: dict
    controller: dict
    allocator: dict
    previous_action: np.ndarray


def capture(env,previous_action):
    previous=np.asarray(previous_action,np.float32)
    if previous.shape!=(4,) or not np.isfinite(previous).all() or env.render_mode is not None:
        raise ValueError('headless nominal snapshot and finite previous action4 required')
    return Snapshot(env.model,copy.copy(env.data),
        copy.deepcopy({k:v for k,v in vars(env).items() if k not in RESOURCES}),
        copy.deepcopy(vars(env.velocity_controller)),copy.deepcopy(vars(env.allocator)),previous.copy())


def restore(env,snapshot):
    if env.model is not snapshot.model: raise ValueError('snapshot belongs to a different compiled model')
    # Copy EVERY MjData field, including ctrl/act/history/solver warmstart/derived
    # fields. No mj_forward recomputation that could alter warmstart or caches.
    mujoco.mj_copyData(env.data,env.model,snapshot.data)
    for k in list(vars(env)):
        if k not in RESOURCES: del vars(env)[k]
    vars(env).update(copy.deepcopy(snapshot.environment))
    vars(env.velocity_controller).clear(); vars(env.velocity_controller).update(copy.deepcopy(snapshot.controller))
    vars(env.allocator).clear(); vars(env.allocator).update(copy.deepcopy(snapshot.allocator))
    return snapshot.previous_action.copy()


def canonical(value):
    if isinstance(value,np.ndarray): return dict(dtype=str(value.dtype),shape=list(value.shape),values=value.tolist())
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,np.random.Generator): return value.bit_generator.state
    if isinstance(value,dict): return {str(k):canonical(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [canonical(v) for v in value]
    if hasattr(value,'__dict__'): return dict(type=type(value).__qualname__,fields=canonical(vars(value)))
    return value


def fingerprint(snapshot):
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(snapshot.model,spec),np.float64)
    mujoco.mj_getState(snapshot.model,snapshot.data,state,spec)
    return json_hash(dict(integration_sha256=hashlib.sha256(state.astype('<f8').tobytes()).hexdigest(),
        environment=canonical(snapshot.environment),controller=canonical(snapshot.controller),
        allocator=canonical(snapshot.allocator),previous_action=canonical(snapshot.previous_action)))
