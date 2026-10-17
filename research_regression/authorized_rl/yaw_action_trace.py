"""Explicit diagnostic-only action adapter; never edit the stored policy."""
import numpy as np
from .goal_heading_trace import collect

class ZeroAppliedYaw:
 def __init__(self,model):self._model=model
 @property
 def policy(self):return self._model.policy
 @property
 def num_timesteps(self):return self._model.num_timesteps
 def predict(self,obs,*,deterministic):
  if deterministic is not True:raise ValueError('R30 deterministic diagnostic only')
  original,state=self._model.predict(obs,deterministic=True);a=np.asarray(original)
  if a.shape!=(4,) or not np.isfinite(a).all() or np.any(np.abs(a)>1):raise ValueError('R30 invalid original action')
  applied=a.copy();applied[3]=0.
  return applied,state

def collect_zero_yaw(model,task,upstream,*,guard,on_record):
 count=0
 def emit(row):
  nonlocal count
  original=np.clip(np.asarray(row['raw_action_mean']),-1.,1.);applied=np.asarray(row['action_clipped'])
  if (not np.array_equal(original[:3],applied[:3]) or applied[3]!=0.
   or row['yaw_rate_command_enu_rad_s']!=0.):raise ValueError('R30 ablation changed translation or applied nonzero yaw')
  count+=1
  on_record({**row,'diagnostic':'APPLIED_YAW_RATE_ZERO_ABLATION_NOT_UNCHANGED_POLICY',
   'policy_action_before_ablation':original.tolist(),'action_applied_after_ablation':applied.tolist(),
   'translation_unchanged_given_observation':True,'applied_yaw_zero_verified':True})
 result=collect(ZeroAppliedYaw(model),task,upstream,guard=guard,on_record=emit)
 if count!=result['policy_steps']:raise ValueError('R30 incomplete ablation records')
 return {**result,'applied_yaw_zero_verified':True,'translation_unchanged_given_observation':True,
  'action_scope':'only applied action index3 changed to0;policy parameters/input/translation for sameobs unchanged;result is altered-action diagnostic'}
