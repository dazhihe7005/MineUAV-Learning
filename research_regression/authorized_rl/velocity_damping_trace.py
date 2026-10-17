"""Fixed explicit .3 velocity feedback outside the stored network,diagnostic only."""
import numpy as np
from .yaw_action_trace import ZeroAppliedYaw
from .goal_heading_trace import collect

def apply_feedback(yawzero_action,state):
 a=np.asarray(yawzero_action);v=np.asarray(state)
 if (a.shape!=(4,) or v.shape!=(7,) or not np.isfinite(a).all() or not np.isfinite(v).all()
  or np.any(np.abs(a)>1) or a[3]!=0.):raise ValueError('R31 invalid finite bounded baseline/state7')
 output=np.asarray(a,dtype=np.float32).copy()
 velocity=np.asarray(v[3:6],dtype=np.float32)
 if not np.isfinite(velocity).all():raise ValueError('R31 velocity cast overflow')
 output[:3]=np.clip(output[:3]-np.float32(.3)*velocity/np.array([1.5,1.5,1.],dtype=np.float32),-1.,1.)
 if not np.isfinite(output).all():raise ValueError('R31 nonfinite feedback action')
 return output

class DampedTranslationPolicy(ZeroAppliedYaw):
 def predict(self,obs,*,deterministic):
  if not isinstance(obs,dict) or 'state' not in obs:raise ValueError('R31 requires current state7')
  baseline,hidden=super().predict(obs,deterministic=deterministic)
  return apply_feedback(baseline,obs['state']),hidden

def collect_damped(model,task,upstream,*,guard,on_record):
 count=0
 def emit(row):
  nonlocal count
  original=np.clip(np.asarray(row['raw_action_mean'],dtype=np.float32),-1.,1.);baseline=original.copy();baseline[3]=0.
  applied=np.asarray(row['action_clipped'],dtype=np.float32);expected=apply_feedback(baseline,np.asarray(row['state_pre'],dtype=np.float32))
  if not np.array_equal(applied,expected) or row['yaw_rate_command_enu_rad_s']!=0.:raise ValueError('R31 applied feedback does not match frozen physical-unit formula')
  count+=1
  on_record({**row,'diagnostic':'EXPLICIT_VELOCITY_FEEDBACK_HYBRID_ACTION_NOT_UNCHANGED_POLICY',
   'feedback_gain':.3,'feedback_velocity_enu_m_s':row['state_pre'][3:6],
   'policy_action_before_intervention':original.tolist(),'yawzero_baseline_action':baseline.tolist(),
   'action_applied_after_feedback':applied.tolist(),'applied_feedback_verified':True,'applied_yaw_zero_verified':True})
 answer=collect(DampedTranslationPolicy(model),task,upstream,guard=guard,on_record=emit)
 if count!=answer['policy_steps']:raise ValueError('R31 incomplete action intervention records')
 return {**answer,'applied_feedback_verified':True,'applied_yaw_zero_verified':True,
  'action_scope':'explicit physical-unit velocity feedback .3 andYawzero;network parameters unchanged,hybrid diagnostic not original learnedpolicy'}
