"""Reset-only early distance mixing; no training/physics or RNG side effects."""
import math

def _steps(value):
 if type(value) is not int or value<0:raise ValueError('nonnegative completed policy steps required')
 return value

def stage_at_reset(completed_policy_steps,draw):
 step=_steps(completed_policy_steps)
 if step>=32768:
  if draw is not None:raise ValueError('latephase must not consume distance RNG')
  return 'ORIGINAL'
 if type(draw) not in (int,float) or not math.isfinite(draw) or not 0<=draw<1:
  raise ValueError('earlyphase requires finite uniform draw in[0,1)')
 return 'ORIGINAL' if draw<.5 else 'SHORT'

def distance_seed(seed):
 if type(seed) is not int or seed<0:raise ValueError('nonnegative integer training seed required')
 return 330000+seed

def base_weights(completed_policy_steps,layouts):
 step=_steps(completed_policy_steps)
 if (not isinstance(layouts,(list,tuple)) or len(layouts)!=12
  or any(not isinstance(v,str) or not v for v in layouts) or layouts.count('empty')!=3):
  raise ValueError('expected frozen12base layout contract with3empty')
 return tuple([1/12]*12) if step<32768 else tuple(1/6 if v=='empty' else 1/18 for v in layouts)

def configuration():
 return {'version':'EARLY_DISTANCE_RESET_MIX_R33_V1','switch_nominal_policy_step':32768,
  'early_original_probability_per_reset':.5,'early_short_probability_per_reset':.5,
  'early_base_sampling':'UNIFORM12 regardlessdistance,not weightedempty;preserves earlybase marginal',
  'late_base_sampling':'R28_ORIGINAL3empty each1/6 and9nonempty each1/18',
  'distance_rng_seed_offset':330000,'distance_rng_scope':'separate reproducible PRNG state,not a proof of statistical independence',
  'late_distance_rng':'NOT_CONSUMED','switch_semantics':'next natural reset at/after32768,no mid-episode target change',
  'phase_switch_step_semantics':'first reset at/after32768,not first earlyORIGINAL draw',
  'claim_limit':'probability per episode reset,not fraction of environment timesteps;no action/Reward/PI/physics change'}
