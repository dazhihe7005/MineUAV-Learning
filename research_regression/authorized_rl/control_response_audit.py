"""Read-only response arithmetic,not a controller or identified transfer model.

The caller must separately verify raw identities and full physical trace contracts.
25Hz rows contain the LAST100Hz controller request,not its full inner history.
"""
import math,statistics

def _vector(value,n):
 if not isinstance(value,(list,tuple)) or len(value)!=n or any(type(x) not in (int,float) or not math.isfinite(x) for x in value):
  raise ValueError('audit requires complete finite vectors')
 return list(value)

def response_samples(rows):
 if not rows:raise ValueError('empty response trace')
 output=[];previous=0.
 for row in rows:
  pre,post=row['pre_time_s'],row['post_time_s']
  if (type(pre) not in (int,float) or type(post) not in (int,float) or not math.isfinite(pre) or not math.isfinite(post)
   or type(row['physics_steps']) is not int or row['physics_steps']!=20
   or abs(pre-previous)>1e-8 or abs(post-pre-.04)>1e-8):raise ValueError('audit requires full20tick monotonic physical intervals')
  v0=_vector(row['pre_qvel'],6);v1=_vector(row['post_qvel'],6);q=_vector(row['post_qpos'],7)[3:7]
  integral=_vector(row['pi_integral_error_post'],3);request=_vector(row['desired_acceleration_post_enu_m_s2'],3)
  command=_vector(row['velocity_command_enu_m_s'],3)
  if abs(sum(x*x for x in q)-1)>1e-7:raise ValueError('audit quaternion not unit')
  d,s=row['distance_m'],row['speed_m_s']
  if (type(d) not in (int,float) or type(s) not in (int,float) or not math.isfinite(d) or not math.isfinite(s)
   or min(d,s)<0 or abs(s-math.sqrt(sum(x*x for x in v1[:3])))>1e-8):raise ValueError('audit invalid physical distance/speed')
  w,x,y,z=q
  output.append({'interval_start_s':pre,'interval_end_s':post,'interval_mid_time_s':(pre+post)/2,
   'last_control_time_s':post-.01,
   'average_acceleration_enu_m_s2':[(b-a)/(post-pre) for a,b in zip(v0[:3],v1[:3])],
   'desired_acceleration_enu_m_s2':request,'velocity_command_enu_m_s':command,
   'post_velocity_enu_m_s':v1[:3],'post_body_rates_rad_s':v1[3:6],
   'pi_integral_error_m':integral,'pi_integral_acceleration_enu_m_s2':[a*k for a,k in zip(integral,(.5,.5,.8))],
   'roll_rad':math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
   'pitch_rad':math.asin(max(-1.,min(1.,2*(w*y-z*x)))),'distance_m':d,'speed_m_s':s})
  previous=post
 return output

def lag_alignment(samples,lag_steps,*,axis=0,window=(3.,7.)):
 if type(lag_steps) is not int or not 1<=lag_steps<=25 or type(axis) is not int or not 0<=axis<3:
  raise ValueError('positive causal lag1..25 and axis0..2 required')
 lo,hi=window
 if not math.isfinite(lo) or not math.isfinite(hi) or not 0<=lo<hi:raise ValueError('invalid fixed audit window')
 pairs=[];x=[];y=[]
 for j in range(lag_steps,len(samples)):
  ref,response=samples[j-lag_steps],samples[j]
  tr,ta=ref['last_control_time_s'],response['interval_mid_time_s']
  if not lo<=tr<=hi or not lo<=ta<=hi:continue
  if ta<tr or abs(ta-tr-(.04*lag_steps-.01))>1e-8:raise ValueError('noncausal or inconsistent response clock')
  a,b=ref['desired_acceleration_enu_m_s2'][axis],response['average_acceleration_enu_m_s2'][axis]
  if not math.isfinite(a) or not math.isfinite(b):raise ValueError('nonfinite correlation signal')
  x.append(a);y.append(b);pairs.append({'reference_time_s':tr,'response_time_s':ta})
 coefficient=None
 if len(x)>=8:
  mx,my=statistics.fmean(x),statistics.fmean(y)
  dx=[a-mx for a in x];dy=[b-my for b in y]
  xx=math.fsum(a*a for a in dx);yy=math.fsum(b*b for b in dy)
  if xx>1e-20 and yy>1e-20:coefficient=max(-1.,min(1.,math.fsum(a*b for a,b in zip(dx,dy))/math.sqrt(xx*yy)))
 return {'status':'OBSERVED' if coefficient is not None else 'NOT_OBSERVABLE','lag_policy_intervals':lag_steps,
  'reference_to_response_offset_s':.04*lag_steps-.01,'axis':axis,'fixed_window_s':[lo,hi],
  'pairs_count':len(pairs),'pearson_r':coefficient,'pairs':pairs,
  'interpretation':'descriptive cross-correlation of last-request samples and interval-average measured acceleration;not actual control latency,causal identification or plant phase-margin proof'}

def summarize_response(samples):
 if not samples:raise ValueError('empty audit samples')
 alignments=[lag_alignment(samples,i) for i in range(1,26)]
 observed=[v for v in alignments if v['pearson_r'] is not None]
 peak=max(observed,key=lambda v:v['pearson_r']) if observed else None
 near=[v for v in samples if v['distance_m']<.1]
 return {'policy_samples':len(samples),'simulation_duration_s':samples[-1]['interval_end_s'],
  'max_abs_roll_rad':max(abs(v['roll_rad']) for v in samples),'max_abs_pitch_rad':max(abs(v['pitch_rad']) for v in samples),
  'max_abs_body_rates_rad_s':[max(abs(v['post_body_rates_rad_s'][j]) for v in samples) for j in range(3)],
  'max_pi_integral_acceleration_enu_m_s2':[max(abs(v['pi_integral_acceleration_enu_m_s2'][j]) for v in samples) for j in range(3)],
  'near_goal_samples':len(near),'near_goal_joint_low_speed_samples':sum(v['speed_m_s']<.15 for v in near),
  'near_goal_mean_command_norm_m_s':statistics.fmean(math.sqrt(sum(x*x for x in v['velocity_command_enu_m_s'])) for v in near) if near else None,
  'near_goal_mean_actual_speed_m_s':statistics.fmean(v['speed_m_s'] for v in near) if near else None,
  'x_alignment_grid':[{k:a for k,a in v.items() if k!='pairs'} for v in alignments],
  'selected_grid_max_x_alignment':{k:a for k,a in peak.items() if k!='pairs'} if peak else 'NOT_OBSERVABLE',
  'selection_limit':'maximum of preregistered25gridpoints,descriptive selection only,no statistical significance or measured latency claim',
  'motor_saturation':'NOT_OBSERVABLE','causal_root_cause':'UNVERIFIED',
  'missing_evidence':['no100Hz intermediate velocities/requests','no perrotor/saturation/wrench telemetry','no perrow raw LiDAR tensor','no intervention or newphysics performed']}
