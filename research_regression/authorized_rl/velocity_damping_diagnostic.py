"""R31 fixed explicit outer action feedback;not a policy-weight/controller edit."""
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from research_ops.identity import git
from .runtime import ROOT
from .yaw_action_diagnostic import references as yaw_inputs,FINAL_POLICY_SHA

R30_SHA='1cc55a09678d40d0765613ebaf46b1973cce6b0c6451714d606bb668dc27aa49'
R30_COMMIT='64fe3ed22281f3929f8bb3efa36a47e864b85dc9'

def configuration():
 return {'experiment_id':'AUTORL_R31_VELOCITY_DAMPING_DIAGNOSTIC_ONLY',
  'kind':'EXPLICIT_HYBRID_ACTION_FEEDBACK_ABLATION_NOT_POLICY_LEARNING',
  'feedback_gain':.3,'feedback_frame':'WORLD_ENU_VELOCITY_FROM_CURRENT_SIM_STATE7',
  'action_scales':[1.5,1.5,1.,1.],
  'formula':'float32 clip(original_clipped_xyz - .3*current_ENU_velocity/[1.5,1.5,1],-1,1);appliedYaw=0',
  'order':'original SB3 clip first,thenexplicit physical-unit feedback andsecondbounds clip;not unmodified model action',
  'single_variable':'R30 same finalmodel/same heldtask/Yawzero baseline;additional velocityfeedback gain0 to.3',
  'gain_rationale':'R29 approximate local XYposition gain~.43/s;goalboundary .10m and speedthreshold.15m/s imply .43*.10/.15~.287;fixed .3 rounded before evaluation,not an identified global plant or certified safety gain',
  'new_episodes':2,'reused_R30_control_episodes':2,'max_policy_steps_per_episode':375,'wall_seconds':180,'workers':1,'device':'cpu',
  'training_timesteps':0,'optimizer_updates':0,'test_episodes':0,
  'hypothesis':'both new cases meet unchanged stopping criterion vs two existing R30 timeouts supports explicit velocity-feedback effect in this pair only',
  'acceptance':'two successes,no physical/sensor failures;otherwiseFAIL without threshold retuning',
  'claim_limit':'hybrid action intervention,not learned navigation,unmodifiedpolicy success,causal uniqueness,obstacle avoidance or generalization'}

def references(root=ROOT):
 root=Path(root)
 try:
  prior=yaw_inputs(root);path=root/'mujoco/reports/autonomous_rl_yaw_action_r30.json'
  if sha256(path)!=R30_SHA:raise OpsError('R31 released R30 report differs')
  report=read_json(path)
  if (report['implementation_status']!='PASS' or report['scientific_result']!='FAIL' or report['result_review']['status']!='GO'
   or report['optimizer_updates'] or report['training_timesteps'] or report['test_episodes']):raise OpsError('R31 R30 evidence scope differs')
  for ref in report['raw_files']:
   if sha256(root/ref['path'])!=ref['sha256']:raise OpsError('R31 immutable R30 raw differs')
  envelope=read_json(root/'.research_local/ops_v2/versions'/('v2_'+report['registration_sha256'])/'state.json');state=envelope['state']
  if (digest(state)!=envelope['sha256'] or state['phase_complete'] is not True or state['latest_successful_push']['commit']!=R30_COMMIT
   or git(root,'merge-base','--is-ancestor',R30_COMMIT,'HEAD',check=False).returncode):raise OpsError('R31 R30 not actually published')
  controls=report['cases'];cases=prior['cases']
  if (len(controls)!=2 or [v['task_identity_sha256'] for v in controls]!=[v['identity_sha256'] for v in cases]
   or any(v['outcome']!='time_limit' or v['policy_steps']!=375 or v['stop_metrics']['joint_samples']!=0
    or v['applied_yaw_zero_verified'] is not True or v['policy_sha256_before_after']!=FINAL_POLICY_SHA for v in controls)):
   raise OpsError('R31 exact two prior yawzero control cases mismatch')
  return {'model':prior['model'],'cases':cases,'controls':controls,'expected_policy_sha256':FINAL_POLICY_SHA,
   'R30_report_sha256':R30_SHA,'budget_snapshot_sha256':prior['budget_snapshot_sha256']}
 except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid fixed R31 inputs') from error

def make_protocol(reg,root=ROOT):
 return {**configuration(),**references(root),'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity'])}

def evaluate(rows,spec):
 if len(rows)!=2 or [v['task_identity_sha256'] for v in rows]!=[v['identity_sha256'] for v in spec['cases']]:raise ValueError('R31 missing/duplicate/wrong task')
 for v in rows:
  n,ticks,maximum=v['policy_steps'],v['physics_ticks'],v['stop_metrics']['max_consecutive_joint_samples']
  if (type(n) is not int or not 1<=n<=375 or type(ticks) is not int or not 20*(n-1)<ticks<=20*n
   or type(maximum) is not int or not 0<=maximum<=n or v['outcome'] not in ('success','collision','time_limit')
   or v['outcome']=='time_limit' and n!=375 or v['outcome']=='success' and maximum<5
   or v['policy_sha256_before_after']!=FINAL_POLICY_SHA or type(v['training_timesteps_before_after']) is not int
   or v['training_timesteps_before_after']!=65536 or v['optimizer_unchanged'] is not True
   or v['applied_feedback_verified'] is not True or v['applied_yaw_zero_verified'] is not True):raise ValueError('R31 unsafe/unsubstantiated feedback trace')
 count=sum(v['outcome']=='success' for v in rows)
 return {'velocity_feedback_contrast':'PASS' if count==2 else 'FAIL','new_hybrid_action_successes':count,'reused_R30_control_successes':0,
  'scope':'changed-action velocity feedback contribution only,not original learnedpolicy navigation or solecausal proof'}
