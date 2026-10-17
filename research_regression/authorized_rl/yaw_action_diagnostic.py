"""R30 explicit applied-action ablation, not a goal-yaw change or training."""
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from research_ops.identity import git
from .runtime import ROOT
from .policy_function_probe import references as fixed_r28_references,R28_ID
from .held_goal_profile import held_goal_manifest

R29_SHA='e9edcb1523230060b96be28f0a4aee730bf1cf4b62135a3bdbf7229f6b54ec6b'
R29_COMMIT='c3b7197ed715638db7c1d18c962fe28f7b41ba1a'
FINAL_POLICY_SHA='f7349c40158b1f064542bdf8180f22cf9b68f11aee9ab563f732e678bfc85475'

def configuration():
 return {'experiment_id':'AUTORL_R30_YAW_ACTION_ABLATION_ONLY','kind':'READ_ONLY_FROZEN_MODEL_CHANGED_APPLIED_ACTION_NOT_POLICY_LEARNING',
  'ablation':'APPLIED_YAW_RATE_ACTION_ZERO_TRANSLATION_UNCHANGED_FOR_SAME_OBSERVATION',
  'goal_yaw_mode':'HOLD_INITIAL_ENU_YAW','new_episodes':2,'reused_R28_control_episodes':2,
  'max_policy_steps_per_episode':375,'wall_seconds':180,'workers':1,'device':'cpu',
  'training_timesteps':0,'optimizer_updates':0,'test_episodes':0,
  'hypothesis':'both new empty original-distance cases meet unchanged stop standard vs two existing finalmodel timeouts => supports yaw-action involvement in this pair only',
  'acceptance':'bothsuccess and no physical/sensor failure;elseFAIL,do not reduce stopping threshold',
  'claim_limit':'new action-ablation diagnostics,not unchanged learned-policy success,training improvement,solecausal mechanism or generalization'}

def references(root=ROOT):
 root=Path(root)
 try:
  inputs=fixed_r28_references(root)
  p=root/'mujoco/reports/autonomous_rl_policy_function_r29.json'
  if sha256(p)!=R29_SHA:raise OpsError('R30 released R29 report differs')
  report=read_json(p)
  if (report['implementation_status']!='PASS' or report['result_review']['status']!='GO'
   or report['optimizer_updates'] or report['physics_ticks'] or report['new_episodes'] or report['test_episodes']):raise OpsError('R30 R29 proof scope differs')
  envelope=read_json(root/'.research_local/ops_v2/versions'/('v2_'+report['registration_sha256'])/'state.json');s=envelope['state']
  if (digest(s)!=envelope['sha256'] or s['phase_complete'] is not True or s['latest_successful_push']['commit']!=R29_COMMIT
   or git(root,'merge-base','--is-ancestor',R29_COMMIT,'HEAD',check=False).returncode):raise OpsError('R30 R29 not actually published')
  for ref in report['raw_files']:
   if sha256(root/ref['path'])!=ref['sha256']:raise OpsError('R30 R29 immutable raw differs')
  model=inputs['models'][-1]
  prereg=read_json(root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
  cases=[t for t in held_goal_manifest(prereg)['validation']['ORIGINAL'] if t['scenario']['layout']=='empty']
  summary=read_json(root/'.research_local/authorized_rl_v1/runs'/R28_ID/'summary.json')
  event=summary['events'][-1]
  controls=[v for v in event['validation']['episodes'] if v['layout']=='empty']
  if (model['step']!=65536 or event['actual_step']!=65536 or len(cases)!=len(controls) or len(cases)!=2
   or [v['scenario_identity_sha256'] for v in controls]!=[v['identity_sha256'] for v in cases]
   or any(v['terminal_class']!='time_limit' or v['policy_steps']!=375 for v in controls)
   or event['weight_sha256']!=FINAL_POLICY_SHA
   or event['validation']['policy_sha256_before_after']!=FINAL_POLICY_SHA):raise OpsError('R30 exact model/task/control mismatch')
  return {'model':model,'cases':cases,'controls':controls,'expected_policy_sha256':FINAL_POLICY_SHA,
   'R28_summary_sha256':sha256(root/'.research_local/authorized_rl_v1/runs'/R28_ID/'summary.json'),
   'R29_report_sha256':R29_SHA,'budget_snapshot_sha256':inputs['budget_snapshot_sha256']}
 except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid fixed R30 inputs') from error

def make_protocol(reg,root=ROOT):
 return {**configuration(),**references(root),'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity'])}

def evaluate(rows,spec):
 if len(rows)!=2 or [v['task_identity_sha256'] for v in rows]!=[v['identity_sha256'] for v in spec['cases']]:raise ValueError('R30 missing/duplicate/wrong task')
 for v in rows:
  n,ticks,maximum=(v['policy_steps'],v['physics_ticks'],v['stop_metrics']['max_consecutive_joint_samples'])
  if (type(n) is not int or not 1<=n<=375 or type(ticks) is not int or not 20*(n-1)<ticks<=20*n
   or type(maximum) is not int or not 0<=maximum<=n or v['outcome'] not in ('success','collision','time_limit')
   or v['outcome']=='time_limit' and n!=375 or v['outcome']=='success' and maximum<5
   or v['policy_sha256_before_after']!=FINAL_POLICY_SHA or type(v['training_timesteps_before_after']) is not int
   or v['training_timesteps_before_after']!=65536 or v['optimizer_unchanged'] is not True
   or v['applied_yaw_zero_verified'] is not True or v['translation_unchanged_given_observation'] is not True):
   raise ValueError('R30 unsafe/changed/unsubstantiated ablation trace')
 successes=sum(v['outcome']=='success' for v in rows)
 return {'yaw_action_contrast':'PASS' if successes==2 else 'FAIL','new_ablation_successes':successes,'reused_control_successes':0,
  'scope':'applied yaw-action contribution in two changed-policy cases only,not learned navigation or solecausal proof'}
