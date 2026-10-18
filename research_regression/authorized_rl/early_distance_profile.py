"""R33 fixed early-distance-mixing contrast, same independent cumulative grant."""
import re
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from research_ops.identity import git
from .runtime import ROOT
from .held_goal_profile import held_goal_experiment,verify_held_goal_proof
from .early_distance_mix import configuration
from .closed_failure_proof import _reference

R28_REPORT_SHA='54af04e2001f6612fea6a20d6af1837243a3a7b23c728ddf3d61d3b0921b35fc'
R28_SUMMARY_SHA='542f70407f426a15caf60d567411e17ccdc2c2b8fd0e49a275af1868318cf36c'
R31_REPORT_SHA='1714a99a94675af18b1c446f4d0399bcc4d1d805fdc6febfa307861ceae7460a'
R32_REPORT_SHA='ab586479347674b0d8b59558e51c90e85bf18cfa17b8607975a8bf4f2f63134a'

def early_mix_experiment():
 p=held_goal_experiment()
 p.update(schema=33,experiment_id='ARL_R33_EARLY_DISTANCE_MIX_SEED0',execution_profile='HELD_GOAL_EARLY_DISTANCE_MIX_R33_V1',
  early_distance_mix_configuration=configuration(),early_original_probability_per_reset=.5,
  curriculum_rule='first32768: atnaturalreset choose SHORT/ORIGINAL each.5 with separate distance RNG,baseuniform12;thenORIGINAL atnaturalreset andR28empty.5 weights',
  original_train_sampling='early bothdistance baseuniform12;late3empty each1/6 and9nonempty each1/18',
  control_report_sha256=R28_REPORT_SHA,control_summary_sha256=R28_SUMMARY_SHA,
  response_audit_report_sha256=R32_REPORT_SHA,velocity_feedback_diagnostic_report_sha256=R31_REPORT_SHA,
  single_variable='EARLY_ORIGINAL_DISTANCE_PROBABILITY_PER_NATURAL_RESET_0_TO_05_WITH_UNCHANGED_UNIFORM_EARLY_BASE_MARGINAL',
  prefix_policy_relationship='fresh650a weights equalR28 step0;distance draws canchange experience atfirstreset,no same trainedprefix expected',
  historical_control_scope='completeR28 same heldgoal/geometry/initialization/CNN/PPO/Reward/PI/timesteps control;newresetdistance distribution only',
  initialization_scope='fresh unchanged handprior,not R28checkpoint or explicit actionfeedback/Yawzero wrapper')
 p['budget_head_policy']='archive pre-run head32;only sanctioned new RESERVE/terminal append events may advance livehead;old1..32/grant/attempts immutable,no reset/refund'
 p['evaluation']={'selection':'FINAL_CHECKPOINT','actual_new_validation_episodes':36,'reused_original_step0_episodes':0,
  'goal_yaw_mode':'HOLD_INITIAL_ENU_YAW','physical_sensor_failures_max':0,
  'early_mix_benefit_over_R28':'final heldoriginal successes >R28final0;collisions <=R28final2',
  'control_original_final_success':0,'control_original_final_collision_max':2,
  'held_original_learning_over_prior':'final heldoriginal successes > own heldoriginal step0;collisions <= own step0',
  'held_original_candidate_success_min':4,'held_original_candidate_collision_max':1,
  'held_empty_retention':'heldoriginal empty final successes >= own32k empty successes AND own32k empty successes >=1',
  'held_short_candidate_success_min':4,'held_short_candidate_collision_max':1,
  'held_short_prior_retention':'final heldshort successes >= own heldshort step0;collisions <= own step0',
  'original_fixed_yaw0_transfer':'NOT TESTED',
  'claim_limit':'single seed/small heldgoal Validation;retaining handprior not learned navigation,LiDAR dependence,obstacle-generalization or realflight proof'}
 return p

def validate_early_mix_profile(p):
 clean={k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
 if digest(clean)!=digest(early_mix_experiment()):raise OpsError('unsupported fixed R33 early-distance profile')
 for key in ('source_identity_sha256','registration_sha256'):
  if key in p and (not isinstance(p[key],str) or not re.fullmatch('[0-9a-f]{64}',p[key])):raise OpsError('invalid R33 identity')

def require_preregistered(p,declared):
 validate_early_mix_profile(p)
 if declared!=p:raise OpsError('R33 actual execution differs from frozen preregistration/source identity')

def verify_early_mix_proof(p,root=ROOT):
 root=Path(root);validate_early_mix_profile(p)
 try:
  verify_held_goal_proof(held_goal_experiment(),root)
  from .policy_function_probe import references as verify_completed_R28
  verify_completed_R28(root) # Readonly hashes/sidecars;never deserialize or initialize their models.
  r28=read_json(root/'mujoco/reports/autonomous_rl_held_goal_r28.json')
  if sha256(root/'mujoco/reports/autonomous_rl_held_goal_r28.json')!=R28_REPORT_SHA or sha256(_reference(root,r28['summary']))!=R28_SUMMARY_SHA:
   raise OpsError('R33 fixed completedR28 control differs')
  for filename,h,commit in (('autonomous_rl_velocity_damping_r31.json',R31_REPORT_SHA,'24e8b51b94f0ce520ed150459cff3d81dee0427a'),
   ('autonomous_rl_control_response_r32.json',R32_REPORT_SHA,'4e39632430730bbb4670770ca7ce7c2754e142d4')):
   path=root/'mujoco/reports'/filename
   if sha256(path)!=h:raise OpsError('R33 released diagnostic/audit report differs')
   v=read_json(path)
   if v['implementation_status']!='PASS' or v['training_timesteps'] or v['optimizer_updates'] or v['test_episodes']:
    raise OpsError('R33 diagnostics had unexpected execution scope')
   for key in ('review','result_review'):
    actual=read_json(_reference(root,v[key]['evidence']))
    if v[key]['status']!='GO' or actual['verdict']!='GO' or actual['critical'] or actual['important']:raise OpsError('R33 actual independent predecessor review incomplete')
   for ref in v['raw_files']:_reference(root,ref)
   envelope=read_json(root/'.research_local/ops_v2/versions'/('v2_'+v['registration_sha256'])/'state.json');state=envelope['state']
   if (digest(state)!=envelope['sha256'] or state['phase_complete'] is not True or state['latest_successful_push']['commit']!=commit
    or git(root,'merge-base','--is-ancestor',commit,'HEAD',check=False).returncode):raise OpsError('R33 predecessor not actually published')
  return {'status':'PASS','control_report_sha256':R28_REPORT_SHA,'control_summary_sha256':R28_SUMMARY_SHA,
   'scope':'new fixed distribution contrast only,no retry/resume or hand-edited historical authorization'}
 except (OSError,KeyError,ValueError,TypeError) as error:raise OpsError('missing/invalid fixed R33 completed evidence') from error

def verify_early_mix_checks(p,root=ROOT):
 from research_ops.identity import source_identity
 root=Path(root)
 try:
  path=root/'.research_local/autorl_r33_checks_01.json';a=read_json(path);source=p['source_identity_sha256']
  if (a['status']!='PASS' or a['selected_pass']!=1258 or a['excluded']!=15 or a['optimizer_updates']
   or a['source_identity_sha256']!=source or a['registration_sha256']!=p['registration_sha256']):raise OpsError('stale R33 admission')
  values={}
  for key in ('operations','tensor','historical','runtime_binding'):
   ref=a['evidence'][key];relative=f'.research_local/autorl_r33_{key}_final_01.json'
   if ref['path']!=relative or sha256(root/relative)!=ref['sha256']:raise OpsError('R33 admission path/hash differs')
   values[key]=read_json(root/relative)
  op,te,hi,bi=(values[k] for k in ('operations','tensor','historical','runtime_binding'))
  if (not op['complete'] or (op['tests_run'],op['failures'],op['errors'],op['optimizer_updates'])!=(280,0,0,0)
   or len(op['results'])!=280 or any(v['outcome']!='PASS' for v in op['results'])
   or op['source_identity_sha256']!=source or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
   or te['returncode'] or te['tests_run']!=93 or te['optimizer_updates'] or 'Ran 93 tests' not in te['stderr']
   or te['source_identity_sha256']!=source or te['registration_sha256']!=p['registration_sha256']
   or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
   or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results']) or bi['returncode']
   or bi['registration_sha256']!=p['registration_sha256'] or bi['source_before_sha256']!=source or bi['source_after_sha256']!=source
   or bi['final']!=a['evidence']['historical']):raise OpsError('R33 full current no-update tests incomplete')
  return {'status':'PASS','admission_sha256':sha256(path)}
 except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid R33 admission') from error
