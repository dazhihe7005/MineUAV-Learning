"""R28 fixed held-goal-yaw task learning, not original-yaw task success."""
import copy,re
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from research_ops.identity import git
from .runtime import ROOT
from .empty_rehearsal_profile import rehearsal_experiment
from .short_goal import short_manifest,BASE_MANIFEST_SHA
from .goal_heading_diagnostic import heading_task
from .closed_failure_proof import _reference
from .closed_yaw_failure import verify_r24_closed

R27_REPORT_SHA='b1f161de91d1d0aba235760a7361d5cf0dd3658a696d775566b86979205591e6'
R27_RESULT_SHA='a48ea6a81b6d650d8388091c01d493f9da80a5e1c69c112a99e67517d4012563'
R17_REPORT_SHA='d87993588f64db612daab4ee3091f64437a090eeb2b82532906f770a740a12f6'
R17_SUMMARY_SHA='950a588ea593f5eb0ece276748624e590efb231472a5c5aacb67098946da2f72'

def held_goal_manifest(prereg):
    if digest(prereg['scenario_manifest'])!=BASE_MANIFEST_SHA:raise ValueError('held-goal requires fixed original L6 geometry manifest')
    short=short_manifest(prereg)
    return {'schema':'R28_HELD_INITIAL_ENU_GOAL_YAW_TASKS_V1','base_manifest_sha256':BASE_MANIFEST_SHA,
        'short_manifest_sha256':digest(short),
        **{split:{stage:[heading_task(row) for row in rows[split]] for stage,rows in
            (('SHORT',short),('ORIGINAL',prereg['scenario_manifest']))} for split in ('train','validation')},
        'test':'SEALED_NOT_INCLUDED','scope':'new task goals only;unchanged geometry/start/target-position/initialYaw,not original goalYaw0 task'}

def verify_held_goal_manifest(manifest,prereg):
    if manifest!=held_goal_manifest(prereg):raise ValueError('fixed held-goal task manifest differs')

def held_goal_experiment():
    p=rehearsal_experiment();del p['expected_prefix_policy_sha256']
    manifest=held_goal_manifest(read_json(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json'))
    p.update(schema=28,experiment_id='ARL_R28_HOLD_GOAL_YAW_SEED0',execution_profile='CENTERED_STD01_HELD_INITIAL_GOAL_YAW_R28_V1',
        gamma=.99,gae_lambda=.95,goal_yaw_mode='HOLD_INITIAL_ENU_YAW',held_goal_manifest_sha256=digest(manifest),
        held_goal_train_task_count=24,held_goal_validation_task_count=12,
        control_report_sha256=R17_REPORT_SHA,control_summary_sha256=R17_SUMMARY_SHA,heading_diagnostic_report_sha256=R27_REPORT_SHA,
        single_variable='TASK_GOAL_YAW_ZERO_TO_INITIAL_ENU_YAW_IN_TRAIN_AND_DECLARED_VALIDATION',
        prefix_policy_relationship='task observation/action experience changes from first reset;no old prefix equality expected',
        historical_control_scope='completeR17 provenance only;different goalheading tasks,no fair original-task success-rate improvement claim',
        checkpoint_base_metadata_scope='L6 describes base geometry/start/positiongoal and old task;actual newgoalYaw bound by independentmanifest/protocol/reset/episode/eval evidence')
    p['evaluation']={'selection':'FINAL_CHECKPOINT','actual_new_validation_episodes':36,'reused_original_step0_episodes':0,
        'goal_yaw_mode':'HOLD_INITIAL_ENU_YAW','physical_sensor_failures_max':0,
        'held_original_learning_over_prior':'final heldoriginal successes > own heldoriginal step0;collisions <= own step0',
        'held_original_candidate_success_min':4,'held_original_candidate_collision_max':1,
        'held_empty_retention':'heldoriginal empty final successes >= own32k empty successes AND own32k empty successes >=1',
        'held_short_candidate_success_min':4,'held_short_candidate_collision_max':1,
        'held_short_prior_retention':'final heldshort successes >= own heldshort step0;collisions <= own step0',
        'original_fixed_yaw0_transfer':'NOT TESTED','claim_limit':'new held-initial-heading tasks,small Validation/seed0;not originalYaw0,learned LiDAR dependence,realmine or multi-seed proof'}
    return p

def validate_held_goal_profile(p):
    clean={k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
    if digest(clean)!=digest(held_goal_experiment()):raise OpsError('unsupported fixed R28 held-goal profile')
    for key in ('source_identity_sha256','registration_sha256'):
        if key in p and (not isinstance(p[key],str) or not re.fullmatch('[0-9a-f]{64}',p[key])):raise OpsError('invalid R28 identity')

def verify_held_goal_proof(p,root=ROOT):
    root=Path(root);validate_held_goal_profile(p)
    try:
        verify_r24_closed(root)
        from .temporal_credit_profile import verify_temporal_proof,temporal_experiment
        verify_temporal_proof(temporal_experiment(),root)
        path=root/'mujoco/reports/autonomous_rl_goal_heading_r27.json'
        if sha256(path)!=R27_REPORT_SHA:raise OpsError('R28 fixed R27 heading diagnostic differs')
        v=read_json(path)
        if (v['implementation_status']!='PASS' or v['scientific_result']!='PASS' or v['assessment']['new_task_successes']!=2
                or v['training_timesteps'] or v['optimizer_updates'] or v['test_episodes']):raise OpsError('R27 held-goal feasibility evidence missing')
        for key in ('review','result_review'):
            review=v[key];actual=read_json(_reference(root,review['evidence']))
            if review['status']!='GO' or actual['verdict']!='GO' or actual['critical'] or actual['important']:raise OpsError('R27 actual independent review incomplete')
        for ref in v['raw_files']:_reference(root,ref)
        result=root/'.research_local/autorl_r27_goal_heading_diagnostic/run_01/result.json'
        if sha256(result)!=R27_RESULT_SHA or read_json(result.parent/'complete.json')['result_sha256']!=R27_RESULT_SHA:raise OpsError('R27 actual complete result differs')
        archived=root/'.research_local/ops_v2/versions'/('v2_'+v['registration_sha256'])/'state.json'
        envelope=read_json(archived);state=envelope['state'];commit='dc410d29ecc912c92b63ac5ca546d145805e9e65'
        if (digest(state)!=envelope['sha256'] or state['phase_complete'] is not True or state['latest_successful_push']['commit']!=commit
                or git(root,'merge-base','--is-ancestor',commit,'HEAD',check=False).returncode):raise OpsError('R27 not actually published')
        return {'status':'PASS','new_task_diagnostic_report_sha256':R27_REPORT_SHA,'control_summary_sha256':R17_SUMMARY_SHA,
            'scope':'new heldheading tasks;not retry/resume,not original fixedYaw0 success'}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid R28 released proof') from error

def verify_held_goal_checks(p,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        path=root/'.research_local/autorl_r28_checks_01.json';a=read_json(path);source=p['source_identity_sha256']
        if a['source_identity_sha256']!=source or a['registration_sha256']!=p['registration_sha256']:
            raise OpsError('stale R28 no-update admission')
        values={}
        for key in ('operations','tensor','historical','runtime_binding'):
            ref=a['evidence'][key];relative=f'.research_local/autorl_r28_{key}_final_01.json'
            if ref['path']!=relative or sha256(root/relative)!=ref['sha256']:raise OpsError('R28 admission hash/path differs')
            values[key]=read_json(root/relative)
        op,te,hi,bi=(values[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or op['tests_run']!=253 or op['failures'] or op['errors'] or op['optimizer_updates']
                or op['source_identity_sha256']!=source or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
                or len(op['results'])!=253 or any(v['outcome']!='PASS' for v in op['results'])
                or te['returncode'] or te['tests_run']!=83 or te['optimizer_updates'] or 'Ran 83 tests' not in te['stderr']
                or te['source_identity_sha256']!=source or te['registration_sha256']!=p['registration_sha256']
                or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
                or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results']) or bi['returncode']
                or bi['source_before_sha256']!=source or bi['source_after_sha256']!=source
                or bi['registration_sha256']!=p['registration_sha256'] or bi['final']!=a['evidence']['historical']):
            raise OpsError('current R28 full no-update tests incomplete')
        return {'status':'PASS','admission_sha256':sha256(path)}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid R28 no-update tests') from error
