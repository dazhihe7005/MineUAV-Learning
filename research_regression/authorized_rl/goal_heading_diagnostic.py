"""R27 task-goal yaw contrast, no training eligibility or old result mutation."""
import copy,math,re
from pathlib import Path
from research_ops.storage import OpsError,read_json,digest,sha256
from research_ops.identity import git
from .runtime import ROOT
from .interrupted_yaw_diagnostic import references as interrupted_references
from .closed_failure_proof import _reference

R26_REPORT_SHA='1796d350cd472c8f21b572796adaf3a185056fe25fe10f403936f7a2e58385a8'
R26_RESULT_SHA='dfd021c0e4c922fcd5dd353c98d48731f29557ce22c54c881aca0729378e6c45'
R26_COMMIT='4310738b5f7f9472f321242690457e4d2bff561d'

def configuration():
    return {'experiment_id':'AUTORL_R27_GOAL_HEADING_DIAGNOSTIC_ONLY','kind':'NEW_TASK_HEADING_NON_LEARNING_CONTRAST',
        'goal_yaw_mode':'HOLD_INITIAL_ENU_YAW','new_episodes':2,'reused_control_episodes':2,'max_policy_steps_per_episode':375,
        'wall_seconds':180,'workers':1,'device':'cpu','training_timesteps':0,'optimizer_updates':0,'test_episodes':0,
        'single_variable':'declared task goal yaw0 to initialENUyaw;not controller/physics/policy weights/reward change',
        'hypothesis':'both hold-initial original-distance empty tasks succeed vs same prior R26 two timeouts',
        'claim_limit':'two altered-task empty Validation cases;not original-task success,learned improvement,controller bug proof or generalization'}

def heading_task(row):
    yaw=float(row['start_yaw_rad'])
    if not math.isfinite(yaw):raise ValueError('invalid task heading')
    task={'scenario':copy.deepcopy(row),'goal_yaw_mode':'HOLD_INITIAL_ENU_YAW','goal_yaw_rad':math.atan2(math.sin(yaw),math.cos(yaw))}
    return {**task,'identity_sha256':digest(task)}

def references(root=ROOT):
    root=Path(root)
    try:
        prior=interrupted_references(root)
        path=root/'mujoco/reports/autonomous_rl_interrupted_yaw_r26.json'
        if sha256(path)!=R26_REPORT_SHA:raise OpsError('R27 fixed R26 release changed')
        report=read_json(path);review=report['result_review'];actual=read_json(_reference(root,review['evidence']))
        if (report['implementation_status']!='PASS' or report['training_timesteps'] or report['optimizer_updates'] or report['test_episodes']
                or review['status']!='GO' or actual['verdict']!='GO' or actual['critical'] or actual['important']):raise OpsError('R26 diagnostic not reviewed')
        for ref in report['raw_files']:_reference(root,ref)
        source=root/'.research_local/autorl_r26_interrupted_yaw_diagnostic/run_01/result.json'
        if sha256(source)!=R26_RESULT_SHA or read_json(source.parent/'complete.json')['result_sha256']!=R26_RESULT_SHA:raise OpsError('R26 actual raw completion changed')
        archived=root/'.research_local/ops_v2/versions'/('v2_'+report['registration_sha256'])/'state.json'
        envelope=read_json(archived);state=envelope['state']
        if (digest(state)!=envelope['sha256'] or state['phase_complete'] is not True or state['latest_successful_push']['commit']!=R26_COMMIT
                or git(root,'merge-base','--is-ancestor',R26_COMMIT,'HEAD',check=False).returncode):raise OpsError('R26 not actually published')
        model=prior['models'][0];selected=[v for v in prior['cases'] if v['distance_stage']=='ORIGINAL']
        cases=[heading_task(v['scenario']) for v in selected]
        controls=[v for v in read_json(source)['cases'] if v['checkpoint_step']==0 and v['distance_stage']=='ORIGINAL']
        if (len(cases)!=2 or len(controls)!=2 or any(v['outcome']!='time_limit' for v in controls)
                or [v['scenario_identity_sha256'] for v in controls]!=[v['scenario']['identity_sha256'] for v in cases]
                or any(v['policy_sha256_before_after']!=model['expected_policy_sha256'] for v in controls)):
            raise OpsError('R27 fixed control/model/task mismatch')
        return {'model':model,'cases':cases,'controls':controls,'R26_result_sha256':R26_RESULT_SHA,
            'budget_snapshot_sha256':prior['budget_snapshot_sha256'],'old_training_gate_still_closed':True,'new_training_authority':False}
    except (OSError,KeyError,ValueError,TypeError) as error:raise OpsError('missing/invalid R27 fixed released proof') from error

def make_protocol(reg,root=ROOT):
    return {**configuration(),**references(root),'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity'])}

def evaluate(rows,spec):
    if len(rows)!=2 or [r['task_identity_sha256'] for r in rows]!=[v['identity_sha256'] for v in spec['cases']]:raise ValueError('R27 missing/duplicate/wrong task')
    fingerprint=None
    for r,c in zip(rows,spec['cases']):
        n=r['policy_steps'];ticks=r['physics_ticks'];maximum=r['stop_metrics']['max_consecutive_joint_samples'];h=r['policy_sha256_before_after']
        if (r['scenario_identity_sha256']!=c['scenario']['identity_sha256'] or r['goal_yaw_rad']!=c['goal_yaw_rad']
                or type(n) is not int or not 1<=n<=375 or type(ticks) is not int or not 20*(n-1)<ticks<=20*n
                or r['optimizer_unchanged'] is not True or type(r['training_timesteps_before_after']) is not int or r['training_timesteps_before_after']!=0
                or r['outcome'] not in ('success','collision','time_limit') or r['outcome']=='time_limit' and n!=375
                or type(maximum) is not int or not 0<=maximum<=n or r['outcome']=='success' and maximum<5
                or not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) or fingerprint is not None and h!=fingerprint):raise ValueError('R27 invalid changed/mismatched trace state')
        fingerprint=h
    successes=sum(v['outcome']=='success' for v in rows)
    return {'heading_contrast':'PASS' if successes==2 else 'FAIL','new_task_successes':successes,'control_successes':0,
        'claim':'fixed prior,altered task heading,2empty cases only;not learned improvement or original task/generalization proof'}
