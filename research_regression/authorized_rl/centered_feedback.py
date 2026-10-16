"""R15: two newly trained centered policies, fixed historic counterfactual input."""
import copy
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .feedback_spec import verify_file,probe_cases
from .feedback_followup import eligible_budget,verify_released_raw,validate_background

R14_REPORT_SHA='489c3aa7f211f393c8d552a501663d9e4e02556befa9751a0404aa9348fffe70'
R14_SUMMARY_SHA='0cd11a7161b2518ee9aa48c180d31435c723ae6f9978cd785128d788a074fafd'
R13_RESULT_SHA='1107e3fe8388b6b36910a469a1a35324c51efac2a6fabeaf2dd41eb923876c5d'


def configuration():
    return {'experiment_id':'AUTORL_R15_CENTERED_FEEDBACK_ONLY','kind':'READ_ONLY_INFERENCE_NOT_TRAINING',
        'model_count':2,'model_steps':[32768,65536],'case_count':62,'new_inference_rows':124,
        'optimizer_updates':0,'training_timesteps':0,'physics_steps':0,'new_scans':0,'episodes':0,
        'validation_episodes':0,'test_episodes':0,'wall_seconds':120,'workers':1,'device':'cpu',
        'reference_usage':'R13_SAME_CENTERED_CLASS_INITIAL_PANEL_REUSED_NO_NEW_INFERENCE',
        'retention':{'zero_velocity_norm_max_m_s':.05,'max_action_difference_to_reference':.05,
            'all_three_axis_goal_sign_fractions':1.,'all_velocity_response_differences':'WITHIN_REFERENCE_ZERO_TOLERANCE',
            'velocity_response_difference_error_max_m_s':.05,'yaw_response_difference':'STRICTLY_POSITIVE'},
        'claim_limit':'fixed historical-background counterfactual feedback, not physical braking, drift, navigation success or generalization'}


def references(root=ROOT):
    root=Path(root)
    try:
        from .centered_training_profile import verify_centered_proof,centered_experiment
        verify_centered_proof(centered_experiment(),root)
        path=root/'mujoco/reports/autonomous_rl_centered_actor_r14.json';verify_file(path,R14_REPORT_SHA)
        report=read_json(path)
        if report['implementation_status']!='PASS' or report['test_episodes']!=0:
            raise OpsError('R14 implementation/test isolation incomplete')
        for key in ('review','result_review'):
            v=report[key]
            if v['status']!='GO':raise OpsError('R14 actual review not GO')
            verify_file(root/v['evidence']['path'],v['evidence']['sha256'])
        verify_released_raw(report,root)
        run=root/'.research_local/authorized_rl_v1/runs/ARL_R14_CENTERED_ACTOR_SEED0'
        verify_file(run/'summary.json',R14_SUMMARY_SHA);s=read_json(run/'summary.json')
        if (read_json(run/'complete.json')['summary_sha256']!=R14_SUMMARY_SHA
            or s['actual_timesteps']!=65536 or s['test_episode_count']!=0
            or s['experiment_id']!='ARL_R14_CENTERED_ACTOR_SEED0'):
            raise OpsError('R14 actual completed identity differs')
        refs=[]
        for e in s['events']:
            if e['actual_step'] not in (32768,65536):continue
            rec=e['checkpoint'];name=f"{s['experiment_id']}_STEP{e['actual_step']}"
            side=run/(name+'.json');archive=run/(name+'.zip');actual=read_json(side)
            if (actual!=rec or rec['run_id']!=name or rec['training_timesteps']!=e['actual_step']
                or rec['execution_protocol_sha256']!=s['protocol_sha256']
                or digest({k:v for k,v in actual.items() if k!='sidecar_payload_sha256'})!=actual['sidecar_payload_sha256']
                or not e['optimizer_and_prediction_reload_verified']
                or e['actual_actor_parameterization']['version']!='SAME_BACKGROUND_STATE_CENTERED_ACTOR_V1'):
                raise OpsError('R15 actual centered checkpoint identity differs')
            verify_file(archive,rec['checkpoint_sha256'])
            refs.append({'experiment_id':s['experiment_id'],'actual_step':e['actual_step'],'run_id':name,
                'protocol_sha256':s['protocol_sha256'],'summary_sha256':R14_SUMMARY_SHA,
                'checkpoint_path':str(archive.relative_to(root)),'checkpoint_sha256':rec['checkpoint_sha256'],
                'sidecar_path':str(side.relative_to(root)),'sidecar_sha256':sha256(side),
                'policy_sha256':e['weight_sha256'],'usage':'READ_ONLY_INFERENCE_NOT_RESUME'})
        if [v['actual_step'] for v in refs]!=[32768,65536]:raise OpsError('R15 requires exactly two new checkpoints')
        return refs
    except (OSError,KeyError,TypeError,ValueError) as error:
        raise OpsError('missing/corrupt fixed R15 history') from error


def load_background(root=ROOT):
    from .feedback_followup import load_background as old_background
    root=Path(root);obs,seal,_,scripted=old_background(root)
    path=root/'.research_local/autorl_r13_centered_preflight/run_01/result.json';verify_file(path,R13_RESULT_SHA)
    old=read_json(path)
    if (read_json(path.parent/'complete.json')['result_sha256']!=R13_RESULT_SHA
        or old['historical_sensor_snapshot']!=seal or old['optimizer_updates']!=0 or old['physics_steps']!=0):
        raise OpsError('R13 same-class panel provenance differs')
    prior=copy.deepcopy(old['new_policy_panel'])
    if prior['policy_sha256']!='20f5c226abc762e5aa4a9046ba400b2b0f7012b0da6715d14204b097cae06001' or len(prior['rows'])!=62:
        raise OpsError('R15 historical initial centered policy differs')
    for row,c in zip(prior['rows'],probe_cases()):
        if any(row[k]!=v for k,v in c.items()):raise OpsError('R15 historical/current cases differ')
    prior['reference']={'experiment_id':'AUTORL_R13_CENTERED_ACTOR_PREFLIGHT_ONLY','actual_step':0,
        'usage':'HISTORICAL_PANEL_REUSED_NO_NEW_INFERENCE','source_result_sha256':R13_RESULT_SHA}
    return obs,seal,prior,scripted


def make_protocol(registration,root=ROOT):
    from .budget import BudgetStore
    root=Path(root);budget=BudgetStore(root/'.research_local/authorized_rl_v1').status();eligible_budget(budget)
    # Loading arrays is deliberately deferred to the fixed inference worker.
    return {**configuration(),'registration_sha256':digest(registration),'source_identity_sha256':digest(registration['source_identity']),
        'models':references(root),'cases':probe_cases(),'historical_centered_initial_result_sha256':R13_RESULT_SHA,
        'budget_snapshot_sha256':digest(budget)}


def verify_checks(registration,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        path=root/'.research_local/autorl_r15_checks_01.json';a=read_json(path);source=digest(registration['source_identity'])
        if a['registration_sha256']!=digest(registration) or a['source_identity_sha256']!=source:raise OpsError('stale R15 admission')
        vv={}
        for k in ('operations','tensor','historical','runtime_binding'):
            rel=f'.research_local/autorl_r15_{k}_final_01.json';ref=a['evidence'][k]
            if ref['path']!=rel:raise OpsError('unexpected R15 evidence path')
            verify_file(root/rel,ref['sha256']);vv[k]=read_json(root/rel)
        op,te,hi,bi=(vv[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or op['failures'] or op['errors'] or op['optimizer_updates'] or op['tests_run']!=159
            or len(op['results'])!=159 or any(v['outcome']!='PASS' for v in op['results'])
            or op['source_identity_sha256']!=source or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or te['returncode'] or te['tests_run']!=53 or te['optimizer_updates'] or 'Ran 53 tests' not in te['stderr']
            or te['source_identity_sha256']!=source or te['registration_sha256']!=digest(registration)
            or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
            or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results']) or bi['returncode']
            or bi['source_before_sha256']!=source or bi['source_after_sha256']!=source
            or bi['registration_sha256']!=digest(registration) or bi['final']!=a['evidence']['historical']):
            raise OpsError('R15 current complete no-update checks missing')
        return {'status':'PASS','admission_sha256':sha256(path)}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid R15 current tests') from error
