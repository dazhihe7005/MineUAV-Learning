"""R12 fixed, no-learning follow-up to R4 on four newly completed models."""
import copy
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .feedback_spec import probe_cases,verify_file

SUMMARIES={'ARL_R10_PRIOR_DISTANCE_SEED0':'c9fa8bcc9b14fe441903e1f733e7496393e39a36a72c0d2f8e907fc9778ba2ca',
           'ARL_R11_PRIOR_LOWER_LR_SEED0':'7f34a1e98eae2ef289becacb4d1086f7e5dbb0d686d2cae750610452d90b8f26'}
REPORTS={'autonomous_rl_prior_distance_r10.json':'0847525a939fee3eaf23a0cbf32511a7828aeaa7059084ed86ce81ce25ebc876',
         'autonomous_rl_lower_lr_r11.json':'8a5c911619a7905172d94674664011e7cc6b3dbbae7db42e3b6706f53b8c7dc3',
         'autonomous_rl_feedback_r4.json':'110587bd3b0ee51742a1d92397a49fa947c10d2f421254a754abdb8779eb379c'}
R4_RESULT_SHA='83d3766ebdba586282f23e2b3c39a4284cc03791a4017d963af2fa04cef7a673'
OBSERVATION_SHA='453a7723fb4225d141af3c32f847d6b7df12aad9067d488fe343f82aac3702fc'
BACKGROUND_REL='.research_local/autorl_r4_feedback/run_01'


def configuration():
    return {'experiment_id':'AUTORL_R12_LOW_KL_FEEDBACK_ONLY','kind':'READ_ONLY_INFERENCE_NOT_TRAINING',
        'model_count':4,'model_steps':[32768,65536],'case_count':62,'optimizer_updates':0,'training_timesteps':0,
        'new_physics_steps':0,'new_navigation_episodes':0,'new_validation_episodes':0,'test_episodes':0,
        'wall_seconds':120,'workers':1,'device':'cpu','reference_usage':'R4_R3_PRIOR_STEP0_REUSED_NO_NEW_INFERENCE',
        'retention':{'zero_velocity_norm_max_m_s':.05,'max_action_difference_to_reference':.05,
            'all_three_axis_goal_sign_fractions':1.,'all_velocity_response_differences':'WITHIN_REFERENCE_ZERO_TOLERANCE',
            'velocity_response_difference_error_max_m_s':.05,'yaw_response_difference':'STRICTLY_POSITIVE'},
        'claim_limit':'fixed panel prior-function retention, not physical drift, stability, avoidance or navigation success'}


def eligible_budget(budget):
    if budget['stop_requested']:raise OpsError('stop latched; no new audit')
    for a in budget['attempts']:
        if a['status']=='COMPLETE':continue
        if a['status']=='FAIL' and a['experiment_id']=='ARL_R6_DISTANCE_SEED0':continue
        raise OpsError('unresolved/unknown training attempt; no new inference')


def historical_background(root=ROOT):
    root=Path(root);directory=root/BACKGROUND_REL
    verify_file(directory/'result.json',R4_RESULT_SHA);result=read_json(directory/'result.json')
    if (read_json(directory/'complete.json')['result_sha256']!=R4_RESULT_SHA
        or result['optimizer_updates']!=0 or result['physics_steps']!=0 or result['case_count']!=62
        or result['snapshot_files']['observation_identity_sha256']!=OBSERVATION_SHA):raise OpsError('R4 actual background provenance differs')
    for name,value in result['snapshot_files']['files'].items():
        path=directory/name;verify_file(path,value['sha256'])
        if path.stat().st_size!=value['bytes']:raise OpsError('R4 original sensor snapshot size differs')
    return result


def verify_released_raw(report,root):
    from research_ops.identity import safe_path
    for item in report['raw_files']:
        path=safe_path(root,item['path']);verify_file(path,item['sha256'])
        if path.stat().st_size!=item['bytes']:raise OpsError('released raw evidence size differs')


def references(root=ROOT):
    root=Path(root);refs=[]
    try:
        from .lower_lr_profile import verify_lower_lr_proof,lower_lr_experiment
        verify_lower_lr_proof(lower_lr_experiment(),root)
        for name,expected in REPORTS.items():
            path=root/'mujoco/reports'/name;verify_file(path,expected);r=read_json(path)
            if r['implementation_status']!='PASS':raise OpsError('released diagnostic/control implementation incomplete')
            for k in ('review','result_review'):
                if r[k]['status']!='GO':raise OpsError('independent historical review not GO')
                v=r[k]['evidence'];verify_file(root/v['path'],v['sha256'])
            verify_released_raw(r,root)
        for identity,expected in SUMMARIES.items():
            directory=root/'.research_local/authorized_rl_v1/runs'/identity;verify_file(directory/'summary.json',expected)
            s=read_json(directory/'summary.json')
            if (read_json(directory/'complete.json')['summary_sha256']!=expected or s['actual_timesteps']!=65536
                or s['experiment_id']!=identity or s['test_episode_count']!=0):raise OpsError('follow-up requires complete fixed experiments')
            for event in s['events']:
                if event['actual_step'] not in (32768,65536):continue
                record=event['checkpoint'];name=f"{identity}_STEP{event['actual_step']}"
                actual=read_json(directory/(name+'.json'))
                if (actual!=record or record['run_id']!=name or record['training_timesteps']!=event['actual_step']
                    or record['execution_protocol_sha256']!=s['protocol_sha256']
                    or digest({k:v for k,v in actual.items() if k!='sidecar_payload_sha256'})!=actual['sidecar_payload_sha256']
                    or not event['optimizer_and_prediction_reload_verified']):raise OpsError('follow-up checkpoint identity incomplete')
                verify_file(directory/(name+'.zip'),record['checkpoint_sha256'])
                refs.append({'experiment_id':identity,'actual_step':event['actual_step'],'run_id':name,'protocol_sha256':s['protocol_sha256'],
                    'summary_sha256':expected,'checkpoint_path':str((directory/(name+'.zip')).relative_to(root)),
                    'checkpoint_sha256':record['checkpoint_sha256'],'sidecar_path':str((directory/(name+'.json')).relative_to(root)),
                    'sidecar_sha256':sha256(directory/(name+'.json')),'policy_sha256':event['weight_sha256'],'usage':'READ_ONLY_INFERENCE_NOT_RESUME'})
        if len(refs)!=4 or len({v['run_id'] for v in refs})!=4:raise OpsError('follow-up model coverage differs')
        historical_background(root)
        return refs
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid fixed feedback follow-up history') from error


def validate_background(obs):
    from .feedback_probe import observation_identity
    if observation_identity(obs)!=OBSERVATION_SHA:raise OpsError('R12 input is not the sealed R4 historical observation')


def load_background(root=ROOT):
    import numpy as np
    from .feedback_probe import verify_snapshot
    root=Path(root);old=historical_background(root)
    with np.load(root/BACKGROUND_REL/'base_observation.npz',allow_pickle=False) as saved:obs={k:saved[k].copy() for k in saved.files}
    validate_background(obs);verify_snapshot(root/BACKGROUND_REL,obs,old['snapshot_files'])
    prior=next(v for v in old['models'] if v['reference']['experiment_id']=='ARL_R3_PRIOR_STD03_SEED0' and v['reference']['actual_step']==0)
    if prior['policy_sha256']!='6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74':raise OpsError('wrong historical prior panel')
    for row,case in zip(prior['rows'],probe_cases()):
        if any(row[k]!=value for k,value in case.items()):raise OpsError('historical/new counterfactual cases differ')
    if len(prior['rows'])!=62 or len(old['scripted_reference'])!=62:raise OpsError('old panel coverage differs')
    return obs,copy.deepcopy(old['snapshot_files']),copy.deepcopy(prior),copy.deepcopy(old['scripted_reference'])


def make_protocol(registration,root=ROOT):
    from .budget import BudgetStore
    root=Path(root);budget=BudgetStore(root/'.research_local/authorized_rl_v1').status();eligible_budget(budget)
    return {**configuration(),'registration_sha256':digest(registration),'source_identity_sha256':digest(registration['source_identity']),
        'models':references(root),'cases':probe_cases(),'historical_background_result_sha256':R4_RESULT_SHA,
        'source_observation_identity_sha256':OBSERVATION_SHA,'budget_snapshot_sha256':digest(budget)}


def verify_checks(registration,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        a=read_json(root/'.research_local/autorl_r12_checks_02.json')
        if a['registration_sha256']!=digest(registration) or a['source_identity_sha256']!=digest(registration['source_identity']):raise OpsError('stale R12 admission')
        values={}
        for kind in ('operations','tensor','historical','runtime_binding'):
            ref=a['evidence'][kind];rel=f'.research_local/autorl_r12_{kind}_final_02.json'
            if ref['path']!=rel:raise OpsError('unexpected R12 admission path')
            verify_file(root/rel,ref['sha256']);values[kind]=read_json(root/rel)
        op,te,hi,bi=(values[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or op['failures'] or op['errors'] or op['optimizer_updates'] or op['tests_run']!=140
            or len(op['results'])!=140 or any(v['outcome']!='PASS' for v in op['results'])
            or op['source_identity_sha256']!=a['source_identity_sha256'] or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or te['returncode'] or te['tests_run']!=38 or te['optimizer_updates'] or 'Ran 38 tests' not in te['stderr']
            or te['source_identity_sha256']!=a['source_identity_sha256'] or te['registration_sha256']!=a['registration_sha256']
            or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls'] or len(hi['results'])!=885
            or any(v['outcome']!='passed' for v in hi['results']) or bi['returncode']
            or bi['source_before_sha256']!=a['source_identity_sha256'] or bi['source_after_sha256']!=a['source_identity_sha256']
            or bi['registration_sha256']!=a['registration_sha256'] or bi['final']!=a['evidence']['historical']):raise OpsError('R12 complete current no-update tests missing')
        return {'status':'PASS','admission_sha256':sha256(root/'.research_local/autorl_r12_checks_02.json')}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid R12 checks') from error
