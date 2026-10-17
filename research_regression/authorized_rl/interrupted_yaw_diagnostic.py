"""R26 fixed post-interruption inference scope, not a training eligibility gate."""
import copy
import json
import re
import zipfile
from pathlib import Path
from research_ops.identity import git
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .budget import BudgetStore
from .closed_failure_proof import _reference,verify_r20_closed,eligible_after_closed_r20

R24_ID='ARL_R24_TRAIN_YAW_COVERAGE_SEED0'
R24_REPORT_SHA='b1ef4ab998b391f1bf57bae9a5f7c735c19f950706b0ee15da69c2ee5811e07b'
R25_REPORT_SHA='f244ed2dbdeacb29718db9699a4435288c27743f857faba7b0cce05f0fe5de80'
R24_EXEC_SHA='3595c1be7190fa6288439547ab2a472fd2c1cc8f2f783654ac25ad23d56b038a'


def configuration():
    return {'experiment_id':'AUTORL_R26_INTERRUPTED_YAW_DIAGNOSTIC_ONLY',
        'kind':'NEW_FROZEN_CHECKPOINT_DIAGNOSTIC_NOT_FORMAL_RESULT_OR_RESUME',
        'model_count':2,'case_count':4,'episodes':8,'checkpoint_steps':[0,32768],
        'max_policy_steps_per_episode':375,'wall_seconds':600,'workers':1,'device':'cpu',
        'training_timesteps':0,'optimizer_updates':0,'test_episodes':0,'formal_completion_claim':False,
        'hypothesis':'two short empty Validation tasks: prior2success and32k fewer successes supports retention loss in this subset only',
        'claim_limit':'posthoc eight NEW diagnostic episodes;R24 remains interrupted with original five criteria NOT TESTED;not finalmodel/generalization'}


def references(root=ROOT):
    root=Path(root)
    try:
        reports=[]
        for name,expected,commit,complete,verdict in (
            ('autonomous_rl_train_yaw_r24.json',R24_REPORT_SHA,'996d2b10ab39e4303476450f6b510142a35d31a1',False,'GO_REPORT_ONLY'),
            ('autonomous_rl_ledger_fixture_r25.json',R25_REPORT_SHA,'be547c53894317048ea0decc2c13f39cede76cc4',True,'GO')):
            path=root/'mujoco/reports'/name
            if sha256(path)!=expected:raise OpsError('R26 released input report differs')
            report=read_json(path);reports.append(report)
            review=report['result_review'];actual=read_json(_reference(root,review['evidence']))
            if (review['status']!=verdict or actual['verdict']!=verdict or review['critical'] or review['important']
                    or actual['critical'] or actual['important']):raise OpsError('R26 actual result review missing')
            directory=root/'.research_local/ops_v2/versions'/('v2_'+report['registration_sha256'])
            envelope=read_json(directory/'state.json');state=envelope['state']
            if (digest(state)!=envelope['sha256'] or state['phase_complete'] is not complete
                    or state['latest_successful_push']['commit']!=commit
                    or git(root,'merge-base','--is-ancestor',commit,'HEAD',check=False).returncode):
                raise OpsError('R26 input release not preserved/published')
        failed,repair=reports
        if (failed['implementation_status']!='FAIL' or failed['execution_status']!='INTERRUPTED'
                or failed['scientific_result']!='NOT TESTED' or failed['actual_new_validation_episodes']!=0
                or failed['completion_analysis_executed'] or failed['automatic_retry']
                or repair['implementation_status']!='PASS' or repair['budget_unchanged'] is not True
                or repair['new_training_ready']!='NO'):raise OpsError('R26 failed/repair scope differs')
        for ref in failed['raw_files']:_reference(root,ref)
        run=root/'.research_local/authorized_rl_v1/runs'/R24_ID
        if {p.name for p in run.iterdir()}!={Path(v['path']).name for v in failed['raw_files']}:
            raise OpsError('R24 raw directory changed or false completion added')
        disposition=read_json(_reference(root,failed['append_only_disposition']))
        actual_review=read_json(_reference(root,failed['disposition_review']))
        event_path=root/'.research_local/authorized_rl_v1/events/000030.json';event=read_json(event_path)
        if (sha256(event_path)!=disposition['new_event_sha256'] or event['sequence']!=30 or event['kind']!='FAIL'
                or event['data']['experiment_id']!=R24_ID or event['data']['reason']!=disposition['reason']
                or actual_review['verdict']!='GO_DISPOSITION_ONLY' or actual_review['critical'] or actual_review['important']):
            raise OpsError('R26 append-only R24 closure proof invalid')
        budget=BudgetStore(root/'.research_local/authorized_rl_v1').status()
        if budget['stop_requested']:raise OpsError('R26 stop latched')
        entry=next(v for v in budget['attempts'] if v['experiment_id']==R24_ID)
        if entry['status']!='FAIL' or entry['result']!=event['data'] or entry['protocol']!=failed['configuration']:
            raise OpsError('R26 actual failed ledger differs')
        for item in budget['attempts']:
            if item['status']=='COMPLETE':continue
            if item['status']=='FAIL' and item['experiment_id'] in ('ARL_R6_DISTANCE_SEED0','ARL_R20_TEMPORAL_CREDIT_SEED0',R24_ID):continue
            raise OpsError('R26 unknown/pending attempt blocks diagnostic')
        verify_r20_closed(root)
        try:eligible_after_closed_r20(budget,root)
        except OpsError as error:
            if 'unresolved/unknown failed attempt' not in str(error):raise
        else:raise OpsError('old training gate unexpectedly admits R24')
        models=[]
        for ref in failed['checkpoints']:
            step=ref['step'];run_id=f'{R24_ID}_STEP{step}';archive=run/(run_id+'.zip');side=run/(run_id+'.json');record=read_json(side)
            if (step not in (0,32768) or sha256(archive)!=ref['zip_sha256'] or sha256(side)!=ref['sidecar_sha256']
                    or record['checkpoint_sha256']!=ref['zip_sha256'] or record['execution_protocol_sha256']!=R24_EXEC_SHA
                    or digest({k:v for k,v in record.items() if k!='sidecar_payload_sha256'})!=record['sidecar_payload_sha256']):
                raise OpsError('R26 fixed checkpoint/sidecar differs')
            with zipfile.ZipFile(archive) as z:
                data=json.loads(z.read('data'))
                if z.testzip() is not None or 'policy.optimizer.pth' not in z.namelist():raise OpsError('R26 archive integrity failed')
            if (data['num_timesteps'],data['gamma'],data['gae_lambda'])!=(step,.99,.95):raise OpsError('R26 checkpoint temporal configuration differs')
            models.append({'step':step,'run_id':run_id,'checkpoint_path':str(archive.relative_to(root)),
                'checkpoint_sha256':sha256(archive),'sidecar_sha256':sha256(side),'execution_protocol_sha256':R24_EXEC_SHA,
                'expected_policy_sha256':failed['initial_weights']['policy'] if step==0 else None,
                'use':'DESERIALIZE_FOR_READ_ONLY_INFERENCE_ONLY;legacy loader mode namedWARM_START,not training initialization/resume'})
        if [v['step'] for v in models]!=[0,32768]:raise OpsError('R26 model pair differs')
        from .short_goal import short_manifest
        prereg=read_json(root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        cases=[]
        for stage,rows in (('SHORT',short_manifest(prereg)['validation']),('ORIGINAL',prereg['scenario_manifest']['validation'])):
            selected=[v for v in rows if v['layout']=='empty' and v['split']=='validation']
            if len(selected)!=2:raise OpsError('R26 fixed two empty Validation tasks missing')
            cases.extend({'distance_stage':stage,'scenario':copy.deepcopy(v)} for v in selected)
        return {'models':models,'cases':cases,'budget_snapshot_sha256':digest(budget),
            'R24_permanent_FAIL':True,'old_training_gate_still_closed':True,'new_training_authority':False}
    except (OSError,KeyError,TypeError,ValueError,StopIteration,zipfile.BadZipFile) as error:
        raise OpsError('missing/invalid R26 released diagnostic inputs') from error


def make_protocol(registration,root=ROOT):
    return {**configuration(),**references(root),'registration_sha256':digest(registration),
        'source_identity_sha256':digest(registration['source_identity'])}


def evaluate_retention(rows,spec):
    expected=[(m['step'],c['distance_stage'],c['scenario']['identity_sha256']) for m in spec['models'] for c in spec['cases']]
    actual=[(v['checkpoint_step'],v['distance_stage'],v['scenario_identity_sha256']) for v in rows]
    if len(expected)!=8 or len(set(expected))!=8 or actual!=expected:raise ValueError('R26 missing/duplicate/wrong model/case order')
    fingerprints={}
    for row in rows:
        n=row['policy_steps'];ticks=row['physics_ticks'];outcome=row['outcome'];maximum=row['stop_metrics']['max_consecutive_joint_samples']
        if (type(n) is not int or not 1<=n<=375 or type(ticks) is not int or not 20*(n-1)<ticks<=20*n
                or type(row['checkpoint_step']) is not int or type(row['training_timesteps_before_after']) is not int
                or row['optimizer_unchanged'] is not True or row['training_timesteps_before_after']!=row['checkpoint_step']
                or outcome not in ('success','collision','time_limit') or type(maximum) is not int or not 0<=maximum<=n
                or outcome=='success' and maximum<5 or outcome=='time_limit' and n!=375):raise ValueError('R26 invalid/unsafe/changed training state')
        h=row['policy_sha256_before_after']
        if not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h):raise ValueError('R26 invalid policy fingerprint')
        previous=fingerprints.setdefault(row['checkpoint_step'],h)
        if previous!=h:raise ValueError('R26 mixed policy fingerprints')
    counts={}
    for step in (0,32768):
        for stage in ('SHORT','ORIGINAL'):
            selected=[v for v in rows if v['checkpoint_step']==step and v['distance_stage']==stage]
            counts[f'{step}_{stage}']={k:sum(v['outcome']==k for v in selected) for k in ('success','collision','time_limit')}
    prior=counts['0_SHORT']['success'];later=counts['32768_SHORT']['success']
    h=('NOT_TESTED_PRIOR_NOT_ESTABLISHED' if prior!=2 else
       'SUPPORTED_IN_THIS_SUBSET' if later<prior else 'NOT_SUPPORTED_IN_THIS_SUBSET')
    return {'counts':counts,'short_retention_loss':h,'formal_R24_result':'STILL_NOT_TESTED_INCOMPLETE_RUN'}
