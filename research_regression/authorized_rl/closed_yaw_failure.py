"""Explicit R24 failure closure for NEW reviewed profiles, never old retries."""
from pathlib import Path
from research_ops.identity import git
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .budget import BudgetStore
from .closed_failure_proof import _reference,verify_r20_closed

R24_REPORT_SHA='b1ef4ab998b391f1bf57bae9a5f7c735c19f950706b0ee15da69c2ee5811e07b'
R24_EVENT_SHA='093ec35e5bf0e8c87562ed1c2c109e3c91f7b9222ebdaeb77967e62848ce422b'
R24_ID='ARL_R24_TRAIN_YAW_COVERAGE_SEED0'

def verify_r24_closed(root=ROOT):
    root=Path(root)
    try:
        path=root/'mujoco/reports/autonomous_rl_train_yaw_r24.json'
        if sha256(path)!=R24_REPORT_SHA:raise OpsError('R24 published failure identity differs')
        report=read_json(path);review=report['result_review'];actual=read_json(_reference(root,review['evidence']))
        if (review['status']!='GO_REPORT_ONLY' or actual['verdict']!='GO_REPORT_ONLY' or review['critical'] or review['important']
                or actual['critical'] or actual['important']):raise OpsError('R24 actual failure publication review missing')
        archived=root/'.research_local/ops_v2/versions'/('v2_'+report['registration_sha256'])/'state.json'
        envelope=read_json(archived);state=envelope['state'];commit='996d2b10ab39e4303476450f6b510142a35d31a1'
        if (digest(state)!=envelope['sha256'] or state['phase_complete'] is not False or state['latest_successful_push']['commit']!=commit
                or git(root,'merge-base','--is-ancestor',commit,'HEAD',check=False).returncode):raise OpsError('R24 failure release not preserved/published')
        if (report['implementation_status']!='FAIL' or report['execution_status']!='INTERRUPTED' or report['scientific_result']!='NOT TESTED'
                or report['automatic_retry'] or report['completion_analysis_executed'] or report['completion_finalizer_executed']
                or report['actual_new_validation_episodes']!=0 or report['full_65536_steps_and_3600_seconds_permanently_charged'] is not True):
            raise OpsError('R24 permanent failure/unfinished scope changed')
        for ref in report['raw_files']:_reference(root,ref)
        run=root/'.research_local/authorized_rl_v1/runs'/R24_ID
        if {v.name for v in run.iterdir()}!={Path(v['path']).name for v in report['raw_files']}:raise OpsError('R24 raw directory changed or false completion added')
        _reference(root,report['interruption_audit'])
        disposition=read_json(_reference(root,report['append_only_disposition']))
        disposition_review=read_json(_reference(root,report['disposition_review']))
        event_path=root/'.research_local/authorized_rl_v1/events/000030.json';event=read_json(event_path)
        if (sha256(event_path)!=R24_EVENT_SHA or disposition['new_event_sha256']!=R24_EVENT_SHA
                or disposition['after_head_count']!=30 or disposition['automatic_retry']
                or event['sequence']!=30 or event['kind']!='FAIL' or event['data']['experiment_id']!=R24_ID
                or event['data']['reason']!=disposition['reason'] or disposition_review['verdict']!='GO_DISPOSITION_ONLY'
                or disposition_review['critical'] or disposition_review['important']):raise OpsError('R24 append-only failed disposition invalid')
        budget=BudgetStore(root/'.research_local/authorized_rl_v1').status();entry=next(v for v in budget['attempts'] if v['experiment_id']==R24_ID)
        if entry['status']!='FAIL' or entry['result']!=event['data'] or entry['protocol']!=report['configuration']:
            raise OpsError('R24 actual ledger no longer matches permanent failure')
        if (entry['protocol']['timesteps'],entry['protocol']['wall_seconds'])!=(65536,3600):raise OpsError('R24 full reservation altered')
        return {'status':'PASS','R24_status':'PERMANENT_FAIL_NO_RETRY','R24_scientific_result':'NOT TESTED','refund':False,
            'failed_report_sha256':R24_REPORT_SHA,'failed_event_sha256':R24_EVENT_SHA,
            'scope':'new separately reviewed profile only;does not launch,authorize,retry,resume or refund any experiment'}
    except (OSError,KeyError,TypeError,ValueError,StopIteration) as error:raise OpsError('missing/invalid closed R24 failure proof') from error

def eligible_after_closed_r24(budget,root=ROOT):
    if budget['stop_requested']:raise OpsError('stop latched;no new research')
    for attempt in budget['attempts']:
        if attempt['status']=='COMPLETE':continue
        if attempt['status']=='FAIL' and attempt['experiment_id'] in ('ARL_R6_DISTANCE_SEED0','ARL_R20_TEMPORAL_CREDIT_SEED0',R24_ID):continue
        raise OpsError('unresolved/unknown failed or pending attempt;independent audit required')
    actual=BudgetStore(Path(root)/'.research_local/authorized_rl_v1').status()
    if digest(actual)!=digest(budget):raise OpsError('caller budget is not actual immutable grant ledger')
    from .repaired_distance_profile import verify_repaired_proof,repaired_distance_experiment
    verify_repaired_proof(repaired_distance_experiment(),root);verify_r20_closed(root)
    return verify_r24_closed(root)
