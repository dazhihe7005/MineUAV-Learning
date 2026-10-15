"""Reviewed immutable manifests; only fixed Level0/1 task capabilities."""
from __future__ import annotations

import re
from pathlib import Path

from research_ops.identity import file_record,git,safe_path,source_identity
from research_ops.storage import AUTHORIZATION,ZERO_BUDGET,OpsError,digest

TASK_PARAMETERS={
    'offline-audit':{'scope':{'history'}},
    'regression-check':{'suite':{'historical','operations'}},
    'documentation-update':{},
    'experiment-analysis':{'experiment_id':{'L9','L9S'}},
}
MANDATORY_SOURCE_PATHS=('research_ops_v2','research_regression','researchctl-v2','tests/research_ops_v2')


def delivery_path(root,relative):
    p=Path(relative);safe_path(root,relative)
    if relative in ('README.md','researchctl-v2'):
        return p
    allowed=(relative.startswith('docs/') and p.suffix in ('.md','.json')
        or relative.startswith(('research_ops_v2/','research_regression/')) and p.suffix in ('.py','.json')
        or relative.startswith('tests/research_ops_v2/') and p.suffix=='.py'
        or relative.startswith('mujoco/lidar_integration/test_') and p.suffix=='.py'
        or relative.startswith('mujoco/reports/') and (len(p.parts)==3 and p.suffix=='.json'
            or 'figures' in p.parts and p.suffix=='.png'))
    if not allowed or p.name.startswith('.') or p.name in ('CODEX_MASTER_SPEC.md','PHASE_OPS_01_PERSISTENCE.md','RESEARCH_OPERATIONS_PROTOCOL.md'):
        raise OpsError('delivery path is not an approved source/document/report/PNG')
    return p


def validate_task(task):
    if set(task)!={'task_id','task_type','parameters','wall_time_s','workers'}:
        raise OpsError('task schema or budget fields invalid')
    if not isinstance(task['task_id'],str) or not re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}',task['task_id']):
        raise OpsError('invalid task identity')
    kind=task['task_type'];parameters=task['parameters']
    if kind not in TASK_PARAMETERS or not isinstance(parameters,dict) or set(parameters)!=set(TASK_PARAMETERS[kind]):
        raise OpsError('unsupported task type/parameter; no scripts or shell commands')
    if any(v not in TASK_PARAMETERS[kind][k] for k,v in parameters.items()):
        raise OpsError('task parameter is not in whitelist')
    if type(task['workers']) is not int or task['workers']!=1 or type(task['wall_time_s']) not in (float,int) or not 0<task['wall_time_s']<=600:
        raise OpsError('task needs one worker and finite wall-time<=600s')


def build_registration(root,*,phase_id,tasks,sync_files,source_paths,approved_changes=None,
                       report_path='mujoco/reports/research_operations_v2.json'):
    source_paths=sorted(set(source_paths)|set(MANDATORY_SOURCE_PATHS)|set(approved_changes or {}))
    value={'schema_version':2,'operations_version':'RESEARCH_OPS_V2','phase_id':phase_id,
           'branch':git(root,'branch','--show-current').stdout.strip(),
           'base_commit':git(root,'rev-parse','HEAD').stdout.strip(),
           'authorization_scope':AUTHORIZATION,'budget_ledger':ZERO_BUDGET,'training_budget':0,
           'working_directory':'.','output_root':'.research_local/ops_v2/runs',
           'tasks':tasks,'sync_files':sync_files,'source_paths':source_paths,'report_path':report_path,
           'source_identity':source_identity(root,source_paths),'approved_changes':approved_changes or {}}
    validate_registration(root,value)
    return value


def validate_registration(root,value,*,check_source=True):
    expected={'schema_version','operations_version','phase_id','branch','base_commit','authorization_scope',
        'budget_ledger','training_budget','working_directory','output_root','tasks','sync_files',
        'source_paths','source_identity','approved_changes','report_path'}
    if set(value)!=expected or value['schema_version']!=2 or value['operations_version']!='RESEARCH_OPS_V2':
        raise OpsError('unrecognized V2 registration')
    if not re.fullmatch('[A-Z][A-Z0-9_-]{1,40}',value['phase_id']) or value['branch'] in ('','main','master'):
        raise OpsError('invalid phase/branch')
    if digest(value['authorization_scope'])!=digest(AUTHORIZATION) or digest(value['budget_ledger'])!=digest(ZERO_BUDGET) or type(value['training_budget']) is not int or value['training_budget']!=0:
        raise OpsError('no training authority or budget may be registered')
    if value['working_directory']!='.' or value['output_root']!='.research_local/ops_v2/runs':
        raise OpsError('task runtime/output directory not allowed')
    if not value['tasks'] or len(value['tasks'])>16 or len({t['task_id'] for t in value['tasks']})!=len(value['tasks']):
        raise OpsError('empty/duplicate/oversized task set')
    for task in value['tasks']:validate_task(task)
    if len(set(value['sync_files']))!=len(value['sync_files']):raise OpsError('duplicate delivery path')
    for path in value['sync_files']:delivery_path(root,path)
    if value['report_path'] not in value['sync_files'] or not value['report_path'].startswith('mujoco/reports/') or Path(value['report_path']).suffix!='.json':
        raise OpsError('stage JSON report must be explicitly registered')
    for path in value['source_identity']:
        delivery_path(root,path)
        if Path(path).suffix not in ('.py','.json') and path!='researchctl-v2':
            raise OpsError('invalid source identity path')
    if not set(MANDATORY_SOURCE_PATHS)<=set(value['source_paths']):raise OpsError('mandatory runtime/test sources cannot be omitted')
    if check_source:
        if source_identity(root,value['source_paths'])!=value['source_identity']:
            raise OpsError('reviewed runtime source changed')
        required=source_identity(root,MANDATORY_SOURCE_PATHS)
        if any(value['source_identity'].get(k)!=v for k,v in required.items()):raise OpsError('runtime identity coverage incomplete')
    for path,change in value['approved_changes'].items():
        if not path.startswith('mujoco/lidar_integration/test_') or not path.endswith('.py') or set(change)!={'before','after'}:
            raise OpsError('only explicit test configuration changes can overlay V1 protected history')
        if file_record(safe_path(root,path))!=change['after']:
            raise OpsError('approved test source identity differs')


def verify_review(registration,review):
    if (review.get('verdict')!='GO' or review.get('registration_sha256')!=digest(registration)
            or review.get('source_identity_sha256')!=digest(registration['source_identity'])
            or not review.get('method')):
        raise OpsError('review does not approve this exact registration/source')
