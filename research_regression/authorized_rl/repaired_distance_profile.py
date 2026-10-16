"""Independent R8 identity, never a retry or mutation of failed R6."""
import re
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .distance_profile import distance_experiment,verify_distance_proof

REPAIR_REPORT_SHA='ecbce36daebac0a561e88225430432679be85046f61101b6812989c511588efa'
REPAIR_SOURCE_SHA='1eaa6b1f515ab32bce7b5ca15d1f1a5889ceb9505317fa6da3752794962031fb'
REPAIR_REG_SHA='486b2730207cdc96ea231cd845a433ff1eea7b1cb44537e2aa0c1beefe1deb16'


def repaired_distance_experiment():
    p=distance_experiment()
    p.update(schema=8,experiment_id='ARL_R8_REPAIRED_DISTANCE_SEED0',
        execution_profile='DISTANCE_CURRICULUM_R8_REPAIRED_V1',
        repair_report_sha256=REPAIR_REPORT_SHA,
        failed_predecessor_experiment_id='ARL_R6_DISTANCE_SEED0')
    return p


def validate_repaired_profile(p):
    clean={k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
    if digest(clean)!=digest(repaired_distance_experiment()):raise OpsError('unsupported repaired distance profile')
    for k in ('source_identity_sha256','registration_sha256'):
        if k in p and (not isinstance(p[k],str) or not re.fullmatch('[0-9a-f]{64}',p[k])):
            raise OpsError('invalid repaired distance source identity')


def verify_repaired_proof(p,root=ROOT):
    validate_repaired_profile(p);root=Path(root)
    try:
        path=root/'mujoco/reports/autonomous_rl_task_accounting_r7.json'
        if sha256(path)!=REPAIR_REPORT_SHA:raise OpsError('released R7 repair report changed')
        report=read_json(path)
        if (report['implementation_status']!='PASS' or report['review']['status']!='GO'
            or report['registration_sha256']!=REPAIR_REG_SHA or report['source_identity_sha256']!=REPAIR_SOURCE_SHA):
            raise OpsError('R7 repair release not approved')
        ref=report['review']['evidence']
        if sha256(root/ref['path'])!=ref['sha256']:raise OpsError('R7 actual review missing/changed')
        reg=read_json(root/'mujoco/reports/autonomous_rl_task_accounting_r7_registration.json')
        if digest(reg)!=REPAIR_REG_SHA:raise OpsError('R7 released registration changed')
        for rel in ('research_regression/authorized_rl/task_accounting.py','research_regression/authorized_rl/task_accounting_tests.py'):
            if sha256(root/rel)!=reg['source_identity'][rel]['sha256']:raise OpsError('R7 tested accounting changed')
        old=read_json(root/'mujoco/reports/autonomous_rl_distance_curriculum_r6.json')
        if old['implementation_status']!='FAIL' or old['optimizer_updates']!=0:raise OpsError('R6 failed historical result changed')
        verify_distance_proof(distance_experiment(),root)
        return {'status':'PASS','repair_report_sha256':REPAIR_REPORT_SHA,'old_R6':'PERMANENT_FAIL_NO_RETRY'}
    except (OSError,KeyError,TypeError,ValueError) as e:raise OpsError('missing/invalid repaired distance proof') from e


def verify_repaired_checks(p,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        a=read_json(root/'.research_local/autorl_r8_checks.json')
        if a['registration_sha256']!=p['registration_sha256'] or a['source_identity_sha256']!=p['source_identity_sha256']:
            raise OpsError('stale R8 admission')
        values={}
        for kind in ('operations','tensor','historical'):
            rel=f'.research_local/autorl_r8_{kind}_final.json';ref=a['evidence'][kind]
            if ref['path']!=rel or sha256(root/rel)!=ref['sha256']:raise OpsError('R8 test evidence changed')
            values[kind]=read_json(root/rel)
        op,te,hi=(values[k] for k in ('operations','tensor','historical'))
        if (not op['complete'] or op['errors'] or op['failures'] or op['optimizer_updates'] or op['tests_run']!=110
            or op['source_identity_sha256']!=a['source_identity_sha256']
            or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or len(op['results'])!=110 or any(v['outcome']!='PASS' for v in op['results'])
            or te['returncode'] or te['tests_run']!=33 or te['optimizer_updates'] or 'Ran 33 tests' not in te['stderr']
            or te['source_identity_sha256']!=a['source_identity_sha256'] or te['registration_sha256']!=a['registration_sha256']
            or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
            or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results'])):
            raise OpsError('R8 admission incomplete or unsafe')
        b=a['historical_runtime_binding'];path=root/b['path']
        if sha256(path)!=b['sha256']:raise OpsError('historical runtime binding changed')
        value=read_json(path)
        if (value['source_before_sha256']!=a['source_identity_sha256']
            or value['source_after_sha256']!=a['source_identity_sha256']
            or value['registration_sha256']!=a['registration_sha256'] or value['returncode']
            or value['final']!=a['evidence']['historical']):raise OpsError('history run/source mismatch')
        return {'status':'PASS','admission_sha256':sha256(root/'.research_local/autorl_r8_checks.json')}
    except (OSError,KeyError,TypeError) as e:raise OpsError('missing/invalid repaired admission') from e
