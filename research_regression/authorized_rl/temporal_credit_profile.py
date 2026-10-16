"""R20 fixed gamma-only contrast; never extends the cumulative grant."""
import re
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .empty_rehearsal_profile import rehearsal_experiment,verify_rehearsal_proof

R17_REPORT_SHA='d87993588f64db612daab4ee3091f64437a090eeb2b82532906f770a740a12f6'
R17_SUMMARY_SHA='950a588ea593f5eb0ece276748624e590efb231472a5c5aacb67098946da2f72'
R19_REPORT_SHA='3762e526e46314fd369b6cc404dcb4966c9a87b69e8b3feccfe811f3157b6ebc'


def temporal_experiment():
    p=rehearsal_experiment();del p['expected_prefix_policy_sha256']
    p.update(schema=20,experiment_id='ARL_R20_TEMPORAL_CREDIT_SEED0',execution_profile='CENTERED_STD01_GAMMA0995_R20_V1',
        gamma=.995,gae_lambda=.95,control_report_sha256=R17_REPORT_SHA,control_summary_sha256=R17_SUMMARY_SHA,
        audit_report_sha256=R19_REPORT_SHA,single_variable='PPO_AND_ROLLOUT_GAMMA_099_TO_0995',
        prefix_policy_relationship='different gamma from first rollout; not expected identical to R17 at32k')
    return p


def validate_temporal_profile(p):
    clean={k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
    if digest(clean)!=digest(temporal_experiment()):raise OpsError('unsupported fixed R20 temporal profile')
    for k in ('source_identity_sha256','registration_sha256'):
        if k in p and (not isinstance(p[k],str) or not re.fullmatch('[0-9a-f]{64}',p[k])):raise OpsError('invalid R20 identity')


def verify_temporal_proof(p,root=ROOT):
    root=Path(root);validate_temporal_profile(p)
    try:
        verify_rehearsal_proof(rehearsal_experiment(),root)
        from .feedback_followup import verify_released_raw
        for name,expected in (('autonomous_rl_empty_rehearsal_r17.json',R17_REPORT_SHA),('autonomous_rl_reward_credit_r19.json',R19_REPORT_SHA)):
            path=root/'mujoco/reports'/name
            if sha256(path)!=expected:raise OpsError('R20 released control/audit changed')
            r=read_json(path)
            if r['implementation_status']!='PASS':raise OpsError('R20 control/audit incomplete')
            for key in ('review','result_review'):
                v=r[key]
                if v['status']!='GO' or v['critical'] or v['important'] or sha256(root/v['evidence']['path'])!=v['evidence']['sha256']:
                    raise OpsError('R20 actual independent control/audit review differs')
            verify_released_raw(r,root)
        summary=root/'.research_local/authorized_rl_v1/runs/ARL_R17_EMPTY_REHEARSAL_SEED0/summary.json'
        if sha256(summary)!=R17_SUMMARY_SHA or read_json(summary.parent/'complete.json')['summary_sha256']!=R17_SUMMARY_SHA:
            raise OpsError('R20 actual R17 control summary differs')
        return {'status':'PASS','control_summary_sha256':R17_SUMMARY_SHA,'audit_report_sha256':R19_REPORT_SHA,
            'initialization':'fresh seed0 hand prior; no checkpoint initialization'}
    except (OSError,KeyError,TypeError,ValueError) as e:raise OpsError('missing/invalid R20 fixed evidence') from e


def verify_temporal_checks(p,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        path=root/'.research_local/autorl_r20_checks_01.json';a=read_json(path);source=p['source_identity_sha256']
        if a['registration_sha256']!=p['registration_sha256'] or a['source_identity_sha256']!=source:raise OpsError('stale R20 admission')
        vv={}
        for k in ('operations','tensor','historical','runtime_binding'):
            rel=f'.research_local/autorl_r20_{k}_final_01.json';ref=a['evidence'][k]
            if ref['path']!=rel or sha256(root/rel)!=ref['sha256']:raise OpsError('R20 evidence path/hash differs')
            vv[k]=read_json(root/rel)
        op,te,hi,bi=(vv[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or op['failures'] or op['errors'] or op['optimizer_updates'] or op['tests_run']!=193
            or len(op['results'])!=193 or any(v['outcome']!='PASS' for v in op['results'])
            or op['source_identity_sha256']!=source or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or te['returncode'] or te['tests_run']!=66 or te['optimizer_updates'] or 'Ran 66 tests' not in te['stderr']
            or te['source_identity_sha256']!=source or te['registration_sha256']!=p['registration_sha256']
            or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
            or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results']) or bi['returncode']
            or bi['source_before_sha256']!=source or bi['source_after_sha256']!=source
            or bi['registration_sha256']!=p['registration_sha256'] or bi['final']!=a['evidence']['historical']):raise OpsError('R20 complete no-update checks missing')
        return {'status':'PASS','admission_sha256':sha256(path)}
    except (OSError,KeyError,TypeError,ValueError) as e:raise OpsError('missing/invalid R20 current checks') from e
