"""R24 frozen Train yaw distribution contrast; complete R17 is the control."""
import re
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .empty_rehearsal_profile import rehearsal_experiment
from .temporal_credit_profile import verify_temporal_proof,temporal_experiment,R17_REPORT_SHA,R17_SUMMARY_SHA
from .closed_failure_proof import verify_r20_closed
from .yaw_coverage import yaw_manifest

R23_REPORT_SHA='6da738bf6600df0e601a9f2f3c9211faf745fa9c14b9aa4e0f46c3a8ad7c802d'
R23_RESULT_SHA='b745afbb910aead9f03af225bd1ca1123fe1defffa1678226fd3e397c902202c'


def yaw_experiment():
    p=rehearsal_experiment();del p['expected_prefix_policy_sha256']
    manifest=yaw_manifest(read_json(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json'))
    p.update(schema=24,experiment_id='ARL_R24_TRAIN_YAW_COVERAGE_SEED0',execution_profile='CENTERED_STD01_TRAIN_YAW_COVERAGE_R24_V1',
        gamma=.99,gae_lambda=.95,yaw_manifest_sha256=digest(manifest),train_yaw_task_count=120,
        yaw_sampling='original and four cardinal options each1/5 using independent Random(seed+0x52323400)',
        control_report_sha256=R17_REPORT_SHA,control_summary_sha256=R17_SUMMARY_SHA,audit_report_sha256=R23_REPORT_SHA,
        single_variable='TRAIN_INITIAL_YAW_DISTRIBUTION_ORIGINAL_TO_ORIGINAL_PLUS_FOUR_CARDINAL_OPTIONS',
        prefix_policy_relationship='Yaw reset changes experience from first episode;no R17 prefix equality expected',
        checkpoint_base_metadata_scope='L6 describes unchanged base geometry/start/oldyaw only;actual R24 yaw tasks bound by independent manifest/protocol/scheduler/episode records')
    p['evaluation']=dict(p['evaluation']);del p['evaluation']['rehearsal_benefit']
    p['evaluation']['yaw_coverage_benefit_over_R17']='original final successes > R17 final0;collisions <= R17 final3'
    return p


def validate_yaw_profile(p):
    clean={k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
    if digest(clean)!=digest(yaw_experiment()):raise OpsError('unsupported frozen R24 yaw profile')
    for key in ('source_identity_sha256','registration_sha256'):
        if key in p and (not isinstance(p[key],str) or not re.fullmatch('[0-9a-f]{64}',p[key])):raise OpsError('invalid R24 identity')


def verify_yaw_proof(p,root=ROOT):
    root=Path(root);validate_yaw_profile(p)
    verify_temporal_proof(temporal_experiment(),root);verify_r20_closed(root)
    try:
        from .feedback_followup import verify_released_raw
        report=root/'mujoco/reports/autonomous_rl_distribution_r23.json'
        if sha256(report)!=R23_REPORT_SHA:raise OpsError('R24 fixed released R23 audit differs')
        v=read_json(report)
        if v['implementation_status']!='PASS' or v['actual_optimizer_updates'] or v['test_episodes']:
            raise OpsError('R23 actual no-learning audit incomplete')
        for key in ('review','result_review'):
            a=v[key]
            if a['status']!='GO' or sha256(root/a['evidence']['path'])!=a['evidence']['sha256']:
                raise OpsError('R23 actual review missing')
        verify_released_raw(v,root)
        result=root/'.research_local/autorl_r23_distribution_audit/run_01/result.json'
        if (sha256(result)!=R23_RESULT_SHA or read_json(result.parent/'complete.json')['result_sha256']!=R23_RESULT_SHA):
            raise OpsError('R23 actual complete raw identity differs')
        return {'status':'PASS','control_summary_sha256':R17_SUMMARY_SHA,'descriptive_audit_result_sha256':R23_RESULT_SHA}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid R24 fixed proof') from error


def verify_yaw_checks(p,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        path=root/'.research_local/autorl_r24_checks_01.json';a=read_json(path);source=p['source_identity_sha256']
        if a['source_identity_sha256']!=source or a['registration_sha256']!=p['registration_sha256']:
            raise OpsError('stale R24 no-update admission')
        values={}
        for key in ('operations','tensor','historical','runtime_binding'):
            ref=a['evidence'][key];relative=f'.research_local/autorl_r24_{key}_final_01.json'
            if ref['path']!=relative or sha256(root/relative)!=ref['sha256']:raise OpsError('R24 admission hash/path differs')
            values[key]=read_json(root/relative)
        op,te,hi,bi=(values[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or op['tests_run']!=222 or op['failures'] or op['errors'] or op['optimizer_updates']
                or op['source_identity_sha256']!=source or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
                or len(op['results'])!=222 or any(v['outcome']!='PASS' for v in op['results'])
                or te['returncode'] or te['tests_run']!=74 or te['optimizer_updates'] or 'Ran 74 tests' not in te['stderr']
                or te['source_identity_sha256']!=source or te['registration_sha256']!=p['registration_sha256']
                or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
                or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results']) or bi['returncode']
                or bi['source_before_sha256']!=source or bi['source_after_sha256']!=source
                or bi['registration_sha256']!=p['registration_sha256'] or bi['final']!=a['evidence']['historical']):
            raise OpsError('current R24 full no-update tests incomplete')
        return {'status':'PASS','admission_sha256':sha256(path)}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid R24 no-update tests') from error
