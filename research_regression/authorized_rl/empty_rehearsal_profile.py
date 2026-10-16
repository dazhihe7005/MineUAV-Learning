"""Fixed R17 original-distance empty rehearsal, same cumulative user grant."""
import re
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .centered_noise_profile import noise_experiment,verify_noise_proof

R16_REPORT_SHA='60c231a2d42afed7ba16b890f7be6774bc62f13bfb7501f5bd9f219893f3f174'
R16_SUMMARY_SHA='bf3814e86f0402ed02748483453a8279079aabca3e4df0c93264a82fcc71668c'
PREFIX_POLICY_SHA='33bf6699158a35842fc05c3d4ca97f391f778db60c759fdad2564a6e94c87f60'
BASELINE_SHA='5f482d0711652f14c8d925a8d4d6f24f467686f1a10eb2787c65c66048c4c70c'


def verify_rehearsal_prefix(step,policy_sha):
    if step==32768 and policy_sha!=PREFIX_POLICY_SHA:
        raise OpsError('R17 first-phase policy is not the identical frozen R16 prefix')


def rehearsal_experiment():
    p=noise_experiment()
    p.update(schema=17,experiment_id='ARL_R17_EMPTY_REHEARSAL_SEED0',execution_profile='CENTERED_STD01_EMPTY_REHEARSAL_R17_V1',
        original_empty_probability=.5,original_train_sampling='three empty each1/6; nine nonempty each1/18; reset only after32768',
        expected_prefix_policy_sha256=PREFIX_POLICY_SHA,control_summary_sha256=R16_SUMMARY_SHA,
        control_report_sha256=R16_REPORT_SHA,single_variable='POST_SWITCH_ORIGINAL_TRAIN_EMPTY_PROBABILITY_025_TO_050')
    p['evaluation']={'selection':'FINAL_CHECKPOINT','actual_new_validation_episodes':36,'reused_original_step0_episodes':0,
        'short_candidate_success_min':4,'short_candidate_collision_max':1,'physical_sensor_failures_max':0,
        'short_prior_retention':'final short successes >= own step0; collisions <= own step0',
        'original_learning_over_prior':'original final successes > own original step0; collisions <= own step0',
        'rehearsal_benefit':'original final successes > R16 final0; collisions <= R16 final3',
        'control_original_final_success':0,'control_original_final_collision_max':3,
        'empty_retention':'original empty final successes >= own32k empty successes AND own32k empty successes >=1',
        'claim_limit':'single seed/small fixed Validation; more empty Train and hand prior success not generalization'}
    return p


def validate_rehearsal_profile(p):
    clean={k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
    if digest(clean)!=digest(rehearsal_experiment()):raise OpsError('unsupported fixed R17 rehearsal profile')
    for k in ('source_identity_sha256','registration_sha256'):
        if k in p and (not isinstance(p[k],str) or not re.fullmatch('[0-9a-f]{64}',p[k])):raise OpsError('invalid R17 identity')


def verify_rehearsal_proof(p,root=ROOT):
    root=Path(root);validate_rehearsal_profile(p)
    try:
        verify_noise_proof(noise_experiment(),root)
        from .feedback_followup import verify_released_raw
        report_path=root/'mujoco/reports/autonomous_rl_centered_noise_r16.json'
        if sha256(report_path)!=R16_REPORT_SHA:raise OpsError('R17 released R16 control differs')
        report=read_json(report_path)
        if report['implementation_status']!='PASS':raise OpsError('R17 control incomplete')
        for k in ('review','result_review'):
            v=report[k]
            if v['status']!='GO' or sha256(root/v['evidence']['path'])!=v['evidence']['sha256']:
                raise OpsError('R17 control independent review missing')
        verify_released_raw(report,root)
        summary=root/'.research_local/authorized_rl_v1/runs/ARL_R16_CENTERED_STD01_SEED0/summary.json'
        if sha256(summary)!=R16_SUMMARY_SHA:raise OpsError('R16 actual control summary differs')
        baseline=root/'.research_local/authorized_rl_v1/baseline.json'
        if sha256(baseline)!=BASELINE_SHA:raise OpsError('historical empty feasibility differs')
        b=read_json(baseline)
        if not b['all_success'] or len(b['rows'])!=3 or b['optimizer_updates'] or b['test_episode_count']:
            raise OpsError('historical feasibility incomplete')
        for path,value in b['runtime_sources'].items():
            if sha256(root/path)!=value:raise OpsError('historical feasibility runtime changed')
        return {'status':'PASS','control_summary_sha256':R16_SUMMARY_SHA,'reused_feasibility_sha256':BASELINE_SHA}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/corrupt R17 fixed controls') from error


def verify_rehearsal_checks(p,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        path=root/'.research_local/autorl_r17_checks_01.json';a=read_json(path)
        if a['registration_sha256']!=p['registration_sha256'] or a['source_identity_sha256']!=p['source_identity_sha256']:
            raise OpsError('stale R17 admission')
        vv={};source=a['source_identity_sha256']
        for k in ('operations','tensor','historical','runtime_binding'):
            rel=f'.research_local/autorl_r17_{k}_final_01.json';ref=a['evidence'][k]
            if ref['path']!=rel or sha256(root/rel)!=ref['sha256']:raise OpsError('R17 admission evidence differs')
            vv[k]=read_json(root/rel)
        op,te,hi,bi=(vv[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or op['failures'] or op['errors'] or op['optimizer_updates'] or op['tests_run']!=174
            or len(op['results'])!=174 or any(v['outcome']!='PASS' for v in op['results'])
            or op['source_identity_sha256']!=source or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or te['returncode'] or te['tests_run']!=60 or 'Ran 60 tests' not in te['stderr'] or te['optimizer_updates']
            or te['source_identity_sha256']!=source or te['registration_sha256']!=p['registration_sha256']
            or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
            or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results']) or bi['returncode']
            or bi['source_before_sha256']!=source or bi['source_after_sha256']!=source
            or bi['registration_sha256']!=p['registration_sha256'] or bi['final']!=a['evidence']['historical']):
            raise OpsError('R17 current no-update checks incomplete')
        return {'status':'PASS','admission_sha256':sha256(path)}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid R17 current admission') from error
