"""Independent R11 global LR experiment; original grant and protocols immutable."""
import re
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .prior_distance_profile import prior_distance_experiment,verify_prior_distance_proof

R10_SUMMARY_SHA='c9fa8bcc9b14fe441903e1f733e7496393e39a36a72c0d2f8e907fc9778ba2ca'
R10_REPORT_SHA='0847525a939fee3eaf23a0cbf32511a7828aeaa7059084ed86ce81ce25ebc876'
LEARNING_RATE=3e-5


def lower_lr_experiment():
    p=prior_distance_experiment()
    p.update(schema=11,experiment_id='ARL_R11_PRIOR_LOWER_LR_SEED0',execution_profile='PRIOR_GLOBAL_LOWER_LR_R11_V1',
        learning_rate=LEARNING_RATE,control_summary_sha256=R10_SUMMARY_SHA,control_report_sha256=R10_REPORT_SHA,
        optimizer_scope='GLOBAL_ACTOR_AND_CRITIC; base L6/L7 identities are provenance, R11 overrides effective LR only')
    p['evaluation'].pop('initialization_benefit')
    p['evaluation'].update(learning_rate_benefit='original final successes > historical R10 final0; collisions <= R10 final1',
        control_original_final_collision_max=1,first32_approx_kl_mean_max=.07136825448833406*.5,
        kl_claim='last epoch minibatch approximate reverse KL; mechanism result is not navigation success')
    return p


def validate_lower_lr_profile(p):
    clean={k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
    if digest(clean)!=digest(lower_lr_experiment()):raise OpsError('unsupported fixed lower-LR profile')
    for key in ('source_identity_sha256','registration_sha256'):
        if key in p and (not isinstance(p[key],str) or not re.fullmatch('[0-9a-f]{64}',p[key])):raise OpsError('invalid lower-LR identity')


def verify_lower_lr_proof(p,root=ROOT):
    root=Path(root);validate_lower_lr_profile(p)
    try:
        verify_prior_distance_proof(prior_distance_experiment(),root)
        if sha256(root/'mujoco/reports/autonomous_rl_prior_distance_r10.json')!=R10_REPORT_SHA:raise OpsError('released R10 report changed')
        report=read_json(root/'mujoco/reports/autonomous_rl_prior_distance_r10.json')
        if report['implementation_status']!='PASS' or report['result_review']['status']!='GO':raise OpsError('R10 engineering/result review incomplete')
        for section in ('review','result_review'):
            ref=report[section]['evidence']
            if sha256(root/ref['path'])!=ref['sha256']:raise OpsError('R10 actual independent review changed')
        run=root/'.research_local/authorized_rl_v1/runs/ARL_R10_PRIOR_DISTANCE_SEED0'
        if sha256(run/'summary.json')!=R10_SUMMARY_SHA or read_json(run/'complete.json')['summary_sha256']!=R10_SUMMARY_SHA:
            raise OpsError('R10 completed fixed control changed')
        summary=read_json(run/'summary.json')
        if summary['actual_timesteps']!=65536 or summary['test_episode_count']!=0 or not summary['finite_model_and_gradients']:
            raise OpsError('R10 control accounting invalid')
        for item in report['raw_files']:
            path=root/item['path']
            if sha256(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:raise OpsError('R10 raw evidence changed')
        return {'status':'PASS','control_summary_sha256':R10_SUMMARY_SHA,'control_report_sha256':R10_REPORT_SHA}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid lower-LR historical proof') from error


def verify_lower_lr_checks(p,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        a=read_json(root/'.research_local/autorl_r11_checks.json')
        if a['registration_sha256']!=p['registration_sha256'] or a['source_identity_sha256']!=p['source_identity_sha256']:
            raise OpsError('stale R11 admission')
        values={}
        for kind in ('operations','tensor','historical','runtime_binding'):
            ref=a['evidence'][kind];rel=f'.research_local/autorl_r11_{kind}_final_01.json'
            if ref['path']!=rel or sha256(root/rel)!=ref['sha256']:raise OpsError('R11 admission evidence changed')
            values[kind]=read_json(root/rel)
        op,te,hi,bi=(values[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or op['errors'] or op['failures'] or op['optimizer_updates'] or op['tests_run']!=131
            or op['source_identity_sha256']!=a['source_identity_sha256']
            or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or len(op['results'])!=131 or any(v['outcome']!='PASS' for v in op['results'])
            or te['returncode'] or te['tests_run']!=36 or te['optimizer_updates'] or 'Ran 36 tests' not in te['stderr']
            or te['source_identity_sha256']!=a['source_identity_sha256'] or te['registration_sha256']!=a['registration_sha256']
            or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
            or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results'])
            or bi['source_before_sha256']!=a['source_identity_sha256'] or bi['source_after_sha256']!=a['source_identity_sha256']
            or bi['registration_sha256']!=a['registration_sha256'] or bi['returncode'] or bi['final']!=a['evidence']['historical']):
            raise OpsError('R11 current no-update tests incomplete/unsafe')
        return {'status':'PASS','admission_sha256':sha256(root/'.research_local/autorl_r11_checks.json')}
    except (OSError,KeyError,TypeError) as error:raise OpsError('missing/invalid R11 admission') from error
