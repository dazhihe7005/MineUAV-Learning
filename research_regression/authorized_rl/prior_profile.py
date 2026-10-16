"""One new reviewed PPO execution profile, using the SAME immutable grant."""
from pathlib import Path
import re
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT

PRIOR_REPORT_SHA='99586894ee6295cca5f58e201bafbec41d6fd833152c63c5f2a52eae9ae24d11'
CONTROL_SUMMARY_SHA='f12df386cf9a0cadd8ac979dbbb8799db9e0af4b2a0127af1c31d7d14b61c927'
PRIOR_POLICY_SHA='1f3e9a0a6779d9b7290ba138bda5e6f916e5d73380b291cc9a97d1650c5e835c'


def prior_experiment():
    from .budget import experiment
    p=experiment('UNIFORM',65536,3600,revision='R2')
    p.update(schema=2,experiment_id='ARL_R2_PRIOR_SEED0',
        initialization='ANALYTIC_STATE_MEAN_TANH_V1',execution_profile='ANALYTIC_PRIOR_R2_V1',
        curriculum_rule='uniform twelve registered Train scenes for entire run',
        validation_steps=[0,32768,65536],prior_report_sha256=PRIOR_REPORT_SHA,
        control_summary_sha256=CONTROL_SUMMARY_SHA,expected_prior_policy_sha256=PRIOR_POLICY_SHA,
        evaluation={'selection':'FINAL_CHECKPOINT','initialization_effect':'final successes > R1 final successes and collisions <= R1 final collisions',
            'learning_over_prior':'final successes > own step0 successes and collisions <= own step0 collisions',
            'candidate_success_min':4,'candidate_collision_max':1,'candidate_physical_sensor_failures_max':0,
            'claim_limit':'single seed small Validation; no learned LiDAR dependence or generalization proof'})
    return p


def validate_prior_profile(p):
    clean={k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
    if digest(clean)!=digest(prior_experiment()):raise OpsError('unsupported fixed prior execution profile')
    for k in ('source_identity_sha256','registration_sha256'):
        if k in p and (not isinstance(p[k],str) or not re.fullmatch('[0-9a-f]{64}',p[k])):
            raise OpsError('invalid prior source/registration identity')


def verify_prior_proof(p,root=ROOT):
    """Scientific feasibility proof, NOT a grant and never a flight smoke proof."""
    validate_prior_profile(p);root=Path(root)
    report_path=root/'mujoco/reports/autonomous_rl_actor_prior_preflight.json'
    if sha256(report_path)!=p['prior_report_sha256']:raise OpsError('prior preflight report changed')
    report=read_json(report_path)
    reg=read_json(root/'mujoco/reports/autonomous_rl_actor_prior_registration.json')
    if (digest(reg)!=report['registration_sha256'] or digest(reg['source_identity'])!=report['source_identity_sha256']
        or report['review']['status']!='GO' or report['result_review']['status']!='GO'):
        raise OpsError('prior independent proof identity invalid')
    for record in report['raw_evidence']:
        path=root/record['path']
        if sha256(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:
            raise OpsError('prior raw proof missing or changed')
    for rel in ('research_regression/authorized_rl/actor_prior.py',):
        if sha256(root/rel)!=reg['source_identity'][rel]['sha256']:
            raise OpsError('tested prior implementation changed')
    result_path=root/'.research_local/autorl_r2_prior/run_01/result.json'
    result=read_json(result_path)
    if (read_json(result_path.parent/'complete.json')['result_sha256']!=sha256(result_path)
        or result['all_success'] is not True or result['assigned_policy_sha256']!=PRIOR_POLICY_SHA
        or any(result[k]!=0 for k in ('optimizer_updates','training_timesteps','validation_episodes','test_episodes'))):
        raise OpsError('prior physics/zero-update proof invalid')
    for rel,expected in result['runtime_sources'].items():
        if sha256(root/rel)!=expected:raise OpsError('prior tested physical/network runtime changed')
    control_path=root/'.research_local/authorized_rl_v1/runs/ARL_R1_UNIFORM_SEED0/summary.json'
    control=read_json(control_path)
    if (sha256(control_path)!=CONTROL_SUMMARY_SHA
        or read_json(control_path.parent/'complete.json')['summary_sha256']!=CONTROL_SUMMARY_SHA
        or control['actual_timesteps']!=65536 or control['test_episode_count']!=0
        or control['runtime_sources']!=result['runtime_sources']
        or control['initial_weights']['policy']!=result['initial_fresh_policy_sha256']):
        raise OpsError('frozen R1 control identity differs')
    return {'status':'PASS','kind':'NO_LEARNING_PHYSICS_FEASIBILITY_NOT_AUTHORIZATION',
            'prior_report_sha256':PRIOR_REPORT_SHA,'control_summary_sha256':CONTROL_SUMMARY_SHA}


def verify_training_checks(p,root=ROOT):
    """Require current actual pre-run evidence before consuming any reservation."""
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        path=root/'.research_local/autorl_r2_checks.json';admission=read_json(path)
        if (admission['source_identity_sha256']!=p['source_identity_sha256']
            or admission['registration_sha256']!=p['registration_sha256']):raise OpsError('stale pre-run admission')
        values={}
        for name in ('operations','tensor','historical'):
            relative=f'.research_local/autorl_r2_{name}_final.json'
            record=admission['evidence'][name]
            if record['path']!=relative or sha256(root/relative)!=record['sha256']:
                raise OpsError('pre-run evidence hash/path differs')
            values[name]=read_json(root/relative)
        ops=values['operations'];tensor=values['tensor'];history=values['historical']
        if (ops['complete'] is not True or ops['failures']!=0 or ops['errors']!=0 or ops['optimizer_updates']!=0
            or ops['source_identity_sha256']!=p['source_identity_sha256'] or ops['tests_run']<78
            or len(ops['results'])!=ops['tests_run'] or any(r['outcome']!='PASS' for r in ops['results'])
            or ops['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or tensor['returncode']!=0 or tensor['optimizer_updates']!=0 or tensor['tests_run']!=9
            or tensor['source_identity_sha256']!=p['source_identity_sha256']
            or tensor['registration_sha256']!=p['registration_sha256']
            or 'Ran 9 tests' not in tensor['stderr']
            or history['complete'] is not True or history['exit_code']!=0 or history['actual_optimizer_updates']!=0
            or history['forbidden_calls'] or len(history['results'])!=885
            or any(r['outcome']!='passed' for r in history['results'])):
            raise OpsError('pre-run tests incomplete/failed/noncurrent')
        return {'status':'PASS','admission_sha256':sha256(path)}
    except (OSError,KeyError,TypeError) as error:
        raise OpsError('missing or invalid current pre-run admission') from error
