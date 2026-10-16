"""Fixed R10 initialization-only distance curriculum; original grant unchanged."""
import re
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .repaired_distance_profile import repaired_distance_experiment,verify_repaired_proof

R8_SUMMARY_SHA='a6760dc61ff03a3895eb451f99d61d77513c643a3e8d4f8e31a55f556475dce4'
R8_REPORT_SHA='9771e19e56568a186c0f150729795411a5df756cf4cdbb3399cc5b221968e844'
R9_REPORT_SHA='437a1ca34846777754a95d766b7a8d59913f083268d85524507e1d2415907e60'
R9_RESULT_SHA='3abcb783342e3f4429904a1a2188a0741e297df3eece98315af17589f61915d1'
R3_PRIOR_SHA='afbeac0d092aa4f38c318f4b75909006db65f2f8eaa4435783ae6b7a887ada4e'
PRIOR_POLICY_SHA='6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74'


def prior_distance_experiment():
    p=repaired_distance_experiment()
    p.update(schema=10,experiment_id='ARL_R10_PRIOR_DISTANCE_SEED0',execution_profile='PRIOR_DISTANCE_R10_V1',
        initialization='ANALYTIC_STATE_MEAN_TANH_V1',expected_initial_policy_sha256=PRIOR_POLICY_SHA,
        control_summary_sha256=R8_SUMMARY_SHA,short_prior_report_sha256=R9_REPORT_SHA,
        short_prior_result_sha256=R9_RESULT_SHA,original_step0_summary_sha256=R3_PRIOR_SHA)
    p['evaluation']={'selection':'FINAL_CHECKPOINT','actual_new_validation_episodes':30,
        'original_step0':'reuse fixed R3 PRIOR/std03 identical full policy; zero new episodes',
        'short_candidate_success_min':4,'short_candidate_collision_max':1,'physical_sensor_failures_max':0,
        'short_prior_retention':'final short successes >= own step0 short successes; collisions <= own step0',
        'initialization_benefit':'original final successes > historical R8 final0; collisions <= R8 final2',
        'original_learning_over_prior':'original final successes > own original step0; collisions <= own original step0',
        'claim_limit':'hand-written initial ability is not learning; single seed small Validation, no LiDAR/generalization proof'}
    return p


def validate_prior_distance_profile(p):
    clean={k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
    if digest(clean)!=digest(prior_distance_experiment()):raise OpsError('unsupported fixed prior-distance profile')
    for key in ('source_identity_sha256','registration_sha256'):
        if key in p and (not isinstance(p[key],str) or not re.fullmatch('[0-9a-f]{64}',p[key])):raise OpsError('invalid prior-distance identity')


def verify_prior_distance_proof(p,root=ROOT):
    root=Path(root);validate_prior_distance_profile(p)
    try:
        verify_repaired_proof(repaired_distance_experiment(),root)
        for rel,expected in [('mujoco/reports/autonomous_rl_repaired_distance_r8.json',R8_REPORT_SHA),
                             ('mujoco/reports/autonomous_rl_short_prior_r9.json',R9_REPORT_SHA)]:
            if sha256(root/rel)!=expected:raise OpsError('released R8/R9 proof changed')
        report=read_json(root/'mujoco/reports/autonomous_rl_short_prior_r9.json')
        if (report['implementation_status']!='PASS' or report['feasibility_status']!='PASS'
            or report['result_review']['status']!='GO' or report['optimizer_updates']!=0 or report['training_timesteps']!=0):
            raise OpsError('R9 actual non-learning proof not approved')
        for section in ('review','result_review'):
            ref=report[section]['evidence']
            if sha256(root/ref['path'])!=ref['sha256']:raise OpsError('R9 actual independent review missing/changed')
        run=root/'.research_local/autorl_r9_short_prior/run_01';result=read_json(run/'result.json')
        if sha256(run/'result.json')!=R9_RESULT_SHA or read_json(run/'complete.json')['result_sha256']!=R9_RESULT_SHA:
            raise OpsError('R9 complete proof identity differs')
        from .short_prior import validate_short_prior_result
        from .short_goal import short_manifest
        prereg=read_json(root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        validate_short_prior_result(result,short_manifest(prereg))
        oldreg=read_json(root/'mujoco/reports/autonomous_rl_short_prior_r9_registration.json')
        if result['registration_sha256']!=digest(oldreg) or result['source_identity_sha256']!=digest(oldreg['source_identity']):
            raise OpsError('R9 result runtime identity differs')
        for name in ('actor_prior.py','std_training.py','short_prior.py','short_goal_env.py','task_accounting.py'):
            rel='research_regression/authorized_rl/'+name
            if sha256(root/rel)!=oldreg['source_identity'][rel]['sha256']:raise OpsError('tested prior/task helper changed')
        for identity,expected in [('ARL_R8_REPAIRED_DISTANCE_SEED0',R8_SUMMARY_SHA),('ARL_R3_PRIOR_STD03_SEED0',R3_PRIOR_SHA)]:
            path=root/'.research_local/authorized_rl_v1/runs'/identity/'summary.json'
            if sha256(path)!=expected or read_json(path.parent/'complete.json')['summary_sha256']!=expected:
                raise OpsError('R10 frozen historical comparison missing/changed')
        return {'status':'PASS','R9_result_sha256':R9_RESULT_SHA,'R8_control_sha256':R8_SUMMARY_SHA}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid prior-distance proof') from error


def verify_prior_distance_checks(p,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        a=read_json(root/'.research_local/autorl_r10_checks.json')
        if a['registration_sha256']!=p['registration_sha256'] or a['source_identity_sha256']!=p['source_identity_sha256']:
            raise OpsError('stale R10 admission')
        values={}
        for kind in ('operations','tensor','historical','runtime_binding'):
            ref=a['evidence'][kind];rel=f'.research_local/autorl_r10_{kind}_final_02.json'
            if ref['path']!=rel or sha256(root/rel)!=ref['sha256']:raise OpsError('R10 admission evidence changed')
            values[kind]=read_json(root/rel)
        op,te,hi,bi=(values[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or op['errors'] or op['failures'] or op['optimizer_updates'] or op['tests_run']!=125
            or op['source_identity_sha256']!=a['source_identity_sha256']
            or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or len(op['results'])!=125 or any(v['outcome']!='PASS' for v in op['results'])
            or te['returncode'] or te['tests_run']!=35 or te['optimizer_updates'] or 'Ran 35 tests' not in te['stderr']
            or te['source_identity_sha256']!=a['source_identity_sha256'] or te['registration_sha256']!=a['registration_sha256']
            or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
            or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results'])
            or bi['source_before_sha256']!=a['source_identity_sha256'] or bi['source_after_sha256']!=a['source_identity_sha256']
            or bi['registration_sha256']!=a['registration_sha256'] or bi['returncode'] or bi['final']!=a['evidence']['historical']):
            raise OpsError('R10 current no-update tests incomplete/unsafe')
        return {'status':'PASS','admission_sha256':sha256(root/'.research_local/autorl_r10_checks.json')}
    except (OSError,KeyError,TypeError) as error:raise OpsError('missing/invalid R10 admission') from error
