"""Fixed R16 initial-noise contrast, same cumulative authorization and mean policy."""
import re
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .centered_training_profile import centered_experiment,verify_centered_proof

INITIAL_POLICY_SHA='650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4'
R14_REPORT_SHA='489c3aa7f211f393c8d552a501663d9e4e02556befa9751a0404aa9348fffe70'
R14_SUMMARY_SHA='0cd11a7161b2518ee9aa48c180d31435c723ae6f9978cd785128d788a074fafd'
R15_REPORT_SHA='cd99705ebc359fdad321e05913deeabf09a13704c5e9d1ef4da5968930a067f1'
R15_RESULT_SHA='85d36387eff30e4ee02e0c4ee8c9bcd398c72e3f2886469ee65e9406dbeac846'


def noise_experiment():
    p=centered_experiment()
    p.update(schema=16,experiment_id='ARL_R16_CENTERED_STD01_SEED0',execution_profile='CENTERED_INITIAL_STD01_R16_V1',
        initial_std=.1,expected_initial_policy_sha256=INITIAL_POLICY_SHA,
        initialization='FRESH_SEED0_HAND_PRIOR_CENTERED_INITIAL_STD01',
        control_summary_sha256=R14_SUMMARY_SHA,control_report_sha256=R14_REPORT_SHA,
        feedback_report_sha256=R15_REPORT_SHA,feedback_result_sha256=R15_RESULT_SHA,
        single_variable='INITIAL_LOG_STD_LOG03_TO_LOG01_REMAINS_TRAINABLE')
    p['evaluation']={'selection':'FINAL_CHECKPOINT','actual_new_validation_episodes':36,'reused_original_step0_episodes':0,
        'original_step0':'NEW actual evaluation; no old-class or initial-noise performance reuse',
        'short_candidate_success_min':4,'short_candidate_collision_max':1,'physical_sensor_failures_max':0,
        'short_prior_retention':'final short successes >= own step0; collisions <= own step0',
        'original_learning_over_prior':'original final successes > own original step0; collisions <= own step0',
        'noise_benefit':'original final successes > R14 final1; collisions <= R14 final3',
        'control_original_final_success':1,'control_original_final_collision_max':3,
        'short_train_success_ratio_control':[34,108],'near_goal_fast_ratio_control':[1867,2077],
        'stopping_experience_mechanism':'short Train success ratio >34/108 AND near-goal-fast ratio <1867/2077; missing denominators FAIL; not navigation proof',
        'claim_limit':'single seed/fixed small Validation; hand prior and lower-noise Train success not learned navigation or generalization'}
    return p


def validate_noise_profile(p):
    clean={k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
    if digest(clean)!=digest(noise_experiment()):raise OpsError('unsupported fixed R16 initial-noise profile')
    for k in ('source_identity_sha256','registration_sha256'):
        if k in p and (not isinstance(p[k],str) or not re.fullmatch('[0-9a-f]{64}',p[k])):raise OpsError('invalid R16 identity')


def verify_noise_proof(p,root=ROOT):
    root=Path(root);validate_noise_profile(p)
    try:
        verify_centered_proof(centered_experiment(),root)
        from .feedback_followup import verify_released_raw
        from .centered_feedback import references
        references(root)
        for name,expected in (('autonomous_rl_centered_actor_r14.json',R14_REPORT_SHA),
                              ('autonomous_rl_centered_feedback_r15.json',R15_REPORT_SHA)):
            path=root/'mujoco/reports'/name
            if sha256(path)!=expected:raise OpsError('R16 released control/feedback changed')
            report=read_json(path)
            if report['implementation_status']!='PASS':raise OpsError('R16 historical implementation incomplete')
            for k in ('review','result_review'):
                v=report[k]
                if v['status']!='GO' or sha256(root/v['evidence']['path'])!=v['evidence']['sha256']:
                    raise OpsError('R16 historical independent review changed')
            verify_released_raw(report,root)
        run=root/'.research_local/autorl_r15_centered_feedback/run_01'
        if sha256(run/'result.json')!=R15_RESULT_SHA or read_json(run/'complete.json')['result_sha256']!=R15_RESULT_SHA:
            raise OpsError('R15 actual fixed-input evidence missing')
        return {'status':'PASS','control_summary_sha256':R14_SUMMARY_SHA,'feedback_result_sha256':R15_RESULT_SHA}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/corrupt R16 fixed controls') from error


def verify_noise_checks(p,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        path=root/'.research_local/autorl_r16_checks_01.json';a=read_json(path)
        if a['registration_sha256']!=p['registration_sha256'] or a['source_identity_sha256']!=p['source_identity_sha256']:
            raise OpsError('stale R16 admission')
        vv={};source=a['source_identity_sha256']
        for k in ('operations','tensor','historical','runtime_binding'):
            rel=f'.research_local/autorl_r16_{k}_final_01.json';ref=a['evidence'][k]
            if ref['path']!=rel or sha256(root/rel)!=ref['sha256']:raise OpsError('R16 admission evidence differs')
            vv[k]=read_json(root/rel)
        op,te,hi,bi=(vv[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or op['failures'] or op['errors'] or op['optimizer_updates'] or op['tests_run']!=166
            or len(op['results'])!=166 or any(v['outcome']!='PASS' for v in op['results'])
            or op['source_identity_sha256']!=source or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or te['returncode'] or te['tests_run']!=55 or 'Ran 55 tests' not in te['stderr'] or te['optimizer_updates']
            or te['source_identity_sha256']!=source or te['registration_sha256']!=p['registration_sha256']
            or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
            or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results']) or bi['returncode']
            or bi['source_before_sha256']!=source or bi['source_after_sha256']!=source
            or bi['registration_sha256']!=p['registration_sha256'] or bi['final']!=a['evidence']['historical']):
            raise OpsError('R16 current no-update checks incomplete')
        return {'status':'PASS','admission_sha256':sha256(path)}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid R16 current admission') from error
