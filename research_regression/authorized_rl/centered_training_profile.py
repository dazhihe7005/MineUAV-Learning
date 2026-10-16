"""Fixed R14 Actor-function contrast under the unchanged cumulative grant."""
import re
from pathlib import Path
from research_ops.storage import OpsError, read_json, sha256, digest
from .runtime import ROOT
from .lower_lr_profile import lower_lr_experiment, verify_lower_lr_proof

R11_SUMMARY_SHA = '7f34a1e98eae2ef289becacb4d1086f7e5dbb0d686d2cae750610452d90b8f26'
R11_REPORT_SHA = '8a5c911619a7905172d94674664011e7cc6b3dbbae7db42e3b6706f53b8c7dc3'
R13_REPORT_SHA = 'cece507b9522455906dd7b31bb7c2da8a55319ae2e31bff45219ea3716842065'
R13_RESULT_SHA = '1107e3fe8388b6b36910a469a1a35324c51efac2a6fabeaf2dd41eb923876c5d'
R13P_REPORT_SHA = 'c38cd4629da70a4ba61a1004fd73a478ff6d25bca7306804ca5764b99cb1fb2b'
INITIAL_POLICY_SHA = '20f5c226abc762e5aa4a9046ba400b2b0f7012b0da6715d14204b097cae06001'


def centered_experiment():
    p = lower_lr_experiment()
    p.update(schema=14, experiment_id='ARL_R14_CENTERED_ACTOR_SEED0', execution_profile='CENTERED_ACTOR_R14_V1',
        actor_parameterization='SAME_BACKGROUND_STATE_CENTERED_ACTOR_V1',
        expected_initial_policy_sha256=INITIAL_POLICY_SHA, control_summary_sha256=R11_SUMMARY_SHA,
        control_report_sha256=R11_REPORT_SHA, preflight_report_sha256=R13_REPORT_SHA,
        preflight_result_sha256=R13_RESULT_SHA, publication_repair_report_sha256=R13P_REPORT_SHA,
        initialization='FRESH_SEED0_HAND_PRIOR_CENTERED_FUNCTION',
        optimizer_scope='GLOBAL_ACTOR_AND_CRITIC_LR3e-5; L6/L7 network hashes are base provenance, not centered function identity',
        zero_state_claim='STRUCTURAL_MEAN_IDENTITY_NOT_LEARNED_SUCCESS')
    p.pop('original_step0_summary_sha256')  # New function requires own actual original-target step0 evaluation.
    p['evaluation'] = {'selection':'FINAL_CHECKPOINT', 'actual_new_validation_episodes':36,
        'reused_original_step0_episodes':0, 'original_step0':'NEW actual evaluation; never reuse old-class performance',
        'short_candidate_success_min':4, 'short_candidate_collision_max':1, 'physical_sensor_failures_max':0,
        'short_prior_retention':'final short successes >= own step0; collisions <= own step0',
        'original_learning_over_prior':'original final successes > own original step0; collisions <= own step0',
        'actor_function_benefit':'original final successes > historical R11 final0; collisions <= R11 final6',
        'control_original_final_collision_max':6,
        'claim_limit':'single seed and finite Validation; structural zero mean and hand-written initial ability are not learning'}
    return p


def validate_centered_profile(p):
    clean = {k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
    if digest(clean) != digest(centered_experiment()):
        raise OpsError('unsupported fixed centered Actor profile')
    for k in ('source_identity_sha256','registration_sha256'):
        if k in p and (not isinstance(p[k],str) or not re.fullmatch('[0-9a-f]{64}',p[k])):
            raise OpsError('invalid centered profile identity')


def verify_centered_proof(p, root=ROOT):
    root = Path(root)
    validate_centered_profile(p)
    try:
        verify_lower_lr_proof(lower_lr_experiment(),root)
        from .feedback_followup import verify_released_raw
        for name, expected in (('autonomous_rl_lower_lr_r11.json',R11_REPORT_SHA),
                               ('autonomous_rl_centered_preflight_r13.json',R13_REPORT_SHA),
                               ('autonomous_rl_publication_repair_r13p.json',R13P_REPORT_SHA)):
            path = root/'mujoco/reports'/name
            if sha256(path) != expected:
                raise OpsError('centered historical release changed')
            report = read_json(path)
            if report['implementation_status'] != 'PASS':
                raise OpsError('centered release implementation incomplete')
            for key in ('review','result_review'):
                v = report[key]
                if v['status'] != 'GO' or sha256(root/v['evidence']['path']) != v['evidence']['sha256']:
                    raise OpsError('centered actual independent review changed')
            if 'raw_files' in report:
                verify_released_raw(report,root)
        control = root/'.research_local/authorized_rl_v1/runs/ARL_R11_PRIOR_LOWER_LR_SEED0'
        if (sha256(control/'summary.json') != R11_SUMMARY_SHA
                or read_json(control/'complete.json')['summary_sha256'] != R11_SUMMARY_SHA):
            raise OpsError('R11 actual complete control missing')
        result_path = root/'.research_local/autorl_r13_centered_preflight/run_01/result.json'
        result = read_json(result_path)
        if (sha256(result_path) != R13_RESULT_SHA or read_json(result_path.parent/'complete.json')['result_sha256'] != R13_RESULT_SHA
                or result['new_policy_sha256'] != INITIAL_POLICY_SHA or result['optimizer_updates'] != 0
                or not result['save_reload_exact_rows'] or result['initial_mean_max_absolute_difference'] != 0):
            raise OpsError('R13 actual preflight differs')
        old = read_json(root/'mujoco/reports/autonomous_rl_centered_preflight_r13_registration.json')
        rel = 'research_regression/authorized_rl/state_centered_policy.py'
        if sha256(root/rel) != old['source_identity'][rel]['sha256']:
            raise OpsError('tested centered policy implementation changed')
        return {'status':'PASS','preflight_result_sha256':R13_RESULT_SHA,'control_summary_sha256':R11_SUMMARY_SHA,
                'preflight_scope':'old actual R13 aggregate; model module unchanged; R13P management repair tested independently'}
    except (OSError,KeyError,TypeError,ValueError) as error:
        raise OpsError('missing/invalid centered Actor preflight/control') from error


def verify_centered_checks(p, root=ROOT):
    from research_ops.identity import source_identity
    root = Path(root)
    try:
        a = read_json(root/'.research_local/autorl_r14_checks_01.json')
        if a['registration_sha256'] != p['registration_sha256'] or a['source_identity_sha256'] != p['source_identity_sha256']:
            raise OpsError('stale R14 admission')
        values = {}
        for k in ('operations','tensor','historical','runtime_binding'):
            ref = a['evidence'][k]
            rel = f'.research_local/autorl_r14_{k}_final_01.json'
            if ref['path'] != rel or sha256(root/rel) != ref['sha256']:
                raise OpsError('R14 admission evidence changed')
            values[k] = read_json(root/rel)
        op,te,hi,bi = (values[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or op['failures'] or op['errors'] or op['optimizer_updates'] or op['tests_run'] != 152
                or op['source_identity_sha256'] != a['source_identity_sha256'] or len(op['results']) != 152
                or any(v['outcome'] != 'PASS' for v in op['results'])
                or op['test_source_identity'] != source_identity(root,['tests/research_ops_v2'])
                or te['returncode'] or te['tests_run'] != 51 or 'Ran 51 tests' not in te['stderr'] or te['optimizer_updates']
                or te['source_identity_sha256'] != a['source_identity_sha256'] or te['registration_sha256'] != a['registration_sha256']
                or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
                or len(hi['results']) != 885 or any(v['outcome'] != 'passed' for v in hi['results'])
                or bi['returncode'] or bi['source_before_sha256'] != a['source_identity_sha256']
                or bi['source_after_sha256'] != a['source_identity_sha256']
                or bi['registration_sha256'] != a['registration_sha256'] or bi['final'] != a['evidence']['historical']):
            raise OpsError('R14 current no-update admission incomplete')
        return {'status':'PASS','admission_sha256':sha256(root/'.research_local/autorl_r14_checks_01.json')}
    except (OSError,KeyError,TypeError,ValueError) as error:
        raise OpsError('missing/invalid R14 current checks') from error
