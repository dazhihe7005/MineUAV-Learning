"""R22: GAE lambda-only contrast to complete R17, not a retry of incomplete R20."""
import re
from pathlib import Path
from research_ops.storage import OpsError, read_json, sha256, digest
from .runtime import ROOT
from .empty_rehearsal_profile import rehearsal_experiment
from .temporal_credit_profile import verify_temporal_proof, temporal_experiment, R17_REPORT_SHA, R17_SUMMARY_SHA, R19_REPORT_SHA
from .closed_failure_proof import verify_r20_closed, R20_REPORT_SHA, R21_REPORT_SHA


def gae_experiment():
    p = rehearsal_experiment(); del p['expected_prefix_policy_sha256']
    p.update(schema=22, experiment_id='ARL_R22_GAE_CREDIT_SEED0', execution_profile='CENTERED_STD01_GAMMA099_LAMBDA099_R22_V1',
             gamma=.99, gae_lambda=.99, control_report_sha256=R17_REPORT_SHA, control_summary_sha256=R17_SUMMARY_SHA,
             audit_report_sha256=R19_REPORT_SHA, closed_failure_report_sha256=R20_REPORT_SHA, guard_report_sha256=R21_REPORT_SHA,
             single_variable='PPO_AND_ROLLOUT_GAE_LAMBDA_095_TO_099',
             prefix_policy_relationship='different GAE advantages from first rollout;not expected identical to R17 at32k')
    p['evaluation'] = dict(p['evaluation'])
    del p['evaluation']['rehearsal_benefit']
    p['evaluation']['gae_credit_benefit_over_R17'] = 'original final successes > R17 final0;collisions <= R17 final3'
    return p


def validate_gae_profile(p):
    clean = {k: v for k, v in p.items() if k not in ('source_identity_sha256', 'registration_sha256')}
    if digest(clean) != digest(gae_experiment()):
        raise OpsError('unsupported fixed R22 GAE profile')
    for key in ('source_identity_sha256', 'registration_sha256'):
        if key in p and (not isinstance(p[key], str) or not re.fullmatch('[0-9a-f]{64}', p[key])):
            raise OpsError('invalid R22 source identity')


def verify_gae_proof(p, root=ROOT):
    validate_gae_profile(p)
    verify_temporal_proof(temporal_experiment(), root)  # complete R17/R19 controls,not R20 completion.
    return verify_r20_closed(root)


def verify_gae_checks(p, root=ROOT):
    from research_ops.identity import source_identity
    root = Path(root)
    try:
        path = root/'.research_local/autorl_r22_checks_01.json'; admission = read_json(path)
        source = p['source_identity_sha256']
        if admission['source_identity_sha256'] != source or admission['registration_sha256'] != p['registration_sha256']:
            raise OpsError('stale R22 admission')
        values = {}
        for key in ('operations', 'tensor', 'historical', 'runtime_binding'):
            ref = admission['evidence'][key]; rel = f'.research_local/autorl_r22_{key}_final_01.json'
            if ref['path'] != rel or sha256(root/rel) != ref['sha256']:
                raise OpsError('R22 admission path/hash differs')
            values[key] = read_json(root/rel)
        op, te, hi, bi = (values[k] for k in ('operations', 'tensor', 'historical', 'runtime_binding'))
        if (not op['complete'] or op['failures'] or op['errors'] or op['optimizer_updates'] or op['tests_run'] != 211
                or len(op['results']) != 211 or any(v['outcome'] != 'PASS' for v in op['results'])
                or op['source_identity_sha256'] != source or op['test_source_identity'] != source_identity(root, ['tests/research_ops_v2'])
                or te['returncode'] or te['tests_run'] != 70 or te['optimizer_updates'] or 'Ran 70 tests' not in te['stderr']
                or te['source_identity_sha256'] != source or te['registration_sha256'] != p['registration_sha256']
                or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
                or len(hi['results']) != 885 or any(v['outcome'] != 'passed' for v in hi['results'])
                or bi['returncode'] or bi['source_before_sha256'] != source or bi['source_after_sha256'] != source
                or bi['registration_sha256'] != p['registration_sha256'] or bi['final'] != admission['evidence']['historical']):
            raise OpsError('R22 complete current no-update admission missing')
        return {'status': 'PASS', 'admission_sha256': sha256(path)}
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise OpsError('missing/invalid R22 current tests') from error
