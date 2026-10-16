"""R13 versioned fixed neural preflight; no training profile or arbitrary commands."""
from pathlib import Path
from research_ops.storage import OpsError, read_json, sha256, digest
from .runtime import ROOT, HOME
from .feedback_followup import eligible_budget, load_background, verify_released_raw
from .feedback_spec import verify_file, probe_cases

REPORT_SHA = 'a35382ca93b7d1c191afd03bd7b76cd0c83353c6ec1686f00c021d615115bced'


def configuration():
    return {'experiment_id': 'AUTORL_R13_CENTERED_ACTOR_PREFLIGHT_ONLY',
            'kind': 'ZERO_UPDATE_ACTOR_PARAMETERIZATION_PREFLIGHT', 'training_timesteps': 0,
            'optimizer_updates': 0, 'physics_steps': 0, 'new_scans': 0, 'episodes': 0,
            'validation_episodes': 0, 'test_episodes': 0, 'workers': 1, 'device': 'cpu',
            'wall_seconds': 120, 'case_count': 62,
            'policy_version': 'SAME_BACKGROUND_STATE_CENTERED_ACTOR_V1',
            'claim_limit': 'structural/gradient/SB3 serialization compatibility, NOT learned navigation or braking'}


def references(root=ROOT):
    root = Path(root)
    try:
        p = root/'mujoco/reports/autonomous_rl_low_kl_feedback_r12.json'
        verify_file(p, REPORT_SHA)
        report = read_json(p)
        if report['implementation_status'] != 'PASS':
            raise OpsError('R12 not published as complete implementation')
        for key in ('review', 'result_review'):
            review = report[key]
            if review['status'] != 'GO':
                raise OpsError('R12 independent review not GO')
            verify_file(root/review['evidence']['path'], review['evidence']['sha256'])
        verify_released_raw(report, root)
        return {'r12_report_sha256': REPORT_SHA, 'usage': 'READ_ONLY_FAILURE_HYPOTHESIS_NOT_TRAINING_INITIALIZATION'}
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise OpsError('missing/corrupt R13 fixed historical evidence') from error


def make_protocol(registration, root=ROOT):
    from .budget import BudgetStore
    root = Path(root)
    budget = BudgetStore(root/'.research_local/authorized_rl_v1').status()
    eligible_budget(budget)
    return {**configuration(), 'references': references(root), 'cases': probe_cases(),
            'registration_sha256': digest(registration),
            'source_identity_sha256': digest(registration['source_identity']),
            'budget_snapshot_sha256': digest(budget)}


def verify_checks(registration, root=ROOT):
    from research_ops.identity import source_identity
    root = Path(root)
    try:
        p = root/'.research_local/autorl_r13_checks_01.json'
        a = read_json(p)
        source = digest(registration['source_identity'])
        if a['registration_sha256'] != digest(registration) or a['source_identity_sha256'] != source:
            raise OpsError('stale R13 admission identity')
        values = {}
        for kind in ('operations', 'tensor', 'historical', 'runtime_binding'):
            ref = a['evidence'][kind]
            rel = f'.research_local/autorl_r13_{kind}_final_01.json'
            if ref['path'] != rel:
                raise OpsError('unexpected R13 admission path')
            verify_file(root/rel, ref['sha256'])
            values[kind] = read_json(root/rel)
        op, te, hi, bi = (values[k] for k in ('operations', 'tensor', 'historical', 'runtime_binding'))
        if (not op['complete'] or op['failures'] or op['errors'] or op['optimizer_updates'] or op['tests_run'] != 144
                or len(op['results']) != 144 or any(v['outcome'] != 'PASS' for v in op['results'])
                or op['source_identity_sha256'] != source
                or op['test_source_identity'] != source_identity(root, ['tests/research_ops_v2'])
                or te['returncode'] or te['tests_run'] != 50 or 'Ran 50 tests' not in te['stderr'] or te['optimizer_updates']
                or te['source_identity_sha256'] != source or te['registration_sha256'] != digest(registration)
                or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
                or len(hi['results']) != 885 or any(v['outcome'] != 'passed' for v in hi['results'])
                or bi['returncode'] or bi['source_before_sha256'] != source or bi['source_after_sha256'] != source
                or bi['registration_sha256'] != digest(registration) or bi['final'] != a['evidence']['historical']):
            raise OpsError('R13 current complete no-update admission missing')
        return {'status': 'PASS', 'admission_sha256': sha256(p)}
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise OpsError('missing/invalid R13 checks') from error
