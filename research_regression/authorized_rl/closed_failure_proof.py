"""Versioned R20 closure proof for NEW research, never retry/resume/refund authority."""
from pathlib import Path
from research_ops.identity import git
from research_ops.storage import OpsError, read_json, sha256, digest
from .runtime import ROOT
from .budget import BudgetStore

R20_REPORT_SHA = 'ba600c42d4b851b390d3787e47c43635ef05112c89b7c12949de1aa4251ec014'
R21_REPORT_SHA = '0d6bc17fdbb88f23cec4ffa219d46da2677558ae31027f457095103e3f6a9404'
R20_EVENT_SHA = 'fe19a084055970bdcd8e4d1fe068decece3d9dc9140c313931b1efc02b13d7a2'
R20_ID = 'ARL_R20_TEMPORAL_CREDIT_SEED0'


def _reference(root, item):
    from research_ops.identity import safe_path
    path = safe_path(root, item['path'])
    if sha256(path) != item['sha256'] or 'bytes' in item and path.stat().st_size != item['bytes']:
        raise OpsError('closed failure evidence missing/changed')
    return path


def verify_r20_closed(root=ROOT):
    root = Path(root)
    try:
        reports = []
        for name, expected, commit, complete in (
                ('autonomous_rl_temporal_credit_r20.json', R20_REPORT_SHA, '5e95e3f9373753f408b6e2b92beb2670193e00cd', False),
                ('autonomous_rl_shutdown_guard_r21.json', R21_REPORT_SHA, '597ff08fd35d2d18c81c67e2f2072f718a918c33', True)):
            path = root/'mujoco/reports'/name
            if sha256(path) != expected:
                raise OpsError('R20 closure/R21 guard release differs')
            report = read_json(path); reports.append(report)
            for key in ('review', 'result_review'):
                review = report[key]
                if review['status'] != 'GO' or review['critical'] or review['important']:
                    raise OpsError('closure independent review is not GO')
                actual = read_json(_reference(root, review['evidence']))
                if actual['verdict'] != 'GO' or actual['critical'] or actual['important']:
                    raise OpsError('closure actual review is not GO')
            directory = root/'.research_local/ops_v2/versions'/('v2_'+report['registration_sha256'])
            envelope = read_json(directory/'state.json'); state = envelope['state']
            if (digest(state) != envelope['sha256'] or state['phase_complete'] is not complete
                    or state['latest_successful_push']['commit'] != commit
                    or git(root, 'merge-base', '--is-ancestor', commit, 'HEAD', check=False).returncode):
                raise OpsError('closure release not actually published/preserved')
        old, guard = reports
        if (old['implementation_status'] != 'FAIL' or old['execution_status'] != 'INTERRUPTED'
                or old['scientific_result'] != 'NOT TESTED' or old['automatic_retry']
                or guard['implementation_status'] != 'PASS' or guard['budget_unchanged'] is not True):
            raise OpsError('failed history/repaired scope changed')
        for item in old['raw_files'] + old['additional_posthoc_files']:
            _reference(root, item)
        run = root/'.research_local/authorized_rl_v1/runs'/R20_ID
        if set(p.name for p in run.iterdir()) != {Path(v['path']).name for v in old['raw_files']}:
            raise OpsError('old interrupted directory changed or false completion added')
        disposition = read_json(_reference(root, old['append_only_disposition']))
        actual_review = read_json(_reference(root, old['disposition_review']))
        event_path = root/'.research_local/authorized_rl_v1/events/000026.json'
        event = read_json(event_path)
        if (sha256(event_path) != R20_EVENT_SHA or disposition['new_event_sha256'] != R20_EVENT_SHA
                or event['kind'] != 'FAIL' or event['sequence'] != 26 or event['data']['experiment_id'] != R20_ID
                or event['data']['reason'] != disposition['reason'] or disposition['automatic_retry']
                or disposition['after_head_count'] != 26 or actual_review['verdict'] != 'GO_DISPOSITION_ONLY'
                or actual_review['critical'] or actual_review['important']):
            raise OpsError('R20 unique append-only closure not verified')
        budget = BudgetStore(root/'.research_local/authorized_rl_v1').status()
        entry = next(a for a in budget['attempts'] if a['experiment_id'] == R20_ID)
        if (entry['status'] != 'FAIL' or entry['result'] != event['data']
                or entry['protocol'] != old['configuration']):
            raise OpsError('R20 actual ledger no longer matches closed failed attempt')
        reg = read_json(root/'mujoco/reports/autonomous_rl_shutdown_guard_r21_registration.json')
        if digest(reg) != guard['registration_sha256']:
            raise OpsError('R21 published registration changed')
        for name in ('research_regression/authorized_rl/shutdown_guard.py', 'tests/research_ops_v2/test_shutdown_guard.py'):
            if sha256(root/name) != reg['source_identity'][name]['sha256']:
                raise OpsError('reviewed SIGTERM helper or regression changed')
        return {'status': 'PASS', 'R20_status': 'PERMANENT_FAIL_NO_RETRY',
                'R20_scientific_result': 'NOT TESTED', 'refund': False,
                'failed_report_sha256': R20_REPORT_SHA, 'guard_report_sha256': R21_REPORT_SHA,
                'scope': 'new separately reviewed research only;no old identity retries/checkpoint resume'}
    except (OSError, KeyError, TypeError, ValueError, StopIteration) as error:
        raise OpsError('missing/invalid closed R20/R21 proof') from error


def eligible_after_closed_r20(budget, root=ROOT):
    if budget['stop_requested']:
        raise OpsError('stop latched; no new research')
    for attempt in budget['attempts']:
        if attempt['status'] == 'COMPLETE':
            continue
        if attempt['status'] == 'FAIL' and attempt['experiment_id'] in ('ARL_R6_DISTANCE_SEED0', R20_ID):
            continue
        raise OpsError('unresolved/unknown failed attempt;new independent audit required')
    actual = BudgetStore(Path(root)/'.research_local/authorized_rl_v1').status()
    if digest(actual) != digest(budget):
        raise OpsError('caller budget is not the actual immutable grant ledger')
    # R6 closure remains verified by the existing frozen repair proof chain.
    from .repaired_distance_profile import verify_repaired_proof, repaired_distance_experiment
    verify_repaired_proof(repaired_distance_experiment(), root)
    return verify_r20_closed(root)
