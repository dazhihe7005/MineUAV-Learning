"""Fixed R22 prepare/start;closed failures are not retry/resume permission."""
import argparse
import json
from research_ops.storage import OpsError, FileLock, read_json, atomic_json, digest
from .runtime import ROOT, HOME, verify_current, run_directory, supervise
from .gae_credit_profile import gae_experiment, verify_gae_proof, verify_gae_checks
from .closed_failure_proof import eligible_after_closed_r20
from .budget import BudgetStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True); commands.add_parser('prepare')
    p = commands.add_parser('start'); p.add_argument('--expected-protocol-sha256', required=True)
    args = parser.parse_args()
    try:
        manager, registration = verify_current()
        if registration['phase_id'] != 'AUTORL-R22-GAE-CREDIT':
            raise OpsError('R22 reviewed phase required')
        if args.command == 'prepare':
            with FileLock(manager.lock_path):
                p = gae_experiment()
                p.update(source_identity_sha256=digest(registration['source_identity']), registration_sha256=digest(registration))
                verify_gae_proof(p); verify_gae_checks(p)
                budget = BudgetStore(HOME).status(); eligible_after_closed_r20(budget)
                if (budget['remaining_timesteps'] < p['timesteps'] or budget['remaining_wall_seconds'] < p['wall_seconds']
                        or any(a['experiment_id'] == p['experiment_id'] for a in budget['attempts'])
                        or run_directory(HOME, p['experiment_id']).exists()):
                    raise OpsError('R22 insufficient budget or consumed identity')
                atomic_json(HOME/'protocols'/(p['experiment_id']+'.json'), p, exclusive=True)
                result = {'status': 'PASS', 'protocol_sha256': digest(p), 'experiment_id': p['experiment_id'], 'reservation': 'NOT_YET'}
        else:
            p = read_json(HOME/'protocols/ARL_R22_GAE_CREDIT_SEED0.json')
            if digest(p) != args.expected_protocol_sha256:
                raise OpsError('R22 explicit prepared protocol SHA differs')
            result = supervise(p)
        print(json.dumps(result, allow_nan=False))
    except OpsError as error:
        print(json.dumps({'status': 'BLOCKED', 'error': str(error)})); raise SystemExit(2)


if __name__ == '__main__': main()
