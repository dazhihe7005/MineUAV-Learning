"""Explicit fixed R20 prepare/start only; no scripts, checkpoints or arbitrary parameters."""
import argparse,json
from research_ops.storage import OpsError,FileLock,read_json,atomic_json,digest
from .runtime import ROOT,HOME,verify_current,run_directory,supervise
from .temporal_credit_profile import temporal_experiment,verify_temporal_proof,verify_temporal_checks
from .feedback_followup import eligible_budget
from .budget import BudgetStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True);subs.add_parser('prepare')
    p=subs.add_parser('start');p.add_argument('--expected-protocol-sha256',required=True)
    args=parser.parse_args()
    try:
        m,reg=verify_current()
        if reg['phase_id']!='AUTORL-R20-TEMPORAL-CREDIT':raise OpsError('R20 reviewed phase required')
        if args.command=='prepare':
            with FileLock(m.lock_path):
                p=temporal_experiment();p.update(source_identity_sha256=digest(reg['source_identity']),registration_sha256=digest(reg))
                verify_temporal_proof(p);verify_temporal_checks(p);budget=BudgetStore(HOME).status();eligible_budget(budget)
                if (budget['remaining_timesteps']<p['timesteps'] or budget['remaining_wall_seconds']<p['wall_seconds']
                    or any(a['experiment_id']==p['experiment_id'] for a in budget['attempts'])
                    or run_directory(HOME,p['experiment_id']).exists()):raise OpsError('R20 insufficient budget or consumed identity')
                atomic_json(HOME/'protocols'/(p['experiment_id']+'.json'),p,exclusive=True)
                result={'status':'PASS','protocol_sha256':digest(p),'experiment_id':p['experiment_id'],'reservation':'NOT_YET'}
        else:
            p=read_json(HOME/'protocols'/'ARL_R20_TEMPORAL_CREDIT_SEED0.json')
            if digest(p)!=args.expected_protocol_sha256:raise OpsError('R20 explicit prepared protocol SHA differs')
            result=supervise(p)
        print(json.dumps(result,allow_nan=False))
    except OpsError as e:
        print(json.dumps({'status':'BLOCKED','error':str(e)}));raise SystemExit(2)


if __name__=='__main__':main()
