"""Explicit fixed research operations; never accepts arbitrary scripts or shell."""
import argparse
import json
from research_ops.storage import OpsError,read_json,atomic_json,digest
from .budget import BudgetStore,grant,experiment
from .runtime import ROOT,HOME,supervise,verify_current,run_directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('status','doctor','recover','request-stop','baseline'):sub.add_parser(name)
    p=sub.add_parser('initialize');p.add_argument('--user-grant-sha256',required=True)
    p=sub.add_parser('prepare');p.add_argument('--arm',choices=('UNIFORM','CURRICULUM'),required=True)
    p=sub.add_parser('start');p.add_argument('--arm',choices=('UNIFORM','CURRICULUM'),required=True)
    p.add_argument('--expected-protocol-sha256',required=True)
    sub.add_parser('prepare-prior')
    sub.add_parser('prepare-distance')
    p=sub.add_parser('start-distance');p.add_argument('--expected-protocol-sha256',required=True)
    p=sub.add_parser('start-prior');p.add_argument('--expected-protocol-sha256',required=True)
    p=sub.add_parser('prepare-std');p.add_argument('--variant',choices=('RANDOM','PRIOR'),required=True)
    p=sub.add_parser('start-std');p.add_argument('--variant',choices=('RANDOM','PRIOR'),required=True)
    p.add_argument('--expected-protocol-sha256',required=True)
    args=parser.parse_args();store=BudgetStore(HOME)
    try:
        if args.command=='initialize':
            m,_=verify_current()
            from research_ops.storage import FileLock
            with FileLock(m.lock_path):store.initialize(expected_grant_sha256=args.user_grant_sha256)
            result=store.status()
        elif args.command in ('status','doctor','recover'):
            result=store.status()
            if args.command!='status':
                m,_=verify_current()
                result['history_and_environment']='PASS'
                result['recovery']='AUDIT_ONLY; NEVER REPLAY RESERVED/FAILED ATTEMPTS'
                result['shared_lock_held']=__import__('research_ops.storage',fromlist=['lock_held']).lock_held(m.lock_path)
        elif args.command=='request-stop':
            store.request_stop();result={'status':'STOP_REQUESTED','scheduling':'DISABLED'}
        elif args.command=='baseline':
            from .baseline import run_baseline
            result=run_baseline()
        elif args.command in ('prepare','prepare-prior','prepare-std','prepare-distance'):
            m,r=verify_current()
            from research_ops.storage import FileLock
            with FileLock(m.lock_path):
                if args.command=='prepare-distance':
                    from .distance_profile import distance_experiment,verify_distance_proof
                    if r['phase_id']!='AUTORL-R6-DISTANCE':raise OpsError('distance prepare requires reviewed R6 phase')
                    p=distance_experiment();verify_distance_proof(p)
                elif args.command=='prepare-std':
                    from .std_profile import std_experiment,verify_std_controls
                    if r['phase_id']!='AUTORL-R3-STD':raise OpsError('std prepare requires its reviewed training phase')
                    p=std_experiment(args.variant);verify_std_controls(p)
                elif args.command=='prepare-prior':
                    from .prior_profile import prior_experiment,verify_prior_proof
                    if r['phase_id']!='AUTORL-R2-PRIOR':raise OpsError('prior prepare requires its reviewed training phase')
                    p=prior_experiment();verify_prior_proof(p)
                else:p=experiment(args.arm,65536,3600)
                p.update(source_identity_sha256=digest(r['source_identity']),registration_sha256=digest(r))
                if args.command=='prepare-distance':
                    from .distance_profile import verify_distance_checks
                    verify_distance_checks(p)
                elif args.command=='prepare-prior':
                    from .prior_profile import verify_training_checks
                    verify_training_checks(p)
                elif args.command=='prepare-std':
                    from .std_profile import verify_std_checks
                    verify_std_checks(p)
                dest=HOME/'protocols'/f"{p['experiment_id']}.json"
                atomic_json(dest,p,exclusive=True)
            result={'protocol_sha256':digest(p),'protocol':p,'active':False}
        else:
            identity=('ARL_R6_DISTANCE_SEED0' if args.command=='start-distance' else
                f'ARL_R3_{args.variant}_STD03_SEED0' if args.command=='start-std' else
                'ARL_R2_PRIOR_SEED0' if args.command=='start-prior' else f'ARL_R1_{args.arm}_SEED0')
            p=read_json(HOME/'protocols'/f'{identity}.json')
            if digest(p)!=args.expected_protocol_sha256:raise OpsError('explicit experiment SHA mismatch')
            result=supervise(p)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    except (OpsError,OSError,ValueError,KeyError,TypeError) as error:
        print(json.dumps({'status':'BLOCKED','error':str(error),'error_type':type(error).__name__},ensure_ascii=False));return 2


if __name__=='__main__':raise SystemExit(main())
