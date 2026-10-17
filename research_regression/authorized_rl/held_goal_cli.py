"""R28 fixed prepare/start only; never a retry or checkpoint initialization CLI."""
import argparse
import json
from research_ops.storage import OpsError,FileLock,read_json,atomic_json,digest
from .runtime import ROOT,HOME,verify_current,run_directory,supervise
from .held_goal_profile import held_goal_experiment,verify_held_goal_proof,verify_held_goal_checks
from .held_goal_profile import held_goal_manifest
from .closed_yaw_failure import eligible_after_closed_r24
from .budget import BudgetStore


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('prepare');start=commands.add_parser('start');start.add_argument('--expected-protocol-sha256',required=True)
    args=parser.parse_args()
    try:
        manager,reg=verify_current()
        if reg['phase_id']!='AUTORL-R28-HELD-GOAL-YAW':raise OpsError('R28 reviewed registration required')
        if args.command=='prepare':
            with FileLock(manager.lock_path):
                p=held_goal_experiment();p.update(source_identity_sha256=digest(reg['source_identity']),registration_sha256=digest(reg))
                verify_held_goal_proof(p);verify_held_goal_checks(p)
                manifest=read_json(ROOT/'mujoco/reports/autonomous_rl_held_goal_r28_manifest.json')
                if manifest!=held_goal_manifest(read_json(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')):
                    raise OpsError('published R28 yaw manifest differs')
                budget=BudgetStore(HOME).status();eligible_after_closed_r24(budget)
                if (budget['remaining_timesteps']<p['timesteps'] or budget['remaining_wall_seconds']<p['wall_seconds']
                        or any(a['experiment_id']==p['experiment_id'] for a in budget['attempts'])
                        or run_directory(HOME,p['experiment_id']).exists()):raise OpsError('R28 insufficient budget or consumed identity')
                atomic_json(HOME/'protocols'/(p['experiment_id']+'.json'),p,exclusive=True)
                result={'status':'PASS','protocol_sha256':digest(p),'experiment_id':p['experiment_id'],'reservation':'NOT_YET'}
        else:
            p=read_json(HOME/'protocols/ARL_R28_HOLD_GOAL_YAW_SEED0.json')
            if digest(p)!=args.expected_protocol_sha256:raise OpsError('R28 explicit protocol SHA mismatch')
            result=supervise(p)
        print(json.dumps(result,allow_nan=False))
    except OpsError as error:
        print(json.dumps({'status':'BLOCKED','error':str(error)}));raise SystemExit(2)


if __name__=='__main__':main()
