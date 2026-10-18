"""Fixed new R33 prepare/start only;no checkpoint initialization or retry option."""
import argparse,json
from research_ops.storage import OpsError,FileLock,read_json,atomic_json,digest
from .runtime import ROOT,HOME,verify_current,run_directory,supervise
from .early_distance_profile import early_mix_experiment,verify_early_mix_proof,verify_early_mix_checks,require_preregistered
from .held_goal_profile import held_goal_manifest
from .closed_yaw_failure import eligible_after_closed_r24
from .budget import BudgetStore

def main():
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
 sub.add_parser('prepare');start=sub.add_parser('start');start.add_argument('--expected-protocol-sha256',required=True)
 args=parser.parse_args()
 try:
  manager,reg=verify_current()
  if reg['phase_id']!='AUTORL-R33-EARLY-DISTANCE-MIX':raise OpsError('R33 reviewed registration required')
  if args.command=='prepare':
   with FileLock(manager.lock_path):
    p=early_mix_experiment();p.update(source_identity_sha256=digest(reg['source_identity']),registration_sha256=digest(reg))
    require_preregistered(p,read_json(ROOT/'mujoco/reports/autonomous_rl_early_distance_r33_prereg.json'))
    verify_early_mix_proof(p);verify_early_mix_checks(p)
    manifest=read_json(ROOT/'mujoco/reports/autonomous_rl_early_distance_r33_manifest.json')
    if manifest!=held_goal_manifest(read_json(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')):raise OpsError('R33 fixed unchanged task manifest differs')
    b=BudgetStore(HOME).status();eligible_after_closed_r24(b)
    if (b['remaining_timesteps']<p['timesteps'] or b['remaining_wall_seconds']<p['wall_seconds']
     or any(v['experiment_id']==p['experiment_id'] for v in b['attempts']) or run_directory(HOME,p['experiment_id']).exists()):raise OpsError('R33 budget/consumed identity')
    atomic_json(HOME/'protocols'/(p['experiment_id']+'.json'),p,exclusive=True)
    result={'status':'PASS','protocol_sha256':digest(p),'experiment_id':p['experiment_id'],'reservation':'NOT_YET'}
  else:
   p=read_json(HOME/'protocols/ARL_R33_EARLY_DISTANCE_MIX_SEED0.json')
   require_preregistered(p,read_json(ROOT/'mujoco/reports/autonomous_rl_early_distance_r33_prereg.json'))
   if digest(p)!=args.expected_protocol_sha256:raise OpsError('R33 explicit protocol SHA differs')
   result=supervise(p)
  print(json.dumps(result,allow_nan=False))
 except OpsError as error:
  print(json.dumps({'status':'BLOCKED','error':str(error)}));raise SystemExit(2)

if __name__=='__main__':main()
