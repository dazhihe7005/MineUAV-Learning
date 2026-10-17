"""R31 unique bounded explicit velocity-feedback diagnostic;no training CLI."""
import argparse,json,os,time
from research_ops.storage import OpsError,FileLock,read_json,atomic_json,digest,sha256,process_identity
from research_ops.identity import source_identity
from research_ops.environment import enforce_resources
from .runtime import ROOT,HOME,verify_current
from .budget import BudgetStore
from .velocity_damping_diagnostic import make_protocol,evaluate

OUT=ROOT/'.research_local/autorl_r31_velocity_damping/run_01'

def verify_checks(reg):
 p=ROOT/'.research_local/autorl_r31_checks_01.json';v=read_json(p);source=digest(reg['source_identity'])
 if v['status']!='PASS' or v['registration_sha256']!=digest(reg) or v['source_identity_sha256']!=source or v['optimizer_updates']:raise OpsError('R31 stale regression identity')
 actual={}
 for k in ('operations','tensor','historical','runtime_binding'):
  ref=v['evidence'][k];path=ROOT/ref['path']
  if ref['path']!=f'.research_local/autorl_r31_{k}_final_01.json' or sha256(path)!=ref['sha256']:raise OpsError('R31 changed regression evidence')
  actual[k]=read_json(path)
 op,te,hi,bi=(actual[k] for k in ('operations','tensor','historical','runtime_binding'))
 if (not op['complete'] or (op['tests_run'],op['failures'],op['errors'],op['optimizer_updates'])!=(266,0,0,0)
  or len(op['results'])!=266 or any(v['outcome']!='PASS' for v in op['results']) or op['source_identity_sha256']!=source
  or op['test_source_identity']!=source_identity(ROOT,['tests/research_ops_v2'])
  or te['returncode'] or te['tests_run']!=89 or te['optimizer_updates'] or 'Ran 89 tests' not in te['stderr']
  or te['registration_sha256']!=digest(reg) or te['source_identity_sha256']!=source
  or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
  or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results'])
  or bi['returncode'] or bi['registration_sha256']!=digest(reg) or bi['source_before_sha256']!=source
  or bi['source_after_sha256']!=source or bi['final']!=v['evidence']['historical']):raise OpsError('R31 full selected checks incomplete')
 return sha256(p)

def probe(reg,spec):
 import resource,torch
 from stable_baselines3 import PPO
 from unittest.mock import patch
 from lidar_integration.ppo_formal_scheduler import load_l6_prereg
 from lidar_integration.ppo_training_runner import _weights_sha
 from .policy_function_run import load_model
 from .policy_function_probe import references as model_references
 from .velocity_damping_trace import collect_damped
 def forbidden(*args,**kwargs):raise AssertionError('R31_FORBIDS_TRAINING_OPTIMIZER')
 started=time.monotonic();cpu=time.process_time();minimum=float('inf')
 frozen={str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()};budget=BudgetStore(HOME).status()
 def guard():
  nonlocal minimum
  minimum=min(minimum,enforce_resources(ROOT)['mem_available_mib'])
  if (time.monotonic()-started>spec['wall_seconds']-5 or BudgetStore(HOME).status()['stop_requested']
   or source_identity(ROOT,reg['source_paths'])!=reg['source_identity']):raise OpsError('R31 duration/stop/source protection')
 torch.set_num_threads(1);guard();cases=[]
 with patch.object(PPO,'learn',forbidden),patch.object(PPO,'train',forbidden),patch.object(torch.optim.Adam,'step',forbidden):
  prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
  model=load_model(spec['model'],prereg,model_references())
  if _weights_sha(model.policy)!=spec['expected_policy_sha256']:raise OpsError('R31 final model identity differs')
  for index,task in enumerate(spec['cases']):
   guard();path=OUT/f'trace_{index}.jsonl'
   with path.open('x',encoding='utf8') as stream:
    def emit(row):stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');stream.flush()
    result=collect_damped(model,task,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',guard=guard,on_record=emit)
   if result['outcome'] not in ('success','collision','time_limit'):raise OpsError('R31 unsafe physical/sensor failure')
   result['trace_file']={'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'bytes':path.stat().st_size}
   atomic_json(OUT/f'case_{index}.json',result,exclusive=True);cases.append(result)
 guard();assessment=evaluate(cases,spec)
 if (BudgetStore(HOME).status()!=budget or spec!=make_protocol(reg)
  or {str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()}!=frozen):raise OpsError('R31 authorization/raw/protocol changed')
 result={'status':'PASS','registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity']),'protocol_sha256':digest(spec),
  'cases':cases,'assessment':assessment,'actual_policy_steps':sum(v['policy_steps'] for v in cases),'physics_ticks':sum(v['physics_ticks'] for v in cases),
  'new_feedback_diagnostic_episodes':2,'reused_R30_control_episodes':2,'training_timesteps':0,'optimizer_updates':0,'test_episodes':0,
  'budget_snapshot_sha256':digest(budget),'authorization_files_unchanged':True,'authorization_file_count':len(frozen),
  'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,'worker_hwm_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
  'per_policy_guard_sampled_min_MemAvailable_mib':minimum,'claim_limit':spec['claim_limit']}
 atomic_json(OUT/'result.json',result,exclusive=True);atomic_json(OUT/'complete.json',{'result_sha256':sha256(OUT/'result.json')},exclusive=True)
 return {'status':'PASS','result_sha256':sha256(OUT/'result.json'),'episodes':2,'training_timesteps':0}

def run(expected_sha):
 if os.getpgrp()!=os.getpid():raise OpsError('R31 requires owned isolated foreground process group')
 from research_ops_v2.manager import OperationsV2
 from .lifetime import start_guard,stop_guard
 m=OperationsV2(ROOT);parent=process_identity(os.getppid())
 with FileLock(m.lock_path):
  _,reg=verify_current()
  if reg['phase_id']!='AUTORL-R31-VELOCITY-DAMPING':raise OpsError('R31 wrong fixed reviewed phase')
  verify_checks(reg);enforce_resources(ROOT);spec=read_json(ROOT/'mujoco/reports/autonomous_rl_velocity_damping_r31_prereg.json')
  if digest(spec)!=expected_sha or spec!=make_protocol(reg):raise OpsError('R31 frozen protocol differs')
  watcher=start_guard(parent,time.monotonic()+spec['wall_seconds'],ROOT)
  try:
   OUT.mkdir(parents=True,exist_ok=False)
   atomic_json(OUT/'attempt.json',{'owner':process_identity(),'parent':parent,'protocol_sha256':digest(spec),'kind':spec['kind']},exclusive=True)
   try:return probe(reg,spec)
   except BaseException as error:
    import traceback
    atomic_json(OUT/'failure.json',{'exception_type':type(error).__name__,'message':str(error)[:2000],'traceback':traceback.format_exc()[-12000:],
     'partial_traces':[{'path':str(p.relative_to(ROOT)),'sha256':sha256(p),'bytes':p.stat().st_size} for p in OUT.glob('trace_*.jsonl')],
     'context':'unique hybrid action-feedback diagnostic,no retry/no training'},exclusive=True);raise
  finally:stop_guard(watcher)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-sha',required=True)
 print(json.dumps(run(parser.parse_args().expected_sha),allow_nan=False))

if __name__=='__main__':main()
