"""Run OPS-only regressions with ML imports forbidden; publish unique evidence."""
import argparse
import importlib.abc
from pathlib import Path
import resource
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research_ops.identity import source_identity
from research_ops.storage import atomic_json,digest,now
from research_ops.environment import resource_status


class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','stable_baselines3','mujoco','mujoco_lidar'):
            raise RuntimeError('OPS unit regression forbids ML/simulator imports')


class Result(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.rows=[];self.minimum_mem=1e30
    def startTest(self,test):
        self.minimum_mem=min(self.minimum_mem,resource_status(ROOT)['mem_available_mib'])
        super().startTest(test)
    def addSuccess(self,test):
        self.rows.append({'nodeid':test.id(),'outcome':'PASS'});super().addSuccess(test)
    def addFailure(self,test,error):
        self.rows.append({'nodeid':test.id(),'outcome':'FAIL'});super().addFailure(test,error)
    def addError(self,test,error):
        self.rows.append({'nodeid':test.id(),'outcome':'ERROR'});super().addError(test,error)
    def addSkip(self,test,reason):
        self.rows.append({'nodeid':test.id(),'outcome':'SKIP'});super().addSkip(test,reason)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',required=True,type=Path)
    args=parser.parse_args()
    if args.evidence.exists():raise FileExistsError('evidence already exists')
    started=time.monotonic();cpu=time.process_time()
    source=source_identity(ROOT,['research_ops','researchctl'])
    test_sources=source_identity(ROOT,['tests/research_ops'])
    finder=NoML();sys.meta_path.insert(0,finder)
    try:
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/research_ops'),pattern='test_*.py')
        result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    finally:sys.meta_path.remove(finder)
    assert source_identity(ROOT,['research_ops','researchctl'])==source
    assert source_identity(ROOT,['tests/research_ops'])==test_sources
    evidence={'phase_id':'OPS-01','recorded_at':now(),'complete':True,'tests_run':result.testsRun,
              'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
              'optimizer_updates':0,'ML_imports':'FORBIDDEN_IN_UNIT_PROCESS','results':result.rows,
              'runtime_identity_sha256':digest(source),'runtime_identity':source,
              'test_source_identity':test_sources,'wall_seconds':time.monotonic()-started,
              'process_cpu_s':time.process_time()-cpu,
              'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
              'child_peak_rss_mib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024.,
              'minimum_sampled_mem_available_mib':result.minimum_mem,
              'scope':'OPS unit/real temporary Git/process fixtures; no PPO, PX4 or real devices'}
    atomic_json(args.evidence,evidence,exclusive=True)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
