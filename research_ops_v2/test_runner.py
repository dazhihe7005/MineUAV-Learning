"""Fixed stdlib-only operations suite; no ML module is imported."""
import io
import argparse
import importlib.abc
import resource
import sys
import time
import unittest
from research_ops.identity import source_identity
from research_ops.storage import digest,atomic_json,read_json
from research_ops.environment import resource_status
from .registration import validate_registration,MANDATORY_SOURCE_PATHS


class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','stable_baselines3','mujoco','mujoco_lidar'):
            raise RuntimeError('operations unit tests forbid ML/simulator imports')


def run_tests(root,registration=None):
    started=time.monotonic();cpu=time.process_time();minimum=resource_status(root)['mem_available_mib']
    loader=unittest.TestLoader()
    suite=loader.discover(str(root/'tests/research_ops_v2'),pattern='test_*.py')
    output=io.StringIO()
    class Result(unittest.TextTestResult):
        rows=[]
        def addSuccess(self,test):self.rows.append({'nodeid':test.id(),'outcome':'PASS'});super().addSuccess(test)
        def addFailure(self,test,error):self.rows.append({'nodeid':test.id(),'outcome':'FAIL'});super().addFailure(test,error)
        def addError(self,test,error):self.rows.append({'nodeid':test.id(),'outcome':'ERROR'});super().addError(test,error)
    finder=NoML();sys.meta_path.insert(0,finder)
    try:result=unittest.TextTestRunner(stream=output,verbosity=2,resultclass=Result).run(suite)
    finally:sys.meta_path.remove(finder)
    identity=source_identity(root,registration['source_paths'] if registration else MANDATORY_SOURCE_PATHS)
    return {'complete':result.wasSuccessful(),'tests_run':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'optimizer_updates':0,'scope':'fixed stdlib operations V2 unit tests',
            'source_identity_sha256':digest(identity), 'results':result.rows,
            'test_source_identity':source_identity(root,['tests/research_ops_v2']),
            'output':output.getvalue(),'ML_imports':'FORBIDDEN_IN_UNIT_PROCESS',
            'wall_seconds':time.monotonic()-started,'process_cpu_s':time.process_time()-cpu,
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
            'child_peak_rss_mib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024.,
            'minimum_endpoint_mem_available_mib':min(minimum,resource_status(root)['mem_available_mib'])}


def main():
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];registration=read_json(args.manifest)
    validate_registration(root,registration)
    before=source_identity(root,registration['source_paths'])
    value=run_tests(root,registration)
    assert before==source_identity(root,registration['source_paths'])
    atomic_json(args.evidence,value,exclusive=True)
    print(value['output']);return 0 if value['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
