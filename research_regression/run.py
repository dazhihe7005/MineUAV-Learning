"""Fixed no-update regression launcher; no arbitrary pytest/module arguments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from research_ops.environment import runtime_environment
from research_ops.identity import safe_path
from research_ops.storage import OpsError, atomic_json, read_json
from .upstream import ROOT, source_identity


def deny_legacy_tmp_access():
    """Explicit fault injection: the old dependency is inaccessible to this process.

    This does not rename/delete/mount over any actual user's directory and does
    not replace a scanner or fixture function. An attempted access fails the run.
    """
    legacy='/tmp/mineuav-mujoco-lidar-audit'
    def hook(event,args):
        if event in ('open','os.listdir','os.scandir','subprocess.Popen'):
            for argument in args:
                values=argument if isinstance(argument,(list,tuple)) else (argument,)
                if any(isinstance(v,(str,bytes)) and legacy in (v.decode(errors='replace') if isinstance(v,bytes) else v) for v in values):
                    raise OpsError('SIMULATED_LEGACY_TMP_UNAVAILABLE: forbidden dependency access')
    sys.addaudithook(hook)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=('historical',),required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args(argv)
    path=args.evidence.resolve()
    if not path.is_relative_to(ROOT/'.research_local') or path.exists():
        raise OpsError('new regression evidence must stay under ignored local storage')
    if not args.child:
        lock=read_json(ROOT/'research_ops/environment_lock.json')
        env=runtime_environment(ROOT,lock)
        return subprocess.run([lock['interpreter'],'-m','research_regression.run','--suite',args.suite,
            '--evidence',str(path),'--child'],cwd=ROOT,env=env).returncode
    deny_legacy_tmp_access()
    before=source_identity()
    from lidar_integration.verify_repaired_integration import main as historical
    result=historical(['--evidence',str(path)])
    after=source_identity()
    if before!=after:
        raise OpsError('regression source changed during execution')
    data=read_json(path)
    data['upstream_identity']=before
    data['legacy_tmp_access']='DENIED_BY_EXPLICIT_PYTHON_AUDIT_HOOK; no actual directory was changed'
    # This is this new run's result, not any original historical evidence file.
    atomic_json(path,data)
    return result


if __name__=='__main__':raise SystemExit(main())
