"""Explicit V2 registration and lifecycle; never initializes or rewrites V1."""
import argparse
import json
from pathlib import Path
from research_ops.storage import OpsError,atomic_json,digest,read_json
from .manager import OperationsV2
from .registration import build_registration


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('status','recover','verify','sync'):sub.add_parser(name)
    doctor=sub.add_parser('doctor');doctor.add_argument('--restore-dependencies',action='store_true')
    start=sub.add_parser('start');start.add_argument('--task',required=True)
    stop=sub.add_parser('request-stop');stop.add_argument('--no-sync',action='store_true')
    for name in ('migrate','register'):
        p=sub.add_parser(name);p.add_argument('--manifest',type=Path,required=True)
        p.add_argument('--review',type=Path,required=True);p.add_argument('--expected-sha',required=True)
    p=sub.add_parser('prepare');p.add_argument('--phase',required=True);p.add_argument('--task-id',required=True)
    p.add_argument('--task-type',required=True);p.add_argument('--parameters',default='{}')
    p.add_argument('--wall-time',type=float,required=True);p.add_argument('--sync-file',action='append',required=True)
    p.add_argument('--source-path',action='append',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report-path',default='mujoco/reports/research_operations_v2.json')
    args=parser.parse_args(argv);root=Path(__file__).resolve().parents[1];manager=OperationsV2(root)
    try:
        if args.command in ('migrate','register'):
            manifest=read_json(args.manifest)
            if digest(manifest)!=args.expected_sha:raise OpsError('explicit manifest SHA mismatch')
            result=manager.register(manifest,read_json(args.review),migrating=args.command=='migrate')
        elif args.command=='prepare':
            value=build_registration(root,phase_id=args.phase,tasks=[{'task_id':args.task_id,
                'task_type':args.task_type,'parameters':json.loads(args.parameters),
                'wall_time_s':args.wall_time,'workers':1}],sync_files=args.sync_file,source_paths=args.source_path,
                report_path=args.report_path)
            from research_ops.identity import safe_path
            path=safe_path(root,str(args.output))
            if not path.is_relative_to(root/'.research_local'):raise OpsError('drafts must remain under local storage')
            atomic_json(path,value,exclusive=True);result={'manifest':str(path),'sha256':digest(value),'active':False}
        elif args.command=='start':result=manager.start(args.task)
        elif args.command=='request-stop':result=manager.request_stop(sync=not args.no_sync)
        elif args.command=='doctor':result=manager.doctor(restore_dependencies=args.restore_dependencies)
        else:result=getattr(manager,args.command)()
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 2 if result.get('status') in ('BLOCKED','FAIL','SYNC_PENDING') else 0
    except (OpsError,OSError,ValueError,KeyError,TypeError) as error:
        print(json.dumps({'status':'BLOCKED','error':str(error)[:1000],'type':type(error).__name__},ensure_ascii=False))
        return 2


if __name__=='__main__':raise SystemExit(main())
