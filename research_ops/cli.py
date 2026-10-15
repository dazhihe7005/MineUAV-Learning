"""Foreground CLI; unsupported tasks fail before any simulator/model is loaded."""
import argparse
import json
from pathlib import Path
import sys

from .lifecycle import Operations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('init', 'status', 'recover', 'verify', 'sync'):
        commands.add_parser(name)
    doctor = commands.add_parser('doctor')
    doctor.add_argument('--restore-dependencies', action='store_true')
    doctor.add_argument('--source', type=Path)
    start = commands.add_parser('start')
    start.add_argument('--task', default='ops-audit')
    start.add_argument('--max-seconds', type=float, default=0.)
    stop = commands.add_parser('request-stop')
    stop.add_argument('--local-only', action='store_true', help='request stop without Git synchronization')
    args = parser.parse_args(argv)
    try:
        manager = Operations(args.root)
        if args.command == 'init': result = manager.initialize()
        elif args.command == 'doctor':
            result = manager.doctor(restore_dependencies=args.restore_dependencies, source=args.source)
        elif args.command == 'start': result = manager.start(args.task, max_seconds=args.max_seconds)
        elif args.command == 'request-stop': result = manager.request_stop(sync=not args.local_only)
        else: result = getattr(manager, args.command)()
        print(json.dumps(result, indent=2, allow_nan=False))
        statuses = {result.get('status'), result.get('state', {}).get('stage_status')}
        return 2 if statuses & {'BLOCKED', 'SYNC_PENDING', 'STOP_REQUESTED', 'INTERRUPTED', 'FAIL'} else 0
    except Exception as error:
        # No traceback or environment variables containing credentials in routine output.
        print(json.dumps({'status': 'BLOCKED', 'error_type': type(error).__name__,
                          'reason': str(error) if isinstance(error, RuntimeError) else 'operation failed; inspect local evidence',
                          'AUTO_TRAINING_AUTHORIZED': 'NO', 'TRAINING_BUDGET_AVAILABLE': 0}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
