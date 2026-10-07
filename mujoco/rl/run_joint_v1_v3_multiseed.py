"""Eight fixed-budget new branches, seed0 reuse, report or read-only verification."""
import argparse
import json
from pathlib import Path

from joint_multiseed_replication import run_branch, assemble_report, verify_experiment, REPORT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--train', choices=('v1', 'v3'))
    operation.add_argument('--assemble', action='store_true')
    operation.add_argument('--verify', action='store_true')
    parser.add_argument('--seed', type=int, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.train:
        if args.seed is None:
            parser.error('--train requires --seed1..4; seed0 is reused')
        row = run_branch(root, args.train, args.seed)
        return 0 if row['status'] == 'completed' else 1
    if args.seed is not None:
        parser.error('--seed applies only to --train')
    if args.assemble:
        row = assemble_report(root)
        print(json.dumps(row['summary'], indent=2))
    else:
        print(json.dumps(verify_experiment(root/'mujoco/reports'/REPORT), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
