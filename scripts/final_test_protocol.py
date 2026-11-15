"""Reserve final seeds or lock candidates; this command never evaluates them."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.training.final_protocol import reserve_seeds, lock_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    reserve = commands.add_parser('reserve')
    reserve.add_argument('--output', type=Path, required=True)
    reserve.add_argument('--history', type=Path, nargs='+', default=[REPO / 'runs', REPO / 'output'])
    reserve.add_argument('--count', type=int, default=200)
    lock = commands.add_parser('lock')
    lock.add_argument('--manifest', type=Path, required=True)
    lock.add_argument('--runs', type=Path, nargs='+', required=True)
    lock.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'reserve':
        result = reserve_seeds(args.output, args.history, args.count)
        print(f"Reserved {len(result['seeds'])} seeds; audited {len(result['historical_records'])} JSON files. No evaluation performed.")
    else:
        result = lock_candidates(args.manifest, args.runs, args.output)
        print(f"Locked {len(result['candidates'])} candidates. No evaluation performed.")


if __name__ == '__main__':
    main()
