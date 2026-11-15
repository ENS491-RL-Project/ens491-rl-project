"""Run/resume the preregistered 24 development runs with two CPU workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('output/parity_2026_11_12'))
    parser.add_argument('--budget', type=int, default=5120)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant in ['old', 'new']:
        if not (args.out_dir / f'controls_{variant}.json').exists():
            raise RuntimeError('Run both --controls-only variants first')
    fixed = json.loads((args.out_dir / 'controls_new.json').read_text())['controls']
    for name, result in fixed.items():
        if result['train'] != result['eval'] or result['train_actions'] != result['eval_actions']:
            raise RuntimeError(f'Failed fixed-runtime control {name}')
    jobs = [(task, strict, seed, variant) for seed in [42, 43, 44]
            for task, strict in [('choice', False), ('choice', True), ('baseline', False), ('depth', False)]
            for variant in ['old', 'new']]
    plan = {'budget': args.budget, 'jobs': jobs, 'final_test_used': False,
            'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted([*REPO.joinpath('src').rglob('*.py'),
                                               REPO / 'scripts/run_parity_experiment.py', REPO / 'scripts/parity_tasks.py'])}}
    plan_path = args.out_dir / 'cohort_plan.json'
    serialized = json.dumps(plan, indent=2, sort_keys=True)
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != json.loads(serialized):
            raise RuntimeError('Plan or source changed; choose a new output directory')
    else:
        with plan_path.open('x') as f:
            f.write(serialized)
    def run(job):
        task, strict, seed, variant = job
        name = f'{task}_{"strict" if strict else "goal"}_{seed}_{variant}'
        result = args.out_dir / f'{name}.json'
        if result.exists():
            data = json.loads(result.read_text())
            if data['budget_decisions'] != args.budget or data['final_test_used']:
                raise RuntimeError(f'Unexpected existing result {result}')
            return name, 0, 'existing'
        command = [sys.executable, str(REPO / 'scripts/run_parity_experiment.py'),
                   '--task', task, '--seed', str(seed), '--variant', variant,
                   '--budget', str(args.budget), '--out-dir', str(args.out_dir)]
        if strict:
            command.append('--strict')
        with (args.out_dir / f'{name}.log').open('a') as log:
            code = subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        return name, code, 'completed' if code == 0 else 'failed'
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            result = future.result()
            results.append(result)
            print(*result, flush=True)
    (args.out_dir / 'cohort_status.json').write_text(json.dumps(results, indent=2))
    raise SystemExit(any(code for _, code, _ in results))


if __name__ == '__main__':
    main()
