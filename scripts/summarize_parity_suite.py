"""Summarize matched runtime pairs without reading the reserved final seed set."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(root):
    rows = []
    common_path = root / 'common_evaluator.json'
    common = json.loads(common_path.read_text()) if common_path.exists() else None
    common_rows = {r['result_file']: r for r in common['rows']} if common else {}
    for task, strict in [('choice', False), ('choice', True), ('baseline', False), ('depth', False)]:
        for seed in [42, 43, 44]:
            stem = f'{task}_{"strict" if strict else "goal"}_{seed}'
            paths = [root / f'{stem}_{variant}.json' for variant in ['old', 'new']]
            if not all(p.exists() for p in paths):
                continue
            old, new = [json.loads(p.read_text()) for p in paths]
            fields = ['before', 'mask', 'action', 'after', 'reward', 'term', 'trunc']
            first_transition = next(({'index': i, 'fields': [k for k in fields if a[k] != b[k]]}
                for i, (a, b) in enumerate(zip(old['transitions'], new['transitions']))
                if any(a[k] != b[k] for k in fields)), None)
            first_update = next(({'update': i+1, 'fields': [k for k in a['buffer'] if a['buffer'][k] != b['buffer'][k]]}
                for i, (a, b) in enumerate(zip(old['updates'], new['updates'])) if a['buffer'] != b['buffer']), None)
            # Weight equality alone is insufficient; buffer digests identify when learning diverged.
            row = {'task': task, 'strict': strict, 'seed': seed,
                   'weights_equal': old['weights'] == new['weights'],
                   'first_learner_transition_difference': first_transition,
                   'first_buffer_difference': first_update,
                   'old': {k: old[k] for k in ['actual_train_calls', 'events', 'development']},
                   'new': {k: new[k] for k in ['actual_train_calls', 'events', 'development']},
                   'result_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
            for variant, data in [('old', old), ('new', new)]:
                name = f'{stem}_{variant}.json'
                if name in common_rows:
                    row[variant]['common_fixed_evaluation'] = common_rows[name]
                row[variant]['env_steps'] = data['report']['env_steps_own']
                row[variant]['decisions'] = data['report']['decisions_trained']
                row[variant]['stable'] = data['report']['stable']
                row[variant]['run_dir'] = data['report']['run_dir']
                if 'first_key' in data:
                    row[variant]['first_key'] = data['first_key']
            rows.append(row)
    return {'purpose': 'development sensitivity study', 'final_test_used': False,
            'budget_decisions': json.loads((root / 'cohort_plan.json').read_text())['budget'],
            'common_evaluator_sha256': hashlib.sha256(common_path.read_bytes()).hexdigest() if common else None,
            'pairs_completed': len(rows), 'pairs_expected': 12, 'pairs': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('output/parity_2026_11_12'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.out_dir)
    output = args.output or args.out_dir / 'summary.json'
    output.write_text(json.dumps(result, indent=2))
    print('pairs', result['pairs_completed'], '/', result['pairs_expected'])
    for row in result['pairs']:
        print(row['task'], 'strict' if row['strict'] else 'goal', row['seed'],
              'SR', row['old']['development']['success_rate'], '->', row['new']['development']['success_rate'],
              'equal weights', row['weights_equal'], 'first buffer', row['first_buffer_difference'])


if __name__ == '__main__':
    main()
