"""Evaluate 9-task GRU: episode-balanced metrics, confusion matrix, per-PPO metrics.

Run from project root:
 python -m scripts.evaluate_gru_task --data-dir data/gru/ppo_500 --model-dir models/gru_task_500 --seq-len 20
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.train_gru_task import TASKS, POLICIES, POLICY_TO_TASK, WindowDataset, collate, load_episodes, logits_for
from src.task_detection.gru_identifier import GRUTaskIdentifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path('data/gru/ppo_500'))
    p.add_argument('--model-dir', type=Path, default=Path('models/gru_task_500'))
    p.add_argument('--seq-len', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    model_path = args.model_dir / f'gru_seq{args.seq_len}_best.pt'
    split_path = args.model_dir / f'gru_seq{args.seq_len}_splits.json'
    split_info = json.loads(split_path.read_text(encoding='utf-8'))
    episodes = load_episodes(args.data_dir)
    lookup = {(e['policy'], int(e['episode_id'])): i for i, e in enumerate(episodes)}
    test_indices = [lookup[(r['policy'], int(r['episode_id']))] for r in split_info['test']]
    assert len(test_indices) == len(set(test_indices))
    for r, i in zip(split_info['test'], test_indices):
        assert episodes[i]['seed'] == int(r['seed'])
        assert episodes[i]['success'] == bool(r['success'])
    ds = WindowDataset(episodes, test_indices, args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    assert checkpoint['seq_len'] == args.seq_len
    assert checkpoint['num_tasks'] == len(TASKS)
    model = GRUTaskIdentifier(obs_dim=checkpoint['obs_dim'], hidden_size=checkpoint['hidden_size'], num_tasks=len(TASKS)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, lengths, _ in loader:
            logits = logits_for(model, x.to(device), lengths)
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    assert len(predictions) == len(ds.windows)
    conf = np.zeros((len(TASKS), len(TASKS)), dtype=np.int64)
    ep_correct = {i: [] for i in test_indices}
    for (i, _), pred in zip(ds.windows, predictions):
        actual = episodes[i]['label']
        conf[actual, pred] += 1
        ep_correct[i].append(int(actual == pred))
    ep_acc = {i: float(np.mean(v)) for i, v in ep_correct.items()}
    per_task = {name: float(np.mean([ep_acc[i] for i in test_indices if episodes[i]['label'] == task_id])) for task_id, name in enumerate(TASKS)}
    per_policy = {name: float(np.mean([ep_acc[i] for i in test_indices if episodes[i]['policy'] == name])) for name in POLICIES}
    window_acc = float(np.trace(conf) / conf.sum())
    window_macro = float(np.mean(np.diag(conf) / np.maximum(conf.sum(axis=1), 1)))
    episode_balanced = float(np.mean(list(ep_acc.values())))
    task_episode_macro = float(np.mean(list(per_task.values())))
    print(f'Device: {device} | sequence length: {args.seq_len}')
    print(f'Test: {len(test_indices)} episodes, {len(ds.windows)} windows')
    print(f'Window-level accuracy:         {window_acc:.4f}')
    print(f'Window macro accuracy:         {window_macro:.4f}')
    print(f'Episode-balanced accuracy:     {episode_balanced:.4f}')
    print(f'Task-episode macro accuracy:   {task_episode_macro:.4f}')
    print('\nPer-task episode-balanced accuracy:')
    for name, value in per_task.items():
        print(f'  {name:22s} {value:.4f}')
    print('\nPer-PPO episode-balanced accuracy:')
    for name, value in per_policy.items():
        print(f'  {name:22s} {value:.4f} (task={TASKS[POLICY_TO_TASK[name]]})')
    print('\nConfusion matrix (rows=actual, columns=predicted; window counts):')
    print('Task order: ' + ', '.join(TASKS))
    print(conf)
    out = args.model_dir / f'gru_seq{args.seq_len}_evaluation'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['actual\\predicted', *TASKS])
        for task, row in zip(TASKS, conf):
            writer.writerow([task, *row.tolist()])
    normalized = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    with (out / 'confusion_matrix_normalized.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['actual\\predicted', *TASKS])
        for task, row in zip(TASKS, normalized):
            writer.writerow([task, *[f'{v:.6f}' for v in row]])
    results = {'sequence_length': args.seq_len, 'test_episodes': len(test_indices), 'test_windows': len(ds.windows),
               'window_accuracy': window_acc, 'window_macro_accuracy': window_macro,
               'episode_balanced_accuracy': episode_balanced, 'task_episode_macro_accuracy': task_episode_macro,
               'per_task_episode_balanced_accuracy': per_task, 'per_ppo_episode_balanced_accuracy': per_policy,
               'confusion_matrix_task_order': list(TASKS), 'confusion_matrix_window_counts': conf.tolist(),
               'note': 'Episode-balanced accuracy averages window correctness within each episode, not episode-majority-vote accuracy.'}
    (out / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'\nSaved evaluation outputs to: {out}')
    print('Note: episode-balanced accuracy averages window correctness within each episode.')


if __name__ == '__main__':
    main()
