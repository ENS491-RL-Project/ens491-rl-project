# 0004 — A node is STABLE at success ≥ 0.90 with Wilson lower bound ≥ 0.80, twice

Status: Accepted
Date: 2026-10-01 · amended 2026-10-02 (H-6.8)   Approved by: Emir (2026-11-12)   Source: v2 §5.3
Reviewed by Emir: yes — 2026-11-12

## Decision
Every `eval_every` steps, 50 episodes on a fixed seed set. STABLE when two consecutive evaluations, at least one policy update apart, both have success ≥ 0.90 and Wilson 95% lower bound ≥ 0.80 (binds at 46/50). A held-out seed set is scored at freeze.

## Why
- The v1 rule (mean reward > 0.85, std < 0.05) depends on path length.
- At N = 50, σ ≈ 0.042: 'SR ≥ 0.90 twice' with resampled seeds stops on a lucky pair.
- H-6.8 found two evaluations of the same network counted as two; `solve_doorkey` was declared STABLE after one update. Hence the one-update floor.

## Rejected alternatives
- Looser threshold: errors compound up the hierarchy (three 0.90 skills in a chain ≈ 0.73). `pickup_box`'s failure is traced mainly to deterministic execution (loops), not to the threshold: see [`plans/2026-10-02_H7_summary.md`](../../plans/2026-10-02_H7_summary.md) §12.
- Stricter LB ≥ 0.85: silently demands 48/50.
- Mean episode length as a criterion: reported, not used.

## Evidence
- H-6.8: two evaluations of one network counted as two (identical pairs in every H-6 history): v2 §5.3.
- Threshold alternatives (stricter / looser) not compared yet: empirical question 'Stabilization thresholds' in [`open-questions.md`](../open-questions.md).
- Results in [`results.md`](../results.md): H-7a `drop_held` + `pickup_box` spec 2; execution-mode audit.

## Affects
`EvaluationCallback`; `pickup_box` never met it (0.64–0.70 after 1M steps).
