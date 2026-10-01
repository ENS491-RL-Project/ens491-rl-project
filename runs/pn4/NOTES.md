# PN-4 run log notes (written 2026-10-01, H-0)

Retrospective notes on the artifacts in this directory. Nothing here was regenerated;
these are corrections to the record.

## Stabilization threshold mismatch

`scripts/train_pn4.py` trained `{0,2}` (FourRooms, lateral from `{0,1}`) against a
**relaxed 0.75** mean-reward threshold, while `docs/ENS491_Project_State.md` and
`CLAUDE.md` both documented the criterion as **0.85**. The inline comment in
`train_pn4.py` records the relaxation ("relaxed for PN-4 milestone; FourRooms requires
~2M steps to reach 0.85") but the docs were never updated to match.

The distinction turned out not to matter for the outcome:

- `{0,1}` Empty-8x8: stabilized at **40,000 steps** (1,607 episodes), mean reward **0.958**. Frozen.
- `{0,2}` FourRooms: ran the **full 2,000,000 steps** (28,312 episodes) and **never stabilized**,
  ending at mean **0.240**, oscillating ~0.19-0.48 throughout. Saved unfrozen.

The gap to either threshold is ~0.51, not marginal, so the 0.75 vs 0.85 question never
became the deciding factor. Diagnosis of the FourRooms failure is roadmap task C-1.

Both thresholds are retired in Target Architecture v2 §5.3, which replaces reward
mean/std with a success-rate criterion.

## Checkpoint metadata is incomplete

`col01/meta.json` and `col02/meta.json` contain only three keys:

    {"n": 0, "m": 1, "frozen": true}
    {"n": 0, "m": 2, "frozen": false}

They were written by an older `ColumnTrainer.save()` than the one currently in
`src/continual_learning/pn_trainer.py`, which writes nine keys (seed, env_id,
reward_threshold, std_threshold, requested_device, actual_device, timestamp_utc).

Consequence: **these checkpoints are not reproducible from their own metadata** — no
seed, no env id, no threshold is recorded. v2 addresses this with a `spec_hash` in the
node registry (plan item F18); H-3 re-evaluates `col01` under the new criterion and
will record what is known about it then.

`full_run_log.txt` is UTF-16LE (a PowerShell `*>` redirect) and was produced on a
different machine (paths show `C:\Users\Emir\...`).
