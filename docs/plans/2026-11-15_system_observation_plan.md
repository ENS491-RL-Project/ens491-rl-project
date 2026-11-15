# System observation — how the registered roots behave on fresh seeds, and where they fail

> Written 2026-11-15 by Mehmet P. with Claude Code. Evaluation only: nothing is trained, frozen or
> registered, and no file under `src/` changes. Seeds, rows and failure labels are fixed here before
> the run. It is an observation that prepares W8, not W8: the task id is given by hand (R-3 is not
> built) and the final test seeds (0015) are not used.

## Rows

Every row is argmax (the registered mode), through the `Executor`.

| Row | Node | Env | Seeds | Kind |
|---|---|---|---|---|
| unlock | `{1,1} unlock_door` | Unlock | 70000–70499 | home |
| doorkey | `{2,1} solve_doorkey` | DoorKey-6x6 | 71000–71499 | home |
| unlockpickup | `{2,2} unlock_pickup` | UnlockPickup | 72000–72499 | home |
| doorkey5 | `{2,1} solve_doorkey` | DoorKey-5x5 | 73000–73199 | zero-shot |
| doorkey8 | `{2,1} solve_doorkey` | DoorKey-8x8 | 74000–74199 | zero-shot |
| doorkey16 | `{2,1} solve_doorkey` | DoorKey-16x16 | 75000–75199 | zero-shot |
| flat_unlock | `{0,B} flat_unlock_ppo` (seed 44) | Unlock | 70000–70499 | reference |
| flat_doorkey | `{0,B} flat_doorkey_ppo` (seed 42) | DoorKey-6x6 | 71000–71499 | reference |

Zero-shot rows are observations, not results for the node: it was trained on 6x6 and its timeout
(360) is unchanged; DoorKey-5x5 ends at 250 env steps.

## Recorded per episode

Success, env steps, the root's selection sequence, the number of child timeouts, and for a failure
one label, in this order:
`env_ended` (the env ended first) · `loop:<node>` (a full-state repeat; the primitive acting when it
first repeated) · `timeout:<node>` (no repeat; the primitive acting at the end).

## Reported

Success rate with Wilson bounds, env steps on successful episodes (mean, median), the failure
table by label, example seeds per label, the most frequent selection sequences. Hierarchy and flat
PPO are compared on the same seeds where a flat model is saved.

Self-check: on the 50 held-out seeds each root must reproduce `runs/v2/regression.json`.
