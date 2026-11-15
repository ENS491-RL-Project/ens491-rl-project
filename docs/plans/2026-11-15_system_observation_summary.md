# System observation — summary

Plan: [`2026-11-15_system_observation_plan.md`](2026-11-15_system_observation_plan.md).
Result file: [`runs/v2/system_observation.json`](../../runs/v2/system_observation.json)
(every episode: seed, success, steps, label, selection sequence).
Run 2026-11-15, evaluation only, argmax, task id given by hand. The script first reproduced
`regression.json` for the three roots on the held-out seeds.

## Result

| Row | Node, env | Success | Wilson 95 % | Env steps on success (mean / median) | Failures |
|---|---|---|---|---|---|
| unlock | `{1,1} unlock_door`, Unlock | 496/500 | 0.980–0.997 | 14.5 / 14 | 4 loop in `get_key` |
| doorkey | `{2,1} solve_doorkey`, DoorKey-6x6 | 500/500 | 0.992–1.000 | 13.8 / 13 | — |
| unlockpickup | `{2,2} unlock_pickup`, UnlockPickup | 491/500 | 0.966–0.991 | 23.2 / 23 | 8 loop in `get_key`, 1 loop in `pickup_box` |
| doorkey5 (zero-shot) | `{2,1} solve_doorkey`, DoorKey-5x5 | 200/200 | 0.981–1.000 | 10.4 / 10 | — |
| doorkey8 (zero-shot) | `{2,1} solve_doorkey`, DoorKey-8x8 | 52/200 | 0.204–0.325 | 20.1 / 20 | 80 loop in `go_to_goal`, 58 in `get_key`, 10 in `open_door` |
| doorkey16 (zero-shot) | `{2,1} solve_doorkey`, DoorKey-16x16 | 7/200 | 0.017–0.070 | 55.9 / 61 | 128 loop in `get_key`, 52 in `go_to_goal`, 13 in `open_door` |
| flat_unlock (reference) | `{0,B} flat_unlock_ppo` s44, Unlock | 482/500 | 0.944–0.977 | 10.3 / 10 | 18 loop |
| flat_doorkey (reference) | `{0,B} flat_doorkey_ppo` s42, DoorKey-6x6 | 494/500 | 0.974–0.995 | 15.4 / 15 | 6 loop |

## What it shows

- **Every failure, in every row, is a deterministic loop**: the full env state repeats while one
  primitive acts. No failure is a plain timeout, a wrong selection order, or the env ending first.
  The selector then re-selects the same child (`get_key > get_key`), which loops again.
- **On the home envs the weak primitive is `{0,1} get_key`**: 12 of the 13 failures. On successful
  episodes the selection sequence is the same in all 496 / 500 / 491 episodes.
- **Zero-shot, the navigation does not carry to larger rooms**: 1.00 on 5x5, 0.26 on 8x8, 0.035 on
  16x16, all loops in `go_to_goal` and `get_key`. These rows are observations, not results for the
  node (trained on 6x6, timeout 360 unchanged).
- **Hierarchy against flat PPO on the same seeds:**
  - Unlock: the hierarchy is more reliable (496 against 482; it alone solves 17 seeds, flat alone 3)
    and slower: on the 479 seeds both solve, 14.4 against 10.3 env steps; flat is faster on 348.
  - DoorKey-6x6: the hierarchy is more reliable (500 against 494) and faster: on the 494 seeds both
    solve, 13.8 against 15.4 env steps; the hierarchy is faster on 277, flat on 110.
  One flat seed per task (the saved models); not a comparison of training cost.

## Limits

One checkpoint per node; argmax only (the open question on execution mode is not touched: under
sampling the audit found no loops in frozen nodes); recognition is not in the loop; zero-shot rows
use sizes the nodes never saw.

## Example seeds to replay

Unlock `get_key` loop: 70050, 70262, 70323. UnlockPickup `get_key` loop: 72085, 72112, 72176;
`pickup_box` loop: 72135. DoorKey-8x8: `go_to_goal` 74001, `get_key` 74006, `open_door` 74034.
