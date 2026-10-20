# H-7a close-out — the decision on `pickup_box`, four corrections, one diagnostic

> Written 2026-10-02 16:50 (+03) by the advisor chat, against commit `b9a8150`
> (`docs/plans/2026-10-02_H7_summary.md`, `runs/v2/`). The advisor chat cannot run the repo's code:
> where a real function signature or a measurement disagrees with this file, the measurement wins
> and is recorded as a deviation.
>
> **Nothing in this file authorises training.** No policy is trained, re-trained, continued or
> frozen in this run. It is documentation plus evaluation of checkpoints that already exist.

## 1. The decision

**H-7a is closed with A2b's failure as its result.** `{0,4} pickup_box` is not re-specified, its
budget is not extended, and `{2,2} unlock_pickup`, the option-set baseline and H-7b are not run.
Amendment 1 said "one re-specification only … no third attempt", and that holds.

Why the rule is kept, although one attempt costs under half an hour of wall time:

- Flat PPO got one configuration and one budget. Every further design on the hierarchy's side
  that is chosen after seeing a result makes the comparison weaker. That is what the rule was for.
- Both specifications were written by the advisor, the second from a one-number diagnosis, and
  both were wrong about where the difficulty was. A third written the same way would be a third
  guess.

**Whether a third design is ever attempted** is decided by the author with the supervisor
(supervisor brief, Q12). If it is, it is a new experiment with its own pre-set budget, and every
table that shows its result also shows the two withdrawn specifications and their cost.

## 2. What the files show, and what they do not

Verified from `runs/v2/` at `b9a8150`: every number in the H-7a summary's §2 and §3 matches its
file, the amortisation sums are right, regression is 9 of 9 at delta 0.0, origin equals HEAD.

Not established: **"the missing piece is the door crossing."** What was measured is that the
seed-42 checkpoint fails from left-room starts (0.58) and at the handoff (0.60) and succeeds from
right-room starts (0.96). At the handoff the agent stands on the cell in front of the open door,
one turn away from facing it, and still fails 20 of 50. Three readings fit that, and the files
cannot tell them apart:

1. **A loop.** The policy has no memory and is evaluated deterministically, and from the left room
   the box is out of view. A deterministic memoryless policy that maps two observations to
   "turn left" and "turn right" never leaves them. The one row with action shares
   (`h7_preflight.json`, `pickup_box` after `unlock_door` with the key in hand) is 76% turns.
2. **The key.** At the handoff the dropped key lies on the cell the agent faces. The training hook
   never produces that layout. The same row shows `pickup` and `drop` at 7% each, which is seven or
   eight pick-up-and-drop cycles per failed episode.
3. **Navigation as such**: it has not learned to find and pass the door.

The diagnostic in section 4 separates them. It decides nothing about what to build.

## 3. Corrections (documents and one analysis script; no training)

- [ ] **K.1 The exploratory arms: "no colour conditioning on any seed" is contradicted by the same
  table.** In H-6 summary §10.1, H-7a summary §5.1 and Project_State's "condition its child choice"
  row. The probe's mean-probability gap is small (+0.05 to +0.08), but the policies sit near 0.5, so
  the deterministic choice moves with it. First key = yellow on the held-out seeds:

  | run (strict mask, 100 updates) | yellow door | blue door | gap | first-key accuracy | a colour-blind policy with the same key rates |
  |---|---|---|---|---|---|
  | H-6.8, seed 42, `ent_coef` 0.01 | 1/23 | 0/27 | +0.04 | 0.56 | 0.54 |
  | H-6.8, seed 43, 0.01 | 15/23 | 6/27 | **+0.43** | 0.72 | 0.51 |
  | H-6.8, seed 44, 0.01 | 10/23 | 8/27 | +0.14 | 0.58 | 0.51 |
  | arm (a), seed 42, 0.01 | 1/23 | 0/27 | +0.04 | 0.56 | 0.54 |
  | arm (b), seed 42, 0.05 | 12/23 | 11/27 | +0.11 | 0.56 | 0.50 |
  | arm (b), seed 43, 0.05 | 14/23 | 7/27 | **+0.35** | 0.68 | 0.51 |
  | arm (b), seed 44, 0.05 | 10/23 | 3/27 | **+0.32** | 0.68 | 0.52 |

  Recompute every cell from `unlock_matching_s*.json` and `choice_diagnosis_arm_*.json`, add a
  two-sided Fisher exact test per row (the advisor's hand computation gives 0.004, 0.021 and 0.021
  for the three bold rows and above 0.35 for the rest; none is corrected for seven runs), and put
  the table in §10.1. Also state the probe gap's course in arm (b) on seeds 43 and 44: about zero
  through update 40, then +0.05 to +0.14 over the last twenty updates.
  Replace the two sentences with what the table supports: **three of seven strict-mask runs, all on
  seeds 43 and 44, show a weak dependence of the first key on the door's colour; seed 42 shows none
  in three runs; no run learned the choice (best first-key accuracy 0.72); the entropy bonus is not
  shown to matter, because seed 43 shows the dependence at 0.01 as well.**
- [ ] **K.2 The end-to-end number is in the repo but is not named.** `evaluate_node` runs `prepare`
  on all 50 held-out seeds without filtering, so the "real handoff" row is the success rate of the
  fixed sequence `unlock_door → drop_held → pickup_box` on UnlockPickup from the environment's own
  reset: **30/50 = 0.60** (Wilson 0.46–0.72), deterministic. Say so in H-7a summary §4 under R2 and
  in §9, with its three labels: the third skill is an unfrozen checkpoint, one seed, and this is a
  scripted sequence, not a node and not a trained composite. Beside it: flat PPO 0/50 on the same
  seeds after 3,000,320 steps. Confirm in the text that the two rows use the same 50 seeds.
- [ ] **K.3 The amortisation table hides two things in footnotes.** Add them to
  `scripts/analyse_h7_amortisation.py` and to the table as lines of their own, with the same
  double computation:
  - **As built**, through DoorKey: the frozen checkpoints cost
    180,000 + 100,000 + 131,817 + 200,000 + 162,138 = **773,955**, which is inside flat PPO's
    650,000–900,000. The 561,756–570,536 figure holds only under the H-6.8 update size, which flat
    PPO never got. Both lines are shown; the summary's R3 sentence names both.
  - **Including the withdrawn specification**, through UnlockPickup, per seed like every other
    cell: more than 1,593,228 – 1,602,008 plus 600,064 = **more than 2,193,292 – 2,202,072**,
    against flat PPO's more than 3,650,320 – 3,900,320. Beside it, the totals over all runs: the six
    `pickup_box` runs are 4,804,608 env steps, flat PPO's three UnlockPickup runs 9,000,960.
    Stated plainly in R3: with the failed design counted the hierarchy has still spent less than
    flat PPO over these three tasks, and neither has a stable solution for the third.
- [ ] **K.4 H-7a summary §9, second bullet.** "The missing piece is the door crossing, measured
  twice" becomes "the failures are in left-room starts, measured twice; why is not established
  (close-out §2)". Same change in Project_State's H-7a row.

**Verify:** the four places changed; `sums_verified: true` still; suite green; one commit per item.

## 4. H-7a.5 — where and how the spec-2 checkpoint fails (evaluation only)

- [ ] `scripts/diagnose_pickup_box.py --trace` → `runs/v2/pickup_box_spec2_trace.json`.
  The seed-42 TRAINING checkpoint, the 50 held-out seeds, two start conditions:
  **H** = after `unlock_door` then `drop_held`; **L** = own hook with the pose in the left room.
  Each under deterministic execution, and under `STOCHASTIC` execution with three sampling seeds
  (0, 1, 2), so 50 + 150 episodes per condition.
  Per episode record: success; steps; the room the agent ends in (left, doorway, right); whether
  it ever entered the right room; whether the box was ever in its view (from the raw observation
  image); what it holds at the end; the number of key pick-ups and of drops; action counts;
  whether the box stands directly behind the door; and, for deterministic episodes, the first step
  at which the full state repeats an earlier state of the episode (agent position and direction,
  what it carries, the positions of the key and the box, the door's state) and the period.
  Under deterministic execution a repeated state is a proof that the episode cannot succeed.
- [ ] Summarise per condition and mode: success rate with Wilson bounds, and over the failures the
  counts of (i) a state repeat, (ii) at least one key pick-up, (iii) never entered the right room,
  (iv) box behind the door, (v) box never seen. The counts may overlap; report all five.
- [ ] Readings, fixed here before the run, for the failures in condition H:
  - **loop**: (i) holds in at least 80% of the deterministic failures. Then also report whether
    stochastic execution reaches 0.90 over the 150 episodes.
  - **key**: (ii) holds in at least 50%.
  - **navigation**: (iii) holds in at least 50% and (i) in fewer than 50%.
  - anything else: no reading.
- [ ] Write the result as §10 of the H-7a summary. Wording rules: this describes one checkpoint;
  a stochastic success rate is not a gate and freezes nothing (v2 §5.3 evaluates deterministically);
  never write that `pickup_box` "works"; no sentence proposes a design.

**Verify:** the trace file has 2 × (50 + 150) episode records; the per-condition counts add up;
`git status` shows no change under any `runs/v2/<node>/` checkpoint directory; regression 9 of 9
at delta 0.0; suite green; origin = HEAD.

Then stop. The next step is the supervisor meeting.
