# W1a diagnostic — why do some `pickup_box` models time out at the handoff? (evaluation only)

> Written 2026-11-15 by the advisor chat, after W1a failed (6 of 10 seeds passed; the rule asked for
> 7). To be run after the W1a results are merged. The advisor chat did not run the repo's code:
> where a real signature or a measurement disagrees with this file, the measurement wins and is
> recorded as a deviation.
>
> **Nothing here trains, freezes, registers or changes a rule, and no file under `src/` changes.**
> It decides nothing about what to build. It tells Emir which of three next plans is worth writing.
> W1b does not start from this plan.

## What W1a showed

Sampling removes the loops (every argmax failure is a state repeat; all 10 seeds pass from their own
hook). What is left is at the handoff: the four failing seeds (46, 47, 50, 52) score 0.874–0.892
there, and their failures are mostly timeouts (53–62 of 491 valid episodes, against 9–40 for the
passing seeds).

One earlier hint, on one exploratory checkpoint only (H-7a.5, seed 42, 10 sampled handoff failures):
10 of 10 contained a key pick-up and 9 of 10 never entered the right room. The training hook
`key_off_door` rejects layouts with a key next to the door, while at the real handoff the dropped
key lies right there. That is a hypothesis, not a finding.

## Three explanations to separate

1. **Key.** The dropped key is in the way; the agent picks it up or circles it. → the fix would be
   in the skill's training distribution (a new specification).
2. **Time.** The agent gets there, only too slowly for a 120-step timeout under sampling. → the fix
   would be the timeout (also a specification change).
3. **Search.** The box is out of view and a memoryless policy wanders. → the fix would be memory.

## Steps

- [ ] **1. Script and gate.** New `scripts/diagnose_w1a_handoff.py`, built from `scripts/eval_w1a.py`
  (model loading, hashes, seeding) and the trace code of `scripts/diagnose_pickup_box.py`. It has
  `--smoke`. **Gate:** on the unchanged handoff condition it must reproduce, for all 10 models, the
  per-seed success counts in `w1a_pickup_box_sampled.json` and `..._ext.json` exactly. If not, stop.
- [ ] **2. Trace the same episodes.** All 10 models (and seed 42, labelled exploratory), handoff row,
  env seeds 51000–51499, sampled, sampling seed 0: the very episodes W1a scored. Per episode:
  success, steps, end room, what is held at the end, key pick-ups, drops, steps spent holding the
  key, whether the right room was ever entered and at which step, whether the box was ever in view
  and at which step. Summarise failures **and successes** per seed (successes give the base rates).
- [ ] **3. Two interventions (diagnostic only, never a gate).** Same models, same env seeds, same
  sampling seed, `pickup_box` started from the same handoff state except for one change:
  - **K — key removed:** the dropped key is taken off the grid before `pickup_box` starts.
  - **T — double time:** `pickup_box` may run 240 steps instead of 120 (the env's 288 limit still applies).
  Report per seed and pooled over the four failing seeds, with Wilson bounds.
- [ ] **4. Write-up.** `runs/v2/w1a_handoff_diagnostic.json`,
  `docs/plans/2026-11-15_W1a_diagnostic_summary.md`, a row in `results.md`, `status.md`. Then stop.

## Readings (fixed now; over the four failing seeds pooled, about 1,960 valid episodes)

- **Key** if intervention K lifts the pooled success rate to **≥ 0.95**.
- **Time** if intervention T lifts it to **≥ 0.95**.
- **Search** if neither does, and in **at least 60%** of the unchanged failures the agent never
  picked up the key and the box was never in view.
- Both Key and Time may hold; report both. Anything else: **no reading**.
- The same numbers are shown for the six passing seeds, so a reading that holds for failing and
  passing seeds alike is reported as a property of the skill, not of the failing seeds.

## What each reading would lead to (decided by Emir, in a new plan)

| Reading | Next plan | Cost |
|---|---|---|
| Key | Re-specify `pickup_box`'s training start states to include the real handoff; then the W1a test again, unchanged | One primitive retrained; a new specification, shown beside the withdrawn ones |
| Time | Re-specify the timeout; same W1a test | As above; T7's 288-step window must still hold |
| Search | Memory arm (frame-stacked or recurrent primitives) | All 9 frozen nodes retrained |
| None | Emir's choice: accept an unreliable recipe and continue with a passing seed under a recorded decision, or stop the UnlockPickup line | — |

## Not allowed

- Presenting K or T numbers as results for `pickup_box`. They are counterfactuals.
- Changing the readings' thresholds after seeing any number.
- Training, freezing or registering anything; overwriting any existing file under `runs/v2/`.
- Any sentence that says what should be built. The summary reports the reading and stops.
