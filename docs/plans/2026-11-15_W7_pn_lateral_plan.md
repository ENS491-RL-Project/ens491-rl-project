# W7 — Does a lateral from a frozen primitive make a new primitive learn faster? (C-2)

> **Revision 7**, 2026-11-15, by the parallel-plans advisor chat against `main` at `73bffca`.
> Revisions 2, 4, 6 and 7 apply four read-only reviews made in the CLI (see the end; revisions 6
> and 7 were edited in the CLI at Emir's request); revision 3 recorded Emir's decisions; revision 5
> records his answers of 17:04 (rows 4–6 of the decision table). Arms, seeds, metric and
> thresholds have not changed since revision 2; revision 7 moves the budget from 400,000 to
> 401,408 (Emir's decision) and tightens the outcome table.
> The advisor chat read the code but **did not run it**: where a real signature or a measurement
> disagrees with this file, the measurement wins and is recorded as a deviation.
> Numbers marked *(proposed)* were the advisor's proposals; Emir agreed to them on 2026-11-15
> (see "What Emir decides"). They are fixed from the moment step 2 starts (decision 0010).
>
> Informs: decision 0003 (Evidence: "None yet"), `open-questions.md` rows "Lateral connections:
> forward transfer at all?" and "Retrain the PN-4 column…?", decision 0006 ("PN forward transfer is
> claimed only if C-2 shows it"), Progress Report I §5.2. **Changes no file under `src/` and no
> existing run file.** Nothing is frozen or registered.
>
> **Needs decision 0013 `Accepted` before step 1** (text in the appendix): this plan departs from
> v2's wording in two places and makes one open question moot. Only Emir sets the status.

## Deviations from v2, stated up front

1. **Source.** v2 §12 (C-2) and §10 name the legacy `col01` as the lateral source; §7 [F23] says
   C-2 retrains its own source under the current pipeline. This plan does neither: it uses the
   frozen `{0,2} open_door` that already exists under the current pipeline. That matches §7's
   default in its index reading (`{0,m-1} → {0,m}`, with the target keeping `go_to_goal`'s
   `m = 3`), not in its "most recently created primitive" reading, which today would be
   `{0,9} drop_held`. It also keeps the reason behind F23 (source and target must share the
   pipeline). The target, `go_to_goal`, is the one §12 names.
2. **Arm C's source is not a FROZEN node.** v2 §7 🔒 takes lateral sources from FROZEN primitive
   nodes. Arm C reads an untrained network that is not a node and cannot be frozen through
   `freeze()`. It is an experimental control only: its weights have `requires_grad_(False)`, it is
   never registered, and no library node may be built this way.
3. **Not Empty-8x8 → FourRooms** (the pair in Progress Report I §5.2): FourRooms never stabilised
   in 2M steps and the reason is not diagnosed (C-1), so "steps to STABLE" cannot be measured there.

## Why this shape

- **One question:** with everything else equal, does a new primitive reach STABLE in fewer env
  steps when it reads a frozen earlier primitive through the lateral?
- **Target `go_to_goal`:** the slowest of H-4's three primitives (200k at seed 42), so among
  those it has the most room to improve. (Later primitives were slower: `get_key_blue` 240k,
  `get_key_yellow` 220k.)
- **Three arms, not two.** A lateral adds 8,192 weights and a second input path. If the lateral arm
  wins against "no lateral" alone, the result cannot tell *transferred knowledge* from *a bigger
  network*. The third arm (a lateral from an untrained network) separates the two.

## Known before the run

- The existing numbers (`get_key` 180k, `open_door` 100k, `go_to_goal` 200k) are **one seed (42)
  at `eval_every` 20,000**. Seed-to-seed spread is unknown. No existing number is reused, and
  numbers from this plan are **not comparable to H-4's** (more frequent evaluation gives the
  stopping rule more chances).
- **The lateral path has never been run in v2** (v2 tests cover only `SkillNode` validation; the
  lateral tests are on the v1 path). Confirmed by reading, not by running:
  - `LateralMlpExtractor` reads `source.policy.mlp_extractor`; a `SkillNode`'s `.policy` is a
    `PrimitivePolicyAdapter`. A holder object (`holder.policy = node.policy.sb3_policy`) is needed.
  - **The holder must be a plain Python object, not an `nn.Module`.** `ColumnPolicy` assigns the
    source before `super().__init__()` (`column_policy.py:115-116`), so an `nn.Module` holder
    fails at construction ("cannot assign module before Module.__init__() call"). If it were
    registered, the source would be re-initialised by SB3's orthogonal init and enter the
    optimizer. (Read from the installed torch and SB3 source by the fourth review, not run;
    step 1 checks the source's hash.)
  - `.claude/rules/training.md` says to stop and report before working around lateral access
    in the custom policy. The holder is such a workaround; it is reported here, and Emir's
    approval of the step-0b PR covers it.
  - `train_primitive` does not connect `node.lateral_sources` to its `lateral_source` argument.
    The script must set both, or `spec_hash` and `run.json` will not record the source.
  - `model.zip` stores `policy_kwargs`, which will contain the holder. If it cannot be saved,
    `record_training` raises after training and the run's report is lost. Fallback for scoring:
    `policy_state_dict.pt` with `attach_sb3_primitive_policy(lateral_source=holder)`.
- Lateral runs do two forward passes per step (v2 §7, F22). Wall time is reported, not compared.
- With `eval_every` 4,096 a run has up to 98 evaluations; while the policy still fails, one
  evaluation can cost more env steps than the training between two evaluations. The wall-time
  estimate comes from step 1, not from this file.

## Design (one design)

| | Arm A — none | Arm B — trained source | Arm C — untrained source |
|---|---|---|---|
| Lateral source | — | frozen `{0,2} open_door` | same network shape, random weights, `requires_grad_(False)`, never trained; **a new draw per training seed** (torch seed = 900000 + training seed) |
| Node id | `{0,3} go_to_goal w7-A` | `… w7-B` | `… w7-C` |
| Source recorded as | — | `lateral_sources=[open_door]` on the node (enters `spec_hash`, `run.json`) and its policy hash in the result JSON | torch seed and policy hash in the result JSON and, through the holder's `as_dict()`, in `run.json` (it is not a node) |

**Arm C's source is built the same way every time:** `torch.manual_seed(900000 + s)`, then
`attach_sb3_primitive_policy` on a throwaway `go_to_goal`-shaped node, then
`requires_grad_(False)` on every parameter; the holder's `.policy` is that SB3 policy.

**The holder records its source.** In arms B and C the holder has an `as_dict()` returning the
source's node id (arm B) or torch seed (arm C) and its `policy_sha256`. The run store calls
`as_dict()` when it writes the config (`run_store.py:38-52`), so `run.json` names the source
without a change under `src/`; without it the holder is stored as a type name only.

**What "everything else equal" covers.** For one training seed the three arms share env seeds and
the layout sequence of each env. Initial weights are not shared between A and B/C (`lat1`/`lat2`
are created before the heads, so the initialisation draws differ, `fc1` included); B and C start
from the same target weights. This limits the pairing's power, not the test's validity.

Same for all arms: `go_to_goal` spec, hook and env (DoorKey-6x6) taken from
`train_primitives.specs()`, with the `SkillNode` built by the new script (`build_node` takes no id
or lateral source); `PPO_DEFAULTS`, 8 envs, CPU; `train_primitive` unchanged; stopping rule of
decision 0004 unchanged (argmax, stopping seeds 10000–10049).

Fixed before step 2 *(all proposed)*:
- **Training seeds:** 1000, 1100, …, 1900, the same ten in every arm (30 runs). Spaced by 100
  because the trainer seeds its 8 envs with `seed + i`; adjacent seeds would share 7 of 8 env seeds.
- **Budget:** 401,408 env steps per run (= 196 rollouts of 2,048 = 98 × 4,096). SB3 trains in
  whole rollouts, so a 400,000 budget would end at 401,408 anyway and a 98th evaluation would fall
  beyond the stated budget; with 401,408 the budget, the last evaluation and the censoring value
  are the same number.
- **`eval_every`:** 4,096 (two policy updates; the floor is 2,048).
- **Metric:** `steps_trained` at STABLE, always a multiple of 4,096. A run not STABLE at the
  budget counts as 401,408 ("censored") and is marked. A run that becomes STABLE at the 98th
  evaluation is STABLE (it counts for the gate) and has the same value as a censored run.
- **Final score (reported, not a criterion):** each final model once on 500 fresh env seeds
  **60000–60499**, argmax, DoorKey-6x6 under `go_to_goal`'s own hook.

## Steps (stop and report after each)

- [x] **0. Compare, run nothing.** Done four times in the CLI on 2026-11-15. Confirmed by reading:
  training seeds 1000…1900 (env seeds 1000–1007, …, 1900–1907) and 60000–60499 are unused in
  `scripts/`, `src/`, `tests/`, `runs/v2` and `runs/training/`; torch seeds 900000 + seed are
  unused; the run store writes lateral sources under `children`.
- [x] **0b. Decision PR (documents only, no code).** One PR containing: this plan; decision
  **0013** from the appendix, `Status: Proposed` (0011 is spec 3's and 0012 is reserved for W1b);
  the v2 wording it makes stale (§7 [F23] sentence "C-2 retrains its own lateral source", §10
  `col01` row, §12 C-2 row, and one line under §13 that states both changes: C-2's source is the
  frozen `{0,2} open_door`, and C-2 alone has a control arm whose lateral source is an untrained
  network that is not a node, so v2 itself carries the exception to §7's FROZEN-source rule); a
  note on the `open-questions.md` row "Fresh test seed set" that 60000–60499 is taken by W7 (the
  roadmap lists only 50000–59999); and a Track C note in `status.md` (C-2 planned, source changed
  by 0013). Emir changes the status to `Accepted` in the PR and merges it.
  **Step 1 starts only after that merge** (CLAUDE.md: a `Proposed` decision is not implemented).
  If the spec-3 PR merges first, rebase.
- [x] **1. Script and checks, no long run.** New `scripts/train_w7_lateral.py`
  (`--arm A|B|C --seed <s>`, `--smoke`). Outputs only under `runs/v2/w7/` and the run store.
  Before step 2, show:
  - a smoke run of each arm finishes, and `model.zip` saves and loads back (or the fallback above);
  - the `open_door` policy hash taken **before the PPO object is built** equals the hash after
    training, in arm B; the same for the random source in arm C;
  - in arms B and C the lateral weights (`lat1`, `lat2`) change during training;
  - arm B's `run.json` lists `open_door` under `children` with its spec hash and policy hash;
    arm C's `run.json` config names the source's torch seed and policy hash;
  - arm A at seed 42 with `eval_every` 20,000: does its evaluation series equal H-4's recorded
    series for `go_to_goal` in `runs/v2/primitives.json`? A difference is a deviation to report,
    not a stop;
  - measured steps/s per arm and the estimated wall time for steps 2 and 3.
- [x] **2. Arm A, ten seeds.** Report steps to STABLE per seed, median, spread, number censored.
  **Gate:** if fewer than 7 of 10 are STABLE within the budget, stop: this target cannot carry the
  metric, and Emir chooses what follows (see "After").
- [x] **3. Arms B and C, ten seeds each.** Not on the machine W1 runs are using at the same time.
- [x] **4. Score and write up, in one PR.** Final score of the 30 models, each scored only after
  its lateral source's hash is checked against the recorded one (arm C's source is re-drawn from
  its torch seed or taken from `model.zip`). The 30 `policy_state_dict.pt` files are copied to
  `runs/v2/w7/models/<arm>_s<seed>.pt`, because the run store is not tracked; no `meta.json` is
  placed under `runs/v2/w7/` (`eval_h7_preflight.py` globs `*/meta.json`).
  `runs/v2/w7/w7_lateral.json`; `docs/plans/2026-11-15_W7_summary.md`; one figure (success rate
  against env steps, three arms, median and range over seeds) and one table (per arm: STABLE count,
  median and range of steps to STABLE, final score, steps/s) for Progress Report II.
  The same PR updates (CLAUDE.md rule): `results.md` (one row), `status.md` (Track C, C-2 row),
  decision 0003 (Evidence; "Affects" no longer needs the PN-4 retrain), decision 0006 (its PN line:
  claimed or not claimed, on this evidence), decision 0013 (Evidence: only a link to the
  `results.md` row of the experiment that used this source; W7 is not evidence for the choice),
  `open-questions.md`
  (the lateral row; remove "Retrain the PN-4 column…"). **Open or rebase this PR after the spec-3
  PR is merged**, so the KB pages do not conflict. The summary states that the metric uses 0004's
  argmax rule; if W1b later changes the rule to sampling, these numbers stay under the old rule.
  Then stop.

## Criterion (fixed before step 2; numbers proposed)

Runs are paired by training seed. **"X beats Y"** means both: one-sided **Wilcoxon signed-rank**
on the ten paired differences in steps to STABLE (censored runs at 401,408) gives **p < 0.05**
with X as the faster-arm candidate, and the median of the paired ratios steps(X) / steps(Y) is
**at most 0.80**. (Paired, because the arms share seeds; the signed-rank test stays valid whether
or not sharing a seed actually makes two runs similar.) The test is used three times: B against
A, B against C, and A against B (the reverse direction, with the ratio steps(A) / steps(B)).

The exact computation, fixed here because step counts are multiples of 4,096 and censored runs
tie at 401,408, so zero differences and tied ranks are expected:
- differences d = steps(faster-arm candidate) − steps(other arm), per seed; **pairs with d = 0 are
  dropped** (this includes pairs where both runs are censored); m pairs remain;
- |d| is ranked with **average ranks for ties**; the statistic W⁺ is the sum of ranks of positive d;
- the candidate is faster when d is negative, so a **small** W⁺ is evidence for it: the p-value is
  **P(W⁺ ≤ observed)**, **exact, by enumerating all 2^m sign assignments** of those ranks (at most
  1,024); the script computes this itself (about ten lines, no dependence on a SciPy version)
  and its test checks it against a hand-worked tie-free example;
- the median ratio steps(candidate) / steps(other) is taken over all ten pairs, censored values
  included.

Consequence: with m < 5 the smallest possible p is above 0.05, so the test cannot pass.

| Outcome | Condition | What may be written |
|---|---|---|
| **Forward transfer shown** | B beats A **and** B beats C | "On this pair, a lateral from a trained primitive reduced steps to STABLE by …" |
| **Faster, but not transfer** | B beats A, B does not beat C | "The lateral path helps; a trained source is not shown to matter" — not a PN claim |
| **No measurable benefit** | B does not beat A **and** A does not beat B | Reported as such, with the smallest effect this design could have seen |
| **Negative transfer** | A beats B | Reported as such |

The four rows are exclusive and cover every case ("B beats A" and "A beats B" cannot both hold).
C against A and C against B (C as the candidate) are reported with the same computation but
decide no row. If C beats B, the second row's sentence is replaced by "the trained source was
slower than an untrained one", still not a PN claim.

What this design can see (simulation, log-normal spread, median 200k, 10 pairs, arms treated as
independent): if the seed-to-seed spread is about 35% of the median, a true 40% reduction passes
about 9 times in 10, a 30% reduction about 6 in 10, a 20% reduction about 3 in 10; with no true
effect about 4 in 100. With a spread of 50%: about 0.65 / 0.4 / 0.2. (These were simulated with
SciPy's default p-value, not the exact one above, and this file does not record how "spread" was
defined in the simulation; treat them as approximate.) **So "no measurable benefit" means "no
reduction of roughly 40% or more", not "no effect".** With 15 seeds a 30% reduction passes about
3 times in 4. The summary recomputes this from arm A's measured spread, defined there as the
standard deviation of log steps to STABLE.

## After

- **Transfer shown:** one more pair before any general claim (v2 §12: "repeat on T1/T2"), as a
  new short plan.
- **Faster, not transfer / no benefit / negative:** that is the finding; 0003 and 0006 record that
  PN forward transfer is not claimed on this evidence. No tuning and re-running of this pair.
  Whether a second pair is worth trying is Emir's call.
- **Gate fails in step 2:** arms B and C have not run, so the target can be changed without
  touching the criterion. Candidates: `get_key` (source `open_door`), or a larger budget. This is
  the one exception to "Not allowed": it needs a new revision of this plan approved by Emir
  before any arm B or C run, and arm A is run again on all ten seeds under the new setting.
- Positioning in the report: CompoNet (ICML 2024) is cited for positioning, not compared against.

## Not allowed

- Editing `executor.py`, `option_env.py`, `evaluation.py`, `node_trainer.py`, `column_policy.py`,
  or anything that exists under `runs/v2/` (outside the new `runs/v2/w7/`) and `runs/training/`.
- Calling `registry.save`, `freeze_and_save` or `node.freeze()` on any W7 model or on arm C's source.
- Changing seeds, budget, `eval_every`, arms, test or thresholds after step 2 starts (only
  exception: the gate-fail path under "After").
- Adding seeds after seeing a result; tuning PPO settings for one arm only.
- Using `scripts/train_primitives.py` to train (it overwrites `runs/v2/primitives.json` and saves
  to the registry).
- Writing "PN works" / "PN does not work"; the claim is about this pair and this metric.

## Cost reporting

At most 30 × 401,408 = 12,042,240 training env steps, plus the seed-42 check and smoke runs, listed
separately. Evaluation episodes are not counted, and the summary says so. Wall time and steps/s per
arm are reported (the lateral's extra forward pass, F22).

## Who can run this

Someone comfortable with Python and PyTorch who can read SB3's policy classes; no knowledge of the
executor, options or the hierarchy is needed. One Claude Code session; new files only for code and
runs, plus the document edits listed in steps 0b and 4.

## What Emir decides

Emir's reply on 2026-11-15 16:59 (+03:00), after reading revision 2: "continue with your
recommendations". Rows marked **Agreed** rest on that sentence; he can still change rows 3–4 until
step 2 starts.

| # | Decision | State |
|---|---|---|
| 1 | Source `{0,2} open_door` instead of legacy `col01` or a retrained source (deviation 1) | **Agreed in chat 2026-11-15.** Formally: decision 0013 goes in as `Proposed` and Emir sets `Accepted` in the step-0b PR; step 1 waits for that |
| 2 | Arm C (untrained source, one draw per seed) as a control outside §7's FROZEN-source rule (deviation 2) | **Agreed 2026-11-15** (same file) |
| 3 | Seeds 1000…1900 (10 per arm), budget 401,408, `eval_every` 4,096 | **Agreed 2026-11-15.** Budget changed from 400,000 to 401,408 by Emir in the CLI after the fourth review ("401.408 olsun"). The advisor offered 10 or 15 seeds without recommending one; 10 is what the plan proposed. Say so before step 2 if 15 is wanted |
| 4 | Criterion: paired Wilcoxon, p < 0.05, median ratio ≤ 0.80; gate 7 of 10 | **Agreed 2026-11-15**, including the exact computation added in revision 4 (zeros dropped, average ranks, exact p), confirmed by Emir at 17:04. The tail (revision 6) and the exclusive outcome rows and third test (revision 7) were added afterwards at Emir's instruction to apply the review findings; his `Accepted` on the step-0b PR confirms them |
| 5 | Seed range 60000–60499 for W7 | **Agreed 2026-11-15.** Recorded by the note on the "Fresh test seed set" row in the step-0b PR; no separate hand-over is needed |
| 6 | Owner | Emir, with his own Claude Code session (his statement, 2026-11-15 17:04) |
| 7 | How Progress Report II words the change of pair from §5.2; date and template of the report | **Open** |

Also agreed (2026-11-15): the project is not bound to repeating Progress Report I's pair.

## Appendix — text for the `Proposed` decision file

```
# 0013 — C-2 uses a frozen v2 primitive as the lateral source, with an untrained-source control

Status: Proposed
Date: 2026-11-15   Source: docs/plans/2026-11-15_W7_pn_lateral_plan.md (revision 7)
Reviewed by Emir: no

## Decision
- C-2's lateral source is the frozen `{0,2} open_door`, not the legacy `runs/pn4/col01` (v2 §10,
  §12) and not a newly retrained source (v2 §7 [F23]).
- C-2 has a control arm whose lateral source is an untrained network. It is not a node, is never
  registered, and is allowed in this experiment only. v2 §7's rule for library nodes is unchanged.

## Why
- `open_door` is already trained under the normalised pipeline; F23's reason (pipeline match) holds.
- Without the control, a faster lateral arm cannot be told apart from a larger network.

## Rejected alternatives
- Retraining `col01` on Empty-8x8: a source unrelated to the curriculum; extra work for no gain.
- Empty-8x8 → FourRooms (Progress Report I §5.2): FourRooms never stabilised (C-1 not done).

## Evidence
None, and W7 does not provide any: W7 uses this source, it does not compare sources. The
alternatives (legacy `col01` retrained, another frozen primitive, all earlier primitives) are not
measured. The experiment that would settle the choice is the empirical row "Lateral sources" in
`open-questions.md`, which is only worth running if W7 shows transfer.

## Affects
v2 §7 [F23], §10, §12 and §13 wording (corrected in the same PR as this file); open-questions
"Retrain the PN-4 column…" (becomes moot; removed in W7's result PR) and "Fresh test seed set"
(60000–60499 is taken by W7); decision 0003 "Affects"; `status.md` Track C.
```

## What changed from draft 1 (CLI review, 2026-11-15)

The source choice and arm C are now stated as deviations from v2 and carried by a `Proposed`
decision file (draft 1 said no decision file was needed and had not read §10/§12); the script must
set both `lateral_sources` and `lateral_source`, and record arm C's source by hash; the holder must
not be an `nn.Module`, and the source hash is taken before the PPO object exists; the test is
paired (Wilcoxon) instead of Mann–Whitney; training seeds are spaced so env seeds do not overlap;
arm C draws a new random source per seed instead of one network; KB pages are updated in the
result PR, opened after spec 3 merges; the final score's env and hook are named; the wall-time
estimate is left to step 1; numbers are declared not comparable to H-4.

## What changed in revision 4 (second CLI review, 2026-11-15)

The decision file has a fixed number (0013); step 1 now explicitly waits for it to be `Accepted`,
and the same PR corrects v2 §10/§12 and notes the seed range on the test-seed-set row; the
decision's Evidence says plainly that W7 does not compare sources; the Wilcoxon computation is
spelled out (zeros, ties, exact p); decision 0006 is in the result PR's list. No change to arms,
seeds, budget, metric or thresholds.

## What changed in revision 6 (third CLI review, 2026-11-15)

v2 §7 [F23] is added to the wording the step-0b PR corrects and to 0013's "Affects"; the Wilcoxon
tail is stated (P(W⁺ ≤ observed)) and the ratio's direction named; step 4 says what goes into
0013's Evidence (a link only); the appendix's source line names this revision. No change to arms,
seeds, budget, metric or thresholds.

## What changed in revision 7 (fourth CLI review, by a subagent, 2026-11-15)

The budget is 401,408 instead of 400,000 (Emir's decision): SB3 trains in whole rollouts, so the
old budget ended at 401,408 with a 98th evaluation beyond it; the censoring value, the evaluation
count (98) and the cost line follow. "X beats Y" is defined once, the outcome rows are exclusive,
the reverse test is counted as a third test, and C against A and B is reported. The gate-fail path
is a stated exception to "Not allowed". The §13 line in the step-0b PR carries arm C's exception
in v2 itself; that PR also notes Track C in `status.md`. Arm C's source has a fixed construction,
the holder records its source in `run.json` through `as_dict()`, and step 4 checks source hashes
before scoring and copies the 30 state dicts to `runs/v2/w7/models/`. The holder's reason is
corrected (an `nn.Module` holder fails at construction) and the `training.md` stop-and-report rule
is cited. Smaller wording: which reading of §7's default is meant, "slowest of H-4's three",
what "everything else equal" covers, the undefined "spread" in the power paragraph, "new files
only". No change to arms, seeds, `eval_every`, metric or thresholds.
