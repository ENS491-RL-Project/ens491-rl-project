# Brief — ENS 492 Progress Report II (due tonight, 15 Nov 2026, 23:55)

> Written by the advisor chat for a **new** Claude Code session (not the W1b session). Revision 2
> (19:40): paths, W1b-1 state, decision list and template gaps corrected after checking the
> template, PR-I and the repo on disk. The report is Emir's; this session produces a complete
> **draft** and a claims table. Emir reads, edits and submits. Do not start any training run.
> Do not commit to `main`.

## Where things are

- **Repo:** `C:\Users\memir\Desktop\Desktop\ACADEMY\ens491\ens491-rl-project` (`main` was at
  `2bc131c` when this was written; run `git log -1` and report the commit you actually read).
  The sibling folder `hierarchical-rl-agent` is a stale copy from 4 October: take no result,
  status or script from it.
- **Another session (W1b) is active on this repo.** Work in a separate worktree on a branch
  (`git worktree add ..\ens491-pr2 -b report/pr2 origin/main`), or write only new files. Do not
  touch `src/`, `runs/v2/` or `docs/kb/`.
- **Report folder:** `C:\Users\memir\Desktop\Desktop\ACADEMY\ens491\ENS 491\Reports\3-Progress Report 2\`
- **PR-I (style source and comparison baseline):**
  `...\Reports\2-Progress Report 1\submission\ENS_491_PROGRESS_REPORT_I.docx`
- **Template:** `...\Reports\3-Progress Report 2\Progress-II Template.doc` (binary `.doc`; its
  full requirements are transcribed in this brief, so you do not need to open it).
- **Proposal (original objectives):**
  `...\Reports\1-Proposal Report\submission\ENS_491_PROJECT_539_PROPOSAL_REPORT.docx`

## Deliverables (in this order; stop and report after 1, then after 4)

1. **Outline with one line per subsection and the list of figures/tables** (10 minutes, no prose).
2. **`claims.md`** in the report folder, built from the JSONs **before** any prose: every number
   the report will use, one row each: value, source file and field, and (filled in at step 4) the
   sentence it appears in. The prose is written from this table. A number with no row does not
   enter the report.
3. **Figures** under `reports/progress2/figures/` (PNG + PDF), from files already in `runs/v2/`.
   Add `scripts/generate_report_figures_pr2.py` (the existing `generate_report_figures.py` is the
   PR-I version; leave it as it is).
4. **`ENS_492_PROGRESS_REPORT_II.docx`** in the report folder. Start from a copy of the PR-I docx
   so cover, fonts, heading styles and caption style carry over; replace the body. English.
   Target 15–20 pages of body (the template sets no limit; PR-I was about 25).
   Emir updates the table of contents in Word and exports the PDF. **Fallback:** if keeping PR-I's
   styling is still not working after 30 minutes, write a plain docx with the correct headings
   and captions, and say so in the report-back.

Cover: "ENS 492 – Graduation Project (Implementation)" / "Progress Report II"; title, the five
group members (with IDs and e-mails) and supervisor exactly as on PR-I's cover; date
November 15th, 2026.

**Section headings, verbatim from the template (numbering and capitals as here):**

1. PROJECT SUMMARY
2. SCIENTIFIC/TECHNICAL DEVELOPMENTS
3. ENCOUNTERED PROBLEMS
4. SUSTAINABILITY AND ECONOMIC ASSESSMENT
5. TASKS TO BE COMPLETED UNTIL FINAL REPORT
6. REFERENCES

**If time runs short, cut in this order:** Fig. 1 redrawn (reuse a simple box diagram) → the
evaluation-infrastructure subsection to one paragraph → the choice-task subsection to one
paragraph. Never cut: Section 2.4, the Section 3 tables, Section 5's timetable, `claims.md`.

## Rules for the text

- **Sources:** `docs/kb/status.md`, `results.md`, `open-questions.md`, `docs/kb/decisions/`,
  `docs/plans/*summary*.md`, `docs/plans/2026-11-15_roadmap.md`,
  `docs/ENS491_Target_Architecture_v2.md`, result JSONs under `runs/v2/`. Where the KB and a JSON
  disagree, the JSON wins and you report the disagreement.
- **Unknown = placeholder.** Anything not in the repo, the Proposal or PR-I (who did what,
  meetings, dates, commercialization activity) is written as `[EMIR: …]`, never invented.
- **Wording the project does not allow:** "solved", "works" for `pickup_box` or UnlockPickup;
  sampled and argmax rates in one comparison or one table column; "the hierarchy learns faster
  than flat PPO" (decision 0006 says this is not claimed); PN forward transfer as a contribution
  (W7: no measurable benefit); "no forgetting" as a headline (frozen skills cannot forget by
  construction); `unlock_pickup` as a result (**not run**).
- **W1b-1 state, stated exactly:** step 1 of the plan has run
  (`runs/v2/w1b_pickup_box_freeze.json`: `pickup_box` spec 3, training seed 45, frozen from its
  fixed-budget model under decision 0014 while that decision is `Proposed`; `met_0004: false`).
  The experiment itself, training `unlock_pickup`, has **not** run. `status.md` still says
  "Nothing frozen" in the Track H row: that is a KB/JSON disagreement to report. Re-read the
  file and `git log` immediately before writing 2.4, because the W1b session may have moved on;
  if it has, report what the files show and flag it for Emir.
- **Negative results are reported as results**, each with what it rules out.
- The supervisor does not know the implementation: concepts first, code names in parentheses once.
- Reuse PR-I's sentences only in Section 1 and Section 4 where still true; everything else is new.
- **References:** PR-I's list [1]–[12], plus only works you can verify in the literature file or
  v2. The literature file is **not** in this repo; read it from
  `..\hierarchical-rl-agent\docs\reviews\2026-10-04_literature_update.md` (this is the only thing
  taken from that folder). No reference from memory.

## Section by section

**1. PROJECT SUMMARY** (1–1.5 pages). The template asks for four things; cover each:
- brief description; the gap in the literature / existing solutions; motivation (as in PR-I,
  shortened);
- **objectives and intended results, recalled from the Proposal first** (list them as the
  Proposal states them), then what changed: the architecture was redesigned in October (v1
  columns + MetaController → v2 skill nodes; decision 0001), and the claim is now stated as in
  decision 0006: the agent composes skills while it learns, reuses the compositions, and does
  this without forgetting; PN forward transfer only if shown. Then the objectives in measurable
  form, mapped from the Proposal's (roadmap §1 table: A1–C2);
- one paragraph on **realistic constraints** (the template names them): compute (every run on
  one CPU), environment (MiniGrid, one task family), time (one semester to the Final Report),
  and the evaluation standard the project holds itself to (decision 0004). The template's
  "objectives, criteria, synthesis, analysis, construction, testing and evaluation" should each
  be recognisable in this section, one clause each is enough.

**2. SCIENTIFIC/TECHNICAL DEVELOPMENTS** (8–10 pages). The template requires the capstone phases
(conceptual design, preliminary design, design decision, detailed design) and says the report
must contain "some conceptual designs and a preliminary design". Name the phases in the text.
- 2.1 *Conceptual, preliminary and detailed design.* **Conceptual:** the skill-graph idea (skill
  node with initiation, goal, timeout, policy, children; DAG; one executor with an active
  stack). **Preliminary:** v1 (PR-I's columns + MetaController) and why it was replaced
  (decision 0001). **Detailed (v2):** composites trained as SMDPs, goal-only masking, the
  stabilisation rule (0.90 twice with Wilson bound, fixed seeds, held-out score).
  **Fig. 1** v2 architecture; **Fig. 2** the skill graph as built (9 frozen nodes, DoorKey at
  depth 3) with the planned UnlockPickup/Blocked nodes dashed.
- 2.2 *Design decisions.* **Table:** one line per decision file with its status. The files are
  0001–0011, 0013 and 0014 (13 rows). **0012 does not exist**: it is reserved for the stopping
  rule under sampling and not written; say so in a note under the table, do not write a row.
- 2.3 *Results since PR-I.* One subsection each, with its table or figure:
  - frozen library: 9 nodes, held-out scores, training cost (**Table**);
  - `solve_doorkey`: 1.00 on 3 seeds at depth 3; cost own / with children / flat PPO, both
    framings side by side (**Table**). The KB carries more than one figure for the hierarchy's
    cost through DoorKey (773,955 as built in `h7_amortisation.json`; ~53k own / ~664k with
    children in `status.md`; 561,756–570,536 in `open-questions.md`): give each with what it
    counts, from the JSON;
  - what composition adds beyond the mask: random choice under the strict mask ties the learned
    composite; divergence count and step efficiency are the measures (open-questions, Q8 row);
  - choice task: no choice learned in 100 updates, and why the task itself was flawed;
  - flat PPO on UnlockPickup and BlockedUnlockPickup: 0/50 after 3,000,320 steps × 3 seeds;
  - `pickup_box`: spec 1 → spec 2 → W1a (6/10, fails) → handoff diagnostic (reading "Key") →
    spec 3 (7/10, passes at the threshold). **Fig. 3** per-seed sampled own-hook and handoff rates,
    spec 2 vs spec 3, with the 0.90 line. Argmax numbers in a separate table;
  - lateral ablation (W7, promised in PR-I §5.2): no measurable benefit; state the detectable
    effect size. **Fig. 4** = `runs/v2/w7/w7_lateral_curves.png` or the per-seed table;
  - GRU task identification (PR-I §5.4): 63 / 83 / 85 / 85 / 85 % at N = 1/5/10/20/50, 9 tasks;
    what is not measured (old-task accuracy, switching latency);
  - evaluation infrastructure: training/evaluation parity, run store, decision records, tests
    (run `pytest tests/ -q` once and report the count).
- 2.4 *Experiment design* (the template asks for this explicitly: "Design a test/experiment for
  your proposed solution… how will you test it?"). Use the template's six steps as the
  structure: 1. define problem, 2. determine objectives, 3. design experiment, 4. conduct
  experiment and collect data, 5. analyze data and interpret results, 6. verify predicted
  results. Worked example: W1a → diagnostic → spec 3 (criterion fixed before the run, fresh
  seeds, one pre-declared extension, the rule kept when it failed). Then W1b-1 as the experiment
  **designed and started, not completed**: question, child-selection rule, ceiling gate, pass
  rule, what follows either way (`docs/plans/2026-11-15_W1b-1_unlock_pickup_plan.md`, decision
  0014 `Proposed`), with its state as in "Rules for the text".
- 2.5 *Commercialization / FTU* (a template bullet): one paragraph. PR-I's paragraph is in its
  §2.3 ("Commercialization and Freedom-to-Use considerations"); keep it unless Emir supplies
  activity since then: `[EMIR: any patent/market search since May?]`.

**3. ENCOUNTERED PROBLEMS** (2–3 pages). The template's points, each answered under its own
subsection: changes in plan and goals with causes; whether the original goals are still
relevant and the corrective measures; progress against the timetable with reasons for delay
and planned corrections; effects of the changes.
- 3.1 *Changes in plan and goals.* The v1 stall (June–September) and the redesign: cause,
  corrective measure. The claim narrowed (0006).
- 3.2 *Technical problems.* UnlockPickup blocked since October by one primitive; deterministic
  execution loops under partial observability; the start-state mismatch found by the diagnostic;
  success rate cannot separate composition from masking on this curriculum.
- 3.3 *Progress against the timetable.* **Table: what PR-I promised vs. now**, one row for each
  of PR-I §5.1–5.7 **and each row of PR-I §5.8's timetable with its target week** (11 rows:
  FourRooms stabilisation W14; autoencoder calibration W14; BWT/FWT/retention logging W15;
  lateral/no-lateral baseline W15; MetaController upgrade W16; GRU baseline and AE–GRU handoff
  W16; dynamic options W17; dashboard on real outputs W17; Phase 2 end-to-end integration W19;
  no-AE / no-GRU ablations W21; Phase 3 scope review W22). Known states: 5.1 FourRooms: taken
  off the critical path (decision 0003); 5.2 lateral ablation: done, negative; 5.3 autoencoder:
  not started; 5.4 GRU: done in part; 5.5 MetaController/options: replaced by composite nodes,
  done through DoorKey; 5.6 GUI on real runs: not started; 5.7 evaluation package: `docs/kb/`.
  For rows whose state is not in the KB, write `[EMIR: …]`. Behind schedule: UnlockPickup,
  Blocked, automatic growth, autoencoder, GUI, end-to-end integration. Say so plainly, with the
  reason for each.
- 3.4 *Are the original goals still relevant, and corrective measures.* Yes, with the claim
  narrowed (0006). Measures: the redesign, fixing criteria before runs (0010), the evaluation
  parity work, the priority order of Section 5.
- 3.5 *Effects of the changes.* What the redesign and the narrowed claim change for the Final
  Report: which Proposal deliverables are kept, reduced or dropped (planning extension, PN as
  the transfer mechanism, GUI scope).

**4. SUSTAINABILITY AND ECONOMIC ASSESSMENT** (1.5–2 pages). Keep PR-I's six subsections, which
match the template's list: 4.1 functional unit, 4.2 preliminary system boundary, 4.3 main
environmental impact contributors, 4.4 initial cost components, 4.5 qualitative comparison with
conventional solutions, 4.6 preliminary assessment. Update:
- functional unit = acquiring one new task given the existing library;
- PR-I describes the impact as "GPU-accelerated" training: every run since was on **one CPU**
  (`device: cpu` in the JSONs); correct the wording;
- compute is measured in env steps (cost tables in the W1a/spec-3 summaries and
  `h7_amortisation.json`); CPU-hours come from the `wall_seconds` fields of the run JSONs;
  convert to energy with one stated, labelled assumption (CPU power draw `[EMIR: confirm W]`);
- 4.5 compares with flat PPO as the conventional solution. State both sides: reuse through
  DoorKey costs about the same as flat (773,955 vs 650,000–900,000), and the research cost of
  `pickup_box` (≈ 28.5M env steps: spec 1, W1a on spec 2, spec 3 with its chain steps) exceeds
  flat's 9.0M on a task flat never solved. No efficiency gain is claimed yet.

**5. TASKS TO BE COMPLETED UNTIL FINAL REPORT** (1–1.5 pages) and timetable (the template
requires a timetable). From `docs/plans/2026-11-15_roadmap.md` §2–§3, in this priority: W1b-1 →
W1b-2 → UnlockPickup/Blocked reuse (W2) → fair baselines (W4) → growth G-1/G-2 → autoencoder R-1
and task registry R-3 → system-level continual evaluation (W8) → GUI on real runs. Mark each
"must" or "if time", and name the capstone phase each belongs to (detailed design, construction,
testing, evaluation). Weeks: Draft Final Report is week 12, Final Report week 14
`[EMIR: confirm against the course deadlines]`; the week-by-week table is a proposal marked
`[EMIR: confirm]`.

**6. REFERENCES.**

## Report back with

The commit you read, the docx path, `claims.md`, the list of every `[EMIR: …]` placeholder, every
KB/JSON disagreement found, and anything in this brief that the repo contradicts.
