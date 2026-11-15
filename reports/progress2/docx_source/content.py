# Body of ENS 492 Progress Report II. Every number has a row in claims.md.
# Blocks: ("h1"|"h2"|"h3"|"p"|"b"|"ref", text) · ("fig", key, width_in, caption)
#         ("tab", caption, header, rows, widths, note) · ("pb",)

FIGURES = {
    "arch": "fig1_v2_architecture.png",
    "graph": "fig2_skill_graph.png",
    "pickup": "fig3_pickup_box_spec2_vs_spec3.png",
}

B = []
h1 = lambda t: B.append(("h1", t))
h2 = lambda t: B.append(("h2", t))
h3 = lambda t: B.append(("h3", t))
p = lambda t: B.append(("p", t))
b = lambda t: B.append(("b", t))


def tab(cap, header, rows, widths, note=None):
    B.append(("tab", cap, header, rows, widths, note))


def fig(key, width, cap):
    B.append(("fig", key, width, cap))


# =====================================================================================
h1("PROJECT SUMMARY")
h2("1.1 Description, Literature Gap and Motivation")
p("This project develops a reinforcement learning (RL) agent that acquires tasks one after another, "
  "keeps what it has learned, reuses learned behaviours as building blocks for harder tasks, and "
  "exposes its internal decisions for inspection. The work is carried out in simulated MiniGrid "
  "environments [1] and produces a research prototype, not a deployed system.")
p("The motivation is unchanged since the Proposal. Standard RL agents are optimised for single, "
  "isolated tasks; when the task changes they tend to lose earlier knowledge or relearn it from "
  "scratch [2, 3]. Continual RL limits forgetting through regularisation, replay or architectural "
  "growth [3, 5, 6, 7], but mostly treats earlier knowledge as something to protect rather than as "
  "a repertoire of skills to compose. Hierarchical RL composes temporally extended skills (options) "
  "[4, 11], but usually within a fixed task and with a fixed skill set. The gap the project "
  "addresses is the combination: an agent whose library of skills grows as tasks arrive, in which "
  "compositions of skills are themselves reusable skills, and which can tell which task it is in "
  "without being given a label [8, 9]. Related recent work freezes earlier policies and combines "
  "them (CompoNet [14], HOP [15]) or invents options for continual planning (CHiRP [13]); a library "
  "in which compositions are themselves children of later skills is, to our search, not covered by "
  "them. We state this as \"we did not find\", not \"nobody has\".")

h2("1.2 Objectives and Intended Results in the Proposal")
p("The Proposal set six objectives, each with an intended result:")
b("**Continual task acquisition without catastrophic interference.** A Progressive Networks backbone "
  "with one frozen column per task; intended result: sequential acquisition with earlier performance "
  "kept.")
b("**Autonomous task detection and identification.** An autoencoder detects novel tasks from "
  "reconstruction error and a GRU identifies known tasks from observation sequences; each task is "
  "represented by its initial conditions, terminal conditions and policy; intended result: no "
  "external task labels.")
b("**Hierarchical policy reuse via dynamic options.** Each frozen column becomes a reusable "
  "macro-action selected by a meta-controller; intended result: complex tasks handled by reusing "
  "earlier behaviours instead of relearning them.")
b("**Planning-compatible high-level control (stretch goal).** A controller that can later support "
  "option-level planning; intended result: an architecture compatible with lightweight planning.")
b("**GUI for transparent introspection.** A Streamlit dashboard showing reconstruction error, task "
  "confidence, the active policy and option history; intended result: an interface to inspect and "
  "debug the agent.")
b("**Evaluation in an adapted MiniGrid task sequence.** Tasks that support continual acquisition and "
  "hierarchical reuse; intended result: a quantitative evaluation of task success, retention and "
  "task-identification accuracy.")

h2("1.3 What Changed Since Progress Report I")
p("Two things changed, both recorded as design decisions (Section 2.2). First, the architecture was "
  "redesigned in October (decision 0001). The design reported in Progress Report I (PR-I) put "
  "Progressive Network columns and a separate MetaController on the critical path; it stalled "
  "because the second column never stabilised on FourRooms. In the new design (\"v2\") everything the "
  "agent knows is a **skill node**, and a composite node's policy is its own meta-controller "
  "(Section 2.1). Second, the claim was narrowed (decision 0006). The agent **composes skills while "
  "it learns, reuses the compositions as building blocks, and does this without forgetting**. "
  "Forward transfer through Progressive Network laterals is claimed only if an experiment shows it; "
  "the one experiment run so far did not (Section 2.3.8). The claim that the hierarchy learns a "
  "single task faster than flat PPO is not made. Because frozen skills cannot forget by "
  "construction, \"without forgetting\" is measured on the whole system, including task "
  "recognition, together with its price in added nodes.")
p("**State on 15 November.** Eleven skills are frozen, among them composites for three of the four "
  "curriculum tasks: Unlock, DoorKey and UnlockPickup. DoorKey and UnlockPickup both reuse the Unlock "
  "composite untouched; UnlockPickup, on which flat PPO scored 0 of 50, has a frozen composite since "
  "15 November (Section 2.3.7). The fourth task, BlockedUnlockPickup, has not been run.")
p("Table 1 maps the Proposal's objectives to the claim in measurable form. Each row names what "
  "would refute it and what exists today.")
tab("Table 1. The claim in measurable form (decision 0006).",
    ["Part", "Measurable form", "Refuted if", "Today", "Proposal objective"],
    [
        ["A1 Compose (structure)", "A composite for a new task reaches STABLE without a human naming its children",
         "Not STABLE in budget, or meaningless children", "No evidence (automatic growth not started)", "3, 6"],
        ["A2 Compose (content)", "The composite learns something the mask does not give it",
         "Same score and steps as random choice under the same mask",
         "Order discovery and step efficiency only, on three tasks (Sections 2.3.3, 2.3.7)", "3"],
        ["B1 Reuse (transfer)", "A frozen composite scores ≥ 0.90 untouched in a task it was not trained in",
         "Each new task needs a near-copy", "`unlock_door` reused untouched in DoorKey and UnlockPickup; alone in UnlockPickup 0.98", "3"],
        ["B2 Reuse (fan-out, depth)", "A composite has ≥ 2 parents; the deeper Blocked variant is cheaper than the flat-composite one",
         "Deeper variant not cheaper", "`unlock_door` has two parents; depth comparison not run", "3"],
        ["B3 Reuse (cost)", "Cumulative cost below flat; learns a task flat cannot",
         "A flat learner with memory does the same", "Through DoorKey about equal; UnlockPickup: recipe about 2.38M steps per run against flat > 3.0M (censored); research cost 28.5M (Section 2.3.7)", "1, 3"],
        ["C1 No forgetting (system)", "After each new task all earlier tasks score the same end to end, recognition in the loop",
         "Recognition degrades on old tasks", "Not measured", "1, 2"],
        ["C2 Price of not forgetting", "Nodes added per task; share of near-copies",
         "Library grows faster than tasks", "One near-copy (`open_door_tk`)", "1"],
    ], [2.2, 3.6, 2.6, 2.8, 1.3],
    "Source: project roadmap of 15 November 2026, §1. Objectives 4 (planning) and 5 (GUI) are not part of the claim; their status is in Section 3.3.")

h2("1.4 Realistic Constraints")
p("**Compute:** every run reported here was trained on one CPU (the `device` field of every run "
  "file is `cpu`); no GPU was used. **Environment:** MiniGrid, one task family (Unlock, DoorKey, "
  "UnlockPickup, BlockedUnlockPickup), decomposed into sub-goals by the authors. **Time:** one "
  "semester remains until the Final Report, shared by five members. **Evaluation standard:** a "
  "skill counts as learned only under a fixed, pre-declared rule (decision 0004): success rate at "
  "least 0.90 with a Wilson 95 % lower bound of at least 0.80 in two consecutive 50-episode "
  "evaluations on fixed seeds, with the reported score taken on a separate held-out seed set. "
  "Criteria for new experiments are fixed before the run (decision 0010).")
p("In capstone terms, the **objectives** are the claim parts A1–C2 above; the **criteria** are "
  "decision 0004 and the pass rules each experiment fixes in advance; **synthesis** is the v2 "
  "skill-node design; **analysis** is the design-decision record and the diagnostic experiments; "
  "**construction** is the implemented library of eleven frozen skills; **testing** is the automated "
  "test suite and the training/evaluation parity checks; and **evaluation** is the set of "
  "experiments in Section 2.3, each recorded with its plan and result files.")

# =====================================================================================
h1("SCIENTIFIC/TECHNICAL DEVELOPMENTS")
p("This section follows the capstone design phases. Section 2.1 gives the conceptual design, the "
  "preliminary design (the v1 architecture of PR-I) and why it was replaced, and the detailed design "
  "now implemented. Section 2.2 is the design-decision record. Section 2.3 reports the results "
  "obtained since PR-I, negative results included. Section 2.4 describes how experiments are "
  "designed, with two completed examples. All results are taken from the main line of the project "
  "repository at commit `d7be0a2` (15 November 2026).")

h2("2.1 Conceptual, Preliminary and Detailed Design")
p("**Conceptual design.** The agent's knowledge is a graph of skills. A skill node has an "
  "*initiation condition* (when it may start), a *goal condition* (when it has succeeded), a "
  "*timeout* (after which it has failed), a *policy*, and a list of *children*. A primitive skill "
  "has no children and its policy chooses a MiniGrid action; a composite skill's policy chooses one "
  "of its children, which then keeps control until its own goal, its timeout, or the termination "
  "of an ancestor. A composite may use any earlier frozen skill, at any level, as a child, so the "
  "graph is a directed acyclic graph rather than a tree (decision 0002): one skill can have several "
  "parents. Its level is derived from its children (a primitive is level 0; a composite is one more "
  "than its highest child). At run time one executor keeps a stack of active skills, from the root "
  "task down to the primitive that acts.")
p("**Preliminary design.** PR-I implemented the Proposal's architecture: Progressive Network "
  "columns, one per task, with lateral connections from frozen columns, and a separate "
  "MetaController that selected columns as options. It was replaced in October (decision 0001) for "
  "three reasons: Progressive Networks sat ahead of the hierarchy on the critical path and the "
  "second column never stabilised on FourRooms (PR-I §3.4), stalling the project from June to "
  "September; the multi-level hierarchy, which the supervisor had asked to prioritise, had been "
  "demoted to a stretch goal; and key questions (how levels are created, how termination "
  "propagates through levels) were open while code was written around them.")
p("**Detailed design (v2).** Figure 1 summarises the implemented design. Primitives are trained "
  "with PPO [12] on an environment wrapper that starts episodes in the skill's own start states and "
  "ends them at its goal. Composites are trained with a masked variant of PPO on a wrapper in which "
  "one step is one complete execution of a child: a semi-Markov decision process in the sense of "
  "the options framework [11]. Every skill, at every level, is rewarded by its own goal condition "
  "(1 − 0.9 · steps / timeout on success, otherwise 0), never by the environment's reward. A child "
  "is selectable unless its goal already holds (*goal-only masking*); the initiation condition is "
  "task metadata and does not mask. Every composite has a cap on its own selections, so a parent "
  "cannot re-select a failing child forever (decision 0009). A node becomes STABLE under decision "
  "0004 (one accepted exception, decision 0014, Section 2.3.7) and is then frozen: no gradients, but its forward pass still runs when parents call it. "
  "Policies see only the 147-dimensional egocentric observation; goal and initiation conditions "
  "read the true environment state, which is how the training signal for each sub-goal is defined.")
fig("arch", 6.3,
    "Figure 1. v2 architecture. (a) A skill node and its life cycle. (b) Execution: one stack from "
    "the root task to the primitive that acts; termination is checked top-down every environment "
    "step. (c) Training one new node while all others stay frozen, and the stabilisation rule. "
    "(d) Recognition: the GRU task identifier exists and is evaluated offline; the autoencoder and "
    "the task registry that would let the GRU choose the root are not built yet.")
p("Figure 2 shows the skill graph as it stands. Eleven nodes are frozen. The DoorKey and "
  "UnlockPickup tasks are each executed by a three-frame stack (root → unlock-door composite → "
  "primitive), and the unlock-door composite, trained on the Unlock task, is reused untouched inside "
  "both: it is the first skill with two parents. The BlockedUnlockPickup roots and their missing "
  "primitive are planned (dashed).")
fig("graph", 6.3,
    "Figure 2. The skill graph as built. Solid boxes: the eleven FROZEN, registered nodes, with "
    "held-out success over 50 episodes; for `{0,4} pickup_box` (marked *), frozen under decision 0014, its "
    "deterministic success from the UnlockPickup handoff on 488 fresh episodes. Dashed: the "
    "BlockedUnlockPickup nodes, planned (H-7b, not run). Dotted: the choice-task composite, trained "
    "but not STABLE. `{n,m}` = level n, index m.")

h2("2.2 Design Decisions")
p("Every design choice since PR-I is recorded as a decision file with its reason, the rejected "
  "alternatives and the experiments that inform it. Decisions are settled by experiment where "
  "possible (decision 0010), and a decision that changes a locked rule of the design waits as "
  "*Proposed* until a team member accepts it. Table 2 lists all thirteen.")
tab("Table 2. Design decisions and their status (15 November 2026).",
    ["No.", "Decision", "Status"],
    [
        ["0001", "Replace the v1 column/MetaController design with v2 skill nodes", "Accepted"],
        ["0002", "The skill hierarchy is a DAG, not a tree", "Accepted by the team; supervisor confirmation not recorded"],
        ["0003", "Progressive Networks off the critical path, not dropped", "Accepted"],
        ["0004", "STABLE at success ≥ 0.90 with Wilson lower bound ≥ 0.80, twice", "Accepted (amended 2 October)"],
        ["0005", "Tasks declared by hand in a fixed bottom-up order (growth Phase A)", "Accepted as Phase A only"],
        ["0006", "The claim: compose while learning, reuse compositions, without forgetting", "Accepted as the direction; evidence partial"],
        ["0007", "Automatic hierarchy growth stays in scope", "Accepted"],
        ["0008", "A code knowledge graph (graphify) as the team's code map", "Accepted"],
        ["0009", "The cap on selections is counted per composite, not per tree", "Accepted 12 November, conditional on the final code review"],
        ["0010", "Design decisions are settled by experiment and linked to them", "Accepted"],
        ["0011", "`pickup_box` specification 3: half of training starts at the real handoff", "Accepted (15 November)"],
        ["0013", "The lateral-transfer experiment uses a frozen v2 primitive as source, with an untrained-source control", "Accepted"],
        ["0014", "A primitive that never met 0004 in training may be frozen on a fixed-budget test", "Accepted (15 November)"],
    ], [0.8, 6.0, 3.6],
    "Number 0012 is reserved for the stopping rule under sampled execution (Section 3.2) and has not been written yet.")

h2("2.3 Results Since Progress Report I")
p("All success rates below are measured with the policy acting deterministically (argmax) on 50 "
  "held-out episodes, unless the text says *sampled*. Sampled and argmax rates are different "
  "measurements and are never compared in one table column.")

h3("2.3.1 The Frozen Skill Library")
p("Eleven skills are frozen (Table 3). Eight are primitives: three for the Unlock and DoorKey "
  "tasks, three built for the choice task (Section 2.3.4), and `drop_held` and `pickup_box` for "
  "UnlockPickup. Three are composites: `unlock_door` (level 1), `solve_doorkey` and `unlock_pickup` "
  "(level 2). Ten were frozen when they met decision 0004; `pickup_box` never met it and was frozen "
  "under the accepted exception of decision 0014 (Section 2.3.7). The held-out scores are quoted as "
  "measured, including one below 0.90 (`get_key_blue`, 43/50), which is possible because the "
  "stopping rule is applied on a different seed set from the held-out score. The first eight skills "
  "(the seven original primitives and `unlock_door`) cost about 1.2 million environment steps "
  "together.")
tab("Table 3. The frozen library (held-out 50 episodes, argmax; training seed 42 unless noted).",
    ["Node", "Level", "Trained on", "Held-out", "Own cost (env steps)"],
    [
        ["`{0,1} get_key`", "0", "Unlock, DoorKey", "50/50", "180,000"],
        ["`{0,2} open_door`", "0", "Unlock, DoorKey", "50/50", "100,000"],
        ["`{0,3} go_to_goal`", "0", "DoorKey", "46/50", "200,000"],
        ["`{0,6} get_key_yellow`", "0", "DoorKey (two keys)", "47/50", "220,000"],
        ["`{0,7} get_key_blue`", "0", "DoorKey (two keys)", "43/50", "240,000"],
        ["`{0,8} open_door_tk`", "0", "DoorKey (two keys)", "47/50", "100,000"],
        ["`{0,9} drop_held`", "0", "UnlockPickup", "50/50", "30,000"],
        ["`{1,1} unlock_door`", "1", "Unlock", "49/50", "131,817 (+ 280,000 children)"],
        ["`{2,1} solve_doorkey`", "2", "DoorKey", "50/50", "162,138 (+ 611,817 children)"],
        ["`{0,4} pickup_box` (spec 3, seed 45)", "0", "UnlockPickup", "485/488 from the handoff*", "1,001,472 (+ 372,063 handoff-chain steps)"],
        ["`{2,2} unlock_pickup` (seed 43)", "2", "UnlockPickup", "49/50", "60,205 (+ 1,443,289 children)"],
    ], [2.6, 0.8, 2.3, 1.2, 3.2],
    "`drop_held` scored 1.00 / 1.00 / 0.98 on three training seeds. Composite costs are those of the frozen checkpoints; Section 2.3.2 gives the re-measured values for `solve_doorkey`. *`pickup_box` was frozen under decision 0014 and has no held-out row of the usual kind: the figure is its deterministic success from the UnlockPickup handoff on fresh env seeds 53000–53499; from its own training start states it scores 305/500 deterministically.")
p("Two transfer results matter for reuse. `unlock_door`, trained on Unlock, runs untouched inside "
  "UnlockPickup at 49/50 (0.98). In the other direction, `open_door` scored only 0.83 in the "
  "two-key variant of DoorKey and had to be replaced by `open_door_tk` (0.98 there): a primitive "
  "transfers only as far as its training start states cover the parent's handoff states. This is "
  "the first near-copy in the library (claim part C2).")

h3("2.3.2 DoorKey at Depth Three")
p("`solve_doorkey` selects between the frozen `unlock_door` composite and the `go_to_goal` "
  "primitive, so executing it drives a three-frame stack. Re-trained on three seeds under the "
  "corrected settings of October (smaller policy updates, `n_steps` 64), it reached STABLE on all "
  "three after 1,536 decisions and scored 50/50 held-out on each. Its cost depends on what is "
  "counted, and the project's records carry three figures that count different things (Table 4). "
  "They are given side by side; none of them supports the statement that the hierarchy learns "
  "DoorKey faster than flat PPO, and that statement is not made.")
tab("Table 4. Cost of DoorKey-6x6 in environment steps: what each figure counts.",
    ["Figure", "What it counts", "Env steps"],
    [
        ["Own cost, re-measured", "`solve_doorkey`'s own steps, 3 seeds, `n_steps` 64 (children frozen and given)", "48,887 – 55,659"],
        ["Own + children, re-measured", "the same plus the children as built (611,817)", "660,704 – 667,476"],
        ["As built", "the frozen checkpoint (162,138 own, `n_steps` 256) plus its children", "773,955"],
        ["Cumulative through DoorKey", "Unlock and DoorKey nodes, composites re-measured at `n_steps` 64", "561,756 – 570,536"],
        ["Flat PPO, DoorKey only", "steps to STABLE, 3 seeds", "300,000 – 450,000"],
        ["Flat PPO, Unlock + DoorKey", "flat Unlock (350,000–450,000) plus flat DoorKey", "650,000 – 900,000"],
        ["Flat option set", "same three skills at depth 1, own / with children", "64,000 – 80,951 / 544,000 – 560,951"],
    ], [2.7, 5.0, 2.6],
    "Sources: run files `solve_doorkey_h68_s*`, `2-1_solve_doorkey/meta.json`, `h7_amortisation.json`, `flat_ppo_s*`, `flat_unlock_ppo_s*`, `flat_options_h68_s*`. Caveats recorded with the figures: the key and door primitives were trained on Unlock and DoorKey together; primitive costs are single-seed; evaluation episodes are counted on neither side; and the hierarchy was given hand-written sub-goals and start states that flat PPO must reach by itself.")
p("The reading is therefore: given frozen children, a new composite is cheap (tens of thousands of "
  "steps); counting the children it is not cheaper than flat PPO on this task, as built or "
  "re-measured. Reuse pays only if the children are used again by later tasks: UnlockPickup now "
  "does so (Section 2.3.7), and BlockedUnlockPickup is meant to test it further.")

h3("2.3.3 What Composition Adds Beyond the Mask")
p("A composite's success rate alone cannot show that it learned anything. On this curriculum every "
  "composite's sub-goals have a strict order, and an initiation mask that allows only the next "
  "sub-goal encodes exactly that order. Under that strict mask a *random* selector ties the learned "
  "composite, at depth 1 and at depth 2 (Table 5). The project reports this as a property of the "
  "curriculum and measures the learned selector's contribution in two other ways: the **divergence "
  "count** — over 20 held-out DoorKey episodes, in the 20 decisions where the strict mask would have "
  "withheld a child, the goal-only-trained selector chose the withheld child 0 times, so it found "
  "the order rather than being given it — and **step efficiency** against random choice under the "
  "project's own goal-only mask: 4.4 times fewer environment steps for `unlock_door` and 8 to 10 "
  "times fewer for `solve_doorkey`. Before any training update, the randomly initialised DoorKey "
  "selector already succeeded in 0.40, 0.10 and 0.78 of episodes on the three seeds, which "
  "measures how much the frozen children do on their own. Trained under the strict mask, it is "
  "STABLE at the first point the rule allows (1,024 decisions): the mask does not speed learning up, "
  "it removes the need for it.")
tab("Table 5. Learned selector against random selection (held-out, argmax).",
    ["Selector", "Mask", "`unlock_door` (depth 1): success / mean env steps", "`solve_doorkey` (depth 2, 3 seeds): success / mean env steps"],
    [
        ["Learned", "goal-only", "0.98 / 19.2", "1.00 / 14.2"],
        ["Random over available children", "goal-only", "0.90 / 85.3", "0.82 – 0.92 / 113 – 143"],
        ["Random over all children", "none (clipped)", "0.94 / 83.0", "0.86 – 0.92 / 113 – 135"],
        ["Random", "strict (initiation)", "0.98 / 19.2", "1.00 / 14.2"],
        ["Learned", "strict (initiation)", "0.98 / 19.2", "1.00 / 14.2"],
    ], [2.6, 1.8, 3.0, 3.2])

h3("2.3.4 The Choice Task: No Choice Learned")
p("To escape the masking problem, a task was built in which two children are valid at the same "
  "decision and the right one depends on what the agent sees: a DoorKey variant with a yellow and a "
  "blue key, a door of one of the two colours each episode, and colour-specific key primitives. "
  "Three seeds under each of two masks were trained for 100 policy updates. **No run learned the "
  "choice**: none became STABLE (held-out 0.78–0.80), the best first-key accuracy over the seven "
  "strict-mask runs was 0.72, and three of the seven showed only a weak dependence of the first key "
  "on the door colour (Fisher p 0.004, 0.021, 0.021, uncorrected). The selector's network can read "
  "the colour — trained with supervision it does so at 0.940 from 1,000 examples and 0.997 from "
  "4,000 — so the failure is not one of representation. The task itself was flawed: a wrong pick is "
  "recoverable, because a colour-specific key primitive handed the wrong key drops it and fetches "
  "the right one in 43 of 50 episodes. A wrong choice therefore costs only time, and the training "
  "signal for choosing correctly is weak. This rules out the task as a test of observation-"
  "conditioned choice in its current form; a redesign in which a wrong pick is unrecoverable is on "
  "the task list (Section 5).")

h3("2.3.5 Flat PPO on UnlockPickup and BlockedUnlockPickup")
p("The flat baseline (one PPO policy on the raw task, same observation and network size) reaches "
  "STABLE on Unlock and on DoorKey. On the two harder tasks it scored **0/50 held-out after "
  "3,000,320 environment steps on each of three seeds**, with a best stopping-evaluation success of "
  "0.00 throughout (Table 6). These runs are censored at the budget, so the flat cost of "
  "UnlockPickup is only known to exceed 3 million steps per run (9,000,960 for the three). One "
  "limitation applies: no flat UnlockPickup model was saved, so this baseline cannot be re-scored on "
  "fresh seeds; it rests on the 50 held-out seeds only. A flat learner with memory has not been "
  "tried (Section 5).")
tab("Table 6. Flat PPO baselines (three seeds each).",
    ["Task", "Budget", "Steps trained", "STABLE", "Held-out"],
    [
        ["Unlock", "1,500,000", "350,000 – 450,000", "3 of 3", "46 – 48 / 50"],
        ["DoorKey-6x6", "1,500,000", "300,000 – 450,000", "3 of 3", "47 – 50 / 50"],
        ["UnlockPickup", "3,000,000", "3,000,320", "0 of 3", "0 / 50 each"],
        ["BlockedUnlockPickup", "3,000,000", "3,000,320", "0 of 3", "0 / 50 each"],
    ], [2.6, 1.6, 2.4, 1.4, 1.8])

h3("2.3.6 The Box Primitive (`pickup_box`)")
p("UnlockPickup requires one new primitive, picking up the box once the door is open and the key "
  "has been dropped. It is the reason UnlockPickup had no frozen composite until 15 November, and it is reported "
  "here in full because each step changed what the project does next.")
b("**Specification 1** (October): not STABLE in 600,064 steps on any of three seeds (held-out 0.28, "
  "0.40, 0.24). Its start states did not match what the parent would hand it (the key could still "
  "be in hand), so the task was split into `drop_held` (frozen) and a new `pickup_box`.")
b("**Specification 2** (October): not STABLE in 1,001,472 steps on any of three seeds (held-out "
  "0.70, 0.66, 0.64). Every deterministic failure inspected was a state repeat: the policy loops "
  "between the same states, because it sees only a partial, egocentric view and acts the same way "
  "each time it sees the same view. Sampled actions break such loops.")
b("**W1a** (15 November): does specification 2 pass when scored as it would act, i.e. sampled? Ten "
  "fresh training seeds, a fixed budget, two rows of 500 fresh episodes each: from the skill's own "
  "start states (own hook) and from the real handoff after the frozen `unlock_door` and `drop_held`. "
  "**It fails: 6 of 10 seeds pass, the rule needs 7.** The own-hook row passes on all ten (sampled "
  "0.918–0.994); the four failing seeds fail at the handoff (sampled 0.874–0.892).")
b("**Handoff diagnostic** (evaluation only): on the failing seeds, removing the dropped key from the "
  "grid raises sampled handoff success from 0.880 to 0.985, while doubling the timeout raises it "
  "only to 0.921. By the reading rule fixed in advance this is reading **Key**: the policy is "
  "distracted by the key it has just dropped, a state it never saw in training. These are "
  "counterfactuals, not results for the skill.")
b("**Specification 3** (decision 0011): half of the training episodes start at the real handoff; "
  "W1a's test is unchanged and run on fresh seed ranges. **It passes at the threshold: 7 of 10.** "
  "The handoff row passes on all ten (sampled 0.984–1.000); the three failing seeds fail on the "
  "own-hook row (0.898, 0.778, 0.896). The model scores higher where the parent needs it and lower "
  "from its old start states.")
fig("pickup", 6.3,
    "Figure 3. `pickup_box`, specification 2 against specification 3, per training seed. Sampled success rate on the own-hook row (circles) "
    "and the handoff row (squares), 500 fresh episodes each, specification 2 (W1a, env seeds "
    "50000–51499) against specification 3 (env seeds 52000–53499). Dashed line: the 0.90 row "
    "threshold. Shaded: seeds that fail (a seed passes only if both rows pass). Sampled rates "
    "only; argmax rates are in Table 7.")
tab("Table 7. The same models scored deterministically (argmax), separately from Figure 3.",
    ["Row", "Specification 2 (W1a), 10 seeds", "Specification 3, 10 seeds"],
    [
        ["Own hook", "0.684 – 0.766", "0.586 – 0.718"],
        ["Handoff", "0.379 – 0.713", "0.902 – 1.000"],
        ["Argmax failures that are state repeats", "all", "all"],
    ], [3.4, 3.4, 3.4],
    "No `pickup_box` model in any specification met decision 0004's deterministic rule during training.")
p("What these results rule out: that specification 2 was only a scoring artefact of deterministic "
  "execution (W1a fails even sampled); that its handoff failures were a matter of time (the timeout "
  "counterfactual does not reach the reading threshold). What they do not settle: whether the "
  "skill, frozen and run deterministically, is good enough inside its parent. That is the question "
  "W1b-1 answered (Section 2.3.7). Specification 3 is not reported as a finished skill: it passed a "
  "two-row test at its threshold, under sampling, its deterministic own-hook rate is 0.586–0.718, and "
  "the one model now frozen is accepted for one parent only.")

h3("2.3.7 UnlockPickup by Reuse (W1b-1)")
p("Experiment W1b-1 asked whether the UnlockPickup composite, with children `unlock_door`, "
  "`drop_held` and `pickup_box`, reaches STABLE under decision 0004 unchanged when every node acts "
  "deterministically. One specification-3 model (training seed 45) was frozen as the child under "
  "decision 0014, an accepted exception to the stabilisation rule: the model never met decision "
  "0004 in training, and was frozen because it had passed W1a's test, fixed before training, and "
  "because the Wilson 95 % lower bound of its deterministic success from this parent's real handoff "
  "was 0.982 (485/488; the rule needs 0.95). The freezing rule was written with the ten seeds' "
  "handoff rates already known, so the rule and the seed it selected are not blind; every result of "
  "the composite was unseen until it was trained. The composite was then trained on three seeds. "
  "**It scored 49/50 held-out and 486/500 on 500 fresh environment seeds (0.972, Wilson 95 % "
  "0.954–0.983), argmax.** Flat PPO scored 0/50 on the same 50 held-out seeds after 3,000,320 "
  "steps per run, censored. This is one result, not three: the three trained selectors act "
  "identically, with the same held-out and fresh episodes (486/500 and 31.1 mean steps on each), "
  "because they share the stopping seeds and the frozen children. The registered composite is the "
  "seed-43 model.")
p("What the selector adds beyond the mask is steps, not success. Under the strict mask a random "
  "selector ties it (49/50 held-out), as on Unlock and DoorKey (Section 2.3.3). Against random "
  "choice among available children on the same fresh seeds, which succeeds in 359–375 of 500, the "
  "learned selector needs 31 environment steps per episode against 139–148. What W1b-1 shows is "
  "reuse: `unlock_door`, trained on Unlock, runs untouched as a child on a task flat PPO did not "
  "learn. The remaining 2.8 % fresh loss lies in the chain before the handoff: in the ceiling gate "
  "run before training, every episode that reached the handoff succeeded.")
tab("Table 8. UnlockPickup composite (W1b-1) against flat PPO, argmax.",
    ["Measure", "`{2,2} unlock_pickup`", "Flat PPO"],
    [
        ["Held-out, env seeds 20000–20049", "49/50", "0/50 (each of 3 seeds)"],
        ["Fresh, env seeds 54000–54499", "486/500 = 0.972 (Wilson 95 % 0.954–0.983)", "not available (no model saved)"],
        ["Mean env steps per fresh episode", "31.1 (random over available: 139–148)", "—"],
        ["STABLE under 0004", "3 of 3 seeds, after 1,536 decisions (one result in effect)", "0 of 3 seeds"],
        ["Training cost per run, env steps", "2.37–2.38M recipe; about 2.79M with the reused library", "> 3,000,320 (censored)"],
    ], [3.0, 4.2, 3.0],
    "Recipe: `drop_held` 30,000 + `pickup_box` (1,001,472 + 372,063 handoff-chain steps) × 10/6, because 6 of the 10 specification-3 runs met the freezing rule, + the composite's own 55,412–60,205 = 2,374,637–2,379,430. The reused `unlock_door` library adds 411,817. The research cost of finding the `pickup_box` specification, about 28.5M, is given beside it in Table 14.")
p("The recipe is the cost of building this task again knowing the specification; it is below "
  "flat PPO's censored per-run figure. It is not what the specification cost to find: that research "
  "cost, about 28.5 million steps on `pickup_box` alone, stands beside it (Section 4.5).")

h3("2.3.8 Lateral Connections: No Measurable Forward Transfer")
p("PR-I §5.2 promised a lateral/no-lateral comparison. It was run on a pair the v2 library allows "
  "(decision 0013): training `go_to_goal` with no lateral (A), with a lateral connection from the "
  "frozen `open_door` (B), and with a lateral from an untrained network of the same shape (C), ten "
  "seeds each, a fixed budget of 401,408 steps and an evaluation every 4,096 steps. **There is no "
  "measurable benefit** (Table 9): no arm beats another on steps to STABLE (B against A: p 0.469, "
  "median ratio 0.977; the pre-set rule needed p < 0.05 and a ratio of at most 0.80). The "
  "experiment was sized so that a true 25 % reduction would have passed 0.80 of the time "
  "(a 20 % reduction 0.47 of the time), so a small effect cannot be excluded, but an effect of "
  "25 % or more is unlikely. Forward transfer through laterals is therefore not claimed. The "
  "result is for one pair only. The learning curves are recorded with the run files; the table "
  "gives the per-arm summary.")
tab("Table 9. Lateral ablation (W7): `go_to_goal`, ten seeds per arm.",
    ["Arm", "STABLE", "Median steps to STABLE", "Range", "Final score, median (argmax, 500 episodes)"],
    [
        ["A: no lateral", "10 / 10", "190,464", "143,360 – 221,184", "0.901"],
        ["B: lateral from frozen `open_door`", "9 / 10", "178,176", "155,648 – 401,408 (one censored)", "0.875"],
        ["C: lateral from untrained network", "10 / 10", "184,320", "147,456 – 196,608", "0.919"],
    ], [3.0, 1.2, 2.0, 2.8, 2.6],
    "Fifteen of the 30 final scores are below 0.90 on the separate final seed set. Total training cost 5,656,576 env steps.")

h3("2.3.9 GRU Task Identification")
p("PR-I §5.4 promised a supervised evaluation of the GRU task identifier [10]. It was trained on "
  "observation windows from nine known tasks (seven primitive skills and two root tasks), with "
  "episode-disjoint splits and 825 test episodes, for window lengths N = 1, 5, 10, 20 and 50 on "
  "three seeds (Table 10). Accuracy rises from 63.3 % for a single observation to 83.2 % at N = 5 "
  "and stays at 84.9–85.3 % for N ≥ 10, against 11 % chance. The hardest task is the Unlock root "
  "(70 % at N = 20), which is often read as `get_key`. Not measured yet: accuracy on old tasks after "
  "a new one is added, and the switching latency between tasks. The GRU's prediction does not yet "
  "select the root skill; that needs the task registry (Section 5).")
tab("Table 10. GRU task identification: task-episode macro accuracy, mean of three seeds.",
    ["Window N", "1", "5", "10", "20", "50"],
    [["Accuracy", "63.3 %", "83.2 %", "84.9 %", "85.3 %", "85.1 %"],
     ["Per seed (42 / 43 / 44)", "62.9 / 63.7 / 63.2", "83.0 / 83.2 / 83.5", "84.1 / 85.6 / 85.0",
      "86.1 / 85.5 / 84.3", "85.6 / 84.8 / 85.0"]],
    [2.0, 1.6, 1.6, 1.6, 1.6, 1.6])

h3("2.3.10 Evaluation Infrastructure")
p("Several pieces of work make the results above checkable. A **training/evaluation parity** change "
  "made the code paths used in training and in evaluation identical; the nine nodes frozen at the time give "
  "identical results on their 450 held-out episodes before and after, and a **training parity "
  "audit** of 18 paired runs found buffers and weights identical at every completed update. Every "
  "training attempt, including failed and budget-exhausted ones, is archived by a **run store**; a "
  "node can only be frozen when STABLE (the one exception, decision 0014, is applied by a script that "
  "cites it), and a saved checkpoint is never overwritten. Every "
  "experiment is recorded in a **knowledge base** with its plan, result files and the decisions it "
  "informs, and pass criteria are written before the run. The automated test suite has **290 "
  "passing tests** on commit `d7be0a2`.")

h2("2.4 Experiment Design")
p("Experiments follow the six steps of the template: define the problem, determine objectives, "
  "design the experiment, conduct it and collect data, analyse and interpret, and verify predicted "
  "results. Two completed experiments are given as worked examples: the chain W1a → handoff "
  "diagnostic → specification 3 (Section 2.3.6), and W1b-1 (Section 2.3.7).")
p("**First worked example: W1a → diagnostic → specification 3.**")
b("**1. Define the problem.** `pickup_box` was not STABLE under two specifications, and every "
  "deterministic failure was a state repeat. Question: does specification 2 pass when scored the "
  "way it would act, i.e. sampled?")
b("**2. Determine objectives.** A pass criterion fixed in a plan before any run: a row passes at "
  "sampled success ≥ 0.90 with at least 475 of 500 valid episodes; a seed passes if both rows pass; "
  "over seeds 43–47, four or more passing is a pass, two or fewer a fail, and exactly three triggers "
  "one pre-declared extension to seeds 48–52, requiring 7 of 10.")
b("**3. Design the experiment.** Unchanged trainer and specification, a fixed budget of 1,001,472 "
  "steps, final models only, 500 fresh environment seeds per row never used before, one sampling "
  "seed, and a validation gate that must reproduce earlier recorded episodes one by one before "
  "anything is scored.")
b("**4. Conduct and collect.** Seeds 43–47 gave exactly 3 of 5, so the declared extension ran; "
  "the final count was 6 of 10.")
b("**5. Analyse and interpret.** The rule was kept when it failed: W1a was reported as a fail, not "
  "re-scored at a lower threshold. The failures were all at the handoff, so a diagnostic with two "
  "counterfactuals and a pre-set reading rule was run, giving reading Key.")
b("**6. Verify predicted results.** Specification 3 was derived from that reading and predicted to "
  "fix the handoff row. On fresh seed ranges with the test unchanged, the handoff row passed on all "
  "ten seeds as predicted, and the own-hook row weakened on three; the overall verdict was 7 of 10, "
  "a pass at the threshold.")
p("**Second worked example: W1b-1.**")
b("**1. Define the problem.** UnlockPickup had no frozen composite. Specification 3 had passed its "
  "test under sampled scoring, but a parent runs its children deterministically, and no `pickup_box` "
  "had ever met decision 0004. Question: with one specification-3 model frozen as a child, does the "
  "UnlockPickup composite reach STABLE under decision 0004 unchanged, every node deterministic?")
b("**2. Determine objectives.** The pass rule and the readings of a failure were fixed in the plan "
  "before anything ran (Table 11): a seed passes if STABLE within budget and ≥ 450/500 on fresh "
  "seeds; the experiment passes at 2 of 3 seeds, with no extension.")
b("**3. Design the experiment.** The exception needed to freeze the child was written first as "
  "decision 0014 and kept *Proposed* while the experiment ran as its evidence. A child-selection "
  "rule, a ceiling gate that could stop the experiment before any training, and three training "
  "seeds were fixed; a random-selection control was added before training and kept out of the "
  "verdict.")
b("**4. Conduct and collect.** Seed 45 was selected and frozen; the ceiling gate passed (48/50 on "
  "the stopping seeds, 49/50 held-out; a pre-check, not a result); the composite was trained on "
  "seeds 43–45 and scored on the fresh seeds in the same process.")
b("**5. Analyse and interpret.** All three seeds passed, but as one result, since the selectors act "
  "identically. Random selection under the strict mask ties the learned selector, so the result is "
  "read as reuse, not as a learned choice. The non-blind child selection is disclosed with it.")
b("**6. Verify predicted results.** The plan predicted a ceiling of about 0.97 for the chain "
  "(488/500 × 485/488); the fresh score was 0.972, and every gate episode that reached the handoff "
  "succeeded, so the remaining loss sits before the handoff as predicted. The team then accepted "
  "decision 0014, and the child and the seed-43 composite were registered.")
tab("Table 11. Experiment W1b-1: the design fixed before the composite was trained.",
    ["Element", "Fixed before any composite was trained"],
    [
        ["Child-selection rule", "Freeze the smallest-numbered spec-3 seed that (a) passed W1a's test and (b) has a Wilson 95 % lower bound ≥ 0.95 on its deterministic handoff rate. Six of ten seeds qualify; seed 45 is chosen (sampled own hook 459/500, handoff 488/488; deterministic handoff 485/488, lower bound 0.982; deterministic own hook 305/500, a known limit)"],
        ["Ceiling gate", "Before training, the scripted sequence `unlock_door` → `drop_held` → `pickup_box` must succeed in at least 46/50 stopping episodes; otherwise the experiment fails with reading \"chain\" and nothing is trained. Expected ceiling ≈ 0.97"],
        ["Training", "Seeds 43, 44, 45; budget 40,960 decisions; evaluation every 512; goal-only mask; all children deterministic"],
        ["Pass rule", "A seed passes if STABLE under 0004 within budget and ≥ 450/500 on fresh seeds 54000–54499; W1b-1 passes if at least 2 of 3 seeds pass. No extension, no second child"],
        ["Control (added before training)", "Random selection over available children on the same fresh seeds, reported as step efficiency, not part of the verdict"],
        ["What follows", "Pass: decision 0014 goes to the team, then the child and the composite are registered and BlockedUnlockPickup (W2) starts. Fail: the reading is set by a table fixed in the plan (chain / selector / undetermined), then sampled execution (W1b-2)"],
    ], [2.4, 7.8])

h2("2.5 Commercialization and Freedom-to-Use")
p("The project remains an academic research prototype. The proposal-stage freedom-to-use review "
  "found no directly applicable patents restricting the core techniques, which are grounded in the "
  "academic literature; the software stack (MiniGrid, PyTorch, Gymnasium, Stable-Baselines3 and its "
  "contributed algorithms, Streamlit) is under open-source licences that permit academic and "
  "research use. No freedom-to-use blocker has been identified for the prototype. Because the "
  "project is scientific research and not a commercial product, no further patent or market search "
  "has been carried out since the Proposal; a broader patent and licence review would be needed only "
  "before any commercial use.")

# =====================================================================================
h1("ENCOUNTERED PROBLEMS")
h2("3.1 Changes in Plan and Goals, and Their Causes")
p("**The v1 stall and the redesign.** From June to September the project was blocked on one "
  "experiment: the second Progressive Network column never stabilised on FourRooms (PR-I §3.4), "
  "and the hierarchy, the part the supervisor had asked to prioritise, was waiting behind it. The "
  "corrective measure was the redesign of decision 0001: skills became nodes in a graph, Progressive "
  "Networks moved off the critical path (decision 0003), and FourRooms was removed from it. The "
  "hierarchy reached a depth-3 stack within the first two days of the new design.")
p("**The claim narrowed.** Measurements on DoorKey showed that the hierarchy, counting the training "
  "of its children, is not cheaper than flat PPO on a single task (Section 2.3.2), and frozen skills "
  "cannot forget by construction. The claim was therefore restated (decision 0006) as composition "
  "and reuse of compositions, with no-forgetting measured on the whole system and with its price. "
  "Forward transfer through laterals is not claimed after the negative W7 result.")
p("**Growth stays in scope.** Building the skill graph automatically (choosing the children of a new "
  "composite without a human) remains a goal (decision 0007). The present phase, in which tasks are "
  "declared by hand in a fixed order (decision 0005), is explicitly temporary.")

h2("3.2 Technical Problems")
b("**UnlockPickup was blocked from October until 15 November by one primitive.** `pickup_box` "
  "needed three specifications and about 28.5 million environment steps of experiments (Section 4.5) "
  "before a version passed a two-row test, at the threshold and under sampled scoring, and it could "
  "only be frozen under an exception to the stabilisation rule (decision 0014).")
b("**Deterministic execution loops under partial observability.** A policy that sees a 7 × 7 "
  "egocentric view and acts by argmax repeats the same action in the same view; in "
  "partially observed rooms this produces loops. Every deterministic failure of a frozen node "
  "inspected in the execution-mode audit (39 of 39) was such a state repeat. A scripted UnlockPickup "
  "sequence succeeded in 32/50 episodes deterministically and 147/150 with every node sampled. "
  "Decision 0014 settled the case of one argmax child; whether frozen skills should act by sampling "
  "in general is still open, and it needs its own stopping rule (decision 0012, reserved) because a "
  "sampled evaluation is itself random.")
b("**A start-state mismatch.** The handoff diagnostic showed that `pickup_box` failed in a state it "
  "had never trained on (the key just dropped in front of it). Specification 3 fixed the handoff and "
  "weakened the skill elsewhere; skills trained on designed start states transfer only as far as "
  "those states cover the parent's handoffs.")
b("**Success rate cannot separate composition from masking** on this curriculum (Section 2.3.3), "
  "and the same holds on UnlockPickup (Section 2.3.7). "
  "The choice task meant to escape this was flawed (Section 2.3.4). Until it is redesigned, the "
  "evidence for what a composite learns is the divergence count and step efficiency.")

h2("3.3 Progress Against the Timetable")
p("Table 12 compares what PR-I promised (its §5.1–5.7 and the eleven rows of its §5.8 timetable) "
  "with the state on 15 November. PR-I's week numbers follow the Proposal's project weeks.")
tab("Table 12. What PR-I promised and where it stands.",
    ["PR-I item", "Target", "State on 15 November", "Reason / correction"],
    [
        ["§5.1 FourRooms / PN training refinement", "—", "Taken off the critical path", "Stalled v1; decision 0003"],
        ["§5.2 Lateral vs no-lateral ablation", "—", "Done (W7), negative", "Section 2.3.8"],
        ["§5.3 Autoencoder novelty validation", "—", "Not started", "Hierarchy first; no owner yet"],
        ["§5.4 GRU task identification", "—", "Done in part", "Old-task accuracy and latency not measured"],
        ["§5.5 MetaController with real trained policies", "—", "Replaced by composite nodes; done through UnlockPickup", "Decision 0001"],
        ["§5.6 GUI on real outputs", "—", "Not started", "Behind schedule"],
        ["§5.7 Evaluation package", "—", "Done as the project knowledge base", "Section 2.3.10"],
        ["FourRooms stabilisation study", "Week 14", "Dropped from the critical path", "Decision 0003"],
        ["Autoencoder calibration", "Week 14", "Not started", "Behind schedule"],
        ["BWT / FWT / retention logging", "Week 15", "Not started; replaced by system-level evaluation (claim C1, C2)", "Frozen skills cannot forget; retention is measured on the system"],
        ["Lateral / no-lateral baseline", "Week 15", "Done on `go_to_goal` ← `open_door`, negative", "FourRooms no longer used"],
        ["MetaController upgrade, dynamic option selection", "Week 16", "Done in a different form: learned composite selectors", "Decision 0001"],
        ["GRU baseline and AE–GRU handoff", "Week 16", "GRU baseline done; handoff not started", "Autoencoder not built"],
        ["Dynamic options: runtime addition of new skills", "Week 17", "Frozen nodes are added as children by hand; automatic growth not started", "Behind schedule; scheduled after W2"],
        ["Dashboard on real outputs", "Week 17", "Not started", "Behind schedule"],
        ["Phase 2 end-to-end integration", "Week 19", "Not started", "Needs task registry and growth"],
        ["No-AE / no-GRU ablations", "Week 21", "Not started", "Depends on integration"],
        ["Phase 3 scope review", "Week 22", "Not held", "Scope changes were recorded as decisions instead (0001, 0006)"],
    ], [3.2, 1.2, 3.3, 3.0])
p("**Behind schedule**, with the reason for each: BlockedUnlockPickup (it builds on UnlockPickup, "
  "which one primitive blocked from October until 15 November, Section 2.3.6); automatic growth (it "
  "edits the same training code as the hierarchy work and is scheduled after W2); the autoencoder "
  "(deprioritised behind the hierarchy, no owner); the GUI on real runs (no time allocated); "
  "end-to-end integration (needs the task registry and growth).")

h2("3.4 Are the Original Goals Still Relevant? Corrective Measures")
p("Yes. Continual acquisition, task recognition without labels, hierarchical reuse and "
  "introspection remain the goals; what changed is how the first and third are realised and what is "
  "claimed (decision 0006). The corrective measures are: the redesign itself; fixing pass criteria "
  "before each run and reporting negative results as results (decision 0010); the training and "
  "evaluation parity work, which removed silent differences between how skills were trained and "
  "how they were scored; and a priority order for the remaining work (Section 5) that puts reuse "
  "on BlockedUnlockPickup and fair baselines before new components.")

h2("3.5 Effects of the Changes on the Final Report")
b("**Kept:** continual acquisition through frozen skills; hierarchical reuse, now evidenced by "
  "composites reused inside later tasks; GRU task identification; evaluation on the MiniGrid task "
  "sequence.")
b("**Reduced:** Progressive Networks become an ablation, not the mechanism of transfer; the GUI "
  "becomes a viewer of recorded runs (skill graph, active stack, traces) rather than a live "
  "dashboard of all signals; the autoencoder is evaluated if time permits.")
b("**Dropped from this project:** the planning extension (Proposal objective 4, already a stretch "
  "goal) and FourRooms as a benchmark.")

# =====================================================================================
h1("SUSTAINABILITY AND ECONOMIC ASSESSMENT")
p("The system is a software-only research prototype, so its environmental and economic impact "
  "comes from computation and development time, not from materials or manufacturing. PR-I described "
  "the impact as coming mainly from GPU-accelerated training; this needs correcting: every run "
  "reported since PR-I was trained on one CPU.")

h2("4.1 Functional Unit")
p("The functional unit is now **acquiring one new task given the existing skill library**, measured "
  "in environment steps of training (the cost unit used in every experiment) and converted to CPU "
  "time from the recorded wall-clock time of each run. Examples: the `unlock_door` composite costs "
  "32,869–34,877 steps of its own given its two frozen children (re-measured on three seeds), and "
  "`solve_doorkey` 48,887–55,659 and `unlock_pickup` 55,412–60,205 given theirs. This unit matches the claim: the project's "
  "economic argument is that later tasks become cheap because earlier skills are reused.")

h2("4.2 Preliminary System Boundary")
p("The boundary is gate-to-gate, as in PR-I: from environment initialisation and training to "
  "checkpoint, metadata, log and figure generation. It includes Python execution, MiniGrid "
  "simulation, PPO and masked-PPO training, CPU computation, evaluation runs, checkpoint and run-"
  "store writing, and report-figure generation. It excludes hardware manufacturing, transport and "
  "end-of-life, and electricity unrelated to the experiments.")

h2("4.3 Main Environmental Impact Contributors")
p("The main contributor is the electricity used by training. Table 13 lists the recorded "
  "wall-clock time of the training runs whose result files are in the repository. The two largest "
  "items are the flat baselines on the two hard tasks (8.5 hours, all censored at 0/50) and the "
  "`pickup_box` experiments (8.6 hours). Runs were often executed several at a time on one "
  "machine, so these are process-hours, not machine-hours; smoke runs, runs repeated because of a "
  "bug and runs kept only in the local run archive are not included, so the total is a lower bound.")
tab("Table 13. Recorded training time since PR-I (sum of per-run wall-clock seconds).",
    ["Group", "Runs", "Hours"],
    [
        ["Frozen library (six primitives, `unlock_door`)", "7", "0.34"],
        ["`pickup_box`: specifications 1, 2, 3 and W1a", "26", "8.62"],
        ["Flat PPO baselines (4 tasks × 3 seeds)", "12", "9.84"],
        ["   of which UnlockPickup and BlockedUnlockPickup", "6", "8.46"],
        ["Lateral ablation W7", "30", "2.47"],
        ["`drop_held` and `unlock_pickup` (3 seeds each), other composite runs, choice task, option-set baseline", "25", "4.54"],
        ["**Total**", "", "**25.8**"],
    ], [6.2, 1.2, 1.6],
    "At an assumed average draw of 90 W per process, 25.8 h is about 2.3 kWh, or about 0.9 kg CO₂e at the 0.4 kg CO₂/kWh factor used in the Proposal and PR-I. The Proposal estimated 10–30 GPU-hours (1.5–4.5 kWh) per full run.")
p("Two other contributors are repeated experiments and storage. The project's rule of fixing pass "
  "criteria before a run and using declared extensions instead of open-ended retraining limits "
  "repeated runs; checkpoints and logs are small (a frozen primitive is a 147→64→64 network).")

h2("4.4 Initial Cost Components")
b("**Software:** zero; all libraries are open source.")
b("**Hardware and operation:** existing personal computers, CPU only; electricity as in Table 13.")
b("**Development time:** the dominant cost; student time for design, implementation, experiments "
  "and reports.")
b("**Storage and maintenance:** minimal; checkpoints and result files are kept in the repository, "
  "and the modular design lets one skill be replaced without retraining others.")

h2("4.5 Qualitative Comparison with Conventional Solutions")
p("The conventional solution is flat PPO trained separately on each task. Both sides of the "
  "comparison are stated (Table 14). Through DoorKey, reuse costs about the same as flat training: "
  "773,955 environment steps for the hierarchy as built against 650,000–900,000 for flat PPO on "
  "Unlock and DoorKey. On UnlockPickup, the recipe for the new nodes costs about 2.38 million steps "
  "per run (2.79 million with the reused library), **below flat PPO's censored figure of more than "
  "3.0 million per run on one task**, which flat PPO never learned. The research cost of finding the "
  "`pickup_box` specification, about 28.5 million steps, is **far above** it, and above flat PPO's "
  "9.0 million for three runs. The gain the design aims at is that each further task is cheaper "
  "because the library is reused; BlockedUnlockPickup is the next test of that.")
tab("Table 14. Cost of UnlockPickup in environment steps: hierarchy against flat PPO.",
    ["Figure", "What it counts", "Env steps"],
    [
        ["Recipe, per run", "`drop_held` + `pickup_box` (with handoff-chain steps) × 10/6 + `unlock_pickup` own", "2,374,637 – 2,379,430"],
        ["Recipe without the selection factor", "the same, × 1 instead of × 10/6", "1,458,947 – 1,463,740"],
        ["Reused library", "`unlock_door` (131,817 own + 280,000 for its children)", "411,817"],
        ["Research total on `pickup_box`", "six October runs, W1a, specification 3 and its handoff-chain steps", "28,549,749"],
        ["Flat PPO, per run / three runs", "0/50 held-out, censored at the budget", "> 3,000,320 / 9,000,960"],
    ], [2.6, 5.2, 2.4],
    "Evaluation episodes are counted on neither side. Source: W1b-1 summary (cost table) and its run files.")

h2("4.6 Preliminary Assessment")
p("The project remains economically feasible and environmentally small: about 26 CPU process-hours "
  "of recorded training in six months, open-source software and existing hardware. The main "
  "sustainability risk identified in PR-I, inefficient repeated training, did occur on `pickup_box`; "
  "the measures that now limit it are pre-declared criteria and budgets, a single declared "
  "extension per experiment, and diagnostics run as evaluations (no training) before any new "
  "specification.")

# =====================================================================================
h1("TASKS TO BE COMPLETED UNTIL FINAL REPORT")
p("W1b-1 is complete (Section 2.3.7). The remaining work is listed in priority order (Table 15). Items marked *must* are needed for "
  "the claim; *if time* items are dropped first. Each item names the capstone phase it belongs to. "
  "Criteria are fixed in each item's own plan before its first run.")
tab("Table 15. Remaining work in priority order.",
    ["#", "Work", "Phase", "Priority"],
    [
        ["1", "W2: BlockedUnlockPickup by reuse, T9a (level 2: `move_blocker`, `unlock_door`, `pickup_box`) against T9b (level 3: `move_blocker` and the frozen UnlockPickup composite); `pickup_box` needs its own handoff row under T9a first (decision 0014 is per parent)", "Construction, evaluation", "Must"],
        ["2", "W1b-2: sampled execution of frozen skills with its own stopping rule (decision 0012); flat UnlockPickup retrained with saved models", "Detailed design, testing", "Must"],
        ["3", "W4: fair baselines: recurrent flat PPO; one flat PPO fine-tuned through the curriculum", "Testing", "Must"],
        ["4", "Growth G-1/G-2: one API from declaration to registration; choosing a new composite's children automatically, with random choice under the same mask as control", "Detailed design, construction", "Must"],
        ["5", "Recognition: autoencoder novelty detection (R-1) and the task registry from GRU task id to root skill (R-3)", "Construction, testing", "R-3 must; R-1 if time"],
        ["6", "W8: system-level continual evaluation: after each new task, all earlier tasks end to end with recognition in the loop; node count per task (claim C1, C2)", "Evaluation", "Must"],
        ["7", "GUI on recorded runs: skill graph, active stack, traces", "Construction", "If time"],
        ["8", "Choice task redesigned so that a wrong pick is unrecoverable", "Detailed design, testing", "If time"],
    ], [0.5, 6.0, 2.2, 1.5])
p("Table 16 is the week-by-week schedule. It is deliberately tight: every experiment ends by week "
  "10, so that week 11 is left for writing the draft and as a buffer for one repeated run. This "
  "report falls in week 7; the Draft Final Report is due in week 12 and the Final Report in week 14.")
tab("Table 16. Timetable to the Final Report.",
    ["Week", "Work", "Done when"],
    [
        ["7", "W2 plan; `move_blocker` primitive trained; `pickup_box` handoff row under T9a; W4 baselines (recurrent and fine-tuned flat PPO) started", "`move_blocker` frozen; W2 criteria fixed"],
        ["8", "W2: T9a and T9b trained and compared; W1b-2 plan and stopping rule (0012); flat UnlockPickup retrained with saved models", "W2 verdict recorded"],
        ["9", "Growth G-1 and G-2 (replay Unlock and DoorKey with automatic child selection); task registry R-3; W4 results", "Growth replay verdict; registry selects roots"],
        ["10", "W8 system-level evaluation with recognition in the loop; autoencoder R-1 and GUI on recorded runs if time", "All experiments closed"],
        ["11", "Draft Final Report written; buffer for one repeated run", "Draft complete"],
        ["12", "Draft Final Report submitted", "—"],
        ["13", "Corrections from the draft review", "—"],
        ["14", "Final Report", "—"],
    ], [0.8, 6.4, 3.0])

# =====================================================================================
h1("REFERENCES")
for r in [
    "[1]\tM. Chevalier-Boisvert *et al.*, \"Minigrid & miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks,\" *Advances in Neural Information Processing Systems,* vol. 36, pp. 73383–73394, 2023.",
    "[2]\tV. Mnih *et al.*, \"Human-level control through deep reinforcement learning,\" *Nature,* vol. 518, no. 7540, pp. 529–533, 2015, doi: 10.1038/nature14236.",
    "[3]\tK. Khetarpal, M. Riemer, I. Rish, and D. Precup, \"Towards Continual Reinforcement Learning: A Review and Perspectives,\" *Journal of Artificial Intelligence Research,* vol. 75, pp. 1401–1476, 2022, doi: 10.1613/jair.1.13673.",
    "[4]\tS. Pateria, B. Subagdja, A.-h. Tan, and C. Quek, \"Hierarchical Reinforcement Learning: A Comprehensive Survey,\" *ACM Comput. Surv.,* vol. 54, no. 5, Article 109, 2021, doi: 10.1145/3453160.",
    "[5]\tJ. Kirkpatrick *et al.*, \"Overcoming catastrophic forgetting in neural networks,\" *Proceedings of the National Academy of Sciences,* vol. 114, no. 13, pp. 3521–3526, 2017, doi: 10.1073/pnas.1611835114.",
    "[6]\tD. Rolnick, A. Ahuja, J. Schwarz, T. P. Lillicrap, and G. Wayne, \"Experience replay for continual learning,\" in *Proceedings of the 33rd International Conference on Neural Information Processing Systems,* 2019, pp. 350–360.",
    "[7]\tA. A. Rusu *et al.*, \"Progressive Neural Networks,\" *ArXiv,* vol. abs/1606.04671, 2016.",
    "[8]\tZ. D. Erden, D. Gasmi, and B. Faltings, \"Continual Reinforcement Learning via Autoencoder-Driven Task and New Environment Recognition,\" 2025, doi: 10.48550/arXiv.2505.09003.",
    "[9]\tM. J. Jacobson, C. Q. Wright, N. Jiang, G. Rodriguez-Rivera, and Y. Xue, \"Task Detection in Continual Learning via Familiarity Autoencoders,\" in *2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC),* 2022, pp. 1–8, doi: 10.1109/SMC53654.2022.9945326.",
    "[10]\tJ. Chung, C. Gulcehre, K. Cho, and Y. Bengio, \"Empirical evaluation of gated recurrent neural networks on sequence modeling,\" *arXiv preprint arXiv:1412.3555,* 2014.",
    "[11]\tR. S. Sutton, D. Precup, and S. Singh, \"Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning,\" *Artificial Intelligence,* vol. 112, no. 1, pp. 181–211, 1999, doi: 10.1016/S0004-3702(99)00052-1.",
    "[12]\tJ. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, \"Proximal policy optimization algorithms,\" *arXiv preprint arXiv:1707.06347,* 2017.",
    "[13]\tR. K. Nayyar and S. Srivastava, \"Autonomous Option Invention for Continual Hierarchical Reinforcement Learning and Planning,\" *Proceedings of the AAAI Conference on Artificial Intelligence,* vol. 39, no. 18, pp. 19642–19650, 2025, doi: 10.1609/aaai.v39i18.34163.",
    "[14]\tM. Malagon, J. Ceberio, and J. A. Lozano, \"Self-Composing Policies for Scalable Continual Reinforcement Learning,\" in *Proceedings of the 41st International Conference on Machine Learning (ICML),* PMLR vol. 235, 2024, pp. 34432–34460.",
    "[15]\tT. P. Cannon and Ö. Şimşek, \"Hierarchical Orchestra of Policies,\" in *NeurIPS 2024 Workshop on Intrinsically-Motivated and Open-Ended Learning (IMOL),* 2024, arXiv:2411.03008.",
]:
    B.append(("ref", r))

BLOCKS = B
