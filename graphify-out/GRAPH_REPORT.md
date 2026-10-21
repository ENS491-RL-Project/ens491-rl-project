# Graph Report - ens491-rl-project  (2026-10-21)

## Corpus Check
- 187 files · ~312,306 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 56 file(s) not represented in the graph (top: .log 37, .pt 15, (none) 4)

## Summary
- 1700 nodes · 5231 edges · 84 communities (59 shown, 25 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 793 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `3d427d3c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_h6_hooks.py
- test_evaluation.py
- Column
- OptionEnv
- eval_h7_preflight.py
- TaskLifecycleManager
- .spec_hash
- evaluation.py
- 9. Corrections (H-6.8, 2026-10-02)
- diagnose_pickup_box.py
- FixedAction
- pn_trainer.py
- open_door
- train_flat_baseline.py
- MetaController
- GRUTaskIdentifier
- pytest
- StateView
- test_c1_controls.py
- json
- build_state_view
- .save
- test_h6_accounting.py
- AutoencoderModule
- torch
- ColumnTrainer
- SkillNode
- node_trainer.py
- Run summary — H-7a (2026-10-02): stopped at H-7a.2 on amendment assumption A2b
- audit_execution_mode.py
- ENS 491 — Recursive Hierarchy Design
- capture_trace
- types.py
- LateralMlpExtractor
- generate_report_figures.py
- test_option_env.py
- Part C — tasks
- analyse_h7_amortisation.py
- key_off_door
- MockPolicy
- app.py
- ENS 491 — Project State & Roadmap
- Plan — H-6: `{2,1} solve_doorkey`, the C1 baselines, and a genuine-choice composite
- MaskableMonitor
- .door
- _MockPolicy
- fit_selector
- Session handoff — 2026-10-02 (H-6 corrections, H-6.9, H-7 up to the A2 stop)
- CompositePolicyAdapter
- column_policy.py
- ENS 491 — Module Interface Specification
- Review of the H-0 → H-5 plan (2026-10-01)
- ENS 491 — Target Architecture v2
- train_choice_task.py
- Run summary — v2 migration, Step 0 → H-5 (2026-10-01)
- door_open
- analyse_choice_conditioning.py
- 6. Task recognition (AE + GRU)
- _MockPolicy
- eval_legacy_col01.py
- MockPolicy
- fit_logreg
- Execution-mode audit — what deterministic execution does across the library (evaluation only)
- PrimitivePolicyAdapter
- gui.md
- test_executor_never_branches_on_level
- test_dummyvecenv_n4_matches_n1_for_a_scripted_policy
- _Recorder
- TurnPolicy
- Supervisor Meeting — Open Questions
- RandomOverAll
- prepare_with
- Excluded
- README.md
- ToyWorld
- test_ablation_is_scored_under_the_mask_it_was_trained_with

## God Nodes (most connected - your core abstractions)
1. `SkillNode` - 127 edges
2. `build_state_view()` - 89 edges
3. `Executor` - 78 edges
4. `NodeRegistry` - 66 edges
5. `Column` - 59 edges
6. `open_door()` - 58 edges
7. `StateView` - 51 edges
8. `heldout_seeds()` - 49 edges
9. `execution_env_factory()` - 48 edges
10. `ResetHook` - 45 edges

## Surprising Connections (you probably didn't know these)
- `C.6 The regression sweep missed a frozen node` --references--> `choice_specs()`  [INFERRED]
  docs/plans/2026-10-02_H6_summary.md → scripts/train_choice_task.py
- `9. Implementation Requirements (for PN-3)` --references--> `Column`  [INFERRED]
  docs/ENS491_Recursive_Hierarchy_Design.md → src/continual_learning/column.py
- `H-0 — v1 bug fixes (1 SP)` --references--> `LateralMlpExtractor`  [INFERRED]
  docs/plans/2026-10-01_v2_H0-H5_plan.md → src/continual_learning/column_policy.py
- `Answers to the plan's "Decision" items (recommended; edit before resuming if you disagree)` --references--> `ChannelNormalize`  [INFERRED]
  docs/plans/2026-10-01_v2_H0-H5_plan_review.md → src/envs/obs.py
- `Locked Decisions` --references--> `OptionEnv`  [INFERRED]
  docs/ENS491_Project_State.md → src/envs/option_env.py

## Import Cycles
- None detected.

## Communities (84 total, 25 thin omitted)

### Community 0 - "test_h6_hooks.py"
Cohesion: 0.06
Nodes (29): Verified facts this plan is built on, _audit(), ResetHookEnv, _dir_sees(), face_toward(), fn(), place_two_keys(), recolor_door() (+21 more)

### Community 1 - "test_evaluation.py"
Cohesion: 0.06
Nodes (28): A.4 Cross-cutting (environment, repo, process), resolve_device(), EvalResult, EvaluationCallback, StabilityCriterion, CountdownEnv, _dummy_node(), make_scripted() (+20 more)

### Community 2 - "Column"
Cohesion: 0.06
Nodes (18): _make_trained_column(), ColumnPolicyAdapter, Column, _to_tensor(), ColumnOutput, MockMlpExtractor, MockPolicy, RecordingPolicy (+10 more)

### Community 3 - "OptionEnv"
Cohesion: 0.11
Nodes (13): Commands, Core model (v2 §2–§6), Docs, graphify, Hard rules, Stack, 4.2 Termination, Part B — Build plan, H-0 → H-5 (+5 more)

### Community 4 - "eval_h7_preflight.py"
Cohesion: 0.09
Nodes (32): _adj(), audit_h7a(), audit_move_blocker(), audit_pickup_box(), f29_row(), main(), _neigh(), _proxy_drop_held() (+24 more)

### Community 5 - "TaskLifecycleManager"
Cohesion: 0.11
Nodes (16): run_demo(), TaskLifecycleManager, _to_tensor(), SystemState, _make_trained_column(), test_column_m_is_1indexed(), test_get_state_returns_system_state(), test_layer_indexing_mc_at_n_plus_1() (+8 more)

### Community 6 - ".spec_hash"
Cohesion: 0.10
Nodes (19): 13. Changelog v1 → v2, Amendments applied 2026-10-01 (H-0.5), Amendments applied 2026-10-02 (H-6), Amendments applied 2026-10-02 (H-6.8), Amendments applied 2026-10-02 (H-7), Execution rules for this plan, H-6.0 — F29 pre-flight for the depth-2 stack (1 SP), H-6.2 — retrain and re-freeze `{1,1} unlock_door` with `initiation` declared (1 SP) (+11 more)

### Community 7 - "evaluation.py"
Cohesion: 0.15
Nodes (8): main(), probe(), give_key_of(), available_accelerator(), describe(), _wilson(), wilson_lower(), wilson_upper()

### Community 8 - "9. Corrections (H-6.8, 2026-10-02)"
Cohesion: 0.06
Nodes (31): 10.1 Exploratory: the two diagnostic RL arms (H-7 amendment 1, §8), 10. H-6.9 — why the selector did not learn a choice it can see, 1. Gate results, 2. Nodes, 4. The genuine-choice task: no choice learned in 100 updates, 6. Open questions this run answered, and one it did not, 7. How H-7 should test reuse, 9. Corrections (H-6.8, 2026-10-02) (+23 more)

### Community 9 - "diagnose_pickup_box.py"
Cohesion: 0.15
Nodes (14): matching_key(), main(), main_spec2(), main_trace(), readings(), summarise_trace(), chain(), load_unstable_pickup_box() (+6 more)

### Community 10 - "FixedAction"
Cohesion: 0.20
Nodes (3): FirstAvailable, FixedAction, test_initiation_does_not_mask()

### Community 11 - "pn_trainer.py"
Cohesion: 0.08
Nodes (11): _make_env(), _pca_2d(), _visualize(), _make_column_with_policy(), _make_env(), test_flat_obs_shape(), test_freeze_sets_flag_and_blocks_grad(), test_frozen_column_still_produces_activations() (+3 more)

### Community 12 - "open_door"
Cohesion: 0.09
Nodes (22): H-2 — predicates, SubgoalEnv, obs pipeline (5 SP), H-4 — train and freeze T1, T2, T4 (5 SP), H-6.6 — the genuine-choice composite (5 SP), Part C — the genuine-choice task, designed, Predicted numbers, written before the run, The nodes, The task: two keys, one door, matching colours, What "genuine choice" has to mean (+14 more)

### Community 13 - "train_flat_baseline.py"
Cohesion: 0.10
Nodes (24): H-7.1 — pre-flight: facts, hooks, solvability, first transfer (3 SP), pose_in_room(), fn(), hold(), fn(), _adjacent(), fn(), _door_approachable() (+16 more)

### Community 14 - "MetaController"
Cohesion: 0.15
Nodes (14): Recursive execution (unbounded), Context, Plan — Implement Target Architecture v2 (H-0 → H-5) + v2 design critique, Verified facts this plan is built on, MetaController, MetaControllerOutput, _make_option(), test_add_option_increases_count() (+6 more)

### Community 15 - "GRUTaskIdentifier"
Cohesion: 0.11
Nodes (11): GRUTaskIdentifier, _to_tensor(), GRUOutput, test_forward_returns_gru_output(), test_get_sequence_zero_pads_short_buffer(), test_observe_and_get_sequence(), test_register_new_task_expands_head(), test_register_new_task_idempotent() (+3 more)

### Community 16 - "pytest"
Cohesion: 0.11
Nodes (24): flag(), prim(), sv(), test_at_most_one_lateral_source(), test_availability_cache_evaluates_a_shared_child_once(), test_availability_without_cache_still_correct(), test_available_is_goal_only_not_initiation(), test_check_termination_success_timeout_none() (+16 more)

### Community 17 - "StateView"
Cohesion: 0.11
Nodes (29): 8. Repository state, Part B — nodes, relocate_blocker(), all_of(), carrying_nothing(), door_approach_blocked(), door_approach_clear(), _adjacent() (+21 more)

### Community 18 - "test_c1_controls.py"
Cohesion: 0.09
Nodes (21): mask_divergence(), RandomOverAll, RandomOverAvailable, _child(), FirstAvailable, _graph(), _step_at_least(), _step_odd() (+13 more)

### Community 19 - "json"
Cohesion: 0.13
Nodes (18): load_library(), primitive_specs(), load_all(), main(), load_choice_children(), load_door_opener(), load_three_primitives(), build_node() (+10 more)

### Community 20 - "build_state_view"
Cohesion: 0.17
Nodes (18): build_state_view(), raw(), test_carrying_predicate_truth_table(), test_curriculum_env_exists(), test_door_approach_clear_predicate_truth_table(), test_door_open_predicate_truth_table(), test_env_step_budgets_match_the_timeout_invariant(), test_give_key_removes_it_from_the_grid() (+10 more)

### Community 21 - ".save"
Cohesion: 0.13
Nodes (11): _slug(), SpecMismatch, make_node(), test_changed_goal_predicate_is_refused(), test_changed_timeout_is_refused(), test_composite_hash_covers_its_children(), test_list_nodes(), test_round_trip_restores_weights() (+3 more)

### Community 22 - "test_h6_accounting.py"
Cohesion: 0.14
Nodes (15): carrying(), children_env_steps(), _composite(), FirstChild, _primitive(), test_children_env_steps_counts_a_shared_child_once(), test_children_env_steps_is_zero_for_a_primitive(), test_children_env_steps_sums_primitive_and_composite_costs() (+7 more)

### Community 23 - "AutoencoderModule"
Cohesion: 0.15
Nodes (10): AutoencoderModule, _to_tensor(), AEOutput, test_compute_errors_returns_per_sample_list(), test_encode_decode_shape(), test_forward_returns_ae_output(), test_novelty_flag_respects_threshold(), test_train_on_batch_reduces_loss() (+2 more)

### Community 24 - "torch"
Cohesion: 0.18
Nodes (10): Option, OptionStepOutput, _make_option(), test_option_action_in_valid_range(), test_option_can_initiate_always_true(), test_option_id_matches_column_coordinates(), test_option_not_terminated_before_max_steps(), test_option_reset_clears_step_count() (+2 more)

### Community 25 - "ColumnTrainer"
Cohesion: 0.08
Nodes (14): H-0.5 — amend the design docs (2 SP), H-0 — v1 bug fixes (1 SP), H-3 — EvaluationMonitor (3 SP), New module layout, Review corrections folded in, Corrections to the plan (verified against the code), Step 0 — branch and commit the docs (first, before any code), main() (+6 more)

### Community 26 - "SkillNode"
Cohesion: 0.16
Nodes (9): Track H — Hierarchy (critical path), 10. Code migration map (current repo → v2), 12. Roadmap: four tracks, 2. Skill Node 🔒, 3. Graph structure: a DAG 🔒, A.3 Node model, graph, recognition, PN, growth (§2, §3, §6, §7, §8), H-1 — SkillNode, StateView, Executor, registry (8 SP), Frame (+1 more)

### Community 27 - "node_trainer.py"
Cohesion: 0.11
Nodes (22): 5. Deviations from the plan, and why, C.8 Stale text and dead configuration, H-7.3 — `{0,4} pickup_box` (3 SP), H-7.4 — `{2,2} unlock_pickup`, 3 seeds, five arms (5 SP), deepest_trace(), heldout(), run(), build_flat_options() (+14 more)

### Community 28 - "Run summary — H-7a (2026-10-02): stopped at H-7a.2 on amendment assumption A2b"
Cohesion: 0.05
Nodes (49): `{0,4} pickup_box`, specification 2 (H-7a.2), `{0,9} drop_held` (H-7a.2), 10.1 Success rates, 10.2 The five failure counts (they may overlap), 10.3 Reading (the rules fixed in the close-out before the run), 10. H-7a.5: where and how the spec-2 checkpoint fails (close-out §4, evaluation only), 11. Close-out deviations (`plans/2026-10-02_H7a_closeout.md`), 12.2 The table (+41 more)

### Community 29 - "audit_execution_mode.py"
Cohesion: 0.11
Nodes (21): main(), e0(), e3(), flat_record(), flat_saved_models(), load_pickup_box_training(), main(), materialise() (+13 more)

### Community 30 - "ENS 491 — Recursive Hierarchy Design"
Cohesion: 0.09
Nodes (22): 10. Open Questions — Supervisor Meeting Agenda, 1. Notation, 2. The Dual Role of a Column, 3.1 Tree structure 🔒, 3.2 Meta-controller scope 🔒, 3. Hierarchy Structure, 4.1 Definition 🔒, 4.2 Scope constraint 🔒 (+14 more)

### Community 31 - "capture_trace"
Cohesion: 0.20
Nodes (3): H-6.3 — `{2,1} solve_doorkey`, 3 seeds, with both masking controls (5 SP), env_success(), capture_trace()

### Community 32 - "types.py"
Cohesion: 0.08
Nodes (11): 3. The C1 result, stated plainly, The masking problem is not solved by depth, ExecutorError, TraceEvent, CompositePolicy, Predicate, PrimitivePolicy, DoorView (+3 more)

### Community 33 - "LateralMlpExtractor"
Cohesion: 0.08
Nodes (17): Code layout, Networks, SB3 integration, Stabilization (v2 §5.3), Training, networks and stabilization, Current Status (2026-10-01), ✅ Done (v1 phase, Feb–May 2026), ⚠️ Known limitations of the v1 code (to be replaced, see v2 §10) (+9 more)

### Community 34 - "generate_report_figures.py"
Cohesion: 0.14
Nodes (10): fig1_architecture(), pb(), rbox(), fig2_training_curves(), fig3_experiment_pipeline(), fig4_copy_activations(), fig5_test_summary(), fig6_milestone_timeline() (+2 more)

### Community 36 - "test_option_env.py"
Cohesion: 0.16
Nodes (18): make_option_env(), primitive(), test_action_space_is_the_child_count(), test_child_outcome_and_k_are_logged_every_decision(), test_initiation_masking_ablation_restores_the_strict_mask(), test_mask_never_all_false(), test_maskable_ppo_can_read_the_masks_through_the_wrapper_stack(), test_maskable_ppo_trains_a_few_steps() (+10 more)

### Community 37 - "Part C — tasks"
Cohesion: 0.11
Nodes (18): Assumptions, stated so a contradiction is recognisable, Constraints (from CLAUDE.md and v2; every task includes them), Execution rules, Goal, H-7.0 — Part A amendments (1 SP), H-7.2 — launch the flat baselines in the background (2 SP), H-7.5 — `{0,5} move_blocker`, and the transfer measurement that decides T9 (5 SP), H-7.6 — T9a and T9b, 3 seeds each (5 SP) (+10 more)

### Community 38 - "analyse_h7_amortisation.py"
Cohesion: 0.15
Nodes (13): Candidate curriculum (⚠️ Q6), make_subgoal_env(), SubgoalEnv, make_node(), test_subgoal_env_observation_contract(), test_subgoal_env_refuses_a_goal_that_holds_at_reset(), test_subgoal_env_reset_hook_is_visible_in_the_observation(), test_subgoal_env_reward_only_on_its_own_goal() (+5 more)

### Community 39 - "key_off_door"
Cohesion: 0.16
Nodes (13): Amendments applied 2026-10-02 (H-7 amendment 1), 1. The decision, 2. Scope: H-7a now, H-7b later, 3. Part A, revised (amend v2 §11 and §13 first, one commit), 4. Node specifications, 5. Assumptions (replace A2; A1, A3, A4 stand), 6. Tasks (one commit each, `--smoke` first, push after each), 7. Wording, added to the plan's rules (+5 more)

### Community 41 - "app.py"
Cohesion: 0.24
Nodes (10): _get_tlm(), _live_state_dict(), _load_demo_state(), main(), _render_ae(), _render_columns(), _render_current(), _render_gru() (+2 more)

### Community 42 - "ENS 491 — Project State & Roadmap"
Cohesion: 0.14
Nodes (13): Answered by H-6 (2026-10-02), Critical Path, ENS 491 — Project State & Roadmap, Locked Decisions, Open Empirical Questions, Roadmap Status, Story Points Reference, Stretch (+5 more)

### Community 43 - "Plan — H-6: `{2,1} solve_doorkey`, the C1 baselines, and a genuine-choice composite"
Cohesion: 0.50
Nodes (4): Assumptions, stated so a contradiction is recognizable, Context, Plan — H-6: `{2,1} solve_doorkey`, the C1 baselines, and a genuine-choice composite, Verified facts this plan is built on

### Community 45 - ".door"
Cohesion: 0.33
Nodes (4): fn(), fn(), fn(), fn()

### Community 47 - "fit_selector"
Cohesion: 0.29
Nodes (4): fit_selector(), fit_selector_fixed(), logits_of(), selector_net()

### Community 48 - "Session handoff — 2026-10-02 (H-6 corrections, H-6.9, H-7 up to the A2 stop)"
Cohesion: 0.14
Nodes (13): 1. Step B: corrections to the H-6 summary (`5a4c353`), 2. Step C: H-6.9 (`79426d2`, H-6 summary §10), 3. Step D: H-7, 4. Written but never run, 5. The open decision (the author's, not the agent's), 6. Environment notes (learned the hard way this session), Commits of this session (oldest first), Documents to read, in order (+5 more)

### Community 49 - "CompositePolicyAdapter"
Cohesion: 0.17
Nodes (7): NodeRegistry, CompositePolicyAdapter, NodeStatus, _real_composite(), test_load_frozen_composite_refuses_a_changed_child(), test_load_frozen_composite_rejects_an_unfrozen_child(), test_load_frozen_composite_round_trips()

### Community 50 - "column_policy.py"
Cohesion: 0.07
Nodes (24): A.1 Execution semantics (§4) — the ones that would bite first, A.2 Learning semantics (§5), Part A — v2 design critique and proposed amendments, 12.1 E.0: what `STOCHASTIC` samples, mask_divergence(), crosscheck(), episode(), sequence_episode() (+16 more)

### Community 51 - "ENS 491 — Module Interface Specification"
Cohesion: 0.17
Nodes (11): Common Types, ENS 491 — Module Interface Specification, Entry point (every step), Hard Constraints, Module 1: Autoencoder (AE), Module 2: GRU Task Identifier, Module 3: Progressive Networks Column, Module 4: Option Wrapper (+3 more)

### Community 52 - "Review of the H-0 → H-5 plan (2026-10-01)"
Cohesion: 0.40
Nodes (4): Answers to the plan's "Decision" items (recommended; edit before resuming if you disagree), Fact to confirm before execution, Resume prompt (paste into Claude Code, in default or plan mode, not auto mode), Review of the H-0 → H-5 plan (2026-10-01)

### Community 53 - "ENS 491 — Target Architecture v2"
Cohesion: 0.13
Nodes (14): 0. Why v2 exists, 11. Evaluation plan: claims → experiments, 1. Separation of concerns 🔒, 4.1 Options persist until they terminate, 4.3 Availability, 4.4 The executor (depth-agnostic), 4. Execution semantics 🔒, 6.1 Autoencoder: "is this situation new?" (+6 more)

### Community 54 - "train_choice_task.py"
Cohesion: 0.11
Nodes (24): collect(), decision1(), first_key_accuracy(), key_probabilities(), main(), make_env(), policy_shape(), stage_arm() (+16 more)

### Community 55 - "Run summary — v2 migration, Step 0 → H-5 (2026-10-01)"
Cohesion: 0.25
Nodes (7): Deviations from the plan, and why, Gate results, Repository state, Run summary — v2 migration, Step 0 → H-5 (2026-10-01), Things worth knowing before H-6, Trained nodes, What the C1 controls actually show

### Community 56 - "door_open"
Cohesion: 0.20
Nodes (6): run_pair(), fn(), door_open(), test_predicates_carry_a_spec_for_hashing(), _real_children(), test_reward_shape_on_success()

### Community 57 - "analyse_choice_conditioning.py"
Cohesion: 0.31
Nodes (6): arm_row(), fisher_two_sided(), p(), h68_row(), main(), row()

### Community 60 - "eval_legacy_col01.py"
Cohesion: 0.09
Nodes (14): Task detection (AE + GRU), main(), env_factory(), apply_obs_pipeline(), ChannelNormalize, make_base_env(), normalize_flat_obs(), obs_pipeline_name() (+6 more)

### Community 63 - "Execution-mode audit — what deterministic execution does across the library (evaluation only)"
Cohesion: 0.29
Nodes (6): 1. Why, 2. Seeds, 3. Measurements, 4. Wording rules, 5. Verify, Execution-mode audit — what deterministic execution does across the library (evaluation only)

### Community 68 - "test_dummyvecenv_n4_matches_n1_for_a_scripted_policy"
Cohesion: 0.50
Nodes (4): test_dummyvecenv_n4_matches_n1_for_a_scripted_policy(), make_option_env_factory(), _f(), rollout()

### Community 72 - "Supervisor Meeting — Open Questions"
Cohesion: 0.50
Nodes (3): New questions (v2), Supervisor Meeting — Open Questions, v1 questions and the answers v2 proposes (please confirm or correct)

### Community 77 - "ToyWorld"
Cohesion: 0.12
Nodes (27): build_depth5(), build_ladder(), comp(), flag(), prim(), run(), ScriptedSelect, test_ancestor_termination_closes_the_whole_substack() (+19 more)

## Knowledge Gaps
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SkillNode` connect `SkillNode` to `test_h6_hooks.py`, `test_evaluation.py`, `OptionEnv`, `eval_h7_preflight.py`, `.spec_hash`, `evaluation.py`, `diagnose_pickup_box.py`, `FixedAction`, `train_flat_baseline.py`, `pytest`, `test_c1_controls.py`, `json`, `build_state_view`, `.save`, `test_h6_accounting.py`, `node_trainer.py`, `audit_execution_mode.py`, `capture_trace`, `types.py`, `LateralMlpExtractor`, `test_option_env.py`, `analyse_h7_amortisation.py`, `CompositePolicyAdapter`, `column_policy.py`, `ENS 491 — Module Interface Specification`, `train_choice_task.py`, `door_open`, `eval_legacy_col01.py`, `Env`, `ToyWorld`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Are the 34 inferred relationships involving `SkillNode` (e.g. with `Code layout` and `ENS 491 — Module Interface Specification`) actually correct?**
  _`SkillNode` has 34 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_h6_hooks.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06475485661424607 - nodes in this community are weakly interconnected._
- **Why does `Executor` connect `column_policy.py` to `types.py`, `LateralMlpExtractor`, `OptionEnv`, `eval_h7_preflight.py`, `.spec_hash`, `evaluation.py`, `diagnose_pickup_box.py`, `FixedAction`, `ToyWorld`, `test_c1_controls.py`, `ENS 491 — Module Interface Specification`, `json`, `train_choice_task.py`, `SkillNode`, `audit_execution_mode.py`, `capture_trace`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `Executor` (e.g. with `Code layout` and `Core model (v2 §2–§6)`) actually correct?**
  _`Executor` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_evaluation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06187202538339503 - nodes in this community are weakly interconnected._
- **Why does `open_door()` connect `open_door` to `test_h6_hooks.py`, `eval_h7_preflight.py`, `.spec_hash`, `9. Corrections (H-6.8, 2026-10-02)`, `diagnose_pickup_box.py`, `train_flat_baseline.py`, `StateView`, `json`, `build_state_view`, `SkillNode`, `Run summary — H-7a (2026-10-02): stopped at H-7a.2 on amendment assumption A2b`, `LateralMlpExtractor`, `test_option_env.py`, `Part C — tasks`, `analyse_h7_amortisation.py`, `key_off_door`, `Plan — H-6: `{2,1} solve_doorkey`, the C1 baselines, and a genuine-choice composite`, `column_policy.py`, `ENS 491 — Target Architecture v2`, `Run summary — v2 migration, Step 0 → H-5 (2026-10-01)`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._