# Graph Report - ens491-rl-project  (2026-10-21)

## Corpus Check
- 186 files · ~312,064 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 56 file(s) not represented in the graph (top: .log 37, .pt 15, (none) 4)

## Summary
- 1698 nodes · 5230 edges · 78 communities (63 shown, 15 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 793 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e3bfb9e8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_h6_hooks.py
- test_evaluation.py
- Column
- New module layout
- h7_nodes.py
- TaskLifecycleManager
- .spec_hash
- execution_env_factory
- 9. Corrections (H-6.8, 2026-10-02)
- eval_h7_preflight.py
- test_initiation_does_not_mask
- pn_trainer.py
- open_door
- subgoal_env.py
- MetaController
- GRUTaskIdentifier
- pytest
- StateView
- test_c1_controls.py
- h7_nodes.py
- build_state_view
- .save
- test_h6_accounting.py
- AutoencoderModule
- types.py
- ColumnTrainer
- SkillNode
- node_trainer.py
- analyse_h7_amortisation.py
- audit_execution_mode.py
- ENS 491 — Recursive Hierarchy Design
- H-6.3 — `{2,1} solve_doorkey`, 3 seeds, with both masking controls (5 SP)
- types.py
- LateralMlpExtractor
- generate_report_figures.py
- test_node_registry.py
- test_option_env.py
- Part C — tasks
- SubgoalEnv
- Part C — tasks
- torch
- app.py
- ENS 491 — Project State & Roadmap
- New and changed modules
- MaskableMonitor
- .door
- StateView
- ColumnPolicy
- Session handoff — 2026-10-02 (H-6 corrections, H-6.9, H-7 up to the A2 stop)
- CompositePolicyAdapter
- Executor
- ENS 491 — Module Interface Specification
- ColumnTrainer
- ENS 491 — Target Architecture v2
- diagnose_choice.py
- Run summary — v2 migration, Step 0 → H-5 (2026-10-01)
- carrying
- analyse_choice_conditioning.py
- Scripted
- test_policy_choosing_a_masked_child_is_an_error
- apply_obs_pipeline
- make_training_vec_env
- test_executor_cannot_hang
- Execution-mode audit — what deterministic execution does across the library (evaluation only)
- PrimitivePolicyAdapter
- gui.md
- test_executor_never_branches_on_level
- Supervisor Meeting — Open Questions
- evaluation.py
- Excluded
- README.md
- ToyWorld
- Checkpoint metadata is incomplete

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
- `7. Progressive Networks: forward transfer, not forgetting 🔒` --references--> `LateralMlpExtractor`  [INFERRED]
  docs/ENS491_Target_Architecture_v2.md → src/continual_learning/column_policy.py
- `H-0 — v1 bug fixes (1 SP)` --references--> `LateralMlpExtractor`  [INFERRED]
  docs/plans/2026-10-01_v2_H0-H5_plan.md → src/continual_learning/column_policy.py
- `Answers to the plan's "Decision" items (recommended; edit before resuming if you disagree)` --references--> `ChannelNormalize`  [INFERRED]
  docs/plans/2026-10-01_v2_H0-H5_plan_review.md → src/envs/obs.py

## Import Cycles
- None detected.

## Communities (78 total, 15 thin omitted)

### Community 0 - "test_h6_hooks.py"
Cohesion: 0.09
Nodes (20): face_toward(), recolor_door(), remove_keys(), _keys(), _standable_from(), test_face_toward_is_idempotent_on_an_already_facing_pose(), test_face_toward_leaves_the_direction_alone_when_nothing_is_visible(), test_place_two_keys_rejects_a_layout_with_too_few_free_cells() (+12 more)

### Community 1 - "test_evaluation.py"
Cohesion: 0.06
Nodes (32): Code layout, Commands, Core model (v2 §2–§6), Docs, graphify, Stack, resolve_device(), EvalResult (+24 more)

### Community 2 - "Column"
Cohesion: 0.07
Nodes (17): ColumnPolicyAdapter, Column, _to_tensor(), ColumnOutput, MockMlpExtractor, MockPolicy, RecordingPolicy, test_column_1indexed_m() (+9 more)

### Community 3 - "New module layout"
Cohesion: 0.14
Nodes (12): Context, H-0.5 — amend the design docs (2 SP), H-0 — v1 bug fixes (1 SP), H-3 — EvaluationMonitor (3 SP), New module layout, Part B — Build plan, H-0 → H-5, Part C — H-6 / H-7 outline, Plan — Implement Target Architecture v2 (H-0 → H-5) + v2 design critique (+4 more)

### Community 4 - "h7_nodes.py"
Cohesion: 0.11
Nodes (25): matching_key(), main(), main_spec2(), main_trace(), readings(), summarise_trace(), chain(), stage_blocked() (+17 more)

### Community 5 - "TaskLifecycleManager"
Cohesion: 0.09
Nodes (17): run_demo(), TaskLifecycleManager, _to_tensor(), SystemState, _make_trained_column(), _MockPolicy, test_column_m_is_1indexed(), test_get_state_returns_system_state() (+9 more)

### Community 6 - ".spec_hash"
Cohesion: 0.18
Nodes (9): Deviations from the plan, and why, H-6.0 — F29 pre-flight for the depth-2 stack (1 SP), H-6.1 — composite loading, env-step accounting, thread control (3 SP), H-6.2 — retrain and re-freeze `{1,1} unlock_door` with `initiation` declared (1 SP), H-6.4 — primary C1 baseline: flat PPO on DoorKey-6x6, 3 seeds (3 SP), H-6.5 — secondary baseline: flat option set over the three primitives, 3 seeds (2 SP), H-6.7 — regression, docs, summary (2 SP), New and changed modules (+1 more)

### Community 7 - "execution_env_factory"
Cohesion: 0.21
Nodes (6): main(), probe(), stage_f29(), run_pair(), give_key_of(), execution_env_factory()

### Community 8 - "9. Corrections (H-6.8, 2026-10-02)"
Cohesion: 0.06
Nodes (31): 10.1 Exploratory: the two diagnostic RL arms (H-7 amendment 1, §8), 10. H-6.9 — why the selector did not learn a choice it can see, 1. Gate results, 2. Nodes, 4. The genuine-choice task: no choice learned in 100 updates, 6. Open questions this run answered, and one it did not, 7. How H-7 should test reuse, 9. Corrections (H-6.8, 2026-10-02) (+23 more)

### Community 9 - "eval_h7_preflight.py"
Cohesion: 0.10
Nodes (28): _adj(), _audit(), audit_h7a(), audit_move_blocker(), audit_pickup_box(), f29_row(), facts(), hold() (+20 more)

### Community 10 - "test_initiation_does_not_mask"
Cohesion: 0.20
Nodes (3): FirstAvailable, FixedAction, test_initiation_does_not_mask()

### Community 11 - "pn_trainer.py"
Cohesion: 0.07
Nodes (11): _make_env(), _pca_2d(), _visualize(), _make_column_with_policy(), _make_env(), test_flat_obs_shape(), test_freeze_sets_flag_and_blocks_grad(), test_frozen_column_still_produces_activations() (+3 more)

### Community 12 - "open_door"
Cohesion: 0.06
Nodes (39): Amendments applied 2026-10-02 (H-7 amendment 1), Candidate curriculum (⚠️ Q6), H-2 — predicates, SubgoalEnv, obs pipeline (5 SP), H-4 — train and freeze T1, T2, T4 (5 SP), H-6.6 — the genuine-choice composite (5 SP), Part C — the genuine-choice task, designed, Predicted numbers, written before the run, The nodes (+31 more)

### Community 13 - "subgoal_env.py"
Cohesion: 0.10
Nodes (19): Amendments applied 2026-10-02 (H-6.8), pose_in_room(), fn(), _adjacent(), _dir_sees(), _door_approachable(), fn(), hook_spec() (+11 more)

### Community 14 - "MetaController"
Cohesion: 0.13
Nodes (11): MetaController, MetaControllerOutput, _make_option(), MockPolicy, test_add_option_increases_count(), test_layer_index_stored(), test_select_option_id_in_valid_range(), test_select_option_raises_without_options() (+3 more)

### Community 15 - "GRUTaskIdentifier"
Cohesion: 0.11
Nodes (11): GRUTaskIdentifier, _to_tensor(), GRUOutput, test_forward_returns_gru_output(), test_get_sequence_zero_pads_short_buffer(), test_observe_and_get_sequence(), test_register_new_task_expands_head(), test_register_new_task_idempotent() (+3 more)

### Community 16 - "pytest"
Cohesion: 0.12
Nodes (23): flag(), prim(), sv(), test_at_most_one_lateral_source(), test_availability_cache_evaluates_a_shared_child_once(), test_availability_without_cache_still_correct(), test_available_is_goal_only_not_initiation(), test_check_termination_success_timeout_none() (+15 more)

### Community 17 - "StateView"
Cohesion: 0.13
Nodes (27): relocate_blocker(), fn(), carrying_nothing(), door_approach_blocked(), door_approach_clear(), _adjacent(), _blocked_pose(), _env() (+19 more)

### Community 18 - "test_c1_controls.py"
Cohesion: 0.09
Nodes (23): H-7.4 — `{2,2} unlock_pickup`, 3 seeds, five arms (5 SP), capture_trace(), mask_divergence(), RandomOverAll, run_arms(), _child(), FirstAvailable, _graph() (+15 more)

### Community 19 - "h7_nodes.py"
Cohesion: 0.14
Nodes (21): load_library(), load_all(), main(), prepare_with(), main(), primitive(), load_choice_children(), load_door_opener() (+13 more)

### Community 20 - "build_state_view"
Cohesion: 0.09
Nodes (25): Task detection (AE + GRU), ChannelNormalize, make_base_env(), normalize_flat_obs(), build_state_view(), raw(), test_carrying_predicate_truth_table(), test_channel_normalize_rejects_an_unflattened_input() (+17 more)

### Community 21 - ".save"
Cohesion: 0.24
Nodes (4): _slug(), SpecMismatch, test_changed_goal_predicate_is_refused(), test_changed_timeout_is_refused()

### Community 22 - "test_h6_accounting.py"
Cohesion: 0.12
Nodes (15): children_env_steps(), _composite(), FirstChild, _primitive(), test_children_env_steps_counts_a_shared_child_once(), test_children_env_steps_is_zero_for_a_primitive(), test_children_env_steps_sums_primitive_and_composite_costs(), test_children_env_steps_understates_rather_than_guesses() (+7 more)

### Community 23 - "AutoencoderModule"
Cohesion: 0.14
Nodes (10): AutoencoderModule, _to_tensor(), AEOutput, test_compute_errors_returns_per_sample_list(), test_encode_decode_shape(), test_forward_returns_ae_output(), test_novelty_flag_respects_threshold(), test_train_on_batch_reduces_loss() (+2 more)

### Community 25 - "ColumnTrainer"
Cohesion: 0.15
Nodes (4): main(), ColumnTrainer, test_save_creates_expected_files(), test_train_until_stable_exits_on_max_total()

### Community 26 - "SkillNode"
Cohesion: 0.15
Nodes (13): Hard rules, Track H — Hierarchy (critical path), 10. Code migration map (current repo → v2), 12. Roadmap: four tracks, 2. Skill Node 🔒, 3. Graph structure: a DAG 🔒, A.3 Node model, graph, recognition, PN, growth (§2, §3, §6, §7, §8), H-1 — SkillNode, StateView, Executor, registry (8 SP) (+5 more)

### Community 27 - "node_trainer.py"
Cohesion: 0.10
Nodes (33): 5. Deviations from the plan, and why, C.8 Stale text and dead configuration, H-7.3 — `{0,4} pickup_box` (3 SP), main(), main(), deepest_trace(), heldout(), run() (+25 more)

### Community 28 - "analyse_h7_amortisation.py"
Cohesion: 0.05
Nodes (50): `{0,4} pickup_box`, specification 2 (H-7a.2), `{0,9} drop_held` (H-7a.2), 10.1 Success rates, 10.2 The five failure counts (they may overlap), 10.3 Reading (the rules fixed in the close-out before the run), 10. H-7a.5: where and how the spec-2 checkpoint fails (close-out §4, evaluation only), 11. Close-out deviations (`plans/2026-10-02_H7a_closeout.md`), 12.1 E.0: what `STOCHASTIC` samples (+42 more)

### Community 29 - "audit_execution_mode.py"
Cohesion: 0.08
Nodes (27): crosscheck(), e0(), e3(), flat_record(), flat_saved_models(), load_pickup_box_training(), main(), materialise() (+19 more)

### Community 30 - "ENS 491 — Recursive Hierarchy Design"
Cohesion: 0.09
Nodes (22): 10. Open Questions — Supervisor Meeting Agenda, 1. Notation, 2. The Dual Role of a Column, 3.1 Tree structure 🔒, 3.2 Meta-controller scope 🔒, 3. Hierarchy Structure, 4.1 Definition 🔒, 4.2 Scope constraint 🔒 (+14 more)

### Community 31 - "H-6.3 — `{2,1} solve_doorkey`, 3 seeds, with both masking controls (5 SP)"
Cohesion: 0.33
Nodes (3): H-6.3 — `{2,1} solve_doorkey`, 3 seeds, with both masking controls (5 SP), H-7.6 — T9a and T9b, 3 seeds each (5 SP), env_success()

### Community 32 - "types.py"
Cohesion: 0.14
Nodes (5): TraceEvent, CompositePolicy, PrimitivePolicy, DoorView, ObjView

### Community 33 - "LateralMlpExtractor"
Cohesion: 0.15
Nodes (6): 5.1 Primitive nodes 🔒, 5.2 Composite nodes: SMDP via `OptionEnv` 🔒, 5.3 Stabilization criterion (replaces mean > 0.85, std < 0.05) 🔒, 5. Learning semantics, H-5 — OptionEnv + MaskablePPO, train T3 (8 SP), LateralMlpExtractor

### Community 34 - "generate_report_figures.py"
Cohesion: 0.14
Nodes (10): fig1_architecture(), pb(), rbox(), fig2_training_curves(), fig3_experiment_pipeline(), fig4_copy_activations(), fig5_test_summary(), fig6_milestone_timeline() (+2 more)

### Community 35 - "test_node_registry.py"
Cohesion: 0.27
Nodes (7): make_node(), test_composite_hash_covers_its_children(), test_list_nodes(), test_round_trip_restores_weights(), test_save_writes_three_artifacts_and_a_full_meta(), test_unfrozen_checkpoint_is_refused_by_default(), TinyPolicy

### Community 36 - "test_option_env.py"
Cohesion: 0.14
Nodes (18): make_option_env(), test_ablation_is_scored_under_the_mask_it_was_trained_with(), test_action_space_is_the_child_count(), test_child_outcome_and_k_are_logged_every_decision(), test_dummyvecenv_n4_matches_n1_for_a_scripted_policy(), make_option_env_factory(), _f(), rollout() (+10 more)

### Community 37 - "Part C — tasks"
Cohesion: 0.14
Nodes (12): Assumptions, stated so a contradiction is recognisable, Constraints (from CLAUDE.md and v2; every task includes them), Execution rules, Goal, H-7.7 — collect the flat baselines (1 SP), How the result is worded, Part A — design amendments (before any code, one commit), Plan — H-7: the reuse experiment (C2) on UnlockPickup and BlockedUnlockPickup (+4 more)

### Community 38 - "SubgoalEnv"
Cohesion: 0.16
Nodes (12): make_subgoal_env(), SubgoalEnv, make_node(), test_subgoal_env_observation_contract(), test_subgoal_env_refuses_a_goal_that_holds_at_reset(), test_subgoal_env_reset_hook_is_visible_in_the_observation(), test_subgoal_env_reward_only_on_its_own_goal(), test_subgoal_env_rewards_the_goal_with_the_right_shape() (+4 more)

### Community 39 - "Part C — tasks"
Cohesion: 0.29
Nodes (7): H-7.0 — Part A amendments (1 SP), H-7.1 — pre-flight: facts, hooks, solvability, first transfer (3 SP), H-7.2 — launch the flat baselines in the background (2 SP), H-7.5 — `{0,5} move_blocker`, and the transfer measurement that decides T9 (5 SP), H-7.8 — the amortisation table, regression, docs, summary (3 SP), Part C — tasks, Step 0 — precondition

### Community 40 - "torch"
Cohesion: 0.20
Nodes (9): _make_option(), MockPolicy, test_option_action_in_valid_range(), test_option_can_initiate_always_true(), test_option_id_matches_column_coordinates(), test_option_not_terminated_before_max_steps(), test_option_reset_clears_step_count(), test_option_step_returns_option_step_output() (+1 more)

### Community 41 - "app.py"
Cohesion: 0.11
Nodes (13): _make_trained_column(), _MockMlpExtractor, _MockPolicy, _get_tlm(), _live_state_dict(), _load_demo_state(), main(), _render_ae() (+5 more)

### Community 42 - "ENS 491 — Project State & Roadmap"
Cohesion: 0.14
Nodes (13): Answered by H-6 (2026-10-02), Critical Path, ENS 491 — Project State & Roadmap, Locked Decisions, Open Empirical Questions, Roadmap Status, Story Points Reference, Stretch (+5 more)

### Community 43 - "New and changed modules"
Cohesion: 0.22
Nodes (8): Assumptions, stated so a contradiction is recognizable, Context, Execution rules for this plan, Part A — design amendments (before any code), Plan — H-6: `{2,1} solve_doorkey`, the C1 baselines, and a genuine-choice composite, Risks accepted, and what would change the plan, Verification, Verified facts this plan is built on

### Community 45 - ".door"
Cohesion: 0.33
Nodes (4): fn(), fn(), fn(), fn()

### Community 46 - "StateView"
Cohesion: 0.11
Nodes (10): 4.1 Options persist until they terminate, 4.2 Termination, 4.3 Availability, 4.4 The executor (depth-agnostic), 4. Execution semantics 🔒, Amendments applied 2026-10-01 (H-0.5), A.1 Execution semantics (§4) — the ones that would bite first, A.4 Cross-cutting (environment, repo, process) (+2 more)

### Community 47 - "ColumnPolicy"
Cohesion: 0.17
Nodes (8): Networks, SB3 integration, Stabilization (v2 §5.3), Training, networks and stabilization, Current Status (2026-10-01), ✅ Done (v1 phase, Feb–May 2026), ⚠️ Known limitations of the v1 code (to be replaced, see v2 §10), ColumnPolicy

### Community 48 - "Session handoff — 2026-10-02 (H-6 corrections, H-6.9, H-7 up to the A2 stop)"
Cohesion: 0.13
Nodes (14): 1. Step B: corrections to the H-6 summary (`5a4c353`), 2. Step C: H-6.9 (`79426d2`, H-6 summary §10), 3. Step D: H-7, 4. Written but never run, 5. The open decision (the author's, not the agent's), 6. Environment notes (learned the hard way this session), Commits of this session (oldest first), Documents to read, in order (+6 more)

### Community 49 - "CompositePolicyAdapter"
Cohesion: 0.16
Nodes (7): CompositePolicyAdapter, option_env_factory(), _make(), _real_composite(), test_load_frozen_composite_refuses_a_changed_child(), test_load_frozen_composite_rejects_an_unfrozen_child(), test_load_frozen_composite_round_trips()

### Community 50 - "Executor"
Cohesion: 0.07
Nodes (22): 5.4 Freezing and execution mode 🔒, A.2 Learning semantics (§5), 3. The C1 result, stated plainly, The masking problem is not solved by depth, mask_divergence(), episode(), sequence_episode(), _find() (+14 more)

### Community 51 - "ENS 491 — Module Interface Specification"
Cohesion: 0.15
Nodes (12): Common Types, ENS 491 — Module Interface Specification, Entry point (every step), Hard Constraints, Module 1: Autoencoder (AE), Module 2: GRU Task Identifier, Module 3: Progressive Networks Column, Module 4: Option Wrapper (+4 more)

### Community 52 - "ColumnTrainer"
Cohesion: 0.12
Nodes (7): Answers to the plan's "Decision" items (recommended; edit before resuming if you disagree), Corrections to the plan (verified against the code), Fact to confirm before execution, Resume prompt (paste into Claude Code, in default or plan mode, not auto mode), Review of the H-0 → H-5 plan (2026-10-01), _EpisodeRewardCallback, StabilizationMonitor

### Community 53 - "ENS 491 — Target Architecture v2"
Cohesion: 0.13
Nodes (14): 0. Why v2 exists, 11. Evaluation plan: claims → experiments, 13. Changelog v1 → v2, 1. Separation of concerns 🔒, 6.1 Autoencoder: "is this situation new?", 6.2 GRU: "which known task is this?", 6.3 Where continual-learning forgetting actually happens, 6. Task recognition (AE + GRU) (+6 more)

### Community 54 - "diagnose_choice.py"
Cohesion: 0.11
Nodes (18): collect(), decision1(), first_key_accuracy(), fit_logreg(), fit(), fit_selector(), fit_selector_fixed(), key_probabilities() (+10 more)

### Community 55 - "Run summary — v2 migration, Step 0 → H-5 (2026-10-01)"
Cohesion: 0.29
Nodes (6): Gate results, Repository state, Run summary — v2 migration, Step 0 → H-5 (2026-10-01), Things worth knowing before H-6, Trained nodes, What the C1 controls actually show

### Community 56 - "carrying"
Cohesion: 0.11
Nodes (12): 8. Repository state, Predicate, all_of(), carrying(), door_open(), _real_children(), primitive(), test_only_frozen_children_may_be_trained_over() (+4 more)

### Community 57 - "analyse_choice_conditioning.py"
Cohesion: 0.31
Nodes (6): arm_row(), fisher_two_sided(), p(), h68_row(), main(), row()

### Community 59 - "test_policy_choosing_a_masked_child_is_an_error"
Cohesion: 0.25
Nodes (3): ExecutorError, ScriptedSelect, test_policy_choosing_a_masked_child_is_an_error()

### Community 60 - "apply_obs_pipeline"
Cohesion: 0.18
Nodes (8): env_factory(), apply_obs_pipeline(), ResetHookEnv, _make(), _make(), test_raw_pipeline_remains_available_for_the_legacy_column(), hooked_env(), test_reset_hook_env_regenerates_the_observation()

### Community 63 - "Execution-mode audit — what deterministic execution does across the library (evaluation only)"
Cohesion: 0.29
Nodes (6): 1. Why, 2. Seeds, 3. Measurements, 4. Wording rules, 5. Verify, Execution-mode audit — what deterministic execution does across the library (evaluation only)

### Community 72 - "Supervisor Meeting — Open Questions"
Cohesion: 0.50
Nodes (3): New questions (v2), Supervisor Meeting — Open Questions, v1 questions and the answers v2 proposes (please confirm or correct)

### Community 73 - "evaluation.py"
Cohesion: 0.15
Nodes (5): main(), RandomOverAll, _wilson(), wilson_upper(), RandomOverAvailable

### Community 77 - "ToyWorld"
Cohesion: 0.15
Nodes (24): Outcome, build_depth5(), build_ladder(), comp(), flag(), prim(), run(), test_ancestor_termination_closes_the_whole_substack() (+16 more)

### Community 87 - "Checkpoint metadata is incomplete"
Cohesion: 0.50
Nodes (3): Checkpoint metadata is incomplete, PN-4 run log notes (written 2026-10-01, H-0), Stabilization threshold mismatch

## Knowledge Gaps
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SkillNode` connect `SkillNode` to `test_h6_hooks.py`, `test_evaluation.py`, `h7_nodes.py`, `.spec_hash`, `test_initiation_does_not_mask`, `subgoal_env.py`, `pytest`, `test_c1_controls.py`, `h7_nodes.py`, `build_state_view`, `.save`, `test_h6_accounting.py`, `node_trainer.py`, `audit_execution_mode.py`, `types.py`, `test_node_registry.py`, `test_option_env.py`, `SubgoalEnv`, `StateView`, `CompositePolicyAdapter`, `Executor`, `ENS 491 — Module Interface Specification`, `diagnose_choice.py`, `carrying`, `make_training_vec_env`, `.spec`, `evaluation.py`, `ToyWorld`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Are the 34 inferred relationships involving `SkillNode` (e.g. with `Code layout` and `ENS 491 — Module Interface Specification`) actually correct?**
  _`SkillNode` has 34 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_h6_hooks.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08571428571428572 - nodes in this community are weakly interconnected._
- **Why does `Column` connect `Column` to `LateralMlpExtractor`, `New module layout`, `TaskLifecycleManager`, `torch`, `app.py`, `pn_trainer.py`, `MetaController`, `ColumnPolicy`, `ColumnTrainer`, `types.py`, `ColumnTrainer`, `node_trainer.py`, `ENS 491 — Recursive Hierarchy Design`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `Executor` (e.g. with `Code layout` and `Core model (v2 §2–§6)`) actually correct?**
  _`Executor` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_evaluation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05608322026232474 - nodes in this community are weakly interconnected._
- **Why does `open_door()` connect `open_door` to `h7_nodes.py`, `.spec_hash`, `9. Corrections (H-6.8, 2026-10-02)`, `eval_h7_preflight.py`, `subgoal_env.py`, `StateView`, `h7_nodes.py`, `build_state_view`, `SkillNode`, `analyse_h7_amortisation.py`, `LateralMlpExtractor`, `test_option_env.py`, `Part C — tasks`, `SubgoalEnv`, `Part C — tasks`, `New and changed modules`, `StateView`, `Session handoff — 2026-10-02 (H-6 corrections, H-6.9, H-7 up to the A2 stop)`, `Run summary — v2 migration, Step 0 → H-5 (2026-10-01)`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._