# Supervisor Meeting — Open Questions

> This file is a meeting agenda, not implementation guidance.
> Claude Code does not need to read this file.
>
> Updated 2026-10-01 for Target Architecture v2. "Q-n" numbers match the ⚠️ markers in `ENS491_Target_Architecture_v2.md`.

---

## New questions (v2)

- [ ] **Q1 — Who defines a task?** Sub-skill goals ("agent carries the key", "door is open") and their start states are defined by us as environment wrappers; the agent learns each skill's policy and how to compose skills. Is this an acceptable scope for "the agent represents each task by initial conditions, terminal conditions and a policy", or do you expect sub-goals to be discovered autonomously?
- [ ] **Q2 — Hierarchy shape.** A composite skill may reuse any earlier frozen skill, from any level; its level is 1 + the highest level among its children, and one skill can serve several parents (a DAG). Does this match your notion of compounded tasks?
- [ ] **Q3 — New layer vs new column.** Proposed heuristic: when a new task arrives, first try to solve it with a new composite over existing skills; if that fails within a budget, learn a new primitive skill. We would report it as a heuristic with an ablation, not as a solved problem. Is that acceptable?
- [ ] **Q4 — Continual-learning claim.** Skills cannot be forgotten because they are frozen, so retention is guaranteed by design. We plan to present Progressive Network laterals as forward transfer, and to measure forgetting where it can actually happen (task recognition: AE and GRU). Is this framing acceptable, or do you expect a stronger continual-learning component for the skills themselves?
- [ ] **Q5 — Initiation sets.** Default: a skill may start anywhere; unusable choices are masked (goal already reached, or a required item missing). Should initiation conditions be explicit predicates for every skill?
- [ ] **Q6 — Evaluation tasks.** Proposed curriculum: Unlock → DoorKey → UnlockPickup → BlockedUnlockPickup, decomposed into get_key, open_door, go_to_goal, pickup_box, move_blocker. FourRooms is dropped from the main sequence. Are these environments appropriate?
- [ ] **Q7 — GRU continual training.** When a new task is added, retrain the GRU on stored sequences from all tasks, or use EWC on the GRU?

---

## v1 questions and the answers v2 proposes (please confirm or correct)

- *How does termination propagate in recursive MC→MC calls?* → Each skill ends on its own goal or timeout. An ending skill also ends everything below it, and control returns to its parent, which chooses again (v2 §4.2).
- *Do lateral connections stay active when a frozen column is used as an option?* → Yes. Frozen means no gradients; the forward pass still runs (v2 §5.4, §7).
- *Does a new task open a new layer or a new column?* → Reformulated as Q3. A layer is no longer chosen; it follows from which skills are composed.
- *GRU continual learning: EWC or retraining?* → Still open as Q7.
- *Task sequence approval (Empty → FourRooms → DoorKey → KeyCorridor → multi-key)* → Replaced by Q6.
- *Is the OPT-1 PoC approved (frozen PPO + meta-controller, no PN)?* → This is now the main path: frozen PPO primitives + a trained composite, with PN as a separate track.
- *Restricted initiation set or everywhere-initiation?* → Q5.
