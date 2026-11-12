# Glossary

| Term | Meaning | v2 |
|---|---|---|
| Skill node | Everything the agent knows: initiation, goal, timeout, policy, children. | §2 |
| Primitive node | Node with no children; its policy outputs MiniGrid actions. Level 0. | §2 |
| Composite node | Node whose policy picks one of its frozen children. Its policy is its meta-controller. | §2 |
| Level `n` | `1 + max(child levels)`, derived, never assigned. | §2 |
| DAG | A node may have many parents; children come from any level. | §3 |
| Frame | One active node on the executor's stack, with its own step and decision counters. (per-frame decision counter after PR #15) | §4 |
| Availability | A node may be chosen if its goal is not reached and (for composites) some child is available. | §4 |
| Strict mask / initiation masking | Also require the child's initiation predicate. A control condition, off in normal runs. | §4, Q8 |
| `max_decisions` | Cap on a composite's child selections, counted per frame; mandatory. See decision 0009. | §4 |
| SMDP / `OptionEnv` | A composite is trained on an env where one step = one whole child run. | §5.2 |
| Execution mode | Whether frozen nodes act deterministically or by sampling. Open. | §5.4 |
| DECLARED → TRAINING → STABLE → FROZEN | Node lifecycle. Only FROZEN nodes become children or roots. | §9 |
| STABLE | Decision 0004: SR ≥ 0.90 and Wilson LB ≥ 0.80 in two evaluations ≥ 1 update apart. | §5.3 |
| Lateral connection | PN link from an earlier frozen primitive into a new primitive's hidden layer. | §7 |
| AE / GRU | Autoencoder: "is this task new?". GRU: "which known task is this?" → chooses the root node. | §6 |
| Phase A / Phase B growth | Hand-declared tasks now; composite-first heuristic later. | §8 |
