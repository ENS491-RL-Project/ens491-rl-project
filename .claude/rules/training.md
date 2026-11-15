---
paths:
  - "src/training/**"
  - "src/continual_learning/**"
  - "src/envs/**"
  - "src/skills/**"
  - "scripts/**"
  - "tests/**"
---

# Training, networks and stabilization

## Networks
- Primitive: `ColumnPolicy` with `LateralMlpExtractor`, MLP `147 → 64 → 64`, then SB3's policy (7 actions) and value heads. Optional laterals: `nn.Linear(64, 64)` from each hidden layer of a source column, **added** to the receiving hidden layer after its ReLU. No concatenation.
- Composite: standard MaskablePPO MLP policy on the same 147-dim observation, action space `Discrete(len(children))`. Composites use `n_steps = 64` (v2 §5.3).

## SB3 integration
- Subclass SB3's `ActorCriticPolicy` for primitives (as `ColumnPolicy` does) and keep SB3's training loop intact. Never reimplement PPO.
- Composites get their SMDP from the `OptionEnv` wrapper so MaskablePPO runs unmodified.
- **Stop and report before attempting a workaround if:** lateral activations are not accessible in the custom policy's forward pass; gradient flow through laterals is broken; VecEnv is incompatible with lateral activation sharing; `OptionEnv` cannot be vectorized because frozen child policies hold per-env state.

## Stabilization (v2 §5.3)
- Every `eval_every` training steps: 50 deterministic episodes in execution mode on the node's training env. Success = the node's `goal` reached before its `timeout`.
- The 50 stopping seeds are fixed and identical at every evaluation. The two consecutive evaluations must be at least one policy update apart: `eval_every ≥ n_steps * n_envs` (`EvaluationCallback` enforces it).
- At freeze time evaluate once on a second seed set never used for stopping; that is the success rate quoted for the node.
- Thresholds are experiment parameters: record them in each run's `meta.json`.
