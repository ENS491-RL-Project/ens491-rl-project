# Final test and paired runtime experiments — 2026-11-12

## Final-only evaluation (reserved; not opened)

`final_test_v1.json` contains 200 seed IDs, sampled once with system entropy
from [2^28, 2^30), after excluding all integer leaves in available `runs/` and
`output/` JSON records and [0, 100000). The audit is limited to available records;
unrecorded past use cannot be ruled out. MiniGrid can produce the same layout for
different seeds. Reserving these IDs does not prove layout-level generalization.

Use stopping seeds 10000–10049 during training. Existing held-out seeds 20000–20049
are development/validation data because their results have repeatedly informed
analysis. Do not use final seeds for stopping, choosing masks, hyperparameters,
checkpoint selection, debugging or the paired experiments below.

Before final evaluation, lock one prespecified configuration per task/arm and all
three training seeds (42, 43, 44). Select the last completed model from each
prespecified run: first-stable stop if achieved, otherwise the full budget's last
model. Report failure to stabilize; do not silently exclude failed seeds. Failed
or incomplete artifact archives cannot be selected. Benchmark/environment/hook,
observation normalization, execution mode (deterministic), child checkpoints and
training budget must be fixed before the lock. Do not choose the best seed.

Commands (from repo root, ens491 Python):

```sh
# Reservation already exists; do not regenerate it.
# For a future version only: python scripts/final_test_protocol.py reserve --output NEW_MANIFEST.json
python scripts/final_test_protocol.py lock --manifest docs/protocols/final_test_v1.json --runs RUN42 RUN43 RUN44 --output FINAL_LOCK.json
# Only after model selection is closed (not run in this study):
python scripts/eval_final.py --open --manifest docs/protocols/final_test_v1.json --lock FINAL_LOCK.json --candidate RUN_ID --output FINAL_RESULT.json
```

The reserve/lock commands never run evaluation. `evaluate_locked` in
`src/training/final_protocol.py` is the evaluation entry point; it checks node,
child, training configuration and evaluator fingerprints and writes a one-use claim before evaluation.
The protocol will remain reserved/unopened during this work: exploratory models
are not automatically promoted to final candidates.

Report each training seed's success fraction and Wilson 95% interval, mean
primitive-step length (including failures), and per-episode outcome/length arrays.
Also give across-training-seed mean/range. Do not report pooled episodes as 600
independent training replicates. If results influence further tuning, retire this
manifest from final use and reserve a new one with an explicit protocol version.

## Paired development experiment plan (specified before running)

Purpose: measure whether the runtime fixes alter learning inputs and outcomes;
identical weights are an admissible result, not a reason to change the experiment.

- Old runtime: executor.py and option_env.py from `917665e`.
- Fixed runtime: those files from `584b4be` (parent cutoff, nested mask, per-frame
  cap and dynamic iteration bound). Other code and child weights are identical.
- Separate processes; CPU, 2 torch threads, 8 envs, 64 decisions/env/rollout,
  batch_size=64, remaining PPO defaults unchanged. Training seeds 42,43,44.
- Fixed development budget: 5120 decisions = 10 complete rollout updates, unless
  the usual stability criterion stops earlier. Evaluation every 512 decisions,
  50 stopping episodes. This is a bounded sensitivity study, not a repetition of
  the original choice experiment's 100-update convergence claim.
- Arms: choice (`unlock_matching`, goal-only and strict, explicitly `open_door_tk`);
  baseline (`flat_doorkey_options`, goal-only); synthetic depth-3 budget task
  (goal-only). 4 conditions × 3 seeds × 2 runtimes = 24 training runs.
- Choice and baseline development scores: 20000–20049, no final seeds. Choice also
  records first-key accuracy by door colour on the development set.
- Depth task: MiniGrid Unlock observations with scripted frozen descendants;
  the root goal is reaching environment step 5. Root cap=2; intermediate and inner
  caps=2; primitive timeout=1. Root chooses the depth-2 child or a direct primitive.
  Old global child-executor accounting limits each deep invocation to one env step;
  new frame accounting allows four. Thus the old training task cannot reach its
  goal within two choices; the new one can. This deliberately constructed positive
  control does not establish useful real-task deep composition.
- Archive every attempted run including failures. Record source refs, child hashes,
  final weights and per-update buffer hashes, actual train calls, env-step cost,
  timeout overruns/cuts, nested mask changes and cap closures. Record the earliest
  mismatching update/transition. Compare old/new matched seeds; no significance
  claim from three training seeds.

## Controlled checks before training

1. Parent timeout: a child would reach the parent goal after its deadline; old
   training accepts it, fixed training/evaluation stop at the deadline.
2. Nested strict mask: a frozen selector initially prefers a forbidden child;
   verify fixed training and evaluation select the allowed child.
3. Per-frame cap: a depth-3 chain must reach the intended cap in each runtime;
   verify new training/evaluation agree and old/new differ as constructed.
4. Existing strict dead-end regression: zero env steps must finish TIMEOUT rather
   than exceed a fixed iteration guard.

Proceed to training only after these checks and the merged test suite pass.

The command currently rebuilds unlock_door, solve_doorkey, unlock_matching and
flat_doorkey_options composites from archived models and verified canonical child
checkpoints. Other tasks require a validated builder before locking. Claims live
beside the seed manifest, so a copied lock cannot silently reopen a candidate.
This is a local audit guard, not protection against deliberate file deletion.
