"""
H-6.6: the genuine-choice task -- T10a, T10b, and T11 `{1,2} unlock_matching`.

Why this exists, in one paragraph. H-5 measured a random policy under the strict
initiation mask matching `{1,1} unlock_door` exactly, and H-6.3 measured the
same thing at depth 2 on `{2,1} solve_doorkey`: both tasks have a strict subgoal
order, the initiation mask *is* that order, so success rate cannot separate
composition from masking however deep the hierarchy goes. The fix is not a deeper
task but a task with a real choice in it.

Two children are valid at the same decision point and the right one depends on
the observation:

    decision 1 (empty-handed, door locked)
      get_key_yellow   goal false, initiation (carrying nothing) TRUE   available
      get_key_blue     goal false, initiation (carrying nothing) TRUE   available
      open_door        goal false, initiation (carrying a key)  FALSE   masked

    decision 2 (the key in hand)
      the strict mask forces open_door, as it should

So the choice lives entirely at decision 1, where the door's colour in the
observation decides it, and `Door.toggle` is colour-matched -- plus `pickup`
needs an empty hand and neither key primitive is trained to drop, so a wrong
pick is unrecoverable within the episode. The masked random control is therefore
capped near 0.50 by construction, while a policy that reads the colour channel
is not. That gap is the measurement.

Stages:
    --stage primitives   train and freeze {0,6} get_key_yellow, {0,7} get_key_blue
    --stage f29          score {0,2} open_door from the state they leave behind
    --stage composite    train {1,2} unlock_matching and run all five arms

Usage:
    python scripts/train_choice_task.py --stage primitives --smoke
    python scripts/train_choice_task.py --stage primitives
    python scripts/train_choice_task.py --stage f29
    python scripts/train_choice_task.py --stage composite --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import (  # noqa: E402
    compose,
    give_key_of,
    randomize_agent_pose,
    two_keys,
)
from src.skills.executor import Executor  # noqa: E402
from src.skills.node import SkillNode  # noqa: E402
from src.skills.predicates import (  # noqa: E402
    all_of,
    carrying,
    carrying_nothing,
    door_locked,
    door_open,
)
from src.skills.registry import NodeRegistry  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.c1_controls import ARM_LABELS, capture_trace, mask_divergence, run_arms  # noqa: E402
from src.training.device import describe  # noqa: E402
from src.training.evaluation import (  # noqa: E402
    StabilityCriterion,
    evaluate_node,
    heldout_seeds,
)
from src.training.node_trainer import (  # noqa: E402
    execution_env_factory,
    freeze_and_save,
    load_frozen_primitive,
    pooled_eval_fn,
    set_torch_threads,
    train_composite,
    train_primitive,
)
from src.types import NodeStatus  # noqa: E402
from train_primitives import DOORKEY, UNLOCK, build_node, specs  # noqa: E402

RUNS = REPO / "runs" / "v2"
COLORS = ("yellow", "blue")

# [A: F13] children are 120 (the key primitives) and 60 (open_door), so the
# parent must outlive 120; DoorKey-6x6's max_steps is 360.
T11_TIMEOUT = 240
T11_MAX_DECISIONS = 6
KEY_TIMEOUT = 120

GATE = 0.90

# A per-node stopping rule, recorded in the node's meta.json. See
# open_door_tk_spec() for why this node needs one.
STRICT_STOP = dict(min_success_rate=0.94, min_wilson_lower=0.85, n_consecutive=3)


def choice_specs():
    """
    T10a and T10b. Same hook as the parent's reset distribution, deliberately.

    `randomize_carrying` is **not** in this hook, and that is the one place where
    H-4's lesson is intentionally not applied. Randomizing the carried object
    would teach the primitive to cope with a full hand -- most likely by dropping
    -- which would make a wrong first pick recoverable and erase the very signal
    this task is built to produce. The agent is always empty-handed at the only
    decision where these children are selected, so the training distribution
    already matches the parent's handoff exactly.
    """
    hook = two_keys(COLORS)
    return {
        "get_key_yellow": dict(
            node_id="{0,6} get_key_yellow", m=6,
            goal=carrying("key", "yellow"),
            initiation=carrying_nothing(),
            timeout=KEY_TIMEOUT, env_ids=[DOORKEY], hook=hook,
        ),
        "get_key_blue": dict(
            node_id="{0,7} get_key_blue", m=7,
            goal=carrying("key", "blue"),
            initiation=carrying_nothing(),
            timeout=KEY_TIMEOUT, env_ids=[DOORKEY], hook=hook,
        ),
    }


def open_door_tk_spec():
    """
    T12 `{0,8} open_door_tk` -- `open_door` retrained for the two-key env.

    [A: F29, plan branch A4] `{0,2} open_door` scores 1.00 on its own hook
    distribution and **0.73** from the state the choice task's key primitives
    actually leave: a key in hand, a second key still on the floor, and a door
    that is blue half the time. It had never seen a spare key lying around. The
    plan's prescribed response is to train a new node on the wider mix and leave
    `{0,2}` frozen and untouched -- a node's weights are never modified, so
    "fixing" open_door is not an option (CLAUDE.md).

    Trained on **DoorKey-6x6 only**, under the two-key hook plus a fully
    randomized pose. The first attempt trained on Unlock and DoorKey together,
    which doubled the difficulty for no benefit -- the composite only ever runs
    on DoorKey -- and stopped at a held-out 0.84 with 0.80 at the real handoff.
    Diagnosed rather than assumed: the shortfall was not an F29 facing or layout
    gap (0.82 on its own hook with the door in view, 0.82 with the facing
    randomized, 0.78 at the handoff) but plain stopping-set optimism, so the
    response is the plan's other prescribed one -- train inside the parent's own
    env with the parent's reset distribution -- plus a stricter stop.

    Held to the project's **standard** stopping rule, not a stricter one. A
    stricter rule (STRICT_STOP below) was tried and abandoned: this node plateaus
    at exactly 46/50 on the stopping seeds, so 0.94-three-times is simply never
    reached and a 600k-step run ends with nothing frozen. The residual 4 failures
    are not the blocked-door-approach bug either -- that bug was real, cost ~15%
    of layouts, and is fixed in place_two_keys, but the plateau is identical with
    and without it. Freezing at 0.92 with the standard rule and reporting the
    held-out number is the honest option; inventing a threshold this node happens
    to clear would not be.
    """
    return dict(
        node_id="{0,8} open_door_tk", m=8,
        goal=door_open(),
        initiation=carrying("key"),
        timeout=60,
        env_ids=[DOORKEY],
        hook=compose(two_keys(COLORS), give_key_of("matching"),
                     randomize_agent_pose(anchor=None)),
    )


def build_t11(children):
    return SkillNode(
        "{1,2} unlock_matching",
        children=children,
        goal=door_open(),
        # [H-6 A.3] Nothing to unlock unless the door is locked, and the choice
        # this node exists to make is only available with an empty hand.
        initiation=all_of(door_locked(), carrying_nothing()),
        timeout=T11_TIMEOUT,
        max_decisions=T11_MAX_DECISIONS,
        env_ids=[DOORKEY],
        m=2,
    )


def load_choice_children(registry, device="cpu", third="auto"):
    """
    The two key primitives, plus a door-opening child.

    `third` picks that child: "open_door" reuses `{0,2}` unchanged, "open_door_tk"
    uses the retrained `{0,8}`, and "auto" prefers `{0,8}` when it is frozen on
    disk. The F29 check decides which is honest, so the choice is explicit and
    recorded in the run's JSON rather than inferred.
    """
    kids = []
    for name, spec in choice_specs().items():
        node = build_node(spec)
        load_frozen_primitive(node, registry, DOORKEY, spec["hook"], device=device)
        if node.status is not NodeStatus.FROZEN:
            raise RuntimeError(f"{node.id} did not load as FROZEN")
        print(f"  loaded {node}  spec_hash={node.spec_hash()}")
        kids.append(node)

    kids.append(load_door_opener(registry, device=device, which=third))
    return kids


def load_door_opener(registry, device="cpu", which="auto"):
    candidates = []
    if which in ("auto", "open_door_tk"):
        candidates.append(open_door_tk_spec())
    if which in ("auto", "open_door"):
        candidates.append(specs()["open_door"])
    last = None
    for spec in candidates:
        node = build_node(spec)
        try:
            load_frozen_primitive(node, registry, spec["env_ids"][0], spec["hook"],
                                  device=device)
        except Exception as exc:
            last = exc
            print(f"  (not available: {node.id} -- {type(exc).__name__})")
            continue
        print(f"  loaded {node}  spec_hash={node.spec_hash()}")
        return node
    raise RuntimeError(f"no door-opening child available: {last}")


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def all_trainable_specs():
    out = dict(choice_specs())
    out["open_door_tk"] = open_door_tk_spec()
    return out


def stage_primitives(args, registry) -> dict:
    reports = {}
    available = all_trainable_specs()
    names = args.only or list(choice_specs())
    for name in names:
        spec = available[name]
        node = build_node(spec)
        print(f"\n{'=' * 72}\n{node.id}  goal={node.goal.spec}  "
              f"envs={spec['env_ids']}  timeout={node.timeout}"
              f"\n  hook={spec['hook'].spec}\n{'=' * 72}")
        crit = spec.get("criterion")
        if crit:
            print(f"  stopping rule (overridden, recorded in meta.json): {crit}")
        report, _model = train_primitive(
            node, spec["env_ids"], spec["hook"],
            total_timesteps=args.budget, eval_every=args.eval_every,
            n_eval=args.n_eval, n_envs=args.n_envs, seed=args.seed,
            device=args.device, verbose=1,
            criterion=StabilityCriterion(**crit) if crit else None,
        )
        print(f"  steps={report['steps_trained']:,} wall={report['wall_seconds']}s "
              f"({report['steps_per_second']} steps/s) stable={report['stable']}")
        if report["stable"]:
            report = freeze_and_save(node, registry, report,
                                     env_ids=spec["env_ids"],
                                     reset_hook=spec["hook"],
                                     n_eval=args.n_eval)
            print(f"  FROZEN. held-out: {report['heldout']}")
        else:
            print("  not stable within budget -- not frozen")
            registry.save(node, extra_meta=report)
        reports[name] = report
    return reports


def roll(node, env, obs):
    version = 0
    sv = build_state_view(env, version)
    ex = Executor(node)
    for _ in range(node.timeout):
        action = ex.act(obs, sv)
        if action is None:
            break
        obs, _, term, trunc, _ = env.step(action)
        version += 1
        sv = build_state_view(env, version)
        ex.observe_reward(0.0)
        if term or trunc:
            break
    return obs


def stage_f29(args, registry) -> dict:
    """
    [A: F29] Score `{0,2} open_door` from the state the key primitives leave.

    `open_door` was trained on Unlock and DoorKey-6x6 holding the layout's single
    key, with no other key on the floor. In the two-key env it is handed a grid
    with a second key still lying around, and on half the episodes a blue door --
    both inputs it has never seen in that combination. Unlock does randomize its
    door colour (minigrid 3.1.0 `add_door`), which is why reuse is plausible at
    all, but plausible is not measured.

    Gating rows: handed the MATCHING key. Reported but not gating: handed the
    decoy, where failing is correct -- that is the signal the parent learns from.
    """
    print("loading the frozen children:")
    yellow, blue, open_door_node = load_choice_children(
        registry, device=args.device, third=args.third_child)
    hook = two_keys(COLORS)
    seeds = heldout_seeds(args.n_eval)
    rows = []

    def run_pair(pick: str):
        """
        Roll the key primitive this `pick` names, then score the door opener.

        `pick` is "matching" or "decoy". Rolling the *decoy* child is not a
        curiosity: it is the state the composite reaches when it chooses wrong,
        and the door must NOT open there. A door opener that somehow succeeded
        with the wrong key would mean the choice task has no teeth.
        """
        env = execution_env_factory(DOORKEY, hook)()
        ok = n = child_failed = 0
        try:
            for seed in seeds:
                obs, _ = env.reset(seed=seed)
                sv = build_state_view(env, 0)
                colour = sv.door().color
                matches_yellow = (colour == "yellow")
                want_yellow = matches_yellow if pick == "matching" else not matches_yellow
                obs = roll(yellow if want_yellow else blue, env, obs)
                sv = build_state_view(env, 1)
                if sv.carrying_type != "key":
                    child_failed += 1       # the key primitive failed; not this test
                    continue
                obs = roll(open_door_node, env, obs)
                sv = build_state_view(env, 2)
                n += 1
                ok += bool(door_open()(sv))
        finally:
            env.close()
        return ok, n, child_failed

    for label, pick, gate in (
        ("[GATE] holding the matching key", "matching", True),
        ("[expected fail] holding the decoy key", "decoy", False),
    ):
        ok, n, failed = run_pair(pick)
        rows.append({
            "start_state": label, "gate": gate, "n_episodes": n, "successes": ok,
            "success_rate": round(ok / n, 4) if n else 0.0,
            "key_primitive_failed": failed,
        })

    matched = next(r for r in rows if r["gate"])
    mismatched = next(r for r in rows if not r["gate"])
    matched_sr = matched["success_rate"]
    mismatch_sr = mismatched["success_rate"]

    print(f"\n{'start state':<42}{'n':>5}{'SR':>7}   gate")
    print("-" * 64)
    for r in rows:
        print(f"{r['start_state']:<42}{r['n_episodes']:>5}{r['success_rate']:>7.2f}"
              f"   {'GATE' if r['gate'] else '-'}")

    out = {"node": open_door_node.id, "spec_hash": open_door_node.spec_hash(),
           "env_id": DOORKEY, "hook": hook.spec,
           "rows": rows, "gate_threshold": GATE}
    if matched_sr < GATE:
        print(f"\nGATE FAILED: {open_door_node.id} scores {matched_sr:.2f} holding the matching "
              f"key in the two-key env.\nPlan H-6.6 branch (A4): train {{0,8}} "
              f"open_door_tk on the wider mix and use it as the third child. "
              f"{{0,2}} open_door itself is never modified.")
        out["gate_passed"] = False
    else:
        print(f"\nGATE PASSED: {open_door_node.id} works in the two-key env "
              f"({matched_sr:.2f} holding the matching key). Reused unchanged.")
        print(f"  (non-gating) holding the decoy: {mismatch_sr:.2f} -- failing "
              f"here is correct, and is what the composite learns from")
        out["gate_passed"] = True
    return out


def stage_composite(args, registry) -> dict:
    print("loading the frozen children:")
    children = load_choice_children(registry, device=args.device,
                                    third=args.third_child)
    hook = two_keys(COLORS)
    node = build_t11(children)
    print(f"\n{'=' * 72}\n{node.id}  n={node.n}  "
          f"children={[c.id for c in node.children]}"
          f"\n  env={DOORKEY} hook={hook.spec}"
          f"\n  timeout={node.timeout} max_decisions={node.max_decisions} "
          f"initiation={node.initiation.spec}"
          f"\n  seed={args.seed} budget={args.budget:,} decisions\n{'=' * 72}")

    out: dict = {"seed": args.seed, "node_id": node.id, "env_id": DOORKEY,
                 "hook": hook.spec,
                 "children": [{"id": c.id, "spec_hash": c.spec_hash()}
                              for c in children]}

    report, _model = train_composite(
        node, [DOORKEY], hook,
        total_timesteps=args.budget, eval_every=args.eval_every,
        n_eval=args.n_eval, n_envs=args.n_envs, seed=args.seed,
        device=args.device, verbose=1,
    )
    print(f"  decisions={report['decisions_trained']:,} "
          f"env_steps_own={report['env_steps_own']:,} "
          f"({report['env_steps_per_decision']}/decision) "
          f"wall={report['wall_seconds']}s stable={report['stable']}")

    learned = node.policy
    if report["stable"]:
        node.freeze()
        heldout = pooled_eval_fn(
            node, [DOORKEY], hook, n_episodes=args.n_eval,
            seed_base=heldout_seeds(1)[0], env_factory_fn=execution_env_factory,
        )()
        report["heldout"] = heldout.as_dict()
        print(f"  held-out: {heldout}")
        if not args.no_freeze:
            registry.save(node, extra_meta=report)
            print(f"  FROZEN and saved to {registry.node_dir(node.id).name}/")
    else:
        print("  not stable within budget -- not frozen")
        if not args.no_freeze:
            # Saved with status TRAINING, exactly as train_primitives.py does for
            # an unstable primitive. It can never be loaded as a child (the
            # registry requires FROZEN), but the arms and the mechanism metrics
            # can be recomputed from it without retraining.
            registry.save(node, extra_meta=report)
            print(f"  saved (status TRAINING) to {registry.node_dir(node.id).name}/")
    out["composite"] = report

    seeds = heldout_seeds(args.n_eval)
    arms = run_arms(lambda: build_t11(children), DOORKEY, hook, seeds,
                    learned_policy=learned, rng_seed=args.seed)
    out["arms"] = {k: {"label": ARM_LABELS[k], **v.as_dict()} for k, v in arms.items()}

    print(f"\n--- C1 arms, {args.n_eval} held-out episodes ---")
    for name, res in arms.items():
        print(f"  {ARM_LABELS[name]:<44} SR {res.success_rate:5.2f}  "
              f"[{res.wilson_lower:.2f},{res.wilson_upper:.2f}]  "
              f"mean env steps {res.mean_length:6.1f}")

    # ---- the gate: the learned arm and the masked control must not overlap ----
    learned_res = arms.get("learned")
    control = arms.get("random_strict_mask")
    if learned_res is not None and control is not None:
        separated = learned_res.wilson_lower > control.wilson_upper
        out["gate"] = {
            "learned_wilson_lower": round(learned_res.wilson_lower, 4),
            "masked_random_wilson_upper": round(control.wilson_upper, 4),
            "separated_at_95": bool(separated),
        }
        print(f"\ngate: learned LB {learned_res.wilson_lower:.3f} vs masked-random "
              f"UB {control.wilson_upper:.3f} -> "
              f"{'SEPARATED' if separated else 'OVERLAPPING'}")
        if not separated and report["stable"]:
            print("  A5 CONTRADICTED: the task is not the genuine choice it was "
                  "designed to be. Stop and report.")
        elif not separated:
            print("  (the node did not stabilize, so the gate is not yet "
                  "meaningful -- A5 is only testable on a converged run)")

    # ---- the mechanism: first-decision choice accuracy by door colour --------
    out["first_decision"] = first_decision_accuracy(node, hook, seeds)
    fd = out["first_decision"]
    print(f"\nfirst KEY the policy commits to, by door colour "
          f"(the mechanism -- a wrong key is fatal):")
    for colour, d in sorted(fd["by_colour"].items()):
        print(f"  door {colour:<7} n={d['n']:<4} matching key "
              f"{d['correct']}/{d['n']} = {d['accuracy']:.2f}")
    print(f"  overall {fd['correct']}/{fd['n']} = {fd['accuracy']:.2f}"
          f"   (episodes that never picked a key: "
          f"{fd['episodes_with_no_key_pick']})")
    print(f"  first child selected overall: {fd['first_child_selected']}")

    out["mask_divergence"] = mask_divergence(node, DOORKEY, hook, seeds[:20])
    out["trace"] = capture_trace(node, DOORKEY, hook, seeds[0])
    print(f"  max stack depth {out['trace']['max_stack_depth']}, "
          f"selected: {' -> '.join(out['trace']['selected_children'])}")
    return out


def first_decision_accuracy(node, hook, seeds) -> dict:
    """
    Does the policy actually read the door's colour?

    The outcome gate says the learned arm beats a masked random policy; this says
    *why*. But "the choice at decision 1" is the wrong thing to measure, and
    measuring it first produced a badly misleading number (0.42 accuracy, which
    looked like "always picks yellow").

    Two of the three children are keys and one is the door opener, and those
    carry completely different consequences:

      * picking the WRONG KEY is fatal. `pickup` needs an empty hand and
        `open_door_tk` never drops -- verified, 0 of 50 episodes ended with an
        empty hand and 92% of its actions are `toggle` -- so the episode is lost.
      * picking the OPENER while empty-handed is merely wasteful. `toggle` does
        nothing without a key, the child times out after its 60 steps, and the
        parent chooses again with the state unchanged.

    So the colour-conditioning question is about the first **key** the policy
    commits to, not about the first child it selects. Both are reported:
    `first_key_pick` is the mechanism, `first_child` shows how much time is
    burned getting there.
    """
    env = execution_env_factory(DOORKEY, hook)()
    by_colour: dict[str, dict] = {}
    first_child = {}
    correct = total = no_key = 0
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            version = 0
            sv = build_state_view(env, version)
            colour = sv.door().color
            ex = Executor(node)
            first_key = None
            first_pick = None
            for _ in range(node.timeout):
                mark = len(ex.trace)
                action = ex.act(obs, sv)
                for e in ex.trace[mark:]:
                    if e.event != "SELECT" or e.depth != 2:
                        continue
                    if first_pick is None:
                        first_pick = e.node_id
                    if first_key is None and "get_key" in e.node_id:
                        first_key = e.node_id
                if action is None or first_key is not None:
                    break
                obs, _, term, trunc, _ = env.step(action)
                version += 1
                sv = build_state_view(env, version)
                ex.observe_reward(0.0)
                if term or trunc:
                    break

            first_child[first_pick] = first_child.get(first_pick, 0) + 1
            d = by_colour.setdefault(colour, {"n": 0, "correct": 0})
            if first_key is None:
                no_key += 1
                continue
            want = ("{0,6} get_key_yellow" if colour == "yellow"
                    else "{0,7} get_key_blue")
            ok = first_key == want
            d["n"] += 1
            d["correct"] += int(ok)
            total += 1
            correct += int(ok)
    finally:
        env.close()
    for d in by_colour.values():
        d["accuracy"] = round(d["correct"] / d["n"], 4) if d["n"] else 0.0
    return {
        "n": total, "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "episodes_with_no_key_pick": no_key,
        "by_colour": by_colour,
        "first_child_selected": first_child,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["primitives", "f29", "composite"],
                    required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--no-freeze", action="store_true")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--third-child", default="auto",
                    choices=["auto", "open_door", "open_door_tk"],
                    help="which door-opening child to use; the F29 check decides")
    args = ap.parse_args()

    if args.budget is None:
        args.budget = 600_000 if args.stage == "primitives" else 20_000
    if args.eval_every is None:
        args.eval_every = 20_000 if args.stage == "primitives" else 1_000
    if args.smoke:
        args.n_eval, args.n_envs = 10, 2
        args.budget = 8_000 if args.stage == "primitives" else 600
        args.eval_every = 2_000 if args.stage == "primitives" else 200

    print(describe())
    print(f"torch threads: {set_torch_threads(args.torch_threads)}")
    registry = NodeRegistry(RUNS)

    if args.stage == "primitives":
        result = stage_primitives(args, registry)
        default = "choice_primitives"
    elif args.stage == "f29":
        result = stage_f29(args, registry)
        default = f"choice_f29_{args.third_child}"
    else:
        result = stage_composite(args, registry)
        default = f"unlock_matching_s{args.seed}"

    suffix = "_smoke" if args.smoke else ""
    path = RUNS / (args.out_name or f"{default}{suffix}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
