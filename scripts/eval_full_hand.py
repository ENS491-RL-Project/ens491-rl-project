"""
H-6.8: what does each frozen child do when it is handed a FULL hand?

[A: F29, generalized] A primitive only ever sees states its own reset hook
produces. A parent hands it whatever state the previous child left behind, and
the one input no hook in this project varies is **what the agent is already
carrying**. MiniGrid `pickup` requires an empty hand, so a child asked to fetch
something while holding something else is outside its training distribution
entirely -- its behaviour there is undefined, not learned.

Undefined is not the same as harmless. H-6.6's choice task was built on the
premise that a wrong first key pick is unrecoverable, and that premise is what
made success rate a valid measure of colour-conditioned choice. It is false:

  * `get_key_<colour>`, run while holding the other key, DROPS it and swaps.
  * `{0,8} open_door_tk`, run while holding the wrong key, does not drop at all.

The first of those silently gave the composite a recovery path the experiment was
designed not to have, and it accounts for the whole of its apparent advantage
over the controls. The numbers lived only in a summary paragraph until this
script; they are measurements and belong in `runs/v2/`.

This is the pre-flight H-7 needs for every child it composes. `{0,4} pickup_box`
will routinely be selected while the agent still holds the key, and
`{0,5} move_blocker` needs an empty hand to pick the ball up at all.

Usage:  python scripts/eval_full_hand.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import compose, give_key_of, two_keys  # noqa: E402
from src.skills.executor import Executor  # noqa: E402
from src.skills.predicates import door_open  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.device import describe  # noqa: E402
from src.training.evaluation import heldout_seeds  # noqa: E402
from src.training.node_trainer import execution_env_factory  # noqa: E402
from train_choice_task import COLORS, DOORKEY, load_choice_children  # noqa: E402

RUNS = REPO / "runs" / "v2"
N_EVAL = 50
ACTION_NAMES = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop",
                5: "toggle", 6: "done"}


def run_child(child, env, obs, sv, version):
    """Roll one frozen child to its own termination, logging every action."""
    acts: Counter = Counter()
    ex = Executor(child)
    for _ in range(child.timeout):
        action = ex.act(obs, sv)
        if action is None:
            break
        acts[int(action)] += 1
        obs, _, term, trunc, _ = env.step(action)
        version += 1
        sv = build_state_view(env, version)
        ex.observe_reward(0.0)
        if term or trunc:
            break
    return obs, sv, version, acts


def probe(child, hook, *, want_colour_of, label):
    """
    Hand the agent the decoy key, then ask `child` to do its job.

    `want_colour_of` picks which child to run per episode: "matching" runs the
    key primitive for the door's colour (so it is being asked to fetch a key it
    cannot pick up), "opener" runs the door opener (so it is holding the wrong
    key and `toggle` will never work).
    """
    env = execution_env_factory(DOORKEY, hook)()
    acts: Counter = Counter()
    episodes = ended_empty = swapped = opened = 0
    try:
        for seed in heldout_seeds(N_EVAL):
            obs, _ = env.reset(seed=seed)
            version = 0
            sv = build_state_view(env, version)
            colour = sv.door().color
            if sv.carrying_type != "key" or sv.carrying_color == colour:
                continue                    # not the state under test
            episodes += 1
            node = child(colour) if callable(child) else child
            obs, sv, version, a = run_child(node, env, obs, sv, version)
            acts.update(a)
            ended_empty += int(sv.carrying_type is None)
            swapped += int(sv.carrying_color == colour)
            opened += int(bool(door_open()(sv)))
    finally:
        env.close()

    total = sum(acts.values())
    row = {
        "label": label,
        "episodes": episodes,
        "actions": {ACTION_NAMES[k]: acts[k] for k in sorted(acts)},
        "action_share": {ACTION_NAMES[k]: round(acts[k] / total, 4)
                         for k in sorted(acts)} if total else {},
        "ended_empty_handed": ended_empty,
        "swapped_to_the_matching_key": swapped,
        "swap_rate": round(swapped / episodes, 4) if episodes else 0.0,
        "door_opened": opened,
    }
    print(f"\n{label}  ({episodes} episodes)")
    for k in sorted(acts):
        print(f"  {ACTION_NAMES[k]:<8} {acts[k]:>5}  ({acts[k] / total:.3f})")
    print(f"  ended empty-handed : {ended_empty}/{episodes}")
    print(f"  swapped to matching: {swapped}/{episodes} = {row['swap_rate']:.2f}")
    print(f"  door opened        : {opened}/{episodes}")
    return row


def main() -> int:
    print(describe())
    registry = NodeRegistry(RUNS)
    print("loading the frozen children:")
    yellow, blue, opener = load_choice_children(registry, third="open_door_tk")

    # The state after a wrong first pick: holding the decoy key.
    hook = compose(two_keys(COLORS), give_key_of("decoy"))

    rows = [
        probe(lambda colour: yellow if colour == "yellow" else blue, hook,
              want_colour_of="matching",
              label="get_key_<matching> while holding the DECOY key"),
        probe(opener, hook, want_colour_of="opener",
              label="open_door_tk while holding the WRONG key"),
    ]

    out = {
        "n_eval": N_EVAL,
        "env_id": DOORKEY,
        "hook": hook.spec,
        "seed_base": heldout_seeds(1)[0],
        "children": [
            {"id": n.id, "spec_hash": n.spec_hash()} for n in (yellow, blue, opener)
        ],
        "rows": rows,
    }
    path = RUNS / "choice_full_hand.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO)}")

    swap = rows[0]["swap_rate"]
    print(f"\nThe choice task's premise -- that a wrong key pick is unrecoverable --")
    print(f"holds only if this swap rate is ~0. It is {swap:.2f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
