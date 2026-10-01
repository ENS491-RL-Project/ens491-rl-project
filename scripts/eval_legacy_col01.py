"""
H-3: re-evaluate the legacy PN-4 column {0,1} under the v2 criterion.

`runs/pn4/col01` was stabilized under the retired rule (reward mean >= 0.85,
std <= 0.05 over 50 training episodes) and reported mean reward 0.958 at 40k
steps. This asks the v2 question instead: in what fraction of evaluation
episodes does the goal predicate hold before the timeout (v2 §5.3)?

Two things to keep straight:

- col01 was trained on RAW integer observations, before the ChannelNormalize of
  [A: F16]. It must be evaluated on the raw pipeline or the numbers are
  meaningless -- which is exactly why `normalize_obs` is part of `spec_hash`.
- This is the first real call to `ColumnTrainer.load_policy_into`, which has
  been dead code since it was written.

Usage:  python scripts/eval_legacy_col01.py [--episodes 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import gymnasium as gym  # noqa: E402

from src.continual_learning.column import Column  # noqa: E402
from src.continual_learning.pn_trainer import ColumnTrainer  # noqa: E402
from src.envs.obs import apply_obs_pipeline, obs_pipeline_name  # noqa: E402
from src.skills.node import Predicate, SkillNode  # noqa: E402
from src.skills.predicates import on_goal_tile  # noqa: E402
from src.training.device import describe  # noqa: E402
from src.training.evaluation import (  # noqa: E402
    StabilityCriterion,
    evaluate_node,
    heldout_seeds,
    stopping_seeds,
)

ENV_ID = "MiniGrid-Empty-8x8-v0"
RUN_DIR = REPO / "runs" / "pn4" / "col01"
OUT = REPO / "runs" / "v2" / "legacy_col01_eval.json"


class ColumnPolicyAdapter:
    """Wraps a v1 Column as a v2 PrimitivePolicy [A: F19]."""

    def __init__(self, column: Column) -> None:
        self.column = column

    def act(self, obs, deterministic: bool = True) -> int:
        action, _ = self.column.policy.predict(obs, deterministic=deterministic)
        return int(action)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    print(describe())
    if not (RUN_DIR / "policy_state_dict.pt").exists():
        print(f"ERROR: no checkpoint at {RUN_DIR}")
        return 1

    stored_meta = json.loads((RUN_DIR / "meta.json").read_text(encoding="utf-8"))
    print(f"stored meta.json: {stored_meta}")
    missing = [k for k in ("seed", "env_id", "reward_threshold") if k not in stored_meta]
    if missing:
        print(
            f"NOTE: this checkpoint records none of {missing}. It predates the "
            "9-key save() now in pn_trainer.py and is not reproducible from its "
            "own metadata -- the reason v2 §2 adds spec_hash [A: F18]."
        )

    # col01 predates ChannelNormalize, so it must be evaluated on the raw scale.
    column = Column(n=0, m=1)
    ColumnTrainer.load_policy_into(column, RUN_DIR, ENV_ID)
    column.freeze()

    node = SkillNode(
        "{0,1} legacy_col01",
        goal=on_goal_tile(),
        initiation=Predicate("always", lambda sv: True),
        timeout=gym.make(ENV_ID).unwrapped.max_steps,   # 256; the env's own budget
        env_ids=[ENV_ID],
        obs_pipeline=obs_pipeline_name(False),
        normalize_obs=False,
    )
    node.policy = ColumnPolicyAdapter(column)
    print(f"\nnode: {node}  timeout={node.timeout}  pipeline={node.obs_pipeline}")

    def env_factory():
        return apply_obs_pipeline(gym.make(ENV_ID), normalize=False)

    # Does the layout actually vary with the seed? If not, N episodes are one
    # episode repeated N times and the success rate carries no variance.
    from src.skills.state import build_state_view

    layouts = set()
    for s in stopping_seeds(args.episodes)[:10] + heldout_seeds(args.episodes)[:10]:
        probe = gym.make(ENV_ID)
        probe.reset(seed=s)
        psv = build_state_view(probe)
        layouts.add((tuple(psv.agent_pos), psv.agent_dir, psv.goal_pos))
    deterministic_layout = len(layouts) == 1
    if deterministic_layout:
        print(
            f"\nCAVEAT: {ENV_ID} is layout-deterministic -- identical start "
            f"{next(iter(layouts))[0]} and goal {next(iter(layouts))[2]} for every "
            "seed. The 'episodes' below are one episode repeated, so the success "
            "rate has no variance to measure and the Wilson bound is vacuous here. "
            "This run exercises the H-3 harness; it is not evidence of robustness."
        )

    crit = StabilityCriterion()
    results = {}
    for label, seeds in (
        ("stopping", stopping_seeds(args.episodes)),
        ("held-out", heldout_seeds(args.episodes)),
    ):
        res = evaluate_node(node, env_factory, seeds=seeds)
        results[label] = res.as_dict()
        print(f"  {label:9s} seeds {seeds[0]}..{seeds[-1]}: {res}")

    stopping = results["stopping"]
    verdict = (
        stopping["success_rate"] >= crit.min_success_rate
        and stopping["wilson_lower"] >= crit.min_wilson_lower
    )
    print(
        f"\nv2 criterion (SR >= {crit.min_success_rate}, Wilson LB >= "
        f"{crit.min_wilson_lower}): {'PASS' if verdict else 'FAIL'} on one evaluation"
    )
    print("  (STABLE additionally requires two consecutive passes during training)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "node_id": node.id,
                "env_id": ENV_ID,
                "timeout": node.timeout,
                "obs_pipeline": node.obs_pipeline,
                "criterion": {
                    "min_success_rate": crit.min_success_rate,
                    "min_wilson_lower": crit.min_wilson_lower,
                },
                "single_eval_passes": verdict,
                "layout_deterministic": deterministic_layout,
                "caveat": (
                    "Empty-8x8 has a fixed start and goal, so every evaluation "
                    "episode is identical. The success rate has no variance and the "
                    "Wilson bound is vacuous; this exercises the H-3 harness rather "
                    "than measuring robustness."
                ) if deterministic_layout else None,
                "results": results,
                "stored_meta": stored_meta,
                "v1_reported": {"mean_reward": 0.958, "steps_to_stable": 40_000},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
