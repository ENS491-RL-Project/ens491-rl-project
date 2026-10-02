"""
H-6.9 -- why did `{1,2} unlock_matching`'s selector not learn a choice it can see?

The H-6.8 rerun (100 policy updates, 3 seeds, two mask conditions) learned no
colour-conditioned choice. Under the strict mask, seed 42 settled on "always
blue" and seeds 43/44 were slowly leaving the one-key plateau when the budget
ran out. The cause is not established. This script separates the two candidate
explanations with supervised measurements, before any further RL is run:

  sanity   over 400 resets of the two-key hook: at decision 1 the door is inside
           the 7x7 view, and the strict mask leaves exactly the two key children.
           If either fails, the task is broken, not the learner.

  decode   is the door's colour readable from the decision-1 observation by a
           network with the selector's OWN architecture? 5,000 resets on seeds
           30000-34999 (disjoint from the stopping set 10000+ and the held-out set
           20000+). A supervised 2-class classifier -- SB3's MaskableActorCritic
           policy head, 147 -> 64 tanh -> 64 tanh -> 2, built by the same class
           MaskablePPO uses -- is trained on the first 250 / 500 / 1,000 / 2,000 /
           4,000 and tested on the last 1,000. Logistic regression on the same
           splits for a linear reference.

  arm      (only if the verdict calls for it) a strict-mask training run of the
           composite that logs, at every evaluation, the mean probability the
           policy gives each key child at decision 1, split by door colour, and
           the policy entropy there. `--ent-coef` overrides the composite's
           entropy coefficient for that run only; no default changes.

Verdict rule (fixed before the numbers, launch prompt for H-6.9):
  sanity fails                                  -> a bug
  classifier >= 0.95 with <= 1,000 examples     -> signal is easy; RL optimisation
  classifier <  0.95 at 4,000 examples          -> limit of the representation
  otherwise                                     -> numbers only, no verdict

    python scripts/diagnose_choice.py --stage sanity
    python scripts/diagnose_choice.py --stage decode
    python scripts/diagnose_choice.py --stage arm --seed 42 [--ent-coef 0.05]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX  # noqa: E402

from src.envs.option_env import make_option_env  # noqa: E402
from src.envs.subgoal_env import two_keys  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.node_trainer import set_torch_threads  # noqa: E402
from train_choice_task import (  # noqa: E402
    COLORS,
    DOORKEY,
    RUNS,
    build_t11,
    load_choice_children,
)

OUT = RUNS / "choice_diagnosis.json"
SEED_BASE = 30_000
N_SANITY = 400
N_TOTAL = 5_000
N_TEST = 1_000
TRAIN_SIZES = (250, 500, 1_000, 2_000, 4_000)
INIT_SEEDS = (0, 1, 2)         # classifier initialisations per training size
LABEL = {c: i for i, c in enumerate(COLORS)}   # yellow 0, blue 1
KEY_CHILD = {"yellow": "{0,6} get_key_yellow", "blue": "{0,7} get_key_blue"}

EASY = 0.95                     # the verdict threshold


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def decision1(env, seed: int) -> dict:
    """Reset the OptionEnv and read what the selector sees at its first decision."""
    obs, _ = env.reset(seed=seed)
    inner = env.unwrapped
    # The raw 7x7x3 partial view, before ChannelNormalize, so "door in view" is
    # an exact integer test rather than a float comparison.
    img = inner.gen_obs()["image"]
    door_cells = np.argwhere(img[:, :, 0] == OBJECT_TO_IDX["door"])
    door = env.sv.door()
    return {
        "obs": np.asarray(obs, dtype=np.float32),
        "colour": door.color,
        "mask": list(env.action_masks()),
        "door_in_view": bool(len(door_cells)),
        "door_view_cell": tuple(int(v) for v in door_cells[0]) if len(door_cells) else None,
        "door_colour_in_view": (
            bool(img[tuple(door_cells[0])][1] == COLOR_TO_IDX[door.color])
            if len(door_cells) else False
        ),
    }


def make_env(node, strict: bool = True):
    return make_option_env(node, DOORKEY, two_keys(COLORS), monitor=False,
                           initiation_masking=strict)


# --------------------------------------------------------------------------
# stage: sanity
# --------------------------------------------------------------------------

def stage_sanity(node) -> dict:
    env = make_env(node, strict=True)
    names = [c.id for c in node.children]
    want = [n in KEY_CHILD.values() for n in names]
    rows = {"door_in_view": 0, "door_colour_in_view": 0, "mask_exactly_two_keys": 0}
    colours = {c: 0 for c in COLORS}
    bad = []
    cells = set()
    try:
        for i in range(N_SANITY):
            seed = SEED_BASE + i
            d = decision1(env, seed)
            colours[d["colour"]] += 1
            rows["door_in_view"] += d["door_in_view"]
            rows["door_colour_in_view"] += d["door_colour_in_view"]
            ok_mask = d["mask"] == want
            rows["mask_exactly_two_keys"] += ok_mask
            if d["door_view_cell"] is not None:
                cells.add(d["door_view_cell"])
            if not (d["door_in_view"] and ok_mask):
                bad.append({"seed": seed, "door_in_view": d["door_in_view"],
                            "mask": d["mask"]})
    finally:
        env.close()
    passed = all(v == N_SANITY for v in rows.values())
    out = {
        "n_resets": N_SANITY, "seeds": [SEED_BASE, SEED_BASE + N_SANITY - 1],
        "children": names, "expected_strict_mask": want,
        **rows, "door_colour_counts": colours,
        "distinct_door_cells_in_view": len(cells),
        "failures": bad[:20], "passed": passed,
    }
    print(f"sanity over {N_SANITY} resets (seeds {SEED_BASE}..{SEED_BASE + N_SANITY - 1}):")
    for k, v in rows.items():
        print(f"  {k:<26} {v}/{N_SANITY}")
    print(f"  door colours {colours}; door seen at {len(cells)} distinct view cells")
    print(f"  {'PASSED' if passed else 'FAILED'}")
    return out


# --------------------------------------------------------------------------
# stage: decode
# --------------------------------------------------------------------------

def collect(node, n: int) -> tuple[np.ndarray, np.ndarray, list]:
    env = make_env(node, strict=True)
    X = np.zeros((n, 147), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    cells = []
    try:
        for i in range(n):
            d = decision1(env, SEED_BASE + i)
            X[i] = d["obs"]
            y[i] = LABEL[d["colour"]]
            cells.append(d["door_view_cell"])
    finally:
        env.close()
    return X, y, cells


def selector_net(init_seed: int):
    """The selector's own network: the class and defaults MaskablePPO("MlpPolicy") builds."""
    import gymnasium as gym
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

    torch.manual_seed(init_seed)
    return MaskableActorCriticPolicy(
        gym.spaces.Box(0.0, 1.0, (147,), np.float32), gym.spaces.Discrete(2),
        lambda _: 3e-4,
    )


def logits_of(policy, x: torch.Tensor) -> torch.Tensor:
    latent_pi = policy.mlp_extractor.forward_actor(policy.extract_features(x, policy.pi_features_extractor))
    return policy.action_net(latent_pi)


def fit_selector(Xtr, ytr, Xte, yte, init_seed: int, *, lr=1e-3, batch=64,
                 max_epochs=400, patience=40) -> dict:
    """
    Supervised fit of the selector's actor head. 20% of the training examples
    are held back for early stopping, so the 1,000 test examples are never used
    to choose anything.
    """
    rng = np.random.default_rng(init_seed)
    idx = rng.permutation(len(Xtr))
    n_val = max(1, len(Xtr) // 5)
    va, tr = idx[:n_val], idx[n_val:]
    Xt, yt = torch.from_numpy(Xtr[tr]), torch.from_numpy(ytr[tr])
    Xv, yv = torch.from_numpy(Xtr[va]), torch.from_numpy(ytr[va])
    policy = selector_net(init_seed)
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    best = (-1.0, None, 0)
    since = 0
    g = torch.Generator().manual_seed(init_seed)
    for epoch in range(max_epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for s in range(0, len(Xt), batch):
            b = perm[s:s + batch]
            opt.zero_grad()
            lossf(logits_of(policy, Xt[b]), yt[b]).backward()
            opt.step()
        with torch.no_grad():
            acc_v = (logits_of(policy, Xv).argmax(1) == yv).float().mean().item()
        if acc_v > best[0]:
            best = (acc_v, {k: v.clone() for k, v in policy.state_dict().items()}, epoch)
            since = 0
        else:
            since += 1
            if since >= patience:
                break
    policy.load_state_dict(best[1])
    with torch.no_grad():
        Xe = torch.from_numpy(Xte)
        acc = (logits_of(policy, Xe).argmax(1).numpy() == yte).mean()
        acc_tr = (logits_of(policy, Xt).argmax(1) == yt).float().mean().item()
    return {"test_acc": round(float(acc), 4), "train_acc": round(acc_tr, 4),
            "val_acc": round(best[0], 4), "best_epoch": best[2]}


def fit_selector_fixed(Xtr, ytr, Xte, yte, init_seed: int, *, lr=1e-3, batch=64,
                       steps=3_000) -> dict:
    """
    Sensitivity check, added AFTER the primary numbers were seen: the same network
    trained for a fixed 3,000 gradient steps on all training examples, last
    weights kept, nothing selected. The primary fit's early stopping on a 20%
    split stopped the n=250 and n=500 runs at train accuracy 0.62-0.74 (epoch
    0-51), so their test accuracy measures the stopping rule as much as the
    network. This variant does not enter the verdict.
    """
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    policy = selector_net(init_seed)
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(init_seed)
    done = 0
    while done < steps:
        perm = torch.randperm(len(Xt), generator=g)
        for s in range(0, len(Xt), batch):
            if done >= steps:
                break
            b = perm[s:s + batch]
            opt.zero_grad()
            lossf(logits_of(policy, Xt[b]), yt[b]).backward()
            opt.step()
            done += 1
    with torch.no_grad():
        acc = (logits_of(policy, torch.from_numpy(Xte)).argmax(1).numpy() == yte).mean()
        acc_tr = (logits_of(policy, Xt).argmax(1) == yt).float().mean().item()
    return {"test_acc": round(float(acc), 4), "train_acc": round(acc_tr, 4)}


def fit_logreg(Xtr, ytr, Xte, yte) -> dict:
    """L2 logistic regression, full-batch LBFGS; the L2 weight is picked on a 20% validation split."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xtr))
    n_val = max(1, len(Xtr) // 5)
    va, tr = idx[:n_val], idx[n_val:]

    def fit(X, y, l2):
        X = torch.from_numpy(X).double()
        y = torch.from_numpy(y).double()
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * (w @ w)
            loss.backward()
            return loss
        opt.step(closure)
        return w.detach(), b.detach()

    def acc(wb, X, y):
        w, b = wb
        return float(((torch.from_numpy(X).double() @ w + b > 0).numpy() == y.astype(bool)).mean())

    scores = {}
    for l2 in (1e-5, 1e-4, 1e-3, 1e-2):
        scores[l2] = acc(fit(Xtr[tr], ytr[tr], l2), Xtr[va], ytr[va])
    l2 = max(scores, key=scores.get)
    wb = fit(Xtr, ytr, l2)
    return {"test_acc": round(acc(wb, Xte, yte), 4), "train_acc": round(acc(wb, Xtr, ytr), 4),
            "l2": l2, "val_acc_by_l2": {str(k): round(v, 4) for k, v in scores.items()}}


def verdict(sanity_passed: bool, sel: dict) -> tuple[str, str]:
    mean = {n: sel[n]["mean"] for n in sel}
    if not sanity_passed:
        return "bug", "sanity failed"
    if any(mean[n] >= EASY for n in TRAIN_SIZES if n <= 1_000):
        return "rl_optimisation", (
            f"the selector's architecture reads the door colour at >= {EASY} from "
            f"<= 1,000 supervised examples")
    if mean[TRAIN_SIZES[-1]] < EASY:
        return "representation", (
            f"the selector's architecture stays below {EASY} at "
            f"{TRAIN_SIZES[-1]:,} supervised examples")
    return "none", "between the two thresholds; numbers only"


def stage_decode(node, sanity: dict) -> dict:
    t0 = time.perf_counter()
    X, y, cells = collect(node, N_TOTAL)
    print(f"collected {N_TOTAL} decision-1 observations in {time.perf_counter() - t0:.0f}s; "
          f"blue share {y.mean():.3f}; {len(set(cells))} distinct door view cells")
    Xte, yte = X[-N_TEST:], y[-N_TEST:]
    pool_X, pool_y = X[:-N_TEST], y[:-N_TEST]

    # A non-learned reference: read the colour channel of whatever cell holds a
    # door. 1.00 here says the information is in the observation at all.
    o = Xte.reshape(-1, 7, 7, 3)
    door = np.isclose(o[..., 0] * 10, OBJECT_TO_IDX["door"])
    col = np.rint(o[..., 1] * 5).astype(int)
    oracle = np.array([LABEL["blue"] if (col[i][door[i]] == COLOR_TO_IDX["blue"]).any()
                       else LABEL["yellow"] for i in range(len(o))])
    oracle_acc = float((oracle == yte).mean())
    print(f"hand-written reader (door cell's colour channel): {oracle_acc:.4f}")

    sel, lin, fixed = {}, {}, {}
    for n in TRAIN_SIZES:
        runs = [fit_selector(pool_X[:n], pool_y[:n], Xte, yte, s) for s in INIT_SEEDS]
        accs = [r["test_acc"] for r in runs]
        sel[n] = {"runs": runs, "mean": round(float(np.mean(accs)), 4),
                  "min": min(accs), "max": max(accs)}
        lin[n] = fit_logreg(pool_X[:n], pool_y[:n], Xte, yte)
        fr = [fit_selector_fixed(pool_X[:n], pool_y[:n], Xte, yte, s) for s in INIT_SEEDS]
        fa = [r["test_acc"] for r in fr]
        fixed[n] = {"runs": fr, "mean": round(float(np.mean(fa)), 4),
                    "min": min(fa), "max": max(fa)}
        print(f"  n={n:>5}  selector MLP test {sel[n]['mean']:.4f} "
              f"[{sel[n]['min']:.4f}, {sel[n]['max']:.4f}]   "
              f"logistic {lin[n]['test_acc']:.4f}   "
              f"fixed-3k-steps MLP {fixed[n]['mean']:.4f} [{fixed[n]['min']:.4f}, {fixed[n]['max']:.4f}]"
              f"  ({time.perf_counter() - t0:.0f}s)")

    v, why = verdict(sanity["passed"], sel)
    print(f"verdict: {v} -- {why}")
    return {
        "seeds": [SEED_BASE, SEED_BASE + N_TOTAL - 1],
        "train_pool": [SEED_BASE, SEED_BASE + N_TOTAL - N_TEST - 1],
        "test": [SEED_BASE + N_TOTAL - N_TEST, SEED_BASE + N_TOTAL - 1],
        "label": LABEL, "blue_share_all": round(float(y.mean()), 4),
        "blue_share_test": round(float(yte.mean()), 4),
        "distinct_door_view_cells": len(set(cells)),
        "hand_written_reader_test_acc": oracle_acc,
        "selector_mlp": {
            "architecture": "MaskableActorCriticPolicy default: 147 -> 64 tanh -> 64 tanh -> 2 "
                            "(actor head only, SB3 orthogonal init)",
            "training": "Adam lr 1e-3, batch 64, <= 400 epochs, early stop on a 20% split "
                        "of the training examples (patience 40); test set never used to select",
            "init_seeds": list(INIT_SEEDS),
            "by_train_size": {str(k): v for k, v in sel.items()},
        },
        "logistic_regression": {
            "training": "L2 logistic, LBFGS; L2 weight picked on a 20% split of the training examples",
            "by_train_size": {str(k): v for k, v in lin.items()},
        },
        "sensitivity_fixed_budget_mlp": {
            "status": "post hoc, added after the primary numbers were seen; not used for the verdict",
            "training": "same network, Adam lr 1e-3, batch 64, exactly 3,000 gradient steps on all "
                        "training examples, last weights, no selection",
            "by_train_size": {str(k): v for k, v in fixed.items()},
        },
        "verdict": v, "verdict_reason": why, "threshold": EASY,
        "verdict_statistic": "mean test accuracy over the 3 classifier initialisations, "
                             "primary (early-stopped) fit",
    }


# --------------------------------------------------------------------------
# stage: arm
# --------------------------------------------------------------------------

def key_probabilities(node, probe_X, probe_colour, probe_mask) -> dict:
    """Mean probability of each key child at decision 1, by door colour, and entropy."""
    sb3 = node.policy.sb3_policy
    with torch.no_grad():
        obs = torch.as_tensor(probe_X)
        dist = sb3.get_distribution(obs, action_masks=probe_mask)
        p = dist.distribution.probs.numpy()
        ent = dist.entropy().numpy()
    names = [c.id for c in node.children]
    out = {}
    for colour in COLORS:
        sel = np.array([c == colour for c in probe_colour])
        out[f"door_{colour}"] = {
            n: round(float(p[sel, i].mean()), 4) for i, n in enumerate(names)
            if n in KEY_CHILD.values()
        }
        out[f"door_{colour}"]["p_correct"] = round(
            float(p[sel, names.index(KEY_CHILD[colour])].mean()), 4)
        out[f"door_{colour}"]["n"] = int(sel.sum())
    out["entropy_mean"] = round(float(ent.mean()), 4)
    out["entropy_max_possible"] = round(float(np.log(2)), 4)
    return out


def stage_arm(args, children) -> dict:
    from src.training.node_trainer import train_composite

    node = build_t11(children)
    hook = two_keys(COLORS)
    # Probe on the 50 stopping seeds' decision-1 states: same distribution the
    # stopping criterion scores, fixed for the whole run.
    env = make_env(node, strict=True)
    from src.training.evaluation import stopping_seeds
    probe = [decision1(env, s) for s in stopping_seeds(args.probe_n)]
    env.close()
    probe_X = np.stack([d["obs"] for d in probe])
    probe_colour = [d["colour"] for d in probe]
    probe_mask = np.array([d["mask"] for d in probe], dtype=bool)

    overrides = {} if args.ent_coef is None else {"ent_coef": args.ent_coef}
    report, _model = train_composite(
        node, [DOORKEY], hook,
        total_timesteps=args.budget, eval_every=args.eval_every,
        n_eval=args.n_eval, n_envs=args.n_envs, seed=args.seed,
        initiation_masking=True, ppo_overrides=overrides,
        extra_metrics=lambda: key_probabilities(node, probe_X, probe_colour, probe_mask),
        verbose=1,
    )
    return {
        "arm": "ent_coef" if args.ent_coef is not None else "unchanged",
        "seed": args.seed, "mask": "strict", "ent_coef": report["ppo"]["ent_coef"],
        "probe_seeds": f"stopping seeds, first {args.probe_n}",
        "composite": report,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sanity", "decode", "all", "arm"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ent-coef", type=float, default=None)
    ap.add_argument("--budget", type=int, default=51_200)     # 100 updates of 512
    ap.add_argument("--eval-every", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--probe-n", type=int, default=50)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    global N_SANITY, N_TOTAL, N_TEST, TRAIN_SIZES, INIT_SEEDS
    if args.smoke:
        N_SANITY, N_TOTAL, N_TEST = 20, 300, 100
        TRAIN_SIZES, INIT_SEEDS = (50, 200), (0,)
        args.budget, args.eval_every, args.n_eval, args.n_envs, args.probe_n = 1_024, 512, 10, 8, 10

    print(f"torch threads: {set_torch_threads(args.torch_threads)}")
    registry = NodeRegistry(RUNS)
    children = load_choice_children(registry)
    suffix = "_smoke" if args.smoke else ""

    if args.stage == "arm":
        result = stage_arm(args, children)
        tag = "ent05" if args.ent_coef is not None else "unchanged"
        path = RUNS / (args.out_name or f"choice_diagnosis_arm_{tag}_s{args.seed}{suffix}.json")
    else:
        node = build_t11(children)
        result = {"node_id": node.id, "hook": two_keys(COLORS).spec,
                  "obs_pipeline": "ImgObs+Flatten+ChannelNormalize"}
        result["sanity"] = stage_sanity(node)
        if args.stage in ("decode", "all"):
            result["decode"] = stage_decode(node, result["sanity"])
        path = RUNS / (args.out_name or f"choice_diagnosis{suffix}.json")

    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
