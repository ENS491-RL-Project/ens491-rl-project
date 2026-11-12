"""
The final test (v2 §5.3): 200 seeds from FINAL_SEED_BASE, opened once.

    python scripts/eval_final.py --freeze-manifest   # record every node's hashes; commit the file
    python scripts/eval_final.py --open              # evaluate once, against that manifest

`--open` refuses if a hash differs from the manifest and refuses a second time. Run it only
after every training run and every hyperparameter choice of the claim being tested is done.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from eval_regression import RUNS, load_library  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.evaluation import DEFAULT_N_FINAL, final_seeds  # noqa: E402
from src.training.final_protocol import (  # noqa: E402
    claim_final_set, manifest_entry, verify_manifest, write_manifest,
)
from src.training.node_trainer import execution_env_factory, pooled_eval_fn  # noqa: E402
from src.training.run_store import policy_hash  # noqa: E402

MANIFEST = RUNS / "final_manifest.json"
RESULTS = RUNS / "final_results.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-manifest", action="store_true")
    mode.add_argument("--open", action="store_true")
    args = ap.parse_args()

    library = load_library(NodeRegistry(RUNS))
    entries = [manifest_entry(n, policy_hash(n.policy)) for n, *_ in library]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()

    if args.freeze_manifest:
        write_manifest(MANIFEST, entries, DEFAULT_N_FINAL, commit)
        print(f"wrote {MANIFEST.relative_to(REPO)} ({len(entries)} nodes, {DEFAULT_N_FINAL} episodes)")
        return 0

    verify_manifest(MANIFEST, entries)
    claim_final_set(RESULTS)
    rows = []
    for node, env_ids, hook, composite in library:
        res = pooled_eval_fn(node, env_ids, hook, n_episodes=DEFAULT_N_FINAL,
                             seed_base=final_seeds(1)[0],
                             **({"env_factory_fn": execution_env_factory} if composite else {}))()
        rows.append({"node": node.id, "spec_hash": node.spec_hash(), **res.as_dict()})
        print(f"{node.id:<24}{res.success_rate:>7.3f}  wilson_lb {res.wilson_lower:.3f}")
    RESULTS.write_text(json.dumps({"state": "complete", "code_commit": commit,
                                   "n_episodes": DEFAULT_N_FINAL, "rows": rows}, indent=2),
                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
