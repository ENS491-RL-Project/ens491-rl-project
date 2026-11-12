"""The final test set stays closed: disjoint seeds, one reader, one opening, a pinned manifest."""
import re
from pathlib import Path

import pytest
import torch

from src.skills.node import NEVER, SkillNode
from src.training.evaluation import (
    DEFAULT_N_FINAL, HELDOUT_SEED_BASE, STOPPING_SEED_BASE, final_seeds,
)
from src.training.final_protocol import (
    FinalSetError, claim_final_set, manifest_entry, verify_manifest, write_manifest,
)

REPO = Path(__file__).resolve().parent.parent


def _node(node_id="{0,1} a"):
    n = SkillNode(node_id, goal=NEVER, timeout=10, env_ids=["MiniGrid-Unlock-v0"])
    n.policy = torch.nn.Linear(2, 2)
    return n


def test_final_seeds_are_disjoint_from_stopping_and_heldout():
    final = set(final_seeds())
    assert len(final) == DEFAULT_N_FINAL
    # 1,000 covers the held-out set and its +500 'parent reachable' offset
    assert not final & set(range(STOPPING_SEED_BASE, STOPPING_SEED_BASE + 1000))
    assert not final & set(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + 1000))


def test_only_eval_final_reads_the_final_set():
    allowed = {"src/training/evaluation.py", "scripts/eval_final.py", "tests/test_final_protocol.py"}
    pattern = re.compile(r"\bfinal_seeds\b|\bFINAL_SEED_BASE\b")
    readers = {p.relative_to(REPO).as_posix()
               for d in ("src", "scripts", "tests") for p in (REPO / d).rglob("*.py")
               if pattern.search(p.read_text(encoding="utf-8"))}
    assert readers <= allowed, f"the final set is read by {sorted(readers - allowed)}"


def test_the_final_set_opens_once(tmp_path):
    claim_final_set(tmp_path / "final_results.json")
    with pytest.raises(FinalSetError, match="already opened"):
        claim_final_set(tmp_path / "final_results.json")


def test_manifest_accepts_the_same_library_and_rejects_any_change(tmp_path):
    a, b = _node("{0,1} a"), _node("{0,2} b")
    pinned = [manifest_entry(a, "h-a"), manifest_entry(b, "h-b")]
    path = tmp_path / "final_manifest.json"
    write_manifest(path, pinned, DEFAULT_N_FINAL, "abc123")

    assert verify_manifest(path, pinned)["n_episodes"] == DEFAULT_N_FINAL
    for changed, text in [
        ([manifest_entry(a, "h-a"), manifest_entry(b, "other")], "changed"),   # retrained
        ([manifest_entry(a, "h-a")], "not loaded"),                           # node missing
        (pinned + [manifest_entry(_node("{0,3} c"), "h-c")], "not in the manifest"),  # node added
    ]:
        with pytest.raises(FinalSetError, match=text):
            verify_manifest(path, changed)
