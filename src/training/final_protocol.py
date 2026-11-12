"""The final test is opened once, against a manifest fixed beforehand (v2 §5.3)."""
from __future__ import annotations

import json
from pathlib import Path


class FinalSetError(RuntimeError):
    """The final set may not be opened: the manifest does not match, or it was already opened."""


def manifest_entry(node, policy_sha256) -> dict:
    return {"node": node.id, "spec_hash": node.spec_hash(), "policy_sha256": policy_sha256}


def write_manifest(path: Path, entries: list[dict], n_episodes: int, code_commit) -> None:
    path.write_text(json.dumps(
        {"n_episodes": n_episodes, "code_commit": code_commit,
         "entries": sorted(entries, key=lambda e: e["node"])}, indent=2), encoding="utf-8")


def verify_manifest(path: Path, entries: list[dict]) -> dict:
    """Every manifest entry must be present and unchanged, and nothing may be added."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pinned = {e["node"]: e for e in manifest["entries"]}
    now = {e["node"]: e for e in entries}
    problems = [f"{k}: not loaded" for k in pinned.keys() - now.keys()]
    problems += [f"{k}: not in the manifest" for k in now.keys() - pinned.keys()]
    problems += [f"{k}: spec or weights changed since the manifest" for k in pinned.keys() & now.keys()
                 if pinned[k] != now[k]]
    if problems:
        raise FinalSetError("; ".join(sorted(problems)))
    return manifest


def claim_final_set(path: Path) -> None:
    """Exclusive create: a second open fails, and so does a run that crashed after the first."""
    try:
        with path.open("x", encoding="utf-8") as f:
            json.dump({"state": "started"}, f)
    except FileExistsError:
        raise FinalSetError(
            f"{path} exists: the final set was already opened. A retrained or re-tuned library "
            "needs a new final seed base, not a deleted file (v2 §5.3)."
        ) from None
