"""
NodeRegistry: persistence for frozen skill nodes.

Target Architecture v2 §2 [A: F18]. A frozen node is reused across experiments,
so its checkpoint must record the spec it was trained under. The v1 checkpoints
in `runs/pn4/` show what happens otherwise: their `meta.json` holds three keys
and records neither seed nor env id, so they cannot be reproduced from their own
metadata.

Layout per node, keeping the 3-artifact shape `ColumnTrainer.save` established:

    <run_dir>/
        model.zip               SB3 model (written by the trainer, optional here)
        policy_state_dict.pt    torch state dict
        meta.json               spec, spec_hash, status, seeds, eval results
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

from src.types import NodeStatus

from src.skills.node import SkillNode

META_NAME = "meta.json"
STATE_DICT_NAME = "policy_state_dict.pt"


class SpecMismatch(RuntimeError):
    """
    The stored checkpoint was trained under a different spec than the node asking
    for it. An error, never a warning: silently loading it would make every
    result built on that node meaningless.
    """


class NodeRegistry:
    """Saves and loads frozen nodes under a root directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def node_dir(self, node_id: str) -> Path:
        return self.root / _slug(node_id)

    # -- save ------------------------------------------------------------

    def save(
        self,
        node: SkillNode,
        *,
        extra_meta: Optional[dict[str, Any]] = None,
        run_dir: Optional[Path | str] = None,
    ) -> Path:
        d = Path(run_dir) if run_dir is not None else self.node_dir(node.id)
        d.mkdir(parents=True, exist_ok=True)

        if node.policy is not None and hasattr(node.policy, "state_dict"):
            torch.save(node.policy.state_dict(), d / STATE_DICT_NAME)

        meta: dict[str, Any] = {
            "id": node.id,
            "n": node.n,
            "m": node.m,
            "status": node.status.value,
            "spec_hash": node.spec_hash(),
            "spec": node.spec(),
            "children": [c.id for c in node.children],
            "lateral_sources": [s.id for s in node.lateral_sources],
        }
        meta.update(node.meta)
        if extra_meta:
            meta.update(extra_meta)

        (d / META_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return d

    # -- load ------------------------------------------------------------

    def read_meta(self, node_id: str, run_dir: Optional[Path | str] = None) -> dict[str, Any]:
        d = Path(run_dir) if run_dir is not None else self.node_dir(node_id)
        path = d / META_NAME
        if not path.exists():
            raise FileNotFoundError(f"no {META_NAME} in {d}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load_into(
        self,
        node: SkillNode,
        *,
        run_dir: Optional[Path | str] = None,
        map_location: str = "cpu",
        require_frozen: bool = True,
    ) -> SkillNode:
        """
        Load a checkpoint into `node`, refusing a spec mismatch [A: F18].

        `node.policy` must already be constructed (the network shape comes from
        the trainer, not from the registry).
        """
        d = Path(run_dir) if run_dir is not None else self.node_dir(node.id)
        meta = self.read_meta(node.id, run_dir=d)

        stored, wanted = meta.get("spec_hash"), node.spec_hash()
        if stored != wanted:
            raise SpecMismatch(
                f"{node.id}: checkpoint in {d} was trained under spec_hash {stored}, "
                f"but this node is {wanted}.\n"
                f"  stored spec: {json.dumps(meta.get('spec', {}), sort_keys=True)}\n"
                f"  wanted spec: {json.dumps(node.spec(), sort_keys=True)}"
            )

        if require_frozen and meta.get("status") != NodeStatus.FROZEN.value:
            raise RuntimeError(
                f"{node.id}: checkpoint status is {meta.get('status')!r}, not frozen. "
                "Only FROZEN nodes may be children or roots (v2 §9)."
            )

        sd_path = d / STATE_DICT_NAME
        if sd_path.exists():
            if node.policy is None:
                raise RuntimeError(f"{node.id}: attach a policy before loading weights")
            state = torch.load(sd_path, map_location=map_location, weights_only=True)
            node.policy.load_state_dict(state)

        node.status = NodeStatus(meta.get("status", NodeStatus.FROZEN.value))
        node.meta.update({k: v for k, v in meta.items() if k not in ("spec", "spec_hash")})
        if node.status is NodeStatus.FROZEN and node.policy is not None:
            params = getattr(node.policy, "parameters", None)
            if callable(params):
                for p in params():
                    p.requires_grad_(False)
        return node

    def list_nodes(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and (p / META_NAME).exists()
        )


def _slug(node_id: str) -> str:
    """'{0,1} get_key' -> '0-1_get_key'."""
    out = node_id.replace("{", "").replace("}", "").replace(",", "-").replace(" ", "_")
    return "".join(ch for ch in out if ch.isalnum() or ch in "-_")
