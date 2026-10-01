"""
SkillNode: the only policy-holding object in the architecture.

Target Architecture v2 §2–§4. A node is primitive (no children, policy emits a
MiniGrid action) or composite (children, policy emits a child index); the two
differ only in `children`, never in type. Nothing here branches on `n`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

from src.types import NodeStatus, Outcome

from src.skills.state import StateView


# --------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------

class Predicate:
    """
    A named predicate over a StateView.

    The name matters: it is what `spec_hash` hashes [A: F18]. A bare lambda has
    no stable identity, so a frozen checkpoint could not record which goal it was
    trained against.
    """

    __slots__ = ("spec", "_fn")

    def __init__(self, spec: str, fn: Callable[[StateView], bool]) -> None:
        self.spec = spec
        self._fn = fn

    def __call__(self, sv: StateView) -> bool:
        return bool(self._fn(sv))

    def __repr__(self) -> str:
        return f"Predicate({self.spec!r})"


ALWAYS = Predicate("always", lambda sv: True)
NEVER = Predicate("never", lambda sv: False)


# --------------------------------------------------------------------------
# Policy adapters [A: F19]
# --------------------------------------------------------------------------

class PrimitivePolicy(Protocol):
    def act(self, obs: Any, deterministic: bool = True) -> int:
        """Return a MiniGrid action, 0-6."""


class CompositePolicy(Protocol):
    def select(self, obs: Any, mask: Sequence[bool], deterministic: bool = True) -> int:
        """Return an index into the node's children. `mask` always has >=1 True."""


# --------------------------------------------------------------------------
# SkillNode
# --------------------------------------------------------------------------

class SkillNode:
    """
    One skill. Primitive if `children` is empty, composite otherwise.

    `children` is immutable after construction (v2 §3), `n` is derived from it and
    never assigned by hand, and the constructor refuses a cyclic graph [A: F21] or
    a timeout that violates the composition invariant [A: F13].
    """

    def __init__(
        self,
        node_id: str,
        *,
        children: Iterable["SkillNode"] = (),
        goal: Predicate = NEVER,
        initiation: Predicate = ALWAYS,
        timeout: int = 100,
        max_decisions: int = 64,
        env_ids: Sequence[str] = (),
        obs_pipeline: str = "ImgObs+Flatten+ChannelNormalize",
        normalize_obs: bool = True,
        lateral_sources: Iterable["SkillNode"] = (),
        policy: Any = None,
        m: int = 1,
    ) -> None:
        self.id = node_id
        self._children: tuple[SkillNode, ...] = tuple(children)
        self.goal = goal
        self.initiation = initiation
        self.timeout = int(timeout)
        self.max_decisions = int(max_decisions)
        self.env_ids = tuple(env_ids)
        self.obs_pipeline = obs_pipeline
        self.normalize_obs = bool(normalize_obs)
        self.lateral_sources = tuple(lateral_sources)
        self.policy = policy
        self.m = int(m)
        self.status = NodeStatus.DECLARED
        self.meta: dict[str, Any] = {}

        self._validate()

    # -- validation ------------------------------------------------------

    def _validate(self) -> None:
        if self.timeout <= 0:
            raise ValueError(f"{self.id}: timeout must be positive, got {self.timeout}")
        if self.max_decisions <= 0:
            raise ValueError(f"{self.id}: max_decisions must be positive")

        # [A: F21] Acyclicity. "Acyclic by construction" holds only while nodes
        # are created in order, which a graph loaded from disk need not respect.
        # Any cycle in the reachable subgraph is fatal, not just one through this
        # node: `n` recurses over children, so a cycle anywhere below would
        # recurse forever the first time the level is read.
        self._assert_acyclic()

        # [A: F13] Timeout composition. Without this a parent can be unable to
        # succeed for reasons that look exactly like a learning failure.
        for c in self._children:
            if self.timeout < c.timeout:
                raise ValueError(
                    f"{self.id}: timeout {self.timeout} < child {c.id} timeout "
                    f"{c.timeout}; a parent must outlive its children"
                )

        # [A: F20] One lateral source until C-2; several would have to be summed,
        # and LateralMlpExtractor implements exactly one.
        if len(self.lateral_sources) > 1:
            raise ValueError(
                f"{self.id}: {len(self.lateral_sources)} lateral sources; "
                "v2 §7 allows at most 1 until C-2"
            )
        if self.lateral_sources and not self.is_primitive:
            raise ValueError(f"{self.id}: laterals are for primitive nodes only (v2 §7)")

    def _assert_acyclic(self) -> None:
        WHITE, GREY, BLACK = 0, 1, 2
        color: dict[int, int] = {}
        # Iterative DFS: (node, children_already_pushed)
        stack: list[tuple[SkillNode, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                color[id(node)] = BLACK
                continue
            state = color.get(id(node), WHITE)
            if state == BLACK:
                continue
            if state == GREY:
                raise ValueError(f"{self.id}: cycle -- {node.id} is reachable from itself")
            color[id(node)] = GREY
            stack.append((node, True))
            for c in node._children:
                if color.get(id(c), WHITE) == GREY:
                    raise ValueError(f"{self.id}: cycle -- {c.id} is reachable from itself")
                if color.get(id(c), WHITE) == WHITE:
                    stack.append((c, False))

    # -- structure -------------------------------------------------------

    @property
    def children(self) -> tuple["SkillNode", ...]:
        """Immutable. To change a composition, create a new node (v2 §3)."""
        return self._children

    @property
    def is_primitive(self) -> bool:
        return len(self._children) == 0

    @property
    def n(self) -> int:
        """Level, derived from children. Never assigned by hand (v2 §2)."""
        if self.is_primitive:
            return 0
        return 1 + max(c.n for c in self._children)

    def descendants(self) -> list["SkillNode"]:
        out: list[SkillNode] = []
        seen: set[int] = set()
        stack = list(self._children)
        while stack:
            nd = stack.pop()
            if id(nd) in seen:
                continue
            seen.add(id(nd))
            out.append(nd)
            stack.extend(nd._children)
        return out

    # -- semantics -------------------------------------------------------

    def available(
        self, sv: StateView, cache: Optional[dict] = None
    ) -> bool:
        """
        v2 §4.3 [A: F1]: `not goal(s) and (primitive or any child available)`.

        `initiation` deliberately does NOT appear. With it, every composite in the
        §11 curriculum has exactly one legal child at each decision point and the
        policy has nothing to learn.

        `cache` memoizes on `(id(node), sv.version)` [A: F4] -- without it this is
        exponential in depth on a DAG with shared children.
        """
        if cache is None:
            cache = {}
        key = (id(self), sv.version)
        hit = cache.get(key)
        if hit is not None:
            return hit

        cache[key] = False  # guards re-entry while this node is being evaluated
        if self.goal(sv):
            result = False
        elif self.is_primitive:
            result = True
        else:
            result = any(c.available(sv, cache) for c in self._children)
        cache[key] = result
        return result

    def check_termination(self, sv: StateView, steps: int) -> Optional[Outcome]:
        """v2 §4.2. INTERRUPTED is assigned by the executor, not here."""
        if self.goal(sv):
            return Outcome.SUCCESS
        if steps >= self.timeout:
            return Outcome.TIMEOUT
        return None

    # -- identity [A: F18] -----------------------------------------------

    def spec(self) -> dict[str, Any]:
        """The canonical description a checkpoint is valid for."""
        return {
            "id": self.id,
            "env_ids": list(self.env_ids),
            "goal": self.goal.spec,
            "initiation": self.initiation.spec,
            "timeout": self.timeout,
            "max_decisions": self.max_decisions,
            "obs_pipeline": self.obs_pipeline,
            "normalize_obs": self.normalize_obs,
            "children": [c.spec_hash() for c in self._children],
            "lateral_sources": [s.spec_hash() for s in self.lateral_sources],
        }

    def spec_hash(self) -> str:
        blob = json.dumps(self.spec(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # -- lifecycle -------------------------------------------------------

    def freeze(self) -> None:
        """
        STABLE -> FROZEN. No gradients, but the forward pass keeps running:
        laterals and parents need it (v2 §5.4).
        """
        if self.policy is None:
            raise RuntimeError(f"{self.id}: cannot freeze -- policy is None")
        params = getattr(self.policy, "parameters", None)
        if callable(params):
            for p in params():
                p.requires_grad_(False)
        self.status = NodeStatus.FROZEN

    @property
    def frozen(self) -> bool:
        return self.status is NodeStatus.FROZEN

    def __repr__(self) -> str:
        kind = "primitive" if self.is_primitive else f"composite[{len(self._children)}]"
        return f"SkillNode({self.id!r}, n={self.n}, {kind}, {self.status.value})"
