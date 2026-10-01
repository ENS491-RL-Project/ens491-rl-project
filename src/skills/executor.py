"""
Executor: the active stack that runs a skill graph.

Target Architecture v2 §4.4. One executor serves runtime, evaluation and
`OptionEnv`. No code path depends on `node.n` -- depth is whatever the graph is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.types import Outcome

from src.skills.node import SkillNode
from src.skills.state import StateView

DETERMINISTIC = "deterministic"
STOCHASTIC = "stochastic"


@dataclass
class Frame:
    node: SkillNode
    steps: int = 0        # env steps since this node was pushed
    ret: float = 0.0      # discounted return since push


@dataclass
class TraceEvent:
    """One line of the execution trace. Serializable by construction [A: F28]."""

    depth: int
    node_id: str
    event: str                      # SELECT | CLOSE
    outcome: Optional[str] = None
    steps: int = 0
    version: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "node_id": self.node_id,
            "event": self.event,
            "outcome": self.outcome,
            "steps": self.steps,
            "version": self.version,
        }


class ExecutorError(RuntimeError):
    """Raised instead of hanging. A hung training run costs far more to find."""


class Executor:
    """
    Runs `root` over an environment, one env action at a time.

    Usage, per env step:
        a = ex.act(obs, sv)
        if a is None: episode over
        obs, r, ... = env.step(a); sv = build_state_view(env, version+1)
        ex.observe_reward(r)
    """

    MAX_ITERS = 256   # [A: F6] bound on one act() call; exceeding it is a bug

    def __init__(
        self,
        root: SkillNode,
        gamma: float = 0.99,                 # [A: F8] not a module global
        execution_mode: str = DETERMINISTIC,  # [A: F17]
        max_decisions: Optional[int] = None,  # [A: F2] mandatory; defaults to the node's
    ) -> None:
        self.root = root
        self.gamma = float(gamma)
        self.execution_mode = execution_mode
        self.max_decisions = int(
            max_decisions if max_decisions is not None else root.max_decisions
        )

        self.stack: list[Frame] = [Frame(root)]
        self.decisions = 0
        self.reselections = 0
        self.last_outcome: Optional[Outcome] = None
        self.trace: list[TraceEvent] = []

        self._cache: dict = {}
        self._cache_version: Optional[int] = None
        self._last_closed: dict[int, tuple[int, Outcome]] = {}  # id(node) -> (version, outcome)

    # -- public ----------------------------------------------------------

    @property
    def deterministic(self) -> bool:
        return self.execution_mode == DETERMINISTIC

    @property
    def depth(self) -> int:
        return len(self.stack)

    @property
    def finished(self) -> bool:
        return not self.stack

    def act(self, obs: Any, sv: StateView) -> Optional[int]:
        """
        Return the next env action, or None when the root has finished.

        A bounded loop, not recursion [A: F6]: closing a frame re-enters the loop
        rather than calling act() again, and MAX_ITERS turns a cyclic
        termination/selection into an exception instead of a hang.
        """
        self._refresh_cache(sv)

        for _ in range(self.MAX_ITERS):
            if not self.stack:
                return None

            # 1. Termination, top-down: close the highest finished node and
            #    everything under it (v2 §4.2).
            closed = False
            for k, frame in enumerate(self.stack):
                outcome = frame.node.check_termination(sv, frame.steps)
                if outcome is not None:
                    self._close_from(k, outcome, sv)
                    closed = True
                    break
            if closed:
                continue
            if not self.stack:
                return None

            # 2. Selection, top-down: descend until a primitive is on top.
            top = self.stack[-1].node
            if top.is_primitive:
                return int(top.policy.act(obs, deterministic=self.deterministic))

            if self.decisions >= self.max_decisions:       # [A: F2]
                self._close_from(0, Outcome.TIMEOUT, sv)
                return None

            mask = [c.available(sv, self._cache) for c in top.children]
            if not any(mask):
                # [A: F3] Never hand an empty mask to a policy. Close this frame
                # and let its parent choose again.
                self._close_from(len(self.stack) - 1, Outcome.TIMEOUT, sv)
                continue

            idx = int(top.policy.select(obs, mask, deterministic=self.deterministic))
            if not (0 <= idx < len(top.children)) or not mask[idx]:
                raise ExecutorError(
                    f"{top.id}: policy selected child {idx}, which is not available "
                    f"(mask={mask})"
                )
            child = top.children[idx]

            prev = self._last_closed.get(id(child))
            if prev is not None and prev[0] == sv.version and prev[1] is Outcome.TIMEOUT:
                # [A: F2] Legal under goal-only masking, and the signal the parent
                # must learn from -- but counted, and bounded by max_decisions.
                self.reselections += 1

            self.decisions += 1
            self.stack.append(Frame(child))
            self.trace.append(
                TraceEvent(len(self.stack), child.id, "SELECT", None, 0, sv.version)
            )

        raise ExecutorError(
            f"act() exceeded MAX_ITERS={self.MAX_ITERS} -- cyclic termination/selection "
            f"under {self.root.id}"
        )

    def observe_reward(self, r: float) -> None:
        """
        [A: F8] A frame pushed at env step t has steps == 0 while acting at t, so
        the discount is applied before the counter advances.
        """
        for frame in self.stack:
            frame.ret += (self.gamma ** frame.steps) * r
            frame.steps += 1

    def trace_dicts(self) -> list[dict[str, Any]]:
        return [e.as_dict() for e in self.trace]

    def selected_children(self) -> list[str]:
        """Node ids actually selected -- the input to 'effective level' [A: F28]."""
        return [e.node_id for e in self.trace if e.event == "SELECT"]

    # -- internals -------------------------------------------------------

    def _refresh_cache(self, sv: StateView) -> None:
        """[A: F4] The availability cache is valid for exactly one env step."""
        if self._cache_version != sv.version:
            self._cache = {}
            self._cache_version = sv.version

    def _close_from(self, k: int, outcome: Outcome, sv: StateView) -> None:
        """
        Close stack[k] with `outcome`; everything below it is INTERRUPTED
        (v2 §4.2). Control returns to stack[k-1], which selects again.
        """
        for depth in range(len(self.stack) - 1, k, -1):
            frame = self.stack[depth]
            self._last_closed[id(frame.node)] = (sv.version, Outcome.INTERRUPTED)
            self.trace.append(
                TraceEvent(
                    depth + 1, frame.node.id, "CLOSE",
                    Outcome.INTERRUPTED.value, frame.steps, sv.version,
                )
            )

        frame = self.stack[k]
        self._last_closed[id(frame.node)] = (sv.version, outcome)
        self.trace.append(
            TraceEvent(k + 1, frame.node.id, "CLOSE", outcome.value, frame.steps, sv.version)
        )
        if k == 0:
            self.last_outcome = outcome

        del self.stack[k:]
