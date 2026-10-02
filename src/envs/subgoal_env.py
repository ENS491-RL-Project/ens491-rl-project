"""
SubgoalEnv: the MDP a primitive node is trained on.

Target Architecture v2 §5.1. `reset()` applies the node's initial-condition hook
and `step()` rewards the node's own goal -- never the environment's reward
[A: F10].

Wrapping order matters. SubgoalEnv wraps the *raw* MiniGrid env and the
observation wrappers go outside it:

    gym.make -> SubgoalEnv -> ImgObs -> Flatten -> ChannelNormalize -> Monitor

A reset hook mutates the grid (puts the key in the agent's hand, opens the door),
so the observation has to be regenerated afterwards. Keeping SubgoalEnv inside
the observation wrappers means it hands out a fresh `gen_obs()` dict and the
wrappers above convert it, instead of SubgoalEnv having to re-run the pipeline
itself.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import gymnasium as gym
import numpy as np

from src.envs.obs import apply_obs_pipeline, obs_pipeline_name
from src.skills.node import SkillNode
from src.skills.state import StateView, build_state_view

MAX_RESET_TRIES = 50


# --------------------------------------------------------------------------
# Reset hooks
# --------------------------------------------------------------------------

class ResetHook:
    """A named mutation of a freshly reset env. The name goes into `spec_hash`."""

    __slots__ = ("spec", "_fn")

    def __init__(self, spec: str, fn: Callable[[gym.Env], None]) -> None:
        self.spec = spec
        self._fn = fn

    def __call__(self, env: gym.Env) -> None:
        self._fn(env)

    def __repr__(self) -> str:
        return f"ResetHook({self.spec!r})"


def compose(*hooks: ResetHook) -> ResetHook:
    spec = "+".join(h.spec for h in hooks) if hooks else "none"

    def fn(env):
        for h in hooks:
            h(env)

    return ResetHook(spec, fn)


NO_HOOK = ResetHook("none", lambda env: None)


def _find(u, type_: str):
    for x in range(u.width):
        for y in range(u.height):
            c = u.grid.get(x, y)
            if c is not None and c.type == type_:
                return (x, y), c
    return None, None


def give_key() -> ResetHook:
    """Put the key in the agent's hand (T2 `open_door` starts holding it)."""

    def fn(env):
        u = env.unwrapped
        if u.carrying is not None:
            return
        pos, key = _find(u, "key")
        if key is None:
            raise RuntimeError("give_key: no key in this layout")
        u.grid.set(pos[0], pos[1], None)
        u.carrying = key
        u.carrying.cur_pos = np.array([-1, -1])

    return ResetHook("give_key", fn)


def open_door() -> ResetHook:
    """Unlock and open the door (T4 `go_to_goal` starts with it open)."""

    def fn(env):
        u = env.unwrapped
        _, door = _find(u, "door")
        if door is None:
            raise RuntimeError("open_door: no door in this layout")
        door.is_locked = False
        door.is_open = True

    return ResetHook("open_door", fn)


def clear_blocker() -> ResetHook:
    """Remove the ball blocking the door (BlockedUnlockPickup)."""

    def fn(env):
        u = env.unwrapped
        pos, ball = _find(u, "ball")
        if ball is not None:
            u.grid.set(pos[0], pos[1], None)

    return ResetHook("clear_blocker", fn)


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def relocate_blocker(*, tries: int = 64) -> ResetHook:
    """
    Move the ball that blocks the door to a reachable cell away from the door
    (H-7 Part B; T6 `pickup_box`, and the A5 transfer rows).

    BlockedUnlockPickup puts a ball on the only cell the door can be opened from.
    `clear_blocker()` deletes it, which is NOT the state `{0,5} move_blocker`
    leaves: after it the ball is still in the room, somewhere else. Whether the
    frozen Unlock skills survive a ball lying on the floor -- an object no frozen
    node has ever had in its observation (H-7 plan, V7) -- is exactly what the
    A5 rows measure, so the hook has to reproduce the real state.

    The new cell is drawn with the env's own `_rand_int` from the free cells
    reachable from the agent (with the ball lifted off the grid) that are not
    orthogonally adjacent to the door. A candidate is accepted only if, with the
    ball on it, the agent can still stand next to the key, next to the door and
    (if there is one on its side) next to the box: two obstacles in a 4x4 room can
    isolate a corner, and an episode no policy can solve looks like a learning
    plateau (H-6.6, deviation 2). No ball: a no-op, so one hook serves
    UnlockPickup and BlockedUnlockPickup. No acceptable cell: `UnsuitableLayout`.
    """

    def fn(env):
        u = env.unwrapped
        ball_pos, ball = _find(u, "ball")
        if ball is None:
            return
        door_pos, door = _find(u, "door")
        if door is None:
            raise RuntimeError("relocate_blocker: no door in this layout")
        start = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        u.grid.set(ball_pos[0], ball_pos[1], None)

        free = [c for c in reachable_cells(u, start)
                if u.grid.get(*c) is None and c != start and not _adjacent(c, door_pos)]
        # Whatever the agent could reach with the ball lifted must stay reachable
        # with the ball put down -- the key always, the box only when it is on
        # the agent's side (the door may be shut).
        targets = [p for p, _ in (_find(u, "key"), _find(u, "box"))
                   if p is not None and _standable(u, p, start)]
        door_ok = _door_approachable(u, door_pos, start)

        for _ in range(tries if free else 0):
            c = free[u._rand_int(0, len(free))]
            u.grid.set(c[0], c[1], ball)
            ball.cur_pos = np.array(c)
            if (_door_approachable(u, door_pos, start) == door_ok
                    and all(_standable(u, t, start) for t in targets)):
                return
            u.grid.set(c[0], c[1], None)

        u.grid.set(ball_pos[0], ball_pos[1], ball)
        raise UnsuitableLayout(
            f"relocate_blocker: no acceptable cell among {len(free)} free reachable cells"
        )

    return ResetHook("relocate_blocker", fn)


def key_off_door() -> ResetHook:
    """
    Reject a layout with a key on a cell orthogonally adjacent to the door
    (T6 `pickup_box`, H-7 amendment 1).

    `pickup_box` starts with the door open and an empty hand. A key lying on the
    door's approach cell can then only be passed by picking it up, which fills
    the hand again, and `pickup` needs an empty one: the episode asks for a drop
    the node is not meant to learn. At the real handoff it cannot happen,
    because the agent is standing on that cell when it drops the key. The H-7.1
    pre-flight counted 13 such layouts in 500. Raises `UnsuitableLayout`, so the
    env draws another layout; never moves anything.
    """

    def fn(env):
        u = env.unwrapped
        door_pos, door = _find(u, "door")
        if door is None:
            raise RuntimeError("key_off_door: no door in this layout")
        for x in range(u.width):
            for y in range(u.height):
                c = u.grid.get(x, y)
                if c is not None and c.type == "key" and _adjacent((x, y), door_pos):
                    raise UnsuitableLayout(f"key_off_door: a key at {(x, y)} next to the door")

    return ResetHook("key_off_door", fn)


def randomize_carrying(prob: float = 0.5, type_: str = "key") -> ResetHook:
    """
    With probability `prob`, start the episode already holding the object.

    [A: F29] MiniGrid writes the carried object into the agent's OWN cell of the
    observation -- `gen_obs_grid` sets grid[3, 6] to `self.carrying`. A primitive
    whose reset hook never gives it anything to hold therefore sees object index
    1 at (3,6) in every single training state, and reads 5 the moment a parent
    hands it control after `get_key`. That is a systematic shift in an input the
    policy has never varied over, and it cost `go_to_goal` 0.96 -> 0.52 between
    its own hook distribution and the state `unlock_door` actually produces.

    Randomizing it makes the policy invariant instead of tying it to one parent.
    """

    def fn(env):
        u = env.unwrapped
        if u.carrying is not None or not u._rand_bool():
            return
        pos, obj = _find(u, type_)
        if obj is None:
            return
        u.grid.set(pos[0], pos[1], None)
        u.carrying = obj
        u.carrying.cur_pos = np.array([-1, -1])

    return ResetHook(f"randomize_carrying:{type_}:{prob}", fn)


class UnsuitableLayout(RuntimeError):
    """
    This layout cannot host the node's initial conditions; draw another one.

    Distinct from a plain `RuntimeError`, which means the hook is wrong. A reset
    hook defines a node's initial-condition distribution, and some randomly
    generated layouts simply cannot satisfy it -- DoorKey-6x6 draws its splitting
    wall at x=2 or x=3, so the left room is sometimes a single column of four
    cells, and two keys cannot both be independently reachable there unless the
    agent happens to stand between them. `SubgoalEnv.reset` and
    `ResetHookEnv.reset` catch this and draw the next layout, which keeps the
    distribution well defined instead of crashing a long run on one bad draw.
    The retry stream is deterministic given the episode's seed, so an episode is
    still reproducible.
    """


def recolor_door(colors: Sequence[str] = ("yellow", "blue")) -> ResetHook:
    """
    Draw a colour per episode and give the door that colour.

    The first half of the genuine-choice task (H-6 Part C). DoorKey always builds
    a yellow door with a yellow key, so a composite has nothing to choose;
    recolouring makes the door's colour a per-episode signal that is visible in
    the observation's colour channel and, by `Door.toggle`, decisive -- a locked
    door opens only for a key of its own colour.

    Uses the env's own RNG, so the episode stays reproducible from its seed.
    """
    spec = "recolor_door:" + "|".join(colors)

    def fn(env):
        u = env.unwrapped
        _, door = _find(u, "door")
        if door is None:
            raise RuntimeError("recolor_door: no door in this layout")
        door.color = colors[u._rand_int(0, len(colors))]

    return ResetHook(spec, fn)


def remove_keys() -> ResetHook:
    """
    Clear every key off the grid, and out of the agent's hand.

    Runs before the agent's pose is randomized so the flood fill is not blocked
    by the key the layout happened to place: in a one-column room, a key between
    the agent and the rest of the room makes every other cell unreachable.
    """

    def fn(env):
        u = env.unwrapped
        if u.carrying is not None and u.carrying.type == "key":
            u.carrying = None
        for x in range(u.width):
            for y in range(u.height):
                cell = u.grid.get(x, y)
                if cell is not None and cell.type == "key":
                    u.grid.set(x, y, None)

    return ResetHook("remove_keys", fn)


def _standable(u, cell: tuple[int, int], start: tuple[int, int]) -> bool:
    """Can the agent reach a cell it could pick `cell` up from?"""
    reach = set(reachable_cells(u, start)) | {start}
    x, y = cell
    return any((nx, ny) in reach
               for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))


def _door_approachable(u, door_pos: tuple[int, int], start: tuple[int, int]) -> bool:
    """
    Can the agent still reach a cell it could toggle the door from?

    Keeping a key off the approach cell is not enough: in a one-column room a key
    between the agent and that cell blocks the path without occupying it, and the
    door is unopenable for any policy. Measured before this check existed: 3 of
    the 50 stopping seeds and 4 of the 50 held-out seeds, which is exactly the
    gap between {0,8} open_door_tk's 0.94 stopping score and the 0.84 it
    generalized to -- 4 of its 8 held-out failures were episodes no policy could
    have solved.
    """
    dx, dy = door_pos
    reach = set(reachable_cells(u, start)) | {start}
    for nx, ny in ((dx - 1, dy), (dx + 1, dy), (dx, dy - 1), (dx, dy + 1)):
        if not (0 <= nx < u.width and 0 <= ny < u.height):
            continue
        cell = u.grid.get(nx, ny)
        if cell is not None and cell.type == "wall":
            continue
        if (nx, ny) in reach:
            return True
    return False


def _dir_sees(u, direction: int, object_idx: int) -> bool:
    u.agent_dir = direction
    return bool((u.gen_obs()["image"][:, :, 0] == object_idx).any())


def place_two_keys(
    colors: Sequence[str] = ("yellow", "blue"),
    *,
    tries: int = 64,
) -> ResetHook:
    """
    Place one key matching the door and one decoy of another colour.

    The second half of the genuine-choice task. Both keys must be **independently
    reachable**: the agent has to be able to pick up either one without first
    picking up the other, because MiniGrid `pickup` needs an empty hand and
    neither `get_key_<colour>` primitive is trained to drop. If the decoy were
    only reachable through the matching key, `get_key_<decoy>` would be
    unachievable and the composite's choice would not be a choice at all.

    Checked, not assumed: a candidate pair is accepted only if each key still has
    a standable neighbour once the other key is on the grid. Layouts where no
    pair works raise `UnsuitableLayout` and the caller draws another.
    """
    # ':door_clear' is in the spec string on purpose. spec_hash hashes a hook's
    # NAME, not its implementation [A: F18], so tightening the acceptance rule
    # without renaming it would leave every existing checkpoint loadable while
    # the distribution it was trained on had silently changed. Renaming makes the
    # registry refuse them, which is the behaviour F18 exists for.
    spec = "place_two_keys:" + "|".join(colors) + ":door_clear"

    def fn(env):
        from minigrid.core.world_object import Key

        u = env.unwrapped
        door_pos, door = _find(u, "door")
        if door is None:
            raise RuntimeError("place_two_keys: no door in this layout")
        others = [c for c in colors if c != door.color]
        if not others:
            raise RuntimeError(f"place_two_keys: no colour differs from {door.color}")
        decoy_colour = others[u._rand_int(0, len(others))]

        start = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        # Never block the door's approach. MiniGrid `toggle` acts on
        # `front_pos`, so the door can only be opened from a cell orthogonally
        # adjacent to it, and in DoorKey's vertical wall there is exactly ONE
        # such cell on the agent's side. A key there makes the door impossible to
        # open for any policy. Measured before this guard existed: 59 of 400
        # layouts (14.7%), which capped every door-opening primitive at ~0.85 and
        # was misread as a learning limit until the failures were inspected.
        dx, dy = door_pos
        forbidden = {(dx, dy), (dx - 1, dy), (dx + 1, dy), (dx, dy - 1), (dx, dy + 1)}
        free = [c for c in reachable_cells(u, start)
                if u.grid.get(*c) is None and c != start and c not in forbidden]
        if len(free) < 2:
            raise UnsuitableLayout(
                f"place_two_keys: {len(free)} free reachable cells, need 2"
            )

        for _ in range(tries):
            i = u._rand_int(0, len(free))
            j = u._rand_int(0, len(free))
            if i == j:
                continue
            a, b = free[i], free[j]
            u.grid.set(a[0], a[1], Key(door.color))
            u.grid.set(b[0], b[1], Key(decoy_colour))
            if (_standable(u, a, start) and _standable(u, b, start)
                    and _door_approachable(u, door_pos, start)):
                return
            u.grid.set(a[0], a[1], None)
            u.grid.set(b[0], b[1], None)

        raise UnsuitableLayout(
            f"place_two_keys: no pair of independently reachable cells in "
            f"{tries} tries ({len(free)} free cells)"
        )

    return ResetHook(spec, fn)


def give_key_of(which: str = "matching") -> ResetHook:
    """
    Put one specific key in the agent's hand, leaving the other on the floor.

    `which` is "matching" (the door's colour) or "decoy" (any other). Used to
    reproduce the state the choice task's parent hands `open_door`: a key in
    hand, a second key still lying on the grid, and a door that may be either
    colour. [A: F29] `{0,2} open_door` scored 0.73 from that state despite
    scoring 1.00 on its own hook distribution -- it had never seen a spare key on
    the floor -- which is why `{0,8} open_door_tk` is trained with this hook.

    "decoy" is how the expected-failure case is measured: the door must NOT open,
    and that failure is the signal the composite learns from.
    """
    spec = f"give_key_of:{which}"

    def fn(env):
        u = env.unwrapped
        if u.carrying is not None:
            return
        _, door = _find(u, "door")
        if door is None:
            raise RuntimeError("give_key_of: no door in this layout")
        want_match = which == "matching"
        for x in range(u.width):
            for y in range(u.height):
                cell = u.grid.get(x, y)
                if cell is None or cell.type != "key":
                    continue
                if (cell.color == door.color) == want_match:
                    u.grid.set(x, y, None)
                    u.carrying = cell
                    u.carrying.cur_pos = np.array([-1, -1])
                    return
        raise UnsuitableLayout(f"give_key_of: no {which} key in this layout")

    return ResetHook(spec, fn)


def face_toward(type_: str = "door") -> ResetHook:
    """
    Turn the agent so an object of `type_` is inside its 7x7 view.

    [A: F29, H-6 V7] A composite makes its child choice from one observation at
    one instant, with a feedforward policy and no memory. In DoorKey-6x6 with a
    random facing the door is inside that view in only 163 of 300 resets; facing
    it, in 300 of 300. A choice task whose signal is invisible half the time
    measures partial observability, not composition.

    The direction is found by *checking the observation* rather than by
    hard-coding "face right", so the hook works in any layout, and it is drawn
    uniformly among the directions that keep the object visible, so pose
    diversity survives.
    """
    spec = f"face_toward:{type_}"

    def fn(env):
        from minigrid.core.constants import OBJECT_TO_IDX

        u = env.unwrapped
        want = OBJECT_TO_IDX[type_]
        original = int(u.agent_dir)
        ok = [d for d in range(4) if _dir_sees(u, d, want)]
        u.agent_dir = ok[u._rand_int(0, len(ok))] if ok else original

    return ResetHook(spec, fn)


def two_keys(colors: Sequence[str] = ("yellow", "blue")) -> ResetHook:
    """
    The complete initial-condition hook of the genuine-choice task (H-6 Part C).

    One locked door of a per-episode colour, a matching key, a decoy key of
    another colour, a random agent position, and a facing that keeps the door in
    view. The order is the content of this hook:

      1. recolour the door       -- fixes which key is the right one
      2. remove the layout's key -- so step 3's flood fill is unobstructed
      3. randomize the pose      -- over the whole region, not just one side
      4. place both keys         -- both independently reachable, given the pose
      5. face the door           -- so the signal is in the observation

    Steps 3 and 4 cannot be swapped: which cells are reachable depends on where
    the agent stands, and whether a key is pickupable depends on both.
    """
    return compose(
        recolor_door(colors),
        remove_keys(),
        randomize_agent_pose(anchor=None),
        place_two_keys(colors),
        face_toward("door"),
    )


def _traversable(cell) -> bool:
    return cell is None or cell.can_overlap()


def reachable_cells(u, start: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Flood fill from `start`, treating closed or locked doors as walls.

    This is what makes `randomize_agent_pose` work for both env families without
    branching on the env [A: F30]. `RoomGrid.place_agent(i, j)` takes room
    coordinates while `MiniGridEnv.place_agent(top, size)` takes a rectangle, and
    neither understands "the side of the door the key is on" -- so neither is
    used. Spawning the agent behind a locked door would make the episode
    unsolvable and look like a learning failure.
    """
    seen = {start}
    stack = [start]
    out: list[tuple[int, int]] = []
    while stack:
        x, y = stack.pop()
        cell = u.grid.get(x, y)
        if _traversable(cell):
            out.append((x, y))
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nx < u.width and 0 <= ny < u.height) or (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            if _traversable(u.grid.get(nx, ny)):
                stack.append((nx, ny))
    return out


def randomize_agent_pose(anchor: Optional[str] = None) -> ResetHook:
    """
    Place the agent at a random pose in the region reachable from `anchor`.

    [A: F29] A primitive trained from one fixed start pose only ever sees the
    states that pose produces, but a parent hands it whatever state it reached.
    Randomizing the start is the cheap half of closing that gap.

    `anchor` is an object type ("key", "box", "goal") whose region the agent must
    share; None means anywhere reachable from where the agent already is.
    """
    spec = f"randomize_agent_pose:{anchor or 'any'}"

    def fn(env):
        u = env.unwrapped
        if anchor == "goal":
            start, obj = _find(u, "goal")
        elif anchor is not None:
            start, obj = _find(u, anchor)
        else:
            start, obj = (int(u.agent_pos[0]), int(u.agent_pos[1])), True

        if start is None or obj is None:
            start = (int(u.agent_pos[0]), int(u.agent_pos[1]))

        cells = [c for c in reachable_cells(u, start) if u.grid.get(*c) is None]
        if not cells:
            return
        idx = u._rand_int(0, len(cells))
        u.agent_pos = np.array(cells[idx])
        u.agent_dir = u._rand_int(0, 4)

    return ResetHook(spec, fn)


# --------------------------------------------------------------------------
# SubgoalEnv
# --------------------------------------------------------------------------

class SubgoalEnv(gym.Wrapper):
    """
    One primitive node's training MDP.

    reward  = 1 - 0.9 * steps / timeout when the node's goal becomes true, else 0
    terminated = node.goal(s)
    truncated  = node timeout, or the base env ending without the goal
    """

    def __init__(
        self,
        env: gym.Env,
        node: SkillNode,
        reset_hook: ResetHook = NO_HOOK,
        *,
        max_reset_tries: int = MAX_RESET_TRIES,
    ) -> None:
        super().__init__(env)
        self.node = node
        self.reset_hook = reset_hook
        self.max_reset_tries = max_reset_tries
        self.steps = 0
        self.version = 0
        self.sv: Optional[StateView] = None

    # -- gym API ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        for attempt in range(self.max_reset_tries):
            try:
                self.reset_hook(self.env)
            except UnsuitableLayout:
                # The hook's initial conditions cannot be built in this layout;
                # draw the next one. Deterministic given the episode's seed.
                obs, info = self.env.reset()
                continue
            self.version = 0
            self.sv = build_state_view(self.env, version=0)
            # [A: F7] If the goal already holds, the episode would terminate at
            # step 0 with SUCCESS and silently inflate the success rate.
            if not self.node.goal(self.sv):
                break
            obs, info = self.env.reset()
        else:
            raise RuntimeError(
                f"{self.node.id}: goal held at reset in {self.max_reset_tries} "
                f"consecutive layouts -- check the goal predicate against the hook "
                f"({self.reset_hook.spec})"
            )

        self.steps = 0
        return self.env.unwrapped.gen_obs(), info

    def step(self, action):
        # [A: F10] The env's own reward is discarded; this node is paid for its
        # own goal and nothing else.
        _, _, env_term, env_trunc, info = self.env.step(action)
        self.steps += 1
        self.version += 1
        self.sv = build_state_view(self.env, version=self.version)

        reached = bool(self.node.goal(self.sv))
        reward = 1.0 - 0.9 * (self.steps / self.node.timeout) if reached else 0.0
        truncated = (not reached) and (
            self.steps >= self.node.timeout or env_term or env_trunc
        )

        info = dict(info)
        info["subgoal_reached"] = reached
        info["subgoal_steps"] = self.steps
        return self.env.unwrapped.gen_obs(), reward, reached, truncated, info

    @property
    def state_view(self) -> StateView:
        if self.sv is None:
            raise RuntimeError("reset() first")
        return self.sv


def make_subgoal_env(
    node: SkillNode,
    env_id: str,
    reset_hook: ResetHook = NO_HOOK,
    *,
    seed: Optional[int] = None,
    normalize: bool = True,
    monitor: bool = True,
) -> gym.Env:
    """`gym.make -> SubgoalEnv -> ImgObs -> Flatten -> ChannelNormalize -> Monitor`."""
    from stable_baselines3.common.monitor import Monitor

    env = gym.make(env_id)
    env = SubgoalEnv(env, node, reset_hook)
    env = apply_obs_pipeline(env, normalize=normalize)
    if monitor:
        env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def hook_spec(reset_hook: ResetHook) -> str:
    return reset_hook.spec


def pipeline_spec(normalize: bool = True) -> str:
    return obs_pipeline_name(normalize)
