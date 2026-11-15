"""Frozen scripted graphs for runtime sensitivity controls; no learned checkpoints."""
from src.skills.node import SkillNode, Predicate, ALWAYS, NEVER
from src.types import NodeStatus

UNLOCK = 'MiniGrid-Unlock-v0'


class FixedAction:
    def __init__(self, name, log=None):
        self.name, self.log = name, log
    def act(self, obs, deterministic=True):
        if self.log is not None:
            self.log.append(self.name)
        return 2


class FirstAvailable:
    def select(self, obs, mask, deterministic=True):
        return list(mask).index(True)


def at(step):
    return Predicate(f'audit:step>={step}', lambda sv: sv.step_count >= step)


def primitive(name, goal=NEVER, timeout=1, initiation=ALWAYS, log=None):
    node = SkillNode(name, goal=goal, initiation=initiation, timeout=timeout, env_ids=[UNLOCK])
    node.policy = FixedAction(name, log)
    node.status = NodeStatus.STABLE
    node.freeze()
    return node


def composite(name, children, goal=NEVER, timeout=20, cap=2, frozen=True):
    node = SkillNode(name, children=children, goal=goal, timeout=timeout,
                     max_decisions=cap, env_ids=[UNLOCK])
    node.policy = FirstAvailable()
    if frozen:
        node.status = NodeStatus.STABLE
        node.freeze()
    return node


def depth_task(log=None):
    leaf = primitive('audit_leaf', log=log)
    inner = composite('audit_inner', [leaf])
    middle = composite('audit_middle', [inner])
    direct = primitive('audit_direct', log=log)
    return composite('audit_depth3', [middle, direct], goal=at(5), frozen=False)


def timeout_task(log=None):
    a = primitive('audit_a', goal=at(3), timeout=10, log=log)
    b = primitive('audit_b', goal=NEVER, timeout=10, log=log)
    return composite('audit_timeout', [a, b], goal=at(13), timeout=12, cap=20, frozen=False)


def strict_task(log=None):
    bad = primitive('audit_forbidden', initiation=NEVER, timeout=4, log=log)
    good = primitive('audit_allowed', goal=at(2), timeout=4, log=log)
    inner = composite('audit_mask_inner', [bad, good], goal=at(2), timeout=8)
    return composite('audit_mask_root', [inner], goal=at(2), timeout=8, frozen=False)
