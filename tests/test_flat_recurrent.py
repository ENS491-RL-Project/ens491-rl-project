"""W4(a): the recurrent baseline's evaluation loop against the Executor path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.envs.subgoal_env import NO_HOOK
from src.training.evaluation import evaluate_node
from src.training.node_trainer import attach_sb3_primitive_policy, subgoal_env_factory
from train_flat_baseline import TASKS, build_flat_ppo
from train_flat_recurrent import evaluate, train_recurrent


def test_loop_matches_evaluate_node_for_a_memoryless_policy():
    env_id = TASKS["doorkey"]["env"]
    node = build_flat_ppo("doorkey")
    attach_sb3_primitive_policy(node, env_id, NO_HOOK)      # untrained PPO, deterministic
    seeds = [20000, 20001, 20002]

    ref = evaluate_node(node, subgoal_env_factory(node, env_id, NO_HOOK, monitor=False), seeds=seeds)
    got = evaluate(node.policy.sb3_policy.predict, node, env_id, seeds)

    assert got.per_episode == ref.per_episode
    assert got.lengths == ref.lengths


def test_recurrent_training_runs_and_reports():
    node = build_flat_ppo("doorkey", kind="rppo")
    report, model = train_recurrent(node, TASKS["doorkey"]["env"], total_timesteps=600,
                                    eval_every=512, n_eval=2, n_envs=2)
    assert report["policy_updates"] >= 1 and len(report["criterion"]["history"]) >= 1
    assert hasattr(node.policy, "state_dict")
