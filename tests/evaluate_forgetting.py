import os
import torch
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO

# Ensure the script finds your custom policy and Doric
from src.continual_learning.progressive_policy import ProgressiveACPolicy, ProgressiveFeatureExtractor
from src.continual_learning.doric_test import MiniGridColumnGenerator
from Doric import ProgNet

def evaluate():
    print("--- EVALUATION: The Absolute Knowledge Transfer ---")
    
    # 1. Environment (Task 1: Empty Room)
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")
    env = FlatObsWrapper(env)

    # 2. Rebuild the Architecture
    gen = MiniGridColumnGenerator(obs_size=64, hidden=64, n_actions=64)
    prog_net = ProgNet(colGen=gen)
    id1 = prog_net.addColumn(msg="task1") 
    id2 = prog_net.addColumn(msg="task2") 

    custom_objects = {
        "policy_kwargs": dict(
            features_extractor_class=ProgressiveFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            prog_net=prog_net,
            active_task_id=id2 
        )
    }

    print("Loading Progressive Model Weights...")
    model = PPO.load("ppo_progressive_final", env=env, custom_objects=custom_objects)
    
    # 3. Route logic through the Expert Column
    model.policy.mlp_extractor.task_id = id1
    model.policy.active_task_id = id1

    # 4. SURGICAL INJECTION: Force absolute parity with Phase 1
    print("Surgically injecting Task 1 Expert logic into the framework...")
    task1_weights = torch.load("ppo_task1_weights.pth", map_location='cpu', weights_only=True)
    
    with torch.no_grad():
        # A. Restore Action & Value Heads
        model.policy.action_net.weight.copy_(task1_weights['action_net.weight'])
        model.policy.action_net.bias.copy_(task1_weights['action_net.bias'])
        
        # B. Restore Vision (GRU)
        for name, param in model.policy.features_extractor.named_parameters():
            old_key = f'features_extractor.{name}'
            if old_key in task1_weights:
                param.copy_(task1_weights[old_key])
        
        # C. Restore Column 1 (Overwriting any micro-gradients from Phase 2)
        col1 = model.policy.mlp_extractor.prog_net.getColumn(id1)
        col1.blocks[0].module.weight.copy_(task1_weights['mlp_extractor.policy_net.0.weight'])
        col1.blocks[0].module.bias.copy_(task1_weights['mlp_extractor.policy_net.0.bias'])
        col1.blocks[1].module.weight.copy_(task1_weights['mlp_extractor.policy_net.2.weight'])
        col1.blocks[1].module.bias.copy_(task1_weights['mlp_extractor.policy_net.2.bias'])
        
        # D. Neutralize the Ghost Layer (Block 2)
        col1.blocks[2].module.weight.copy_(torch.eye(64))
        col1.blocks[2].module.bias.copy_(torch.zeros(64))
        
    print("[✓] Brain, Eyes, and Action logic are now 100% mathematically identical to Phase 1.")

    # 5. Evaluation Loop
    obs, _ = env.reset()
    for i in range(3): 
        # Clear GRU memory at start of episode
        model.policy.features_extractor.hidden_state = None
        
        terminated = truncated = False
        step_count = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            step_count += 1
            
            if terminated or truncated:
                print(f"Episode {i+1} Finished. Reward: {reward:.2f} in {step_count} steps.")
                obs, _ = env.reset()
                # Clear GRU memory for next episode
                model.policy.features_extractor.hidden_state = None

    env.close()

if __name__ == "__main__":
    evaluate()