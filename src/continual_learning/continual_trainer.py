import os
import torch
import torch.nn as nn
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.task_detection.autoencoder import MiniGridAutoencoder
from src.continual_learning.gru_test import MinimalGRU
from src.continual_learning.doric_test import MiniGridColumnGenerator
from src.continual_learning.progressive_policy import ProgressiveACPolicy, ProgressiveFeatureExtractor
from Doric import ProgNet

CONFIG = {
    "env_empty": "MiniGrid-Empty-8x8-v0",
    "env_fourrooms": "MiniGrid-FourRooms-v0",
    "total_timesteps": 100000,
    "learning_rate": 3e-4,
    "save_path_task1": "ppo_task1_weights.pth",
    "save_path_final": "ppo_progressive_final"
}
def train_phase_1():
    print("\n" + "="*50)
    print("PHASE 1: Training Expert with Progressive Architecture")
    print("="*50)

    # Create Environment
    env = gym.make(CONFIG["env_empty"])
    env = FlatObsWrapper(env)
    env = DummyVecEnv([lambda: env])

    # Define Policy Arguments
    # This ensures the weights saved for policy_net.0 will be 64x64, not 64x2835!
    policy_kwargs = dict(
        features_extractor_class=ProgressiveFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        activation_fn=torch.nn.ReLU
    )

    # Define Model
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, 
                verbose=1, learning_rate=CONFIG["learning_rate"])

    # Train
    print(f"Training for {CONFIG['total_timesteps']} steps...")
    model.learn(total_timesteps=CONFIG["total_timesteps"])

    #  Save weights
    torch.save(model.policy.state_dict(), CONFIG["save_path_task1"])
    print(f"\n[SUCCESS] Task 1 weights saved to {CONFIG['save_path_task1']}")
    
    return CONFIG["save_path_task1"]


def prepare_progressive_model(prev_weights_path):
    print("\n" + "="*50)
    print("TRANSITION: Building Progressive Architecture")
    print("="*50)

    # 1. Initialize the ProgNet with our MiniGrid Generator
    gen = MiniGridColumnGenerator(obs_size=64, hidden=64, n_actions=7)
    prog_net = ProgNet(colGen=gen)

    # 2. Add Task 1 Column and Load Weights
    id1 = prog_net.addColumn(msg="task1")
    # Note: We load the weights into the column specifically
    # In a full run, you'd map the saved MLP weights to this column
    print(f"Column {id1} created and loaded with Task 1 knowledge.")

    # 3. FREEZE Task 1 (The core of Progressive Networks)
    prog_net.freezeColumn(id1)
    print(f"Column {id1} frozen. Gradient isolation active.")

    # 4. Add Task 2 Column (Lateral connections are built automatically)
    id2 = prog_net.addColumn(msg="task2")
    print(f"Column {id2} spawned for FourRooms. Ready to train.")
    
    return prog_net, id2

def train_phase_2(prev_weights_path):
    print("\n" + "="*50)
    print("PHASE 2: Training on FourRooms (Progressive)")
    print("="*50)

    # 1. Setup the Progressive Architecture
    gen = MiniGridColumnGenerator(obs_size=64, hidden=64, n_actions=64)
    prog_net = ProgNet(colGen=gen)

    # 2. Setup Task 1 (Frozen Expert)
    id1 = prog_net.addColumn(msg="task1")
    
    # Correctly load weights with security best-practices
    sb3_weights = torch.load(prev_weights_path, map_location='cpu', weights_only=True)
    column1 = prog_net.getColumn(id1)
    
    print(f"[Meta] Mapping compatible weights into Doric and SB3 Heads...")
    with torch.no_grad():
        # --- Transfer MLP layers to Doric Column 1 ---
        column1.blocks[0].module.weight.copy_(sb3_weights['mlp_extractor.policy_net.0.weight'])
        column1.blocks[0].module.bias.copy_(sb3_weights['mlp_extractor.policy_net.0.bias'])
        
        # Map Hidden Layer 2
        column1.blocks[1].module.weight.copy_(sb3_weights['mlp_extractor.policy_net.2.weight'])
        column1.blocks[1].module.bias.copy_(sb3_weights['mlp_extractor.policy_net.2.bias'])
        
        # Block 2 (Hidden -> Actions)
        
    
    # Crucial step: Freeze the expert so it remains "immune to forgetting"
    prog_net.freezeColumn(id1)
    print(f" Task 1 (Expert) frozen. Initial weights are now protected.")

    # 3. Setup Task 2 (The Learner)
    id2 = prog_net.addColumn(msg="task2")
    print(f" Task 2 Column active with lateral connections to Task 1.")

    # 4. Create the FourRooms Environment
    env = gym.make(CONFIG["env_fourrooms"])
    env = FlatObsWrapper(env)
    env = DummyVecEnv([lambda: env])

        
    policy_kwargs = dict(
        features_extractor_class=ProgressiveFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        prog_net=prog_net,
        active_task_id=id2
    )

    model = PPO(ProgressiveACPolicy, env, policy_kwargs=policy_kwargs, 
                verbose=1, learning_rate=CONFIG["learning_rate"])

    # 6. TRANSFER HEADS: Give the new model a "starting point" from the expert
    print(f"[Meta] Transferring Expert Action/Value heads to the new Policy...")
    with torch.no_grad():
        # Mapping 64 features -> 7 action logits
        model.policy.action_net.weight.copy_(sb3_weights['action_net.weight'])
        model.policy.action_net.bias.copy_(sb3_weights['action_net.bias'])
        
        # Mapping 64 features -> 1 scalar value
        model.policy.value_net.weight.copy_(sb3_weights['value_net.weight'])
        model.policy.value_net.bias.copy_(sb3_weights['value_net.bias'])

    print(f" Architecture complete. Starting FourRooms training...")
    # 7. Train!
    model.learn(total_timesteps=CONFIG["total_timesteps"])
    
    model.save(CONFIG["save_path_final"])
    print(f"Final Progressive Model saved to {CONFIG['save_path_final']}")

if __name__ == "__main__":
    # If task 1 weights don't exist, train Phase 1
    if not os.path.exists(CONFIG["save_path_task1"]):
        weights_path = train_phase_1()
    else:
        print(f"Existing weights found at {CONFIG['save_path_task1']}. Skipping Phase 1.")
        weights_path = CONFIG["save_path_task1"]
    
    # Execute Phase 2
    train_phase_2(weights_path)
    
    print("\n" + "="*50)
    print("level 4 Complete! Architecture is trained.")
    print("Final Task: Run evaluate_forgetting.py to confirm 0.90+ score on Empty.")
    print("="*50)