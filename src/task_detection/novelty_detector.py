import torch
import torch.nn as nn
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import numpy as np

# Import the model structure you just built
from src.task_detection.autoencoder import MiniGridAutoencoder

def get_average_error(env_name, model, num_samples=200):
    env = gym.make(env_name)
    env = FlatObsWrapper(env)
    criterion = nn.MSELoss()
    
    obs, _ = env.reset()
    total_loss = 0
    
    # .eval() freezes the layers (like Dropout/BatchNorm) for testing
    model.eval() 
    
    with torch.no_grad(): # We don't want it to learn anything new!
        for _ in range(num_samples):
            # Same preprocessing as training (add batch dimension and scale 0-1)
            tensor_obs = torch.tensor(obs / 10.0, dtype=torch.float32).unsqueeze(0)
            
            # Extract only the 147 image pixels for the target
            target_image = tensor_obs[:, :147]
            
            reconstruction = model(tensor_obs)
            loss = criterion(reconstruction, target_image)
            total_loss += loss.item()
            
            # Take a random step
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
                
    return total_loss / num_samples

if __name__ == "__main__":
    print("Loading trained Autoencoder...")
    ae = MiniGridAutoencoder(latent_dim=16)
    ae.load_state_dict(torch.load("ae_empty.pth", weights_only=True))
    
    print("-" * 40)
    print("TEST 1: The Familiar Task (Empty Room)")
    err_empty = get_average_error("MiniGrid-Empty-8x8", ae)
    print(f"Average Reconstruction Error: {err_empty:.5f}")
    
    print("-" * 40)
    print("TEST 2: The Novel Task (Four Rooms)")
    err_fourrooms = get_average_error("MiniGrid-FourRooms-v0", ae)
    print(f"Average Reconstruction Error: {err_fourrooms:.5f}")
    print("-" * 40)
    
    if err_fourrooms > err_empty * 2:
        print("\nSUCCESS: The Alarm Bell works! The error spiked on the new task.")
        print(f"Suggested Threshold for AE-2: {(err_empty + err_fourrooms) / 2:.5f}")
    else:
        print("\nWARNING: The autoencoder is too generalized. It reconstructed the new task too easily.")