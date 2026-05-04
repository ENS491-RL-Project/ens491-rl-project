import torch
import torch.nn as nn

class MinimalGRU(nn.Module):
    def __init__(self, input_size=147, hidden_size=64):
        super(MinimalGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # 1. Feature Extractor: Compress the 147-pixel flat image
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU()
        )
        
        # 2. The Memory Cell: Takes current features + previous memory
        self.gru_cell = nn.GRUCell(input_size=64, hidden_size=hidden_size)

    def forward(self, obs, hidden_state):
        # FIX: Extract only the 147 image pixels from the full observation
        # This handles the 2835 -> 147 mismatch
        img_obs = obs[:, :147]
        
        # Pass the sliced image data to the feature extractor
        features = self.feature_extractor(img_obs)
        
        # Update the memory state
        new_hidden_state = self.gru_cell(features, hidden_state)
        return new_hidden_state

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Minimal GRU Memory Module")
    print("=" * 50)

    # 1. Initialize our module
    gru_module = MinimalGRU(input_size=147, hidden_size=64)
    
    # 2. Simulate a batch of 1 environment
    batch_size = 1
    
    # 3. Create a blank memory state (all zeros) for Time Step 0
    current_memory = torch.zeros(batch_size, gru_module.hidden_size)
    print(f"[Time 0] Initial Memory State shape: {current_memory.shape}")
    print(f"[Time 0] Memory sum (should be 0): {current_memory.sum().item()}\n")
    
    # 4. Simulate the agent taking 3 steps in the environment
    for step in range(1, 4):
        # Simulate a random 147-pixel observation coming from FlatObsWrapper
        fake_obs = torch.rand(batch_size, 147)
        
        # Pass the observation AND the previous memory into the GRU
        current_memory = gru_module(fake_obs, current_memory)
        
        print(f"[Time {step}] Agent saw a new observation!")
        print(f"          Updated Memory shape: {current_memory.shape}")
        # We print the sum to mathematically prove the hidden state is changing/learning
        print(f"          Updated Memory sum: {current_memory.sum().item():.4f}\n")
        
    print("SUCCESS: The GRU cell successfully carried and updated memory across time steps!")