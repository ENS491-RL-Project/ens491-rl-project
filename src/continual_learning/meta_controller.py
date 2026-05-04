import os
import torch
import torch.nn as nn

# Import your previous modules!
from src.task_detection.autoencoder import MiniGridAutoencoder
from src.continual_learning.gru_test import MinimalGRU
from src.continual_learning.doric_test import MiniGridColumnGenerator
from Doric import ProgNet

class MetaController(nn.Module):
    def __init__(self, novelty_threshold=0.01829):
        super(MetaController, self).__init__()
        self.threshold = novelty_threshold

        # 1. The Task Detector (AE)
        self.ae = MiniGridAutoencoder(latent_dim=16)
        # We load the weights you saved earlier!
        self.ae.load_state_dict(torch.load("ae_empty.pth", weights_only=True))
        self.ae.eval() # Freeze the AE, it just acts as a sensor now

        # 2. The Short-Term Memory (GRU)
        self.gru = MinimalGRU(input_size=147, hidden_size=64)

        # 3. The Continual Learning Backbone (ProgNet)
        # Note: The input to the columns is now 64 (the output of the GRU), not the raw image!
        gen = MiniGridColumnGenerator(obs_size=64, hidden=64, n_actions=7)
        self.prog_net = ProgNet(colGen=gen)

        # Initialize the system on Task 1
        self.active_task = self.prog_net.addColumn(msg="task1")
        print(f"[System Init] Started in {self.active_task}")

    def detect_task(self, obs):
        target = obs[:, :147]
        with torch.no_grad():
            reconstruction = self.ae(obs)
            error = nn.functional.mse_loss(reconstruction, target).item()

        # If the error spikes above our threshold AND we are still in task 1...
        if error > self.threshold and self.active_task == "task1":
            print(f"\n[ALARM] High Reconstruction Error detected: {error:.5f}")
            print(f"[Meta-Controller] Freezing {self.active_task} to prevent forgetting...")
            self.prog_net.freezeColumn(self.active_task)
            
            # Spin up a new column for the new environment
            self.active_task = self.prog_net.addColumn(msg="task2")
            print(f"[Meta-Controller] Spawned and switched to {self.active_task}")
            
        return self.active_task

    def forward(self, obs, hidden_state):
        # 1. Which environment are we in?
        task_id = self.detect_task(obs)

        # 2. Update short-term memory
        new_hidden = self.gru(obs, hidden_state)

        # 3. Ask the correct ProgNet column what to do
        action_logits = self.prog_net.forward(task_id, new_hidden)

        return action_logits, new_hidden

if __name__ == "__main__":
    print("=" * 10)
    print("Meta-Controller Integration Test")
    print("=" * 10)

    # Initialize the master brain
    brain = MetaController(novelty_threshold=0.01829)
    memory = torch.zeros(1, 64) # Blank slate

    # Simulate Time Step 1: Agent is in the familiar Empty room
    print("\n--- Time Step 1: Agent is in the Empty Room ---")
    # A fake observation that looks like the empty room (low noise)
    obs_empty = torch.zeros(1, 147) 
    action, memory = brain(obs_empty, memory)
    print(f"Action Logits Shape: {action.shape}")

    # Simulate Time Step 2: We drop the agent into FourRooms
    print("\n--- Time Step 2: Agent drops into FourRooms ---")
    # A fake observation that looks drastically different (high noise, acts like a wall)
    obs_fourrooms = torch.rand(1, 147) 
    action, memory = brain(obs_fourrooms, memory)
    print(f"Action Logits Shape: {action.shape}")