import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper

class MiniGridAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(MiniGridAutoencoder, self).__init__()
        
        # Encoder compresses the image into a small latent space
        self.encoder = nn.Sequential(
            # Input shape: (Batch, Channels: 3, Height: 7, Width: 7)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), 
            nn.ReLU(),
            # Shape is now (Batch, 16, 7, 7)
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0), 
            nn.ReLU(),
            # Shape is now (Batch, 32, 5, 5)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0), 
            nn.ReLU(),
            # Shape is now (Batch, 64, 3, 3)
            
            nn.Flatten(),
            # Shape is now (Batch, 576)
            nn.Linear(64 * 3 * 3, latent_dim) 
            # Final output is the compressed bottleneck
        )

        # DECODER: Rebuilds the image from the latent space
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 64 * 3 * 3),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # We use ConvTranspose2d to "upsample" the image back to its original size
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),
            # Shape is now (Batch, 32, 5, 5)
            
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=0),
            nn.ReLU(),
            # Shape is now (Batch, 16, 7, 7)
            
            # Final layer brings it back to 3 channels (RGB/Grid layers)
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid() # Keeps pixel values bounded between 0 and 1
        )

    def forward(self, x):
        # Un-flatten the FlatObsWrapper input.
        # FlatObsWrapper outputs a 1D array per batch. For a 7x7x3 grid, this is 147 elements.
        # We need it to be (Batch, Channels, Height, Width) -> (Batch, 3, 7, 7)
        
        batch_size = x.shape[0]
        # We take only the first 147 elements in case the environment appended extra info (like mission text)
        image_data = x[:, :147] 
        
        # MiniGrid stores it as (Width, Height, Channels) in the flat array.
        # We reshape it to (Batch, 7, 7, 3), and then permute the axes to get (Batch, Channels, Height, Width)
        x_reshaped = image_data.view(batch_size, 7, 7, 3).permute(0, 3, 1, 2)
            
        # 2. Encode
        latent = self.encoder(x_reshaped)
        
        # 3. Prepare for Decoding
        dec_hidden = self.fc_decode(latent)
        dec_hidden = dec_hidden.view(batch_size, 64, 3, 3) 
        
        # 4. Decode
        reconstructed_image = self.decoder(dec_hidden)
        
        # 5. Reverse the permutation and flatten back to the original format
        reconstructed_flat = reconstructed_image.permute(0, 2, 3, 1).reshape(batch_size, 147)
        return reconstructed_flat

if __name__ == "__main__":
    print("Initializing Autoencoder Test...")
    
    # 1. Setup Environment
    env = gym.make("MiniGrid-Empty-8x8")
    env = FlatObsWrapper(env)
    
    # 2. Setup Model & Optimizer
    ae = MiniGridAutoencoder(latent_dim=16)
    criterion = nn.MSELoss() # Mean Squared Error (our Reconstruction Error)
    optimizer = optim.Adam(ae.parameters(), lr=0.001)
    
    # 3. Collect a batch of random observations to train on
    print("Collecting observations...")
    observations = []
    obs, _ = env.reset()
    for _ in range(1000):
        # We divide by 10.0 because MiniGrid state values are integers (0-10). 
        # Neural networks prefer inputs between 0 and 1!
        observations.append(obs / 10.0) 
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            
    # Convert list to a PyTorch tensor
    import numpy as np
    data = torch.tensor(np.array(observations), dtype=torch.float32)
    
    # 4. Train the Autoencoder
    print("Training Autoencoder on Empty Room data...")
    epochs = 50
    batch_size = 64
    
    for epoch in range(epochs):
        epoch_loss = 0
        # Simple mini-batch training loop
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            optimizer.zero_grad()
            reconstruction = ae(batch)
            
            # Extract only the 147 image pixels from the target batch
            target_image = batch[:, :147] 
            loss = criterion(reconstruction, target_image) 
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Reconstruction Error: {epoch_loss / (len(data)/batch_size):.4f}")
            
    print("Training Complete! The Reconstruction Error should be very low.")
    torch.save(ae.state_dict(), "ae_empty.pth")