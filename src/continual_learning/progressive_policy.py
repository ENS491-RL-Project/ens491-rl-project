import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from src.task_detection.autoencoder import MiniGridAutoencoder
from src.continual_learning.gru_test import MinimalGRU
from Doric import ProgNet

# The Memory & Detection Module
class ProgressiveFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, threshold=0.01829):
        # FIX: Restored the correct super() call for the Feature Extractor
        super().__init__(observation_space, features_dim)
        
        # Task Detector - Uses Autoencoder weights from Stage 3
        self.ae = MiniGridAutoencoder(latent_dim=16)
        try:
            self.ae.load_state_dict(torch.load("ae_empty.pth", weights_only=True))
        except FileNotFoundError:
            print("[Warning] ae_empty.pth not found. Using random AE weights.")
            
        self.ae.eval() 
        self.threshold = threshold
        
        # Memory Module (GRU)
        self.gru = MinimalGRU(input_size=147, hidden_size=features_dim)
        self.hidden_state = None

    def forward(self, observations):
        batch_size = observations.shape[0]
        
        if self.hidden_state is None or self.hidden_state.shape[0] != batch_size:
            self.hidden_state = torch.zeros(batch_size, 64).to(observations.device)

        # Detach to prevent "backward through graph a second time" error
        self.hidden_state = self.hidden_state.detach()

        # Update hidden state (slicing 2835 -> 147 happens inside MinimalGRU)
        self.hidden_state = self.gru(observations, self.hidden_state)
        
        return self.hidden_state

class ProgressiveMlpExtractor(nn.Module):
    
    #Bridge that makes the Doric ProgNet look like a standard SB3 MLP extractor.
    
    def __init__(self, prog_net, task_id, latent_dim=64):
        super().__init__()
        self.prog_net = prog_net
        self.task_id = task_id
        # Attributes SB3 heads (Action/Value) use to determine input size
        self.latent_dim_pi = latent_dim
        self.latent_dim_vf = latent_dim

    def forward(self, features):
        """Returns (latent_pi, latent_vf)"""
        latent = self.prog_net.forward(self.task_id, features)
        return latent, latent

    def forward_actor(self, features):
        return self.prog_net.forward(self.task_id, features)

    def forward_critic(self, features):
        return self.prog_net.forward(self.task_id, features)

class ProgressiveACPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, prog_net, active_task_id, *args, **kwargs):
        # BYPASS: We use object.__setattr__ to save these modules.
        # This prevents PyTorch from throwing an AttributeError before super().__init__().
        object.__setattr__(self, 'prog_net_init', prog_net)
        object.__setattr__(self, 'active_task_id_init', active_task_id)
        
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # FIX: Now that super() is done, we can assign these permanently.
        # This ensures evaluate_forgetting.py can access model.policy.prog_net
        self.active_task_id = active_task_id

    def _build_mlp_extractor(self) -> None:
        
        #Called by super().__init__(). We now have access to the 'init' 
       
        self.mlp_extractor = ProgressiveMlpExtractor(
            self.prog_net_init, 
            self.active_task_id_init, 
            latent_dim=64
        )

    def forward(self, obs, deterministic=False):
        # 1. Feature Extraction (GRU + AE Logic)
        features = self.extract_features(obs)
        
        # 2. Latent Processing (ProgNet Columns)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # 3. Policy Head & Value Head
        distribution = self._get_action_dist_from_latent(latent_pi)
        values = self.value_net(latent_vf)
        
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob