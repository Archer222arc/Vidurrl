"""
PPO-based scheduler for toy model queue scheduling.

Uses trained PPO policy to make routing decisions.
"""

import torch
import numpy as np
from typing import Optional

from toymodel.src.entities import Request, Replica
from toymodel.schedulers.base import BaseScheduler
from toymodel.src.rl_components import SimpleActorCritic, QueueStateBuilder


class PPOScheduler(BaseScheduler):
    """
    PPO-based scheduler for toy model queue scheduling.
    
    Uses a trained PPO policy to make routing decisions based on
    current queue states and request characteristics.
    """

    def __init__(
        self,
        num_replicas: int = 2,
        model_path: Optional[str] = None,
        n_requests: int = 3,
        hidden_dim: int = 64,
        device: str = "cpu"
    ):
        """
        Initialize PPO scheduler.
        
        Args:
            num_replicas: Number of replicas in system
            model_path: Path to trained model checkpoint
            n_requests: Number of request types to include from each queue
            hidden_dim: Hidden layer dimension
            device: Device for computations
        """
        super().__init__(num_replicas)
        
        self.device = device
        self.n_requests = n_requests
        self.state_builder = QueueStateBuilder(
            num_replicas=num_replicas, 
            n_requests=n_requests,
            normalize=False
        )
        
        # Get dimensions from state builder
        state_dim = self.state_builder.get_state_dim()
        action_dim = self.state_builder.get_action_dim()
        
        # Initialize policy network
        self.policy = SimpleActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Load trained model if provided
        if model_path:
            self.load_model(model_path)
        else:
            # Initialize with random weights for testing
            self.policy.eval()
    
    def load_model(self, model_path: str):
        """
        Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        try:
            # Load with weights_only=False for compatibility with older PyTorch versions
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
            else:
                # Direct state dict loading
                self.policy.load_state_dict(checkpoint)
            self.policy.eval()
            print(f"Loaded PPO model from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            print("Using randomly initialized model")
    
    def schedule(self, request: Request, replicas: list[Replica]) -> int:
        """
        Make routing decision using PPO policy.
        
        Args:
            request: Incoming request
            replicas: List of available replicas
            
        Returns:
            Replica ID to route request to
        """
        # Build state representation
        state = self.state_builder.build_state(request, replicas)
        state = state.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Get action from policy
        with torch.no_grad():
            action, _, _ = self.policy.get_action_and_value(state)
        
        # Convert to replica ID
        replica_id = action.item()
        
        # Ensure valid replica ID
        replica_id = max(0, min(replica_id, self.num_replicas - 1))
        
        return replica_id
    
    def get_action_probabilities(self, request: Request, replicas: list[Replica]) -> np.ndarray:
        """
        Get action probabilities for analysis.
        
        Args:
            request: Incoming request
            replicas: List of available replicas
            
        Returns:
            Array of action probabilities
        """
        # Build state representation
        state = self.state_builder.build_state(request, replicas)
        state = state.unsqueeze(0).to(self.device)
        
        # Get action logits
        with torch.no_grad():
            action_logits, _ = self.policy.forward(state)
            action_probs = torch.softmax(action_logits, dim=-1)
        
        return action_probs.cpu().numpy().flatten()
    
    def reset(self):
        """Reset scheduler state."""
        self.state_builder.reset_normalization()
    
    def set_eval_mode(self):
        """Set policy to evaluation mode."""
        self.policy.eval()
    
    def set_train_mode(self):
        """Set policy to training mode."""
        self.policy.train()
