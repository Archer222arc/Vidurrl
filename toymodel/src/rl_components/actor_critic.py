"""
Clean Actor-Critic network for PPO in toy model.

Simple MLP architecture without any tricks or complex features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class SimpleActorCritic(nn.Module):
    """
    Simple Actor-Critic network for toy model queue scheduling.
    
    State: [queue_length_0, queue_length_1, request_type]
    Action: replica_id (0 or 1)
    """

    def __init__(self, state_dim: int = 3, action_dim: int = 2, hidden_dim: int = 64):
        """
        Initialize simple actor-critic network.
        
        Args:
            state_dim: State dimension (default: 3 for [q0, q1, req_type])
            action_dim: Action dimension (default: 2 for replica choices)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Tuple of (action_logits, value)
        """
        features = self.feature_extractor(state)
        action_logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        
        return action_logits, value
    
    def get_action_and_value(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and get value for given state.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(state)
        
        # Create categorical distribution
        dist = Categorical(logits=action_logits)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions for policy gradient computation.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size,)
            
        Returns:
            Tuple of (log_prob, entropy, value)
        """
        action_logits, value = self.forward(state)
        
        # Create categorical distribution
        dist = Categorical(logits=action_logits)
        
        # Get log probability and entropy for given actions
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy, value


