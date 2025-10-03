"""
Simple rollout buffer for PPO experience collection.

Clean implementation without complex features, just basic experience storage and GAE computation.
"""

import torch
import numpy as np
from typing import List, Tuple


class SimpleRolloutBuffer:
    """
    Simple rollout buffer for storing and processing PPO experiences.
    
    Collects state-action-reward sequences and computes advantages using
    Generalized Advantage Estimation for PPO training.
    """

    def __init__(
        self,
        state_dim: int,
        buffer_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu"
    ):
        """
        Initialize rollout buffer.
        
        Args:
            state_dim: Dimension of state space
            buffer_size: Maximum buffer size
            gamma: Discount factor for future rewards
            gae_lambda: GAE lambda parameter for bias-variance tradeoff
            device: Device for tensor computations
        """
        self.state_dim = state_dim
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        self.reset()
    
    def reset(self):
        """Reset buffer for new rollout collection."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool
    ):
        """
        Add single step experience to buffer.
        
        Args:
            state: State tensor
            action: Action tensor
            reward: Reward scalar
            value: Value estimate tensor
            log_prob: Log probability of action
            done: Episode termination flag
        """
        if self.size < self.buffer_size:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.dones.append(done)
            self.size += 1
        else:
            # Overwrite oldest experience
            idx = self.ptr % self.buffer_size
            self.states[idx] = state
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.values[idx] = value
            self.log_probs[idx] = log_prob
            self.dones[idx] = done
            self.ptr += 1
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size >= self.buffer_size
    
    def compute_gae(self, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Args:
            next_value: Bootstrap value for final state
            
        Returns:
            Tuple of (states, actions, log_probs, values, returns, advantages)
        """
        # Convert to tensors
        states = torch.stack(self.states).to(self.device)
        actions = torch.stack(self.actions).to(self.device)
        log_probs = torch.stack(self.log_probs).to(self.device)
        values = torch.stack(self.values).to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        
        # Compute GAE advantages
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, log_probs, values, returns, advantages
    
    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all stored experiences.
        
        Returns:
            Tuple of (states, actions, rewards, values, log_probs, dones)
        """
        states = torch.stack(self.states).to(self.device)
        actions = torch.stack(self.actions).to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        values = torch.stack(self.values).to(self.device)
        log_probs = torch.stack(self.log_probs).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        
        return states, actions, rewards, values, log_probs, dones


