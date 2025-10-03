"""
Clean PPO trainer implementation for toy model.

Simple PPO with only essential components: clipped objective, value function learning, and entropy regularization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

from .actor_critic import SimpleActorCritic


class SimplePPOTrainer:
    """
    Simple PPO trainer for toy model queue scheduling.
    
    Implements clean PPO algorithm with:
    - Clipped policy objective
    - Value function learning
    - Entropy regularization
    - No tricks or complex features
    """

    def __init__(
        self,
        policy: SimpleActorCritic,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        epochs: int = 4,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize simple PPO trainer.
        
        Args:
            policy: Actor-critic policy network
            learning_rate: Learning rate for optimizer
            clip_ratio: PPO clipping ratio
            entropy_coef: Entropy regularization coefficient
            value_coef: Value function loss coefficient
            epochs: Number of training epochs per update
            minibatch_size: Minibatch size for training
            max_grad_norm: Maximum gradient norm for clipping
            device: Device for computations
        """
        self.policy = policy.to(device)
        self.device = device
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update policy using PPO objective.
        
        Args:
            states: State tensor (batch_size, state_dim)
            actions: Action tensor (batch_size,)
            old_log_probs: Old log probabilities (batch_size,)
            old_values: Old value estimates (batch_size,)
            returns: Returns (batch_size,)
            advantages: Advantages (batch_size,)
            
        Returns:
            Dictionary of training statistics
        """
        batch_size = states.shape[0]
        
        # Training statistics
        pi_losses, vf_losses, entropies = [], [], []
        kls, clipfracs, gradnorms = [], [], []
        
        # Convert to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        old_values = old_values.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        for epoch in range(self.epochs):
            # Shuffle data
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, self.minibatch_size):
                end_idx = min(start_idx + self.minibatch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # PPO policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * batch_advantages
                pi_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss (MSE with optional clipping)
                vf_loss = 0.5 * (new_values - batch_returns).pow(2).mean()
                
                # Optional: Add value clipping for stability
                if self.clip_ratio > 0:
                    value_clipped = batch_old_values + (new_values - batch_old_values).clamp(
                        -self.clip_ratio, self.clip_ratio
                    )
                    vf_loss_clipped = 0.5 * (value_clipped - batch_returns).pow(2).mean()
                    vf_loss = torch.max(vf_loss, vf_loss_clipped)
                
                # Entropy bonus
                entropy_bonus = self.entropy_coef * entropy.mean()
                
                # Total loss
                total_loss = pi_loss + self.value_coef * vf_loss - entropy_bonus
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                
                # Collect statistics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                
                pi_losses.append(pi_loss.item())
                vf_losses.append(vf_loss.item())
                entropies.append(entropy.mean().item())
                kls.append(approx_kl)
                clipfracs.append(clipfrac)
                gradnorms.append(grad_norm.item())
        
        # Return statistics
        stats = {
            "pi_loss": np.mean(pi_losses),
            "vf_loss": np.mean(vf_losses),
            "entropy": np.mean(entropies),
            "approx_kl": np.mean(kls),
            "clipfrac": np.mean(clipfracs),
            "grad_norm": np.mean(gradnorms),
        }
        
        return stats
