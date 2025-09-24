"""
Exploration bonus mechanism for PPO action selection.

This module provides advanced exploration mechanisms to prevent
both entropy collapse and random policy degradation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ExplorationBonus(nn.Module):
    """
    Advanced exploration bonus mechanism.

    Prevents both entropy collapse (converging to single action) and
    entropy explosion (becoming random policy) through adaptive bonuses.
    """

    def __init__(
        self,
        num_actions: int = 4,
        history_length: int = 100,
        diversity_bonus_scale: float = 0.1,
        uncertainty_bonus_scale: float = 0.05,
        adaptive_temperature: bool = True,
    ):
        """
        Initialize exploration bonus mechanism.

        Args:
            num_actions: Number of possible actions
            history_length: Length of action history to track
            diversity_bonus_scale: Scale for diversity bonus
            uncertainty_bonus_scale: Scale for uncertainty bonus
            adaptive_temperature: Whether to use adaptive temperature
        """
        super().__init__()
        self.num_actions = num_actions
        self.history_length = history_length
        self.diversity_bonus_scale = diversity_bonus_scale
        self.uncertainty_bonus_scale = uncertainty_bonus_scale
        self.adaptive_temperature = adaptive_temperature

        # Action history tracking
        self.register_buffer("action_history", torch.zeros(history_length, dtype=torch.long))
        self.register_buffer("action_counts", torch.zeros(num_actions, dtype=torch.float))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

        # Adaptive temperature parameters
        if adaptive_temperature:
            self.temp_ema_alpha = 0.1
            self.register_buffer("entropy_ema", torch.tensor(0.0))
            self.target_entropy = np.log(num_actions) * 0.15  # 15% of maximum entropy for stronger convergence

    def compute_exploration_bonus(
        self,
        logits: torch.Tensor,
        current_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute exploration bonus for logits.

        Args:
            logits: Raw policy logits [batch_size, num_actions]
            current_action: Current action taken [batch_size]

        Returns:
            Tuple of (bonus_logits, temperature_scale)
        """
        batch_size = logits.size(0)

        # Update action statistics (only for first item in batch to avoid duplication)
        if batch_size > 0:
            self._update_action_stats(current_action[0])

        # Compute diversity bonus - encourage underused actions
        diversity_bonus = self._compute_diversity_bonus(logits)

        # Compute uncertainty bonus - prevent overconfident predictions
        uncertainty_bonus = self._compute_uncertainty_bonus(logits)

        # Adaptive temperature scaling
        temperature_scale = self._compute_adaptive_temperature(logits)

        # Combine bonuses
        bonus_logits = (
            diversity_bonus * self.diversity_bonus_scale +
            uncertainty_bonus * self.uncertainty_bonus_scale
        )

        return bonus_logits, temperature_scale

    def _update_action_stats(self, action: torch.Tensor):
        """Update action history and counts."""
        # Circular buffer for action history
        idx = self.step_count % self.history_length
        old_action = self.action_history[idx]

        # Update counts
        if self.step_count >= self.history_length:
            self.action_counts[old_action] -= 1.0

        self.action_history[idx] = action
        self.action_counts[action] += 1.0
        self.step_count += 1

    def _compute_diversity_bonus(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity bonus to encourage underused actions.
        """
        batch_size = logits.size(0)

        # Normalize action counts to get frequencies
        total_actions = min(self.step_count.float(), float(self.history_length))
        if total_actions > 0:
            action_frequencies = self.action_counts / total_actions
        else:
            action_frequencies = torch.ones_like(self.action_counts) / self.num_actions

        # Compute diversity bonus: higher bonus for less frequent actions
        uniform_freq = 1.0 / self.num_actions
        diversity_bonus = uniform_freq - action_frequencies
        diversity_bonus = torch.clamp(diversity_bonus, min=0.0)  # Only positive bonuses

        # Expand to batch size
        diversity_bonus = diversity_bonus.unsqueeze(0).expand(batch_size, -1)

        return diversity_bonus

    def _compute_uncertainty_bonus(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty bonus to prevent overconfidence.
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)

        # Normalize entropy (0 to 1 range)
        max_entropy = np.log(self.num_actions)
        normalized_entropy = entropy / max_entropy

        # Uncertainty bonus: higher when entropy is very low (overconfident)
        # or very high (too random)
        target_entropy_normalized = self.target_entropy / max_entropy
        entropy_deviation = torch.abs(normalized_entropy - target_entropy_normalized)
        uncertainty_bonus = entropy_deviation.expand(-1, self.num_actions)

        return uncertainty_bonus

    def _compute_adaptive_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive temperature scaling.
        """
        if not self.adaptive_temperature:
            return torch.tensor(1.0, device=logits.device)

        # Compute current entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

        # Update entropy EMA
        if self.step_count > 0:
            self.entropy_ema = (
                (1 - self.temp_ema_alpha) * self.entropy_ema +
                self.temp_ema_alpha * entropy
            )
        else:
            self.entropy_ema = entropy

        # More aggressive temperature control for low-entropy targets
        entropy_ratio = self.entropy_ema / self.target_entropy

        if entropy_ratio < 0.8:  # Slightly low entropy - small increase
            temperature = 1.1
        elif entropy_ratio > 1.5:  # Too high entropy - strong decrease
            temperature = 0.4
        elif entropy_ratio > 1.2:  # Moderately high entropy - moderate decrease
            temperature = 0.6
        else:
            temperature = 0.8  # Default lower temperature for convergence

        return torch.tensor(temperature, device=logits.device)


class StabilizedCategorical(nn.Module):
    """
    Stabilized categorical distribution with exploration bonuses.
    """

    def __init__(self, num_actions: int = 4):
        super().__init__()
        self.num_actions = num_actions
        self.exploration_bonus = ExplorationBonus(num_actions)

    def forward(
        self,
        logits: torch.Tensor,
        action: torch.Tensor = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with stabilized sampling.

        Returns:
            Tuple of (sampled_action, log_prob, entropy)
        """
        # Apply exploration bonus
        if action is not None:
            bonus_logits, adaptive_temp = self.exploration_bonus.compute_exploration_bonus(
                logits, action
            )
            enhanced_logits = logits + bonus_logits
            temperature = temperature * adaptive_temp.item()
        else:
            enhanced_logits = logits

        # Apply temperature scaling
        if temperature != 1.0:
            enhanced_logits = enhanced_logits / temperature

        # Create distribution and sample
        dist = torch.distributions.Categorical(logits=enhanced_logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy