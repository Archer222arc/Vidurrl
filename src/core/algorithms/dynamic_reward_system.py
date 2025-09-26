"""
Dynamic Reward Scaling and Imbalance Penalty System.

This module implements an advanced reward system that adapts to prevent
policy collapse while maintaining training signal quality. It addresses
the reward degradation observed in collapsed training (e.g., -4.29 → -11.24).

Key features:
1. Progressive imbalance penalties that scale with collapse severity
2. Dynamic reward normalization based on performance trends
3. Emergency reward boosting during collapse recovery
4. Multi-objective reward balancing for load balancing tasks
5. Curriculum learning integration for gradual difficulty increase

Based on analysis of reward patterns during collapse phases.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from typing import Dict, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class ProgressiveImbalancePenalty:
    """
    Progressive penalty system that scales with distribution imbalance severity.

    The penalty increases non-linearly as the coefficient of variation increases,
    providing strong incentives to maintain balance without overwhelming the
    reward signal during early training.
    """

    def __init__(
        self,
        base_penalty_weight: float = 1.0,
        cv_scale_factor: float = 2.0,
        penalty_cap: float = 10.0,
        emergency_penalty_multiplier: float = 5.0,
        min_samples: int = 20
    ):
        """
        Initialize progressive imbalance penalty.

        Args:
            base_penalty_weight: Base weight for imbalance penalty
            cv_scale_factor: Scaling factor for CV-based penalties
            penalty_cap: Maximum penalty multiplier
            emergency_penalty_multiplier: Additional multiplier during emergencies
            min_samples: Minimum samples before applying penalties
        """
        self.base_penalty_weight = base_penalty_weight
        self.cv_scale_factor = cv_scale_factor
        self.penalty_cap = penalty_cap
        self.emergency_penalty_multiplier = emergency_penalty_multiplier
        self.min_samples = min_samples

        self.action_history = deque(maxlen=200)
        self.penalty_history = deque(maxlen=100)

    def calculate_penalty(
        self,
        action_distribution: list[float],
        emergency_mode: bool = False
    ) -> Dict[str, float]:
        """
        Calculate progressive imbalance penalty.

        Args:
            action_distribution: Current action frequency distribution
            emergency_mode: Whether system is in emergency collapse mode

        Returns:
            Dictionary with penalty calculation details
        """
        if len(action_distribution) < 2:
            return {
                'penalty': 0.0,
                'cv': 0.0,
                'penalty_multiplier': 1.0,
                'emergency_applied': False
            }

        # Calculate coefficient of variation
        frequencies = np.array(action_distribution)
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        cv = std_freq / (mean_freq + 1e-8)

        # Progressive penalty scaling based on CV
        if cv <= 0.1:
            # Very balanced - no penalty
            penalty_multiplier = 0.0
        elif cv <= 0.5:
            # Mild imbalance - linear scaling
            penalty_multiplier = (cv - 0.1) / 0.4 * 1.0
        else:
            # Severe imbalance - exponential scaling
            penalty_multiplier = 1.0 + math.exp(cv - 0.5) - 1.0

        # Cap the penalty
        penalty_multiplier = min(penalty_multiplier, self.penalty_cap)

        # Apply emergency multiplier if in collapse mode
        emergency_applied = False
        if emergency_mode:
            penalty_multiplier *= self.emergency_penalty_multiplier
            emergency_applied = True

        # Calculate final penalty
        penalty = self.base_penalty_weight * penalty_multiplier

        # Store penalty history
        self.penalty_history.append(penalty)

        return {
            'penalty': penalty,
            'cv': cv,
            'penalty_multiplier': penalty_multiplier,
            'emergency_applied': emergency_applied,
            'mean_frequency': mean_freq,
            'std_frequency': std_freq
        }

    def update_history(self, action: int) -> None:
        """Update action history for penalty calculation."""
        self.action_history.append(action)

    def get_current_distribution(self, num_actions: int) -> list[float]:
        """Get current action distribution from history."""
        if len(self.action_history) < self.min_samples:
            # Return uniform distribution if insufficient data
            return [1.0 / num_actions] * num_actions

        actions = np.array(list(self.action_history))
        action_counts = np.bincount(actions, minlength=num_actions)
        total_actions = len(self.action_history)

        return (action_counts / total_actions).tolist()


class AdaptiveRewardNormalizer:
    """
    Adaptive reward normalizer that maintains training signal quality.

    Prevents reward signal degradation during collapse while preserving
    the relative ordering of rewards for policy learning.
    """

    def __init__(
        self,
        target_reward_range: Tuple[float, float] = (-1.0, 1.0),
        adaptation_rate: float = 0.01,
        stability_window: int = 100,
        emergency_rescale_threshold: float = 5.0,
        min_reward_std: float = 0.1
    ):
        """
        Initialize adaptive reward normalizer.

        Args:
            target_reward_range: Target range for normalized rewards
            adaptation_rate: Rate of adaptation to new reward distributions
            stability_window: Window size for stability analysis
            emergency_rescale_threshold: Threshold for emergency rescaling
            min_reward_std: Minimum reward standard deviation to maintain
        """
        self.target_min, self.target_max = target_reward_range
        self.adaptation_rate = adaptation_rate
        self.stability_window = stability_window
        self.emergency_rescale_threshold = emergency_rescale_threshold
        self.min_reward_std = min_reward_std

        self.reward_history = deque(maxlen=stability_window * 2)
        self.running_mean = 0.0
        self.running_std = 1.0
        self.normalization_count = 0
        self.emergency_rescales = 0

    def normalize_reward(
        self,
        raw_reward: float,
        force_emergency_rescale: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Normalize reward adaptively.

        Args:
            raw_reward: Raw reward value
            force_emergency_rescale: Force emergency rescaling

        Returns:
            Tuple of (normalized_reward, normalization_stats)
        """
        self.reward_history.append(raw_reward)
        self.normalization_count += 1

        # Calculate current statistics
        if len(self.reward_history) < 10:
            # Insufficient data - return raw reward
            return raw_reward, {
                'normalized_reward': raw_reward,
                'raw_reward': raw_reward,
                'scale_factor': 1.0,
                'shift_factor': 0.0,
                'emergency_rescale': False
            }

        # Update running statistics
        current_rewards = list(self.reward_history)
        current_mean = np.mean(current_rewards)
        current_std = np.std(current_rewards) + 1e-8

        # Adaptive update
        if self.normalization_count == 1:
            self.running_mean = current_mean
            self.running_std = current_std
        else:
            self.running_mean += self.adaptation_rate * (current_mean - self.running_mean)
            self.running_std += self.adaptation_rate * (current_std - self.running_std)

        # Check for emergency rescaling
        emergency_rescale = (force_emergency_rescale or
                           abs(raw_reward - self.running_mean) > self.emergency_rescale_threshold * self.running_std)

        if emergency_rescale:
            # Use recent statistics for emergency rescaling
            self.running_mean = current_mean
            self.running_std = max(current_std, self.min_reward_std)
            self.emergency_rescales += 1
            logger.warning(f"[RewardNormalizer] Emergency rescale #{self.emergency_rescales}: "
                          f"mean={self.running_mean:.3f}, std={self.running_std:.3f}")

        # Normalize reward
        normalized = (raw_reward - self.running_mean) / self.running_std

        # Scale to target range
        scale_factor = (self.target_max - self.target_min) / 2.0
        shift_factor = (self.target_max + self.target_min) / 2.0
        final_normalized = normalized * scale_factor + shift_factor

        # Ensure minimum standard deviation
        if self.running_std < self.min_reward_std:
            self.running_std = self.min_reward_std

        return final_normalized, {
            'normalized_reward': final_normalized,
            'raw_reward': raw_reward,
            'scale_factor': scale_factor,
            'shift_factor': shift_factor,
            'running_mean': self.running_mean,
            'running_std': self.running_std,
            'emergency_rescale': emergency_rescale,
            'emergency_count': self.emergency_rescales
        }


class EmergencyRewardBooster:
    """
    Emergency reward boosting system for collapse recovery.

    Provides temporary reward boosts to help policies escape from
    collapsed states while maintaining learning signal integrity.
    """

    def __init__(
        self,
        boost_duration: int = 100,
        boost_strength: float = 2.0,
        decay_rate: float = 0.95,
        activation_threshold: float = 0.8  # CV threshold
    ):
        """
        Initialize emergency reward booster.

        Args:
            boost_duration: Duration of reward boost in steps
            boost_strength: Initial strength of reward boost
            decay_rate: Decay rate for boost over time
            activation_threshold: CV threshold for boost activation
        """
        self.boost_duration = boost_duration
        self.boost_strength = boost_strength
        self.decay_rate = decay_rate
        self.activation_threshold = activation_threshold

        self.boost_remaining = 0
        self.current_boost = 1.0
        self.total_boosts = 0
        self.active = False

    def activate_boost(self, cv: float, force: bool = False) -> bool:
        """
        Activate emergency boost if conditions are met.

        Args:
            cv: Current coefficient of variation
            force: Force activation regardless of conditions

        Returns:
            True if boost was activated
        """
        if force or (cv > self.activation_threshold and self.boost_remaining <= 0):
            self.boost_remaining = self.boost_duration
            self.current_boost = self.boost_strength
            self.total_boosts += 1
            self.active = True

            logger.info(f"[EmergencyBooster] Activated boost #{self.total_boosts} "
                       f"(CV={cv:.3f}, strength={self.boost_strength:.2f})")
            return True

        return False

    def apply_boost(self, reward: float) -> Tuple[float, Dict[str, Any]]:
        """
        Apply reward boost if active.

        Args:
            reward: Original reward

        Returns:
            Tuple of (boosted_reward, boost_stats)
        """
        if self.boost_remaining <= 0:
            self.active = False
            return reward, {
                'boosted_reward': reward,
                'original_reward': reward,
                'boost_factor': 1.0,
                'boost_remaining': 0,
                'boost_active': False
            }

        # Apply boost
        boosted_reward = reward * self.current_boost

        # Decay boost
        self.current_boost *= self.decay_rate
        self.boost_remaining -= 1

        if self.boost_remaining <= 0:
            self.active = False

        return boosted_reward, {
            'boosted_reward': boosted_reward,
            'original_reward': reward,
            'boost_factor': self.current_boost,
            'boost_remaining': self.boost_remaining,
            'boost_active': self.active,
            'total_boosts': self.total_boosts
        }


class MultiObjectiveRewardBalancer:
    """
    Multi-objective reward balancing for complex load balancing tasks.

    Balances multiple objectives (throughput, latency, fairness) while
    preventing any single objective from dominating during collapse.
    """

    def __init__(
        self,
        objectives: Dict[str, float] = None,
        adaptive_weights: bool = True,
        weight_adaptation_rate: float = 0.02,
        min_weight: float = 0.1,
        max_weight: float = 2.0
    ):
        """
        Initialize multi-objective reward balancer.

        Args:
            objectives: Initial objective weights
            adaptive_weights: Whether to adapt weights based on performance
            weight_adaptation_rate: Rate of weight adaptation
            min_weight: Minimum weight for any objective
            max_weight: Maximum weight for any objective
        """
        default_objectives = {
            'throughput': 1.0,
            'latency': 1.0,
            'fairness': 1.0,
            'balance': 1.0
        }
        self.objective_weights = objectives or default_objectives
        self.adaptive_weights = adaptive_weights
        self.weight_adaptation_rate = weight_adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Performance tracking for each objective
        self.objective_performance = {obj: deque(maxlen=100) for obj in self.objective_weights}
        self.weight_history = {obj: deque(maxlen=100) for obj in self.objective_weights}

    def balance_objectives(
        self,
        objective_values: Dict[str, float],
        system_state: Dict[str, Any] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Balance multiple objectives into single reward.

        Args:
            objective_values: Dictionary of objective values
            system_state: Current system state for adaptation

        Returns:
            Tuple of (balanced_reward, balancing_stats)
        """
        if not objective_values:
            return 0.0, {}

        # Update performance tracking
        for obj, value in objective_values.items():
            if obj in self.objective_performance:
                self.objective_performance[obj].append(value)

        # Adapt weights if enabled
        if self.adaptive_weights and system_state:
            self._adapt_weights(objective_values, system_state)

        # Calculate weighted combination
        total_weighted_value = 0.0
        weight_sum = 0.0
        objective_contributions = {}

        for obj, value in objective_values.items():
            if obj in self.objective_weights:
                weight = self.objective_weights[obj]
                contribution = weight * value
                total_weighted_value += contribution
                weight_sum += weight
                objective_contributions[obj] = {
                    'value': value,
                    'weight': weight,
                    'contribution': contribution
                }

        # Normalize by total weight
        balanced_reward = total_weighted_value / (weight_sum + 1e-8) if weight_sum > 0 else 0.0

        # Store weight history
        for obj, weight in self.objective_weights.items():
            if obj in self.weight_history:
                self.weight_history[obj].append(weight)

        return balanced_reward, {
            'balanced_reward': balanced_reward,
            'objective_contributions': objective_contributions,
            'current_weights': self.objective_weights.copy(),
            'total_weight': weight_sum
        }

    def _adapt_weights(self, objective_values: Dict[str, float], system_state: Dict[str, Any]) -> None:
        """Adapt objective weights based on system performance."""
        # Get collapse indicators
        cv = system_state.get('coefficient_variation', 0.0)
        performance_decline = system_state.get('performance_decline', 0.0)

        # During collapse, increase balance/fairness weights
        if cv > 0.5 or performance_decline < -0.1:
            # Increase balance weight, decrease performance weights
            for obj in self.objective_weights:
                if obj in ['balance', 'fairness']:
                    # Increase balance-related weights
                    new_weight = self.objective_weights[obj] * (1 + self.weight_adaptation_rate)
                else:
                    # Decrease performance weights slightly
                    new_weight = self.objective_weights[obj] * (1 - self.weight_adaptation_rate / 2)

                # Apply bounds
                self.objective_weights[obj] = np.clip(new_weight, self.min_weight, self.max_weight)

        # During recovery, gradually restore original balance
        elif cv < 0.3 and performance_decline > -0.02:
            # Gradually move weights back toward uniform
            target_weight = 1.0
            for obj in self.objective_weights:
                current = self.objective_weights[obj]
                new_weight = current + self.weight_adaptation_rate * (target_weight - current)
                self.objective_weights[obj] = np.clip(new_weight, self.min_weight, self.max_weight)


class DynamicRewardSystem:
    """
    Complete dynamic reward system integrating all components.

    This is the main interface for the advanced reward scaling and
    penalty system designed to prevent and recover from collapse.
    """

    def __init__(
        self,
        num_actions: int,
        config: Optional[Dict] = None
    ):
        """
        Initialize dynamic reward system.

        Args:
            num_actions: Number of possible actions
            config: Configuration dictionary for all components
        """
        self.num_actions = num_actions
        self.config = config or {}

        # Initialize components
        self.imbalance_penalty = ProgressiveImbalancePenalty(
            base_penalty_weight=self.config.get('imbalance_penalty_weight', 1.0),
            emergency_penalty_multiplier=self.config.get('emergency_penalty_multiplier', 5.0)
        )

        self.reward_normalizer = AdaptiveRewardNormalizer(
            target_reward_range=self.config.get('target_reward_range', (-1.0, 1.0)),
            adaptation_rate=self.config.get('normalization_rate', 0.01)
        )

        self.emergency_booster = EmergencyRewardBooster(
            boost_strength=self.config.get('emergency_boost_strength', 2.0),
            boost_duration=self.config.get('emergency_boost_duration', 100)
        )

        self.multi_objective_balancer = MultiObjectiveRewardBalancer(
            objectives=self.config.get('objective_weights', None),
            adaptive_weights=self.config.get('adaptive_objective_weights', True)
        )

        self.step_count = 0
        self.total_interventions = 0

        logger.info(f"[DynamicRewardSystem] Initialized for {num_actions} actions")

    def process_reward(
        self,
        raw_reward: float,
        action: int,
        objective_values: Optional[Dict[str, float]] = None,
        system_state: Optional[Dict[str, Any]] = None,
        emergency_mode: bool = False
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Process reward through all dynamic scaling components.

        Args:
            raw_reward: Original raw reward
            action: Action taken
            objective_values: Individual objective values
            system_state: Current system state
            emergency_mode: Whether system is in emergency mode

        Returns:
            Tuple of (processed_reward, processing_stats)
        """
        self.step_count += 1
        processing_stats = {}

        # 1. Update action history
        self.imbalance_penalty.update_history(action)

        # 2. Calculate imbalance penalty
        action_distribution = self.imbalance_penalty.get_current_distribution(self.num_actions)
        penalty_stats = self.imbalance_penalty.calculate_penalty(
            action_distribution, emergency_mode
        )
        processing_stats['imbalance_penalty'] = penalty_stats

        # 3. Multi-objective balancing if objectives provided
        if objective_values:
            balanced_reward, balance_stats = self.multi_objective_balancer.balance_objectives(
                objective_values, system_state
            )
            processing_stats['multi_objective'] = balance_stats
        else:
            balanced_reward = raw_reward

        # 4. Apply imbalance penalty
        penalized_reward = balanced_reward - penalty_stats['penalty']

        # 5. Emergency reward boosting
        cv = penalty_stats['cv']
        self.emergency_booster.activate_boost(cv, force=emergency_mode)
        boosted_reward, boost_stats = self.emergency_booster.apply_boost(penalized_reward)
        processing_stats['emergency_boost'] = boost_stats

        # 6. Adaptive normalization
        normalized_reward, norm_stats = self.reward_normalizer.normalize_reward(
            boosted_reward, force_emergency_rescale=emergency_mode
        )
        processing_stats['normalization'] = norm_stats

        # 7. Collect final statistics
        processing_stats.update({
            'step': self.step_count,
            'raw_reward': raw_reward,
            'balanced_reward': balanced_reward,
            'penalized_reward': penalized_reward,
            'boosted_reward': boosted_reward,
            'final_reward': normalized_reward,
            'action_distribution': action_distribution,
            'cv': cv,
            'emergency_mode': emergency_mode
        })

        # Count interventions
        if (penalty_stats.get('emergency_applied', False) or
            boost_stats.get('boost_active', False) or
            norm_stats.get('emergency_rescale', False)):
            self.total_interventions += 1

        return normalized_reward, processing_stats

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        action_distribution = self.imbalance_penalty.get_current_distribution(self.num_actions)
        cv = np.std(action_distribution) / (np.mean(action_distribution) + 1e-8)

        return {
            'step_count': self.step_count,
            'total_interventions': self.total_interventions,
            'current_cv': cv,
            'action_distribution': action_distribution,
            'emergency_booster': {
                'active': self.emergency_booster.active,
                'remaining': self.emergency_booster.boost_remaining,
                'total_boosts': self.emergency_booster.total_boosts
            },
            'reward_normalizer': {
                'running_mean': self.reward_normalizer.running_mean,
                'running_std': self.reward_normalizer.running_std,
                'emergency_rescales': self.reward_normalizer.emergency_rescales
            },
            'objective_weights': self.multi_objective_balancer.objective_weights.copy()
        }