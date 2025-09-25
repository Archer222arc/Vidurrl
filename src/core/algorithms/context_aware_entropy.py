"""
Context-Aware Intelligent Entropy Regulation for PPO Training.

This module implements an intelligent entropy regulation system that distinguishes between:
- Unhealthy mode collapse (requires intervention)
- Healthy policy convergence (allows graceful entropy decay)

Key principles:
- Analyze state-action correlation to detect context sensitivity
- Monitor performance trends to distinguish failure from success
- Allow natural convergence while preventing catastrophic collapse
- Minimal intervention approach - only act when truly needed
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import mutual_info_score
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("sklearn not available, using fallback mutual information calculation")
    SKLEARN_AVAILABLE = False

    def mutual_info_score(x, y):
        """Fallback mutual information calculation."""
        # Simple histogram-based mutual information
        import numpy as np
        from collections import Counter

        x_counts = Counter(x)
        y_counts = Counter(y)
        xy_counts = Counter(zip(x, y))

        n = len(x)
        mi = 0.0

        for xy, pxy in xy_counts.items():
            px = x_counts[xy[0]] / n
            py = y_counts[xy[1]] / n
            pxy = pxy / n

            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * np.log(pxy / (px * py))

        return mi


class StateActionAnalyzer:
    """
    Analyzes the relationship between states and actions to detect context sensitivity.

    This helps distinguish between:
    - Context-insensitive mode collapse (bad): same action regardless of state
    - Context-sensitive convergence (good): different actions for different states
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        history_window: int = 1000,
        state_discretization_bins: int = 10,
        correlation_threshold: float = 0.1
    ):
        """
        Initialize state-action analyzer.

        Args:
            state_dim: Dimension of state space
            num_actions: Number of possible actions
            history_window: Size of history window for analysis
            state_discretization_bins: Number of bins for state discretization
            correlation_threshold: Minimum correlation for context sensitivity
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.history_window = history_window
        self.state_discretization_bins = state_discretization_bins
        self.correlation_threshold = correlation_threshold

        self.state_history = deque(maxlen=history_window)
        self.action_history = deque(maxlen=history_window)

        # Track state feature ranges for discretization
        self.state_mins = None
        self.state_maxs = None
        self.stats_initialized = False

    def update_history(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        """Update state-action history."""
        states_np = states.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()

        for i in range(len(states)):
            self.state_history.append(states_np[i])
            # Convert action to integer index for proper bincount
            action = actions_np[i]
            if hasattr(action, '__len__'):  # Multi-dimensional action
                action = int(action.flat[0]) if action.size > 0 else 0
            else:  # Scalar action
                action = int(action)
            # Ensure action is within valid range
            action = max(0, min(action, self.num_actions - 1))
            self.action_history.append(action)

        # Initialize statistics
        if not self.stats_initialized and len(self.state_history) >= 20:
            all_states = np.array(list(self.state_history))
            self.state_mins = np.percentile(all_states, 5, axis=0)
            self.state_maxs = np.percentile(all_states, 95, axis=0)
            self.stats_initialized = True

    def compute_mutual_information(self) -> Dict[str, float]:
        """
        Compute mutual information between state features and actions.

        Returns:
            Dictionary with mutual information metrics
        """
        if len(self.state_history) < 20 or not self.stats_initialized:
            # For early return, provide equal distribution as default
            equal_prob = 1.0 / self.num_actions
            default_action_distribution = [equal_prob] * self.num_actions
            return {
                'overall_mutual_info': 0.0,
                'context_sensitivity_score': 0.0,
                'is_context_sensitive': False,
                'sample_size': len(self.state_history),
                'action_distribution': default_action_distribution
            }

        states = np.array(list(self.state_history))
        actions = np.array(list(self.action_history))

        # Discretize states for mutual information calculation
        discretized_states = self._discretize_states(states)

        # Compute mutual information for each state dimension
        mi_scores = []
        for dim in range(self.state_dim):
            try:
                mi = mutual_info_score(discretized_states[:, dim], actions)
                mi_scores.append(mi)
            except Exception:
                mi_scores.append(0.0)

        overall_mi = np.mean(mi_scores)
        max_possible_mi = math.log(min(self.num_actions, self.state_discretization_bins))

        # Normalize mutual information score
        context_sensitivity_score = overall_mi / max_possible_mi if max_possible_mi > 0 else 0.0
        is_context_sensitive = context_sensitivity_score > self.correlation_threshold

        # Calculate action distribution for mode collapse detection
        action_counts = np.bincount(actions, minlength=self.num_actions)
        total_actions = len(actions)
        action_distribution = (action_counts / total_actions).tolist() if total_actions > 0 else [0.0] * self.num_actions

        return {
            'overall_mutual_info': overall_mi,
            'context_sensitivity_score': context_sensitivity_score,
            'is_context_sensitive': is_context_sensitive,
            'max_possible_mi': max_possible_mi,
            'per_dim_mi': mi_scores,
            'sample_size': len(self.state_history),
            'action_distribution': action_distribution  # Critical field for mode collapse detection
        }

    def _discretize_states(self, states: np.ndarray) -> np.ndarray:
        """Discretize continuous states into bins."""
        discretized = np.zeros_like(states, dtype=int)

        for dim in range(self.state_dim):
            state_range = self.state_maxs[dim] - self.state_mins[dim]
            if state_range > 1e-6:
                # Create bins
                bins = np.linspace(self.state_mins[dim], self.state_maxs[dim],
                                 self.state_discretization_bins + 1)
                discretized[:, dim] = np.digitize(states[:, dim], bins) - 1
                discretized[:, dim] = np.clip(discretized[:, dim], 0,
                                            self.state_discretization_bins - 1)

        return discretized

    def analyze_action_patterns(self) -> Dict[str, float]:
        """
        Analyze action selection patterns for mode collapse detection.

        Returns:
            Dictionary with pattern analysis results
        """
        if len(self.action_history) < 10:
            return {
                'action_entropy': 1.0,
                'max_action_freq': 1.0 / self.num_actions,
                'pattern_consistency': 0.0,
                'action_distribution': [1.0 / self.num_actions] * self.num_actions
            }

        actions = np.array(list(self.action_history))

        # Compute action frequencies
        action_counts = np.bincount(actions, minlength=self.num_actions)
        action_frequencies = action_counts / len(actions)

        # Compute action entropy
        action_entropy = -np.sum(action_frequencies * np.log(action_frequencies + 1e-8))
        normalized_entropy = action_entropy / math.log(self.num_actions)

        # Find most frequent action
        max_action_freq = np.max(action_frequencies)

        # Analyze pattern consistency (how stable the distribution is over time)
        window_size = min(100, len(self.action_history) // 4)
        if len(self.action_history) >= window_size * 2:
            recent_actions = actions[-window_size:]
            old_actions = actions[-2*window_size:-window_size]

            recent_freq = np.bincount(recent_actions, minlength=self.num_actions) / len(recent_actions)
            old_freq = np.bincount(old_actions, minlength=self.num_actions) / len(old_actions)

            # KL divergence between recent and old distributions
            pattern_consistency = np.sum(recent_freq * np.log((recent_freq + 1e-8) / (old_freq + 1e-8)))
        else:
            pattern_consistency = 0.0

        return {
            'action_entropy': action_entropy,
            'normalized_action_entropy': normalized_entropy,
            'max_action_freq': max_action_freq,
            'pattern_consistency': pattern_consistency,
            'action_distribution': action_frequencies.tolist()
        }


class PerformanceTracker:
    """
    Tracks performance metrics to distinguish between declining and improving trends.

    This helps identify whether action imbalance is due to:
    - Performance degradation (requires intervention)
    - Natural convergence to optimal policy (allow it)
    """

    def __init__(
        self,
        window_size: int = 200,
        trend_threshold: float = 0.05,
        stability_window: int = 50
    ):
        """
        Initialize performance tracker.

        Args:
            window_size: Size of performance history window
            trend_threshold: Threshold for significant trend detection
            stability_window: Window size for stability analysis
        """
        self.window_size = window_size
        self.trend_threshold = trend_threshold
        self.stability_window = stability_window

        self.reward_history = deque(maxlen=window_size)
        self.value_loss_history = deque(maxlen=window_size)
        self.policy_loss_history = deque(maxlen=window_size)
        self.kl_div_history = deque(maxlen=window_size)

    def update(
        self,
        rewards: torch.Tensor,
        value_loss: float,
        policy_loss: float,
        kl_divergence: float
    ) -> None:
        """Update performance metrics."""
        avg_reward = rewards.mean().item()
        self.reward_history.append(avg_reward)
        self.value_loss_history.append(value_loss)
        self.policy_loss_history.append(policy_loss)
        self.kl_div_history.append(kl_divergence)

    def analyze_performance_trend(self) -> Dict[str, float]:
        """
        Analyze performance trends to detect improvement/degradation.

        Returns:
            Dictionary with trend analysis results
        """
        if len(self.reward_history) < max(5, self.stability_window // 10):
            return {
                'reward_trend': 0.0,
                'value_loss_trend': 0.0,
                'is_improving': False,
                'is_stable': False,
                'performance_score': 0.0
            }

        rewards = np.array(list(self.reward_history))
        value_losses = np.array(list(self.value_loss_history))

        # Compute recent vs old performance
        recent_size = min(self.stability_window, len(rewards) // 3)
        recent_rewards = rewards[-recent_size:]
        old_rewards = rewards[-2*recent_size:-recent_size] if len(rewards) >= 2*recent_size else rewards[:-recent_size]

        recent_value_loss = value_losses[-recent_size:]
        old_value_loss = value_losses[-2*recent_size:-recent_size] if len(value_losses) >= 2*recent_size else value_losses[:-recent_size]

        # Compute trends
        reward_trend = np.mean(recent_rewards) - np.mean(old_rewards)
        value_loss_trend = np.mean(old_value_loss) - np.mean(recent_value_loss)  # Positive if loss is decreasing

        # Performance improvement indicators
        is_improving = reward_trend > self.trend_threshold or value_loss_trend > self.trend_threshold

        # Performance stability (low variance in recent performance)
        recent_reward_std = np.std(recent_rewards)
        is_stable = recent_reward_std < np.std(rewards) * 0.5

        # Overall performance score (higher is better)
        performance_score = reward_trend + value_loss_trend

        return {
            'reward_trend': reward_trend,
            'value_loss_trend': value_loss_trend,
            'is_improving': is_improving,
            'is_stable': is_stable,
            'performance_score': performance_score,
            'recent_reward_mean': np.mean(recent_rewards),
            'recent_reward_std': recent_reward_std
        }

    def detect_performance_plateau(self) -> Dict[str, bool]:
        """
        Detect if performance has plateaued (indicating potential convergence).

        Returns:
            Dictionary with plateau detection results
        """
        if len(self.reward_history) < max(10, self.window_size // 10):
            return {'plateau_detected': False, 'converged': False}

        rewards = np.array(list(self.reward_history))

        # Check if recent performance is stable and not declining
        recent_quarter = len(rewards) // 4
        recent_rewards = rewards[-recent_quarter:]
        older_rewards = rewards[-2*recent_quarter:-recent_quarter]

        # Plateau conditions
        small_change = abs(np.mean(recent_rewards) - np.mean(older_rewards)) < self.trend_threshold
        low_variance = np.std(recent_rewards) < np.std(rewards) * 0.7
        not_declining = np.mean(recent_rewards) >= np.mean(older_rewards) - self.trend_threshold

        plateau_detected = small_change and low_variance and not_declining

        # Convergence detection (stronger condition)
        very_stable = np.std(recent_rewards) < self.trend_threshold
        converged = plateau_detected and very_stable

        return {
            'plateau_detected': plateau_detected,
            'converged': converged,
            'stability_score': 1.0 - (np.std(recent_rewards) / (np.std(rewards) + 1e-8))
        }


class ContextAwareEntropyRegulator:
    """
    Main entropy regulation system that makes intelligent decisions about when to intervene.

    This system:
    1. Monitors state-action relationships for context sensitivity
    2. Tracks performance trends to detect success vs failure
    3. Allows natural convergence while preventing mode collapse
    4. Uses minimal intervention principle
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        device: str = "cpu",
        # Base entropy parameters
        entropy_min: float = 0.01,
        entropy_max: float = 0.5,
        target_entropy_ratio: float = 0.6,  # Fraction of max entropy
        # Intervention thresholds
        mode_collapse_threshold: float = 0.75,  # Max frequency for single action (legacy)
        min_action_freq_threshold: float = 0.01,  # Min frequency per action (load balancing critical)
        context_sensitivity_threshold: float = 0.1,  # Min mutual info for context sensitivity
        performance_decline_threshold: float = -0.1,  # Significant performance decline
        # Adjustment parameters
        emergency_boost_factor: float = 10.0,
        gentle_adjustment_rate: float = 0.01,
        intervention_cooldown: int = 3,  # Steps between interventions (reduced for immediate mode collapse detection)
        # Analysis parameters
        analysis_window: int = 500,
        min_samples_for_analysis: int = 20
    ):
        """
        Initialize context-aware entropy regulator.
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = device

        # Entropy bounds
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max
        self.target_entropy = target_entropy_ratio * math.log(num_actions)

        # Thresholds
        self.mode_collapse_threshold = mode_collapse_threshold  # Legacy max freq threshold
        self.min_action_freq_threshold = min_action_freq_threshold  # Critical min freq threshold
        self.context_sensitivity_threshold = context_sensitivity_threshold
        self.performance_decline_threshold = performance_decline_threshold

        # Adjustment parameters
        self.emergency_boost_factor = emergency_boost_factor
        self.gentle_adjustment_rate = gentle_adjustment_rate
        self.intervention_cooldown = intervention_cooldown
        self.min_samples_for_analysis = min_samples_for_analysis

        # Components
        self.state_action_analyzer = StateActionAnalyzer(
            state_dim=state_dim,
            num_actions=num_actions,
            history_window=analysis_window,
            state_discretization_bins=10,  # Default value
            correlation_threshold=context_sensitivity_threshold
        )

        self.performance_tracker = PerformanceTracker(
            window_size=analysis_window,
            trend_threshold=abs(performance_decline_threshold),
            stability_window=50
        )

        # Current state
        self.current_entropy_coef = 0.1  # Start with moderate entropy
        self.steps_since_intervention = 0
        self.total_steps = 0
        self.intervention_history = []
        self.emergency_mode = False

        logger.info(f"Initialized ContextAwareEntropyRegulator with target_entropy={self.target_entropy:.3f}")

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        value_loss: float,
        policy_loss: float,
        kl_divergence: float,
        current_entropy: float
    ) -> Dict[str, Union[float, bool, str, Dict]]:
        """
        Main update function that analyzes current state and adjusts entropy coefficient.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            value_loss: Current value function loss
            policy_loss: Current policy loss
            kl_divergence: Current KL divergence
            current_entropy: Current policy entropy

        Returns:
            Dictionary with entropy adjustment and analysis results
        """
        print(f"[DEBUG] ContextAwareEntropyRegulator.update() called with {len(actions)} actions")
        print(f"[DEBUG] Actions tensor shape: {actions.shape}, dtype: {actions.dtype}")
        print(f"[DEBUG] First few actions: {actions[:10] if len(actions) > 10 else actions}")

        self.total_steps += 1
        self.steps_since_intervention += 1

        # Update analyzers
        self.state_action_analyzer.update_history(states, actions)
        self.performance_tracker.update(rewards, value_loss, policy_loss, kl_divergence)

        # Get analysis results
        print(f"[DEBUG] History lengths: state={len(self.state_action_analyzer.state_history)}, action={len(self.state_action_analyzer.action_history)}")
        context_analysis = self.state_action_analyzer.compute_mutual_information()
        action_patterns = self.state_action_analyzer.analyze_action_patterns()
        print(f"[DEBUG] Analysis results - context_analysis keys: {context_analysis.keys()}")
        performance_analysis = self.performance_tracker.analyze_performance_trend()
        plateau_analysis = self.performance_tracker.detect_performance_plateau()

        # Make adjustment decision
        adjustment_result = self._decide_entropy_adjustment(
            context_analysis, action_patterns, performance_analysis,
            plateau_analysis, current_entropy
        )

        # Apply adjustment
        if adjustment_result['should_adjust']:
            self._apply_entropy_adjustment(adjustment_result['adjustment_factor'])
            self.steps_since_intervention = 0
            self.intervention_history.append({
                'step': self.total_steps,
                'reason': adjustment_result['reason'],
                'adjustment': adjustment_result['adjustment_factor'],
                'old_coef': adjustment_result['old_entropy_coef'],
                'new_coef': self.current_entropy_coef
            })

        return {
            'entropy_coef': self.current_entropy_coef,
            'should_adjust': adjustment_result['should_adjust'],
            'adjustment_reason': adjustment_result['reason'],
            'emergency_mode': self.emergency_mode,
            'analysis': {
                'context_sensitivity': context_analysis,
                'action_patterns': action_patterns,
                'performance': performance_analysis,
                'plateau': plateau_analysis
            },
            'intervention_count': len(self.intervention_history),
            'steps_since_intervention': self.steps_since_intervention
        }

    def _decide_entropy_adjustment(
        self,
        context_analysis: Dict,
        action_patterns: Dict,
        performance_analysis: Dict,
        plateau_analysis: Dict,
        current_entropy: float
    ) -> Dict[str, Union[bool, float, str]]:
        """Decide whether entropy adjustment is needed and by how much."""

        # Don't adjust too frequently
        if self.steps_since_intervention < self.intervention_cooldown:
            return {
                'should_adjust': False,
                'reason': 'Intervention cooldown active',
                'adjustment_factor': 1.0,
                'old_entropy_coef': self.current_entropy_coef
            }

        # Need sufficient data for reliable analysis
        print(f"[DEBUG] Sample size check: {context_analysis['sample_size']} vs threshold {self.min_samples_for_analysis}")
        if context_analysis['sample_size'] < self.min_samples_for_analysis:
            return {
                'should_adjust': False,
                'reason': 'Insufficient data for analysis',
                'adjustment_factor': 1.0,
                'old_entropy_coef': self.current_entropy_coef
            }

        # Extract key metrics
        max_action_freq = action_patterns['max_action_freq']
        is_context_sensitive = context_analysis['is_context_sensitive']
        is_improving = performance_analysis['is_improving']
        is_stable = performance_analysis['is_stable']
        performance_score = performance_analysis['performance_score']
        plateau_detected = plateau_analysis['plateau_detected']

        # Advanced Early Warning System: Multi-layer collapse detection
        action_frequencies = context_analysis['action_distribution']
        min_action_freq = min(action_frequencies) if action_frequencies else 1.0

        # Calculate distribution variance for early warning
        import numpy as np
        freq_array = np.array(action_frequencies)
        expected_uniform = 1.0 / self.num_actions  # 0.25 for 4 replicas
        variance_from_uniform = np.var(freq_array)
        max_allowed_variance = expected_uniform * 0.5  # Early warning threshold

        # Distribution skew detection (early intervention)
        std_dev = np.std(freq_array)
        coefficient_of_variation = std_dev / np.mean(freq_array) if np.mean(freq_array) > 0 else 0
        high_skew = coefficient_of_variation > 0.6  # Detect early imbalance

        # Progressive thresholds for graduated intervention
        moderate_imbalance = min_action_freq < (self.min_action_freq_threshold * 2.0)  # 2x threshold for early warning
        severe_imbalance_early = min_action_freq < (self.min_action_freq_threshold * 1.5)  # 1.5x for aggressive intervention

        # DEBUG: Enhanced detailed analysis
        print(f"[DEBUG] Advanced collapse analysis:")
        print(f"[DEBUG] - action_frequencies: {action_frequencies}")
        print(f"[DEBUG] - variance_from_uniform: {variance_from_uniform:.6f} (threshold: {max_allowed_variance:.6f})")
        print(f"[DEBUG] - coefficient_of_variation: {coefficient_of_variation:.4f} (threshold: 0.6)")
        print(f"[DEBUG] - high_skew (early warning): {high_skew}")
        print(f"[DEBUG] - moderate_imbalance (2x threshold): {moderate_imbalance}")
        print(f"[DEBUG] - severe_imbalance_early (1.5x threshold): {severe_imbalance_early}")
        print(f"[DEBUG] - min_action_freq: {min_action_freq:.4f} (critical threshold: {self.min_action_freq_threshold})")
        print(f"[DEBUG] - max_action_freq: {max_action_freq:.4f} (threshold: {self.mode_collapse_threshold})")
        print(f"[DEBUG] - context_sensitive: {is_context_sensitive}")
        print(f"[DEBUG] - performance_score: {performance_score:.4f} (threshold: {self.performance_decline_threshold})")

        # Mode collapse conditions:
        # 1. Under-representation: any action/replica gets too little traffic
        # 2. Over-concentration: single action gets too much traffic (legacy check)
        under_representation = min_action_freq < self.min_action_freq_threshold
        over_concentration = max_action_freq > self.mode_collapse_threshold
        severe_imbalance = under_representation or over_concentration

        context_insensitive = not is_context_sensitive
        performance_declining = performance_score < self.performance_decline_threshold

        print(f"[DEBUG] - under_representation: {under_representation}")
        print(f"[DEBUG] - over_concentration: {over_concentration}")
        print(f"[DEBUG] - severe_imbalance: {severe_imbalance}")
        print(f"[DEBUG] - context_insensitive: {context_insensitive}")
        print(f"[DEBUG] - performance_declining: {performance_declining}")
        # Research-based critical entropy threshold: 0.1 relative entropy ratio
        current_entropy_ratio = current_entropy / math.log(self.num_actions)
        critical_low_entropy = current_entropy_ratio < 0.1  # Research recommendation

        print(f"[DEBUG] - current_entropy_ratio: {current_entropy_ratio:.4f} (critical threshold: 0.1)")
        print(f"[DEBUG] - critical_low_entropy: {critical_low_entropy}")

        # Advanced Multi-tier Intervention System
        # Tier 1: Early Warning (Gentle Intervention)
        early_warning_triggers = [
            high_skew and not is_context_sensitive,  # Distribution becoming skewed + context insensitive
            moderate_imbalance and performance_declining,  # Moderate imbalance + declining performance
            variance_from_uniform > max_allowed_variance,  # High variance from uniform distribution
        ]
        early_warning_active = any(early_warning_triggers)

        # Tier 2: Urgent Intervention (Medium Boost)
        urgent_triggers = [
            severe_imbalance_early,  # 1.5x threshold breach
            coefficient_of_variation > 0.8,  # Very high skew
            current_entropy_ratio < 0.2,  # Moderate entropy crisis
        ]
        urgent_intervention_needed = any(urgent_triggers)

        # Tier 3: Emergency Intervention (Maximum Boost)
        emergency_triggers = [
            (severe_imbalance and (context_insensitive or performance_declining)),
            critical_low_entropy,  # Research-based critical entropy threshold
        ]
        emergency_intervention_needed = any(emergency_triggers)

        print(f"[DEBUG] - early_warning_active: {early_warning_active} (triggers: {early_warning_triggers})")
        print(f"[DEBUG] - urgent_intervention_needed: {urgent_intervention_needed} (triggers: {urgent_triggers})")
        print(f"[DEBUG] - emergency_intervention_needed: {emergency_intervention_needed} (triggers: {emergency_triggers})")

        # Execute interventions with graduated response
        if emergency_intervention_needed:
            self.emergency_mode = True
            if critical_low_entropy:
                collapse_reason = f"critical_low_entropy_ratio_{current_entropy_ratio:.3f}"
                reason = f'EMERGENCY: Critical entropy collapse {current_entropy_ratio:.3f} < 0.1 (研究建议紧急干预)'
            else:
                collapse_reason = "severe_under_representation" if severe_imbalance else "over_concentration"
                reason = f'EMERGENCY: Severe imbalance detected (min={min_action_freq:.3f}, context_insensitive={context_insensitive})'

            return {
                'should_adjust': True,
                'reason': reason,
                'adjustment_factor': self.emergency_boost_factor,  # Max boost (10.0)
                'old_entropy_coef': self.current_entropy_coef
            }

        elif urgent_intervention_needed:
            reason = f'URGENT: Distribution imbalance (early_severe={severe_imbalance_early}, cv={coefficient_of_variation:.3f}, entropy_ratio={current_entropy_ratio:.3f})'
            return {
                'should_adjust': True,
                'reason': reason,
                'adjustment_factor': self.emergency_boost_factor * 0.6,  # Medium boost (6.0)
                'old_entropy_coef': self.current_entropy_coef
            }

        elif early_warning_active:
            reason = f'EARLY_WARNING: Preventing collapse (skew={high_skew}, moderate_imbalance={moderate_imbalance}, variance={variance_from_uniform:.4f})'
            return {
                'should_adjust': True,
                'reason': reason,
                'adjustment_factor': self.emergency_boost_factor * 0.3,  # Gentle boost (3.0)
                'old_entropy_coef': self.current_entropy_coef
            }

        # Exit emergency mode if conditions improve
        action_balance_restored = (min_action_freq >= self.min_action_freq_threshold * 1.2 and
                                  max_action_freq < self.mode_collapse_threshold * 0.8)

        if self.emergency_mode and (action_balance_restored or is_context_sensitive):
            self.emergency_mode = False
            return {
                'should_adjust': True,
                'reason': f'Emergency mode deactivated: balance restored (min_freq={min_action_freq:.3f}, max_freq={max_action_freq:.3f})',
                'adjustment_factor': 0.7,  # Gentle reduction
                'old_entropy_coef': self.current_entropy_coef
            }

        # Gentle adjustments for less severe issues
        if severe_imbalance and not context_insensitive and not is_improving:
            return {
                'should_adjust': True,
                'reason': f'Gentle increase: action imbalance without context sensitivity (max_freq={max_action_freq:.3f})',
                'adjustment_factor': 1.0 + self.gentle_adjustment_rate,
                'old_entropy_coef': self.current_entropy_coef
            }

        # Allow natural decay if performance is stable/improving and context-sensitive
        if plateau_detected and is_context_sensitive and (is_stable or is_improving):
            current_entropy_ratio = current_entropy / math.log(self.num_actions)
            if current_entropy_ratio > 0.1 and self.current_entropy_coef > self.entropy_min:
                return {
                    'should_adjust': True,
                    'reason': 'Natural decay: stable performance with context sensitivity',
                    'adjustment_factor': 1.0 - self.gentle_adjustment_rate,
                    'old_entropy_coef': self.current_entropy_coef
                }

        # No adjustment needed
        return {
            'should_adjust': False,
            'reason': 'No intervention needed',
            'adjustment_factor': 1.0,
            'old_entropy_coef': self.current_entropy_coef
        }

    def _apply_entropy_adjustment(self, adjustment_factor: float) -> None:
        """Apply entropy coefficient adjustment with bounds checking."""
        old_coef = self.current_entropy_coef
        new_coef = old_coef * adjustment_factor
        self.current_entropy_coef = np.clip(new_coef, self.entropy_min, self.entropy_max)

        logger.info(f"Entropy coefficient adjusted: {old_coef:.4f} -> {self.current_entropy_coef:.4f} (factor={adjustment_factor:.3f})")

    def get_entropy_coef(self) -> float:
        """Get current entropy coefficient."""
        return self.current_entropy_coef

    def reset(self) -> None:
        """Reset regulator state (useful between training runs)."""
        self.current_entropy_coef = 0.1
        self.steps_since_intervention = 0
        self.total_steps = 0
        self.intervention_history.clear()
        self.emergency_mode = False

        # Reset analyzers
        self.state_action_analyzer = StateActionAnalyzer(
            state_dim=self.state_dim,
            num_actions=self.num_actions,
            history_window=500,
            correlation_threshold=self.context_sensitivity_threshold
        )

        self.performance_tracker = PerformanceTracker(
            window_size=500,
            trend_threshold=abs(self.performance_decline_threshold),
            stability_window=50
        )

    def get_diagnostic_info(self) -> Dict:
        """Get comprehensive diagnostic information."""
        return {
            'current_entropy_coef': self.current_entropy_coef,
            'emergency_mode': self.emergency_mode,
            'total_steps': self.total_steps,
            'steps_since_intervention': self.steps_since_intervention,
            'intervention_count': len(self.intervention_history),
            'recent_interventions': self.intervention_history[-5:] if self.intervention_history else [],
            'target_entropy': self.target_entropy,
            'bounds': {'min': self.entropy_min, 'max': self.entropy_max}
        }