"""
Enhanced PPO Collapse Detection and Recovery System.

This module implements revolutionary collapse detection based on the deep analysis of
policy collapse patterns in load balancing systems. It addresses the critical failure
modes identified in training metrics analysis.

Key improvements over standard approaches:
1. Multi-signal early warning system with aggressive thresholds
2. Coefficient of variation tracking for distribution collapse
3. Gradient vanishing detection and preservation mechanisms
4. Emergency intervention with forced exploration
5. Adaptive threshold adjustment based on system performance

Based on the analysis showing collapse patterns:
- Step 100-397: CV 0.228 (stable)
- Step 598-893: CV 1.317 (early collapse)
- Step 1094-1389: CV 1.627 (severe collapse)
- Step 3078-3473: CV 1.697 (complete failure)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EnhancedCollapseDetector:
    """
    Revolutionary collapse detection system with multi-signal early warning.

    Implements aggressive early detection based on real collapse patterns:
    - Coefficient of variation threshold: 0.3 (vs previous 0.7)
    - Minimum action frequency: 15% (vs previous 1%)
    - Gradient norm tracking for vanishing gradients
    - Emergency entropy boost: 20x (vs previous 2x)
    """

    def __init__(
        self,
        num_actions: int,
        # Enhanced thresholds based on collapse analysis
        cv_warning_threshold: float = 0.3,      # Early warning at CV > 0.3
        cv_emergency_threshold: float = 0.8,    # Emergency at CV > 0.8
        min_action_freq_threshold: float = 0.15,  # Any action < 15% triggers warning
        gradient_norm_threshold: float = 1e-6,  # Gradient vanishing detection
        entropy_collapse_threshold: float = 0.1, # Entropy below 10% of max
        # Enhanced intervention parameters
        emergency_entropy_boost: float = 20.0,  # Aggressive 20x boost
        forced_exploration_steps: int = 50,     # Force exploration for 50 steps
        intervention_cooldown: int = 10,        # Reduced cooldown for faster response
        # Monitoring windows
        detection_window: int = 20,             # Check every 20 steps (vs 50)
        history_window: int = 100,              # Track 100 step history
        # Adaptive thresholds
        enable_adaptive_thresholds: bool = True,
        performance_decline_threshold: float = -0.1,  # 10% performance decline
    ):
        """
        Initialize enhanced collapse detector.

        Args:
            num_actions: Number of possible actions (replicas)
            cv_warning_threshold: Coefficient of variation warning threshold
            cv_emergency_threshold: CV emergency intervention threshold
            min_action_freq_threshold: Minimum frequency for any action
            gradient_norm_threshold: Minimum gradient norm (below = vanishing)
            entropy_collapse_threshold: Entropy collapse detection threshold
            emergency_entropy_boost: Factor to boost entropy during emergency
            forced_exploration_steps: Steps to force exploration after intervention
            intervention_cooldown: Steps between interventions
            detection_window: Frequency of collapse detection checks
            history_window: Size of action history for analysis
            enable_adaptive_thresholds: Whether to adapt thresholds based on performance
            performance_decline_threshold: Performance decline threshold for adaptation
        """
        self.num_actions = num_actions
        self.cv_warning_threshold = cv_warning_threshold
        self.cv_emergency_threshold = cv_emergency_threshold
        self.min_action_freq_threshold = min_action_freq_threshold
        self.gradient_norm_threshold = gradient_norm_threshold
        self.entropy_collapse_threshold = entropy_collapse_threshold
        self.emergency_entropy_boost = emergency_entropy_boost
        self.forced_exploration_steps = forced_exploration_steps
        self.intervention_cooldown = intervention_cooldown
        self.detection_window = detection_window
        self.history_window = history_window
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.performance_decline_threshold = performance_decline_threshold

        # State tracking
        self.action_history = deque(maxlen=history_window)
        self.reward_history = deque(maxlen=history_window)
        self.gradient_norm_history = deque(maxlen=50)
        self.entropy_history = deque(maxlen=50)

        # Intervention state
        self.last_intervention_step = -1000
        self.forced_exploration_remaining = 0
        self.intervention_count = 0
        self.current_step = 0

        # Performance tracking
        self.baseline_performance = None
        self.performance_ema = None
        self.performance_ema_alpha = 0.1

        # Alert state
        self.current_alert_level = "NORMAL"  # NORMAL, WARNING, EMERGENCY
        self.consecutive_warnings = 0

        logger.info(f"[EnhancedCollapseDetector] Initialized with aggressive thresholds: "
                   f"CV_warning={cv_warning_threshold}, CV_emergency={cv_emergency_threshold}, "
                   f"min_freq={min_action_freq_threshold}, entropy_boost={emergency_entropy_boost}x")

    def update_metrics(
        self,
        action: int,
        reward: float,
        entropy: float,
        gradient_norm: float,
        step: int
    ) -> None:
        """Update tracking metrics for collapse detection."""
        self.current_step = step
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.entropy_history.append(entropy)
        self.gradient_norm_history.append(gradient_norm)

        # Update performance EMA
        if self.performance_ema is None:
            self.performance_ema = reward
            if self.baseline_performance is None and len(self.reward_history) >= 20:
                self.baseline_performance = np.mean(list(self.reward_history)[-20:])
        else:
            self.performance_ema = (1 - self.performance_ema_alpha) * self.performance_ema + \
                                 self.performance_ema_alpha * reward

    def detect_collapse(self) -> Dict[str, Any]:
        """
        Detect policy collapse using multi-signal analysis.

        Returns:
            Dictionary with detection results and recommended interventions
        """
        # Skip detection if insufficient data or too recent intervention
        if (len(self.action_history) < 20 or
            self.current_step < self.last_intervention_step + self.intervention_cooldown):
            return self._create_detection_result("NORMAL", 0.0, {})

        # Only check every detection_window steps for efficiency
        if self.current_step % self.detection_window != 0:
            return self._create_detection_result(self.current_alert_level, 0.0, {})

        # 1. Coefficient of Variation Analysis (Primary Signal)
        cv_analysis = self._analyze_coefficient_variation()

        # 2. Action Frequency Analysis (Secondary Signal)
        freq_analysis = self._analyze_action_frequencies()

        # 3. Gradient Vanishing Detection (Tertiary Signal)
        gradient_analysis = self._analyze_gradient_health()

        # 4. Entropy Collapse Detection (Quaternary Signal)
        entropy_analysis = self._analyze_entropy_collapse()

        # 5. Performance Decline Detection (Quinary Signal)
        performance_analysis = self._analyze_performance_trend()

        # Aggregate all signals for final decision
        collapse_analysis = self._aggregate_signals(
            cv_analysis, freq_analysis, gradient_analysis,
            entropy_analysis, performance_analysis
        )

        # Update adaptive thresholds if enabled
        if self.enable_adaptive_thresholds:
            self._update_adaptive_thresholds(collapse_analysis)

        # Log detailed analysis for debugging
        if collapse_analysis['alert_level'] != "NORMAL":
            logger.warning(f"[CollapseDetector] {collapse_analysis['alert_level']} detected at step {self.current_step}")
            logger.warning(f"  CV: {cv_analysis['coefficient_variation']:.3f} (threshold: {self.cv_warning_threshold})")
            logger.warning(f"  Min freq: {freq_analysis['min_action_freq']:.3f} (threshold: {self.min_action_freq_threshold})")
            logger.warning(f"  Gradient norm: {gradient_analysis['avg_gradient_norm']:.2e}")
            logger.warning(f"  Entropy: {entropy_analysis['current_entropy']:.3f}")
            logger.warning(f"  Action dist: {freq_analysis['action_distribution']}")

        return collapse_analysis

    def _analyze_coefficient_variation(self) -> Dict[str, float]:
        """Analyze coefficient of variation for distribution collapse."""
        if len(self.action_history) < 20:
            return {'coefficient_variation': 0.0, 'cv_alert_level': 'NORMAL'}

        # Calculate action frequencies
        actions = np.array(list(self.action_history))
        action_counts = np.bincount(actions, minlength=self.num_actions)
        action_frequencies = action_counts / len(actions)

        # Calculate coefficient of variation
        mean_freq = np.mean(action_frequencies)
        std_freq = np.std(action_frequencies)
        cv = std_freq / (mean_freq + 1e-8)

        # Determine alert level based on CV
        if cv >= self.cv_emergency_threshold:
            cv_alert_level = 'EMERGENCY'
        elif cv >= self.cv_warning_threshold:
            cv_alert_level = 'WARNING'
        else:
            cv_alert_level = 'NORMAL'

        return {
            'coefficient_variation': cv,
            'cv_alert_level': cv_alert_level,
            'action_frequencies': action_frequencies.tolist(),
            'mean_frequency': mean_freq,
            'std_frequency': std_freq
        }

    def _analyze_action_frequencies(self) -> Dict[str, Any]:
        """Analyze individual action frequencies for severe imbalance."""
        if len(self.action_history) < 10:
            equal_freq = 1.0 / self.num_actions
            return {
                'min_action_freq': equal_freq,
                'max_action_freq': equal_freq,
                'freq_alert_level': 'NORMAL',
                'action_distribution': [equal_freq] * self.num_actions,
                'dominant_action': None
            }

        # Calculate current action distribution
        actions = np.array(list(self.action_history))
        action_counts = np.bincount(actions, minlength=self.num_actions)
        action_frequencies = action_counts / len(actions)

        min_freq = np.min(action_frequencies)
        max_freq = np.max(action_frequencies)
        dominant_action = np.argmax(action_frequencies)

        # Determine alert level
        if min_freq < 0.05 or max_freq > 0.85:  # Severe imbalance
            freq_alert_level = 'EMERGENCY'
        elif min_freq < self.min_action_freq_threshold or max_freq > 0.7:
            freq_alert_level = 'WARNING'
        else:
            freq_alert_level = 'NORMAL'

        return {
            'min_action_freq': min_freq,
            'max_action_freq': max_freq,
            'freq_alert_level': freq_alert_level,
            'action_distribution': action_frequencies.tolist(),
            'dominant_action': dominant_action,
            'dominant_freq': max_freq
        }

    def _analyze_gradient_health(self) -> Dict[str, float]:
        """Analyze gradient norms for vanishing gradient detection."""
        if len(self.gradient_norm_history) < 5:
            return {
                'avg_gradient_norm': 1.0,
                'gradient_alert_level': 'NORMAL',
                'gradient_trend': 0.0
            }

        gradient_norms = np.array(list(self.gradient_norm_history))
        avg_grad_norm = np.mean(gradient_norms[-10:])  # Recent average

        # Check for vanishing gradients
        if avg_grad_norm < self.gradient_norm_threshold:
            gradient_alert_level = 'EMERGENCY'
        elif avg_grad_norm < self.gradient_norm_threshold * 10:
            gradient_alert_level = 'WARNING'
        else:
            gradient_alert_level = 'NORMAL'

        # Analyze trend
        if len(gradient_norms) >= 10:
            recent_avg = np.mean(gradient_norms[-5:])
            older_avg = np.mean(gradient_norms[-10:-5])
            gradient_trend = (recent_avg - older_avg) / (older_avg + 1e-8)
        else:
            gradient_trend = 0.0

        return {
            'avg_gradient_norm': avg_grad_norm,
            'gradient_alert_level': gradient_alert_level,
            'gradient_trend': gradient_trend
        }

    def _analyze_entropy_collapse(self) -> Dict[str, float]:
        """Analyze entropy for policy collapse detection."""
        if len(self.entropy_history) < 5:
            return {
                'current_entropy': 1.0,
                'entropy_alert_level': 'NORMAL',
                'max_possible_entropy': math.log(self.num_actions)
            }

        current_entropy = np.mean(list(self.entropy_history)[-5:])  # Recent entropy
        max_possible_entropy = math.log(self.num_actions)
        relative_entropy = current_entropy / max_possible_entropy

        # Determine alert level
        if relative_entropy < self.entropy_collapse_threshold:
            entropy_alert_level = 'EMERGENCY'
        elif relative_entropy < self.entropy_collapse_threshold * 2:
            entropy_alert_level = 'WARNING'
        else:
            entropy_alert_level = 'NORMAL'

        return {
            'current_entropy': current_entropy,
            'relative_entropy': relative_entropy,
            'entropy_alert_level': entropy_alert_level,
            'max_possible_entropy': max_possible_entropy
        }

    def _analyze_performance_trend(self) -> Dict[str, float]:
        """Analyze performance trends to distinguish failure from convergence."""
        if len(self.reward_history) < 20 or self.baseline_performance is None:
            return {
                'performance_decline': 0.0,
                'performance_alert_level': 'NORMAL',
                'current_performance': 0.0
            }

        current_performance = self.performance_ema
        performance_decline = (current_performance - self.baseline_performance) / (abs(self.baseline_performance) + 1e-8)

        # Determine alert level
        if performance_decline < self.performance_decline_threshold:
            performance_alert_level = 'EMERGENCY'
        elif performance_decline < self.performance_decline_threshold / 2:
            performance_alert_level = 'WARNING'
        else:
            performance_alert_level = 'NORMAL'

        return {
            'performance_decline': performance_decline,
            'performance_alert_level': performance_alert_level,
            'current_performance': current_performance,
            'baseline_performance': self.baseline_performance
        }

    def _aggregate_signals(
        self, cv_analysis: Dict, freq_analysis: Dict,
        gradient_analysis: Dict, entropy_analysis: Dict,
        performance_analysis: Dict
    ) -> Dict[str, Any]:
        """Aggregate all detection signals into final decision."""

        # Count emergency and warning signals
        alert_levels = [
            cv_analysis['cv_alert_level'],
            freq_analysis['freq_alert_level'],
            gradient_analysis['gradient_alert_level'],
            entropy_analysis['entropy_alert_level'],
            performance_analysis['performance_alert_level']
        ]

        emergency_count = alert_levels.count('EMERGENCY')
        warning_count = alert_levels.count('WARNING')

        # Decision logic: More aggressive than standard approaches
        if emergency_count >= 1:  # Any emergency signal triggers intervention
            final_alert_level = 'EMERGENCY'
            confidence_score = 0.9 + 0.1 * min(emergency_count / 2, 1.0)
        elif warning_count >= 2 or (warning_count >= 1 and cv_analysis['cv_alert_level'] == 'WARNING'):
            # Two warnings or CV warning (primary signal) triggers intervention
            final_alert_level = 'WARNING'
            confidence_score = 0.6 + 0.2 * min(warning_count / 2, 1.0)
        else:
            final_alert_level = 'NORMAL'
            confidence_score = 0.1

        # Update consecutive warning count
        if final_alert_level == 'WARNING':
            self.consecutive_warnings += 1
            # Escalate to emergency after 3 consecutive warnings
            if self.consecutive_warnings >= 3:
                final_alert_level = 'EMERGENCY'
                confidence_score = 0.85
        else:
            self.consecutive_warnings = 0

        # Generate intervention recommendations
        interventions = self._generate_interventions(
            final_alert_level, cv_analysis, freq_analysis,
            gradient_analysis, entropy_analysis, performance_analysis
        )

        return self._create_detection_result(
            final_alert_level, confidence_score, interventions,
            {
                'cv_analysis': cv_analysis,
                'freq_analysis': freq_analysis,
                'gradient_analysis': gradient_analysis,
                'entropy_analysis': entropy_analysis,
                'performance_analysis': performance_analysis,
                'emergency_signals': emergency_count,
                'warning_signals': warning_count,
                'consecutive_warnings': self.consecutive_warnings
            }
        )

    def _generate_interventions(
        self, alert_level: str, cv_analysis: Dict, freq_analysis: Dict,
        gradient_analysis: Dict, entropy_analysis: Dict, performance_analysis: Dict
    ) -> Dict[str, Any]:
        """Generate specific intervention recommendations based on detected issues."""

        interventions = {
            'entropy_boost_factor': 1.0,
            'force_exploration': False,
            'exploration_steps': 0,
            'gradient_boost': False,
            'reset_to_checkpoint': False,
            'emergency_actions': []
        }

        if alert_level == 'EMERGENCY':
            # Aggressive emergency interventions
            interventions.update({
                'entropy_boost_factor': self.emergency_entropy_boost,  # 20x boost
                'force_exploration': True,
                'exploration_steps': self.forced_exploration_steps,
                'gradient_boost': True,
                'emergency_actions': [
                    'BOOST_ENTROPY', 'FORCE_EXPLORATION', 'GRADIENT_BOOST'
                ]
            })

            # Additional emergency actions based on specific failure modes
            if cv_analysis['coefficient_variation'] > 1.5:
                interventions['emergency_actions'].append('RESET_TO_CHECKPOINT')
                interventions['reset_to_checkpoint'] = True

            if gradient_analysis['avg_gradient_norm'] < self.gradient_norm_threshold:
                interventions['emergency_actions'].append('REINITIALIZE_OPTIMIZER')

        elif alert_level == 'WARNING':
            # Moderate warning interventions
            interventions.update({
                'entropy_boost_factor': min(5.0, self.emergency_entropy_boost / 2),  # 5x boost
                'force_exploration': True,
                'exploration_steps': self.forced_exploration_steps // 2,
                'emergency_actions': ['BOOST_ENTROPY', 'INCREASE_EXPLORATION']
            })

        return interventions

    def _create_detection_result(
        self, alert_level: str, confidence: float,
        interventions: Dict[str, Any],
        details: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create standardized detection result."""

        self.current_alert_level = alert_level

        result = {
            'alert_level': alert_level,
            'confidence_score': confidence,
            'needs_intervention': alert_level in ['WARNING', 'EMERGENCY'],
            'interventions': interventions,
            'step': self.current_step,
            'time_since_last_intervention': self.current_step - self.last_intervention_step,
            'total_interventions': self.intervention_count
        }

        if details:
            result['details'] = details

        return result

    def _update_adaptive_thresholds(self, collapse_analysis: Dict) -> None:
        """Update detection thresholds based on system performance."""
        if not self.enable_adaptive_thresholds or len(self.reward_history) < 50:
            return

        # Get performance metrics
        performance_analysis = collapse_analysis['details']['performance_analysis']
        performance_decline = performance_analysis['performance_decline']

        # Adapt thresholds based on performance - make more sensitive if declining
        if performance_decline < -0.05:  # Performance declining
            # Make thresholds more sensitive (lower)
            adaptation_factor = 0.95
            self.cv_warning_threshold *= adaptation_factor
            self.min_action_freq_threshold *= adaptation_factor

            logger.info(f"[AdaptiveThresholds] Tightened thresholds due to performance decline: "
                       f"CV={self.cv_warning_threshold:.3f}, MinFreq={self.min_action_freq_threshold:.3f}")

        elif performance_decline > 0.05:  # Performance improving
            # Relax thresholds slightly (but keep aggressive)
            adaptation_factor = 1.02
            self.cv_warning_threshold = min(0.5, self.cv_warning_threshold * adaptation_factor)
            self.min_action_freq_threshold = min(0.2, self.min_action_freq_threshold * adaptation_factor)

    def apply_intervention(self, intervention_type: str) -> bool:
        """Apply specific intervention and update internal state."""
        if self.current_step < self.last_intervention_step + self.intervention_cooldown:
            return False  # Too soon for another intervention

        self.last_intervention_step = self.current_step
        self.intervention_count += 1

        if intervention_type in ['FORCE_EXPLORATION', 'EMERGENCY']:
            self.forced_exploration_remaining = self.forced_exploration_steps

        logger.info(f"[CollapseDetector] Applied intervention '{intervention_type}' at step {self.current_step} "
                   f"(intervention #{self.intervention_count})")

        return True

    def should_force_exploration(self) -> bool:
        """Check if forced exploration should be applied."""
        if self.forced_exploration_remaining > 0:
            self.forced_exploration_remaining -= 1
            return True
        return False

    def get_status_summary(self) -> Dict[str, Any]:
        """Get current detector status summary."""
        return {
            'alert_level': self.current_alert_level,
            'step': self.current_step,
            'interventions_count': self.intervention_count,
            'last_intervention_step': self.last_intervention_step,
            'forced_exploration_remaining': self.forced_exploration_remaining,
            'consecutive_warnings': self.consecutive_warnings,
            'thresholds': {
                'cv_warning': self.cv_warning_threshold,
                'cv_emergency': self.cv_emergency_threshold,
                'min_action_freq': self.min_action_freq_threshold,
                'entropy_collapse': self.entropy_collapse_threshold
            }
        }