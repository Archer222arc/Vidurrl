"""
Reward calculation components for reinforcement learning schedulers.

This module provides reward computation logic for different modes:
- Delta mode: rewards based on change in metrics
- Instant mode: rewards based on current metric values
"""

from typing import Dict, List, Protocol, Tuple
from collections import deque

import numpy as np


class ReplicaScheduler(Protocol):
    """Protocol for replica scheduler interface."""

    def get_num_allocated_blocks(self) -> int:
        """Get number of allocated blocks."""
        ...

    def get_num_blocks(self) -> int:
        """Get total number of blocks."""
        ...


class MetricStore(Protocol):
    """Protocol for metric store interface."""

    def get_throughput(self, current_time: float) -> float:
        """Get current throughput."""
        ...

    def get_average_latency(self) -> float:
        """Get average latency."""
        ...


class RewardCalculator:
    """
    Calculates rewards for reinforcement learning based scheduling.

    Supports different reward modes for balancing throughput, latency,
    and load balancing objectives.
    """

    def __init__(
        self,
        mode: str = "delta",
        latency_weight: float = 1.0,
        balance_penalty_weight: float = 0.0,
        latency_threshold: float = 1.5,  # IMPROVED: Reduced from 2.0 to be more strict
        latency_penalty_scale: float = 3.0,  # IMPROVED: Reduced from 5.0 to soften penalty curve
        load_balance_penalty: float = 0.05,  # IMPROVED: Increased from 0.03 for better balance
        # New parameters for restructured reward - OPTIMIZED VALUES
        throughput_target: float = 10.0,
        absolute_weight: float = 0.6,  # IMPROVED: Reduced from 0.7 to balance abs vs delta
        delta_weight: float = 0.4,     # IMPROVED: Increased from 0.3 to emphasize improvement
        alpha: float = 0.4,            # IMPROVED: Reduced from 0.5 to reduce latency dominance
        beta: float = 0.4,             # IMPROVED: Increased from 0.3 for better throughput signal
        gamma: float = 0.3,            # IMPROVED: Increased from 0.2 for stronger latency signal
        kappa: float = 0.25,           # IMPROVED: Reduced from 0.3 to soften logistic penalty
        sigma: float = 1.2,            # IMPROVED: Increased from 1.0 for smoother penalty curve
        ema_alpha: float = 0.15,       # IMPROVED: Increased from 0.1 for faster adaptation
        # Phase 2: Production asymmetric penalty patterns
        use_asymmetric_penalties: bool = True,
        false_positive_penalty: float = 5.0,  # Under-provisioning penalty (Meta's 5:1 ratio)
        over_provision_factor: float = 0.1,   # Over-provisioning penalty
        beta_exploration_enable: bool = True, # Self-adaptive Beta distribution exploration
        temporal_tracking_enable: bool = True, # Temporal performance tracking
    ):
        """
        Initialize enhanced reward calculator.

        Args:
            mode: Reward mode ("delta", "instant", or "hybrid")
            latency_weight: Weight for latency penalty (legacy)
            balance_penalty_weight: Weight for load balancing penalty (legacy)
            latency_threshold: Soft latency threshold for penalty activation (seconds)
            latency_penalty_scale: Scale factor for latency threshold penalty (legacy)
            load_balance_penalty: Weight for replica load balance penalty (legacy)
            throughput_target: Target throughput for normalization
            absolute_weight: Weight for absolute score component (w_abs)
            delta_weight: Weight for delta score component (w_delta)
            alpha: Balance factor in absolute score (throughput vs latency)
            beta: Weight for normalized throughput delta
            gamma: Weight for normalized latency delta
            kappa: Weight for logistic latency penalty
            sigma: Scale for logistic penalty smoothness
            ema_alpha: Alpha for exponential moving averages
        """
        self.mode = mode.lower()
        if self.mode not in ("delta", "instant", "hybrid"):
            raise ValueError(f"Unknown reward mode: {mode}. Must be 'delta', 'instant', or 'hybrid'")

        # Legacy parameters (kept for backward compatibility)
        self.latency_weight = latency_weight
        self.balance_penalty_weight = balance_penalty_weight
        self.latency_threshold = latency_threshold
        self.latency_penalty_scale = latency_penalty_scale
        self.load_balance_penalty = load_balance_penalty

        # New restructured reward parameters
        self.throughput_target = throughput_target
        self.absolute_weight = absolute_weight
        self.delta_weight = delta_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.ema_alpha = ema_alpha

        # State tracking for delta mode
        self.last_throughput = 0.0
        self.last_latency = 0.0

        # Enhanced EMA tracking with variance
        self.throughput_ema = 0.0
        self.throughput_var = 0.0
        self.latency_ema = 0.0
        self.latency_var = 0.0

        # Phase 2: Production asymmetric penalty parameters
        self.use_asymmetric_penalties = use_asymmetric_penalties
        self.false_positive_penalty = false_positive_penalty
        self.over_provision_factor = over_provision_factor
        self.beta_exploration_enable = beta_exploration_enable
        self.temporal_tracking_enable = temporal_tracking_enable

        # Beta distribution parameters for self-adaptive exploration
        self.success_alpha = 1.0  # Success count + 1
        self.failure_beta = 1.0   # Failure count + 1
        self.beta_shaping_weight = 0.05

        # Temporal performance tracking
        self.performance_history = deque(maxlen=100)
        self.scheduling_efficiency_history = deque(maxlen=50)

        # Reward analysis
        self.reward_breakdown = {}
        self.step_count = 0

    def get_current_metrics(self, metric_store: MetricStore, current_time: float) -> Tuple[float, float]:
        """
        Get current throughput and latency metrics with proper handling of cold start.

        Args:
            metric_store: Metric store interface
            current_time: Current simulation time

        Returns:
            Tuple of (throughput, latency)
        """
        # Get raw metrics from store
        raw_throughput = float(metric_store.get_throughput(current_time))
        raw_latency = float(metric_store.get_average_latency())

        # Cold start protection: use alternative metrics when no completed requests exist
        if raw_throughput <= 0.0 and raw_latency <= 0.0:
            # Estimate throughput from scheduling rate
            if current_time > 0:
                estimated_throughput = self.step_count / current_time
            else:
                estimated_throughput = 1.0  # Baseline estimate

            # Estimate latency from system load indicators
            estimated_latency = max(0.5, self.latency_threshold * 0.3)  # Conservative baseline

            # Use estimates only if we have no historical data
            if self.last_throughput <= 0.0 and self.last_latency <= 0.0:
                throughput = max(estimated_throughput, 0.1)  # Minimum viable throughput
                latency = estimated_latency
            else:
                # Use last valid measurements
                throughput = self.last_throughput
                latency = self.last_latency
        else:
            # Use actual metrics when available
            throughput = max(raw_throughput, 0.01)  # Prevent division by zero
            latency = max(raw_latency, 0.01)

        return throughput, latency

    def update_ema_stats(self, throughput: float, latency: float) -> None:
        """
        Update exponential moving averages and variances for normalization.

        Args:
            throughput: Current throughput value
            latency: Current latency value
        """
        self.step_count += 1

        # Initialize EMA on first step
        if self.step_count == 1:
            self.throughput_ema = throughput
            self.latency_ema = latency
            self.throughput_var = 0.0
            self.latency_var = 0.0
            return

        # Update EMAs
        old_throughput_ema = self.throughput_ema
        old_latency_ema = self.latency_ema

        self.throughput_ema = (1 - self.ema_alpha) * self.throughput_ema + self.ema_alpha * throughput
        self.latency_ema = (1 - self.ema_alpha) * self.latency_ema + self.ema_alpha * latency

        # Update variances using EMA of squared differences
        throughput_diff_sq = (throughput - old_throughput_ema) ** 2
        latency_diff_sq = (latency - old_latency_ema) ** 2

        self.throughput_var = (1 - self.ema_alpha) * self.throughput_var + self.ema_alpha * throughput_diff_sq
        self.latency_var = (1 - self.ema_alpha) * self.latency_var + self.ema_alpha * latency_diff_sq

    def is_valid_update(self, raw_throughput: float, raw_latency: float) -> bool:
        """
        Check if metrics update should be processed or skipped.

        Skip updates when:
        1. Both metrics are zero (likely missing data)
        2. We have valid historical data and current values seem corrupted

        Args:
            raw_throughput: Raw throughput value from metric store
            raw_latency: Raw latency value from metric store

        Returns:
            True if update should be processed, False to skip
        """
        # Skip if both metrics are zero and we have valid history
        if (raw_throughput <= 0.0 and raw_latency <= 0.0 and
            self.last_throughput > 0.0 and self.last_latency > 0.0):
            return False

        # Skip if only one metric is zero but the jump would be too large
        if (raw_throughput <= 0.0 and self.last_throughput > 0.0 and
            raw_latency > self.last_latency * 3.0):  # Latency jump > 3x suggests data corruption
            return False

        if (raw_latency <= 0.0 and self.last_latency > 0.0 and
            raw_throughput < self.last_throughput * 0.3):  # Throughput drop > 70% suggests data corruption
            return False

        return True

    def calculate_balance_penalty(
        self,
        replica_ids: List[int],
        get_replica_scheduler_fn,
    ) -> float:
        """
        Calculate load balancing penalty based on utilization variance.

        Args:
            replica_ids: List of replica IDs
            get_replica_scheduler_fn: Function to get replica scheduler

        Returns:
            Balance penalty (max_utilization - min_utilization)
        """
        utilizations: List[float] = []

        for replica_id in replica_ids:
            scheduler = get_replica_scheduler_fn(replica_id)

            # Get allocated and total blocks - direct access, no fallback
            num_alloc = float(scheduler._num_allocated_blocks)
            num_blocks = float(scheduler._config.num_blocks)

            utilization = num_alloc / num_blocks
            utilizations.append(utilization)

        if not utilizations:
            return 0.0

        return float(max(utilizations) - min(utilizations))

    def calculate_reward(
        self,
        metric_store: MetricStore,
        current_time: float,
        replica_ids: List[int],
        get_replica_scheduler_fn,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate enhanced reward with soft latency penalty and load balancing.

        Args:
            metric_store: Metric store interface
            current_time: Current simulation time
            replica_ids: List of replica IDs
            get_replica_scheduler_fn: Function to get replica scheduler

        Returns:
            Tuple of (reward, info_dict) with detailed reward breakdown
        """
        # Get current metrics (both raw and processed values)
        raw_throughput = float(metric_store.get_throughput(current_time))
        raw_latency = float(metric_store.get_average_latency())
        throughput, latency = self.get_current_metrics(metric_store, current_time)

        # Update latency EMA for trend tracking
        if self.latency_ema == 0.0:
            self.latency_ema = latency
        else:
            self.latency_ema = (1 - self.ema_alpha) * self.latency_ema + self.ema_alpha * latency

        # Calculate penalties
        balance_penalty = self.calculate_balance_penalty(replica_ids, get_replica_scheduler_fn)
        load_balance_penalty = self.calculate_load_balance_penalty(replica_ids, get_replica_scheduler_fn)
        latency_threshold_penalty = self.calculate_latency_threshold_penalty(latency)

        # Base reward components
        base_info = {
            "throughput": throughput,
            "latency": latency,
            "latency_ema": self.latency_ema,
            "balance_penalty": balance_penalty,
            "load_balance_penalty": load_balance_penalty,
            "latency_threshold_penalty": latency_threshold_penalty,
        }

        if self.mode == "delta":
            # Delta mode: reward based on change in metrics
            delta_throughput = throughput - self.last_throughput
            delta_latency = latency - self.last_latency

            # Enhanced reward calculation
            raw_reward = (
                delta_throughput  # Encourage throughput increase
                - self.latency_weight * delta_latency  # Penalize latency increase
                - self.balance_penalty_weight * balance_penalty  # Legacy balance penalty
                - self.load_balance_penalty * load_balance_penalty  # New load balance penalty
                - latency_threshold_penalty  # Soft latency threshold penalty
            )

            # Apply adaptive scaling to remove performance ceiling
            # CRITICAL FIX: Remove hard clipping that prevents outperforming baselines
            reward_scale = 1.0  # IMPROVED: Further reduced scale to preserve more gradient info
            reward = self._adaptive_reward_scaling(raw_reward, reward_scale)

            # Store detailed breakdown for analysis with raw/clipped values
            self.reward_breakdown = {
                "base_throughput": delta_throughput,
                "latency_penalty": -self.latency_weight * delta_latency,
                "balance_penalty": -self.balance_penalty_weight * balance_penalty,
                "load_balance_penalty": -self.load_balance_penalty * load_balance_penalty,
                "latency_threshold_penalty": -latency_threshold_penalty,
                "raw_reward": raw_reward,  # Show original value before clipping
                "total_reward": reward,     # Show final clipped value
                "reward_ceiling_removed": True,  # Track that ceiling has been removed
                "uses_adaptive_scaling": True,  # Flag that we're using new scaling method
                "raw_throughput": raw_throughput,    # Show original metric values
                "raw_latency": raw_latency,          # For debugging 0-value issues
            }

            base_info.update({
                "delta_throughput": delta_throughput,
                "delta_latency": delta_latency,
                "mode": "delta",
                **self.reward_breakdown
            })

        elif self.mode == "instant":
            # Instant mode: reward based on current metric values
            raw_reward = (
                throughput  # Base throughput reward
                - self.latency_weight * latency  # Latency penalty
                - self.balance_penalty_weight * balance_penalty  # Legacy balance penalty
                - self.load_balance_penalty * load_balance_penalty  # Load balance penalty
                - latency_threshold_penalty  # Soft latency threshold penalty
            )

            # Apply adaptive scaling consistent with delta mode
            reward_scale = 1.0  # IMPROVED: Same optimized scale as delta mode for consistency
            reward = self._adaptive_reward_scaling(raw_reward, reward_scale)

            # Store breakdown for analysis with raw/scaled values
            self.reward_breakdown = {
                "base_throughput": throughput,
                "latency_penalty": -self.latency_weight * latency,
                "balance_penalty": -self.balance_penalty_weight * balance_penalty,
                "load_balance_penalty": -self.load_balance_penalty * load_balance_penalty,
                "latency_threshold_penalty": -latency_threshold_penalty,
                "raw_reward": raw_reward,  # Show original value before scaling
                "total_reward": reward,     # Show final scaled value
                "reward_ceiling_removed": True,  # Track that ceiling has been removed
                "uses_adaptive_scaling": True,  # Flag that we're using new scaling method
                "raw_throughput": raw_throughput,
                "raw_latency": raw_latency,
            }

            base_info.update({
                "mode": "instant",
                **self.reward_breakdown
            })

        else:  # hybrid mode - restructured reward with asymmetric penalties
            # Update EMA statistics for normalization
            self.update_ema_stats(throughput, latency)

            # Calculate structured reward components
            absolute_score = self.calculate_absolute_score(throughput, latency)
            delta_score = self.calculate_delta_score(throughput, latency)
            logistic_penalty = self.calculate_logistic_penalty(latency)

            # Phase 2: Add production asymmetric penalty patterns
            asymmetric_penalty = 0.0
            beta_exploration_bonus = 0.0
            temporal_trend_bonus = 0.0

            if self.use_asymmetric_penalties:
                asymmetric_penalty = self.calculate_asymmetric_allocation_penalty(
                    throughput, latency, current_time
                )

            if self.beta_exploration_enable:
                beta_exploration_bonus = self.calculate_beta_exploration_bonus(
                    throughput, latency
                )

            if self.temporal_tracking_enable:
                temporal_trend_bonus = self.calculate_temporal_performance_trend(
                    throughput, latency
                )

            # Enhanced load balance penalties
            load_balance_penalty = self.calculate_load_balance_penalty(replica_ids, get_replica_scheduler_fn)
            direct_imbalance_penalty = self.calculate_direct_load_imbalance_penalty(replica_ids, get_replica_scheduler_fn)
            extreme_imbalance_penalty = self.calculate_extreme_imbalance_penalty(replica_ids, get_replica_scheduler_fn)

            # Combine components with asymmetric penalties and temporal tracking
            raw_reward = (
                self.absolute_weight * absolute_score  # Primary: absolute performance
                + self.delta_weight * delta_score      # Secondary: improvement signal
                - self.load_balance_penalty * load_balance_penalty  # Legacy load balance penalty
                - 2.0 * direct_imbalance_penalty       # ENHANCED: Increased penalty weight to discourage hot-spotting
                - extreme_imbalance_penalty            # NEW: Severe penalty for extreme hot-spotting
                - logistic_penalty                     # Smooth latency penalty
                - asymmetric_penalty                   # Phase 2: Meta's asymmetric penalty patterns
                + beta_exploration_bonus               # Phase 2: Beta distribution exploration
                + temporal_trend_bonus                 # Phase 2: Temporal performance tracking
            )

            # Apply adaptive scaling to preserve gradient information
            # CRITICAL FIX: Remove clipping ceiling to allow exceeding baseline performance
            reward_scale = 1.2  # Further reduced scale for better reward variance
            reward = self._adaptive_reward_scaling(raw_reward, reward_scale)

            # Calculate deltas for monitoring
            delta_throughput = throughput - self.last_throughput
            delta_latency = latency - self.last_latency

            # Store detailed breakdown for analysis with Phase 2 components
            self.reward_breakdown = {
                "absolute_score": absolute_score,
                "delta_score": delta_score,
                "logistic_penalty": logistic_penalty,
                "load_balance_penalty": load_balance_penalty,
                "direct_imbalance_penalty": direct_imbalance_penalty,
                "extreme_imbalance_penalty": extreme_imbalance_penalty,
                "asymmetric_penalty": asymmetric_penalty,  # Phase 2
                "beta_exploration_bonus": beta_exploration_bonus,  # Phase 2
                "temporal_trend_bonus": temporal_trend_bonus,  # Phase 2
                "success_alpha": self.success_alpha,  # Phase 2: Beta distribution state
                "failure_beta": self.failure_beta,   # Phase 2: Beta distribution state
                "throughput_ema": self.throughput_ema,
                "latency_ema": self.latency_ema,
                "throughput_var": self.throughput_var,
                "latency_var": self.latency_var,
                "raw_reward": raw_reward,
                "total_reward": reward,
                "raw_throughput": raw_throughput,
                "raw_latency": raw_latency,
                "step_count": self.step_count,
                "reward_scaling": "linear",
                "reward_scale_factor": reward_scale,
            }

            base_info.update({
                "delta_throughput": delta_throughput,
                "delta_latency": delta_latency,
                "absolute_score": absolute_score,
                "delta_score": delta_score,
                "logistic_penalty": logistic_penalty,
                "direct_imbalance_penalty": direct_imbalance_penalty,
                "extreme_imbalance_penalty": extreme_imbalance_penalty,
                "asymmetric_penalty": asymmetric_penalty,  # Phase 2
                "beta_exploration_bonus": beta_exploration_bonus,  # Phase 2
                "temporal_trend_bonus": temporal_trend_bonus,  # Phase 2
                "mode": "hybrid",
                **self.reward_breakdown
            })

        # Update state for next delta calculation
        self.last_throughput = throughput
        self.last_latency = latency

        return reward, base_info

    def calculate_load_balance_penalty(
        self,
        replica_ids: List[int],
        get_replica_scheduler_fn,
    ) -> float:
        """
        Calculate enhanced load balance penalty based on queue length variance.

        This penalty is designed to discourage hot-spotting by penalizing
        high variance in queue lengths across replicas.

        Args:
            replica_ids: List of replica IDs
            get_replica_scheduler_fn: Function to get replica scheduler

        Returns:
            Load balance penalty (coefficient of variation of queue lengths)
        """
        queue_lengths: List[float] = []

        for replica_id in replica_ids:
            scheduler = get_replica_scheduler_fn(replica_id)
            # Direct access to request queue - fail if interface is wrong
            queue = scheduler._request_queue
            queue_length = len(queue)
            queue_lengths.append(float(queue_length))

        if len(queue_lengths) < 2:
            return 0.0

        # Calculate coefficient of variation (CV = std/mean)
        mean_queue = sum(queue_lengths) / len(queue_lengths)
        if mean_queue == 0:
            return 0.0  # Perfect balance when all queues are empty

        variance = sum((q - mean_queue) ** 2 for q in queue_lengths) / len(queue_lengths)
        std_dev = variance ** 0.5
        cv = std_dev / mean_queue

        # Scale the penalty - higher CV means worse balance
        # Add exponential scaling to make balance penalty more significant
        balance_penalty = float(cv * np.exp(cv))  # Exponential scaling for stronger penalty
        return balance_penalty

    def calculate_direct_load_imbalance_penalty(
        self,
        replica_ids: List[int],
        get_replica_scheduler_fn,
    ) -> float:
        """
        Calculate direct load imbalance penalty using standard deviation of loads.

        This is the primary load balancing signal recommended in the improvement plan.

        Args:
            replica_ids: List of replica IDs
            get_replica_scheduler_fn: Function to get replica scheduler

        Returns:
            Standard deviation of queue lengths across replicas
        """
        queue_lengths: List[float] = []

        for replica_id in replica_ids:
            scheduler = get_replica_scheduler_fn(replica_id)
            queue = scheduler._request_queue
            queue_length = len(queue)
            queue_lengths.append(float(queue_length))

        if len(queue_lengths) < 2:
            return 0.0

        # Calculate standard deviation directly
        mean_queue = sum(queue_lengths) / len(queue_lengths)
        variance = sum((q - mean_queue) ** 2 for q in queue_lengths) / len(queue_lengths)
        std_dev = variance ** 0.5

        return float(std_dev)

    def calculate_extreme_imbalance_penalty(
        self,
        replica_ids: List[int],
        get_replica_scheduler_fn,
    ) -> float:
        """
        Calculate additional penalty for extreme load imbalance situations.

        This helps discourage the agent from creating severe hot-spots that could
        lead to system instability or poor user experience.

        Args:
            replica_ids: List of replica IDs
            get_replica_scheduler_fn: Function to get replica scheduler

        Returns:
            Extreme imbalance penalty (exponential penalty for severe imbalance)
        """
        queue_lengths: List[float] = []

        for replica_id in replica_ids:
            scheduler = get_replica_scheduler_fn(replica_id)
            queue = scheduler._request_queue
            queue_length = len(queue)
            queue_lengths.append(float(queue_length))

        if len(queue_lengths) < 2:
            return 0.0

        # Check for extreme imbalance (one replica has >80% of total load)
        total_load = sum(queue_lengths)
        if total_load == 0:
            return 0.0

        max_load_fraction = max(queue_lengths) / total_load

        # Apply exponential penalty for severe imbalance
        if max_load_fraction > 0.8:
            # Severe penalty for hot-spotting
            excess = max_load_fraction - 0.8
            penalty = 5.0 * (excess ** 2)  # Quadratic penalty for extreme cases
            return float(penalty)

        return 0.0

    def calculate_absolute_score(self, throughput: float, latency: float) -> float:
        """
        Calculate absolute score: (throughput / target) - alpha * (latency / threshold)

        Args:
            throughput: Current throughput
            latency: Current latency

        Returns:
            Absolute score in normalized range
        """
        # Normalize throughput to [0, 1+] range (ensure positive target)
        safe_target = max(self.throughput_target, 0.1)
        throughput_score = throughput / safe_target

        # Normalize latency to [0, 1+] range (ensure positive threshold)
        safe_threshold = max(self.latency_threshold, 0.5)
        latency_score = latency / safe_threshold

        # Combine: reward high throughput, penalize high latency
        absolute_score = throughput_score - self.alpha * latency_score

        return float(absolute_score)

    def calculate_delta_score(self, throughput: float, latency: float) -> float:
        """
        Calculate normalized delta score using EMA baselines.

        Args:
            throughput: Current throughput
            latency: Current latency

        Returns:
            Normalized delta score
        """
        if self.step_count <= 1:
            return 0.0  # No delta on first step

        # Calculate normalized throughput change
        ema_throughput_safe = max(self.throughput_ema, 1e-6)
        throughput_delta_norm = (throughput - self.throughput_ema) / ema_throughput_safe

        # Calculate normalized latency change
        ema_latency_safe = max(self.latency_ema, 1e-6)
        latency_delta_norm = (latency - self.latency_ema) / ema_latency_safe

        # Combine: reward throughput increases, penalize latency increases
        delta_score = self.beta * throughput_delta_norm - self.gamma * latency_delta_norm

        return float(delta_score)

    def calculate_logistic_penalty(self, latency: float) -> float:
        """
        Calculate smooth logistic penalty for high latency.

        Args:
            latency: Current latency

        Returns:
            Smooth penalty using sigmoid function
        """
        # Sigmoid function: 1 / (1 + exp(-x))
        x = (latency - self.latency_threshold) / max(self.sigma, 1e-6)
        penalty = self.kappa / (1.0 + np.exp(-x))
        return float(penalty)

    def calculate_latency_threshold_penalty(self, current_latency: float) -> float:
        """
        Calculate soft latency threshold penalty.

        This implements a smooth penalty that activates when latency exceeds
        the threshold, using an exponential function to avoid sharp cliffs.

        Args:
            current_latency: Current average latency in seconds

        Returns:
            Latency threshold penalty (0 if below threshold, increasing above)
        """
        if current_latency <= self.latency_threshold:
            return 0.0

        # Soft penalty using exponential function
        excess_latency = current_latency - self.latency_threshold
        penalty = self.latency_penalty_scale * (1.0 - np.exp(-excess_latency))

        return float(penalty)

    def get_reward_breakdown(self) -> Dict[str, float]:
        """
        Get detailed breakdown of the last reward calculation.

        Returns:
            Dictionary with reward component details
        """
        return self.reward_breakdown.copy()

    def calculate_asymmetric_allocation_penalty(
        self, throughput: float, latency: float, current_time: float
    ) -> float:
        """
        Calculate Meta's production asymmetric penalty for allocation decisions.

        Implements 5:1 penalty ratio for under-provisioning vs over-provisioning
        based on performance indicators rather than explicit allocation data.

        Args:
            throughput: Current system throughput
            latency: Current system latency
            current_time: Current simulation time

        Returns:
            Asymmetric penalty based on system performance indicators
        """
        # Estimate allocation efficiency from latency vs throughput ratio
        efficiency_ratio = throughput / (latency + 1e-8)

        # Determine if system appears under or over-provisioned
        baseline_efficiency = self.throughput_target / max(self.latency_threshold, 0.5)
        relative_efficiency = efficiency_ratio / baseline_efficiency

        penalty = 0.0

        if relative_efficiency < 0.8:  # Under-provisioned (high latency, low throughput)
            # Severe penalty for SLA violations (Meta's approach)
            under_provision_severity = 0.8 - relative_efficiency
            penalty = self.false_positive_penalty * under_provision_severity
            self.failure_beta += 1  # Record failure for Beta distribution

        elif relative_efficiency > 1.5:  # Over-provisioned (low latency, high throughput but wasteful)
            # Mild penalty for resource waste
            over_provision_severity = relative_efficiency - 1.5
            penalty = self.over_provision_factor * over_provision_severity
            self.success_alpha += 0.5  # Partial success

        else:
            # Good allocation range
            self.success_alpha += 1  # Record success

        return float(penalty)

    def calculate_beta_exploration_bonus(
        self, throughput: float, latency: float
    ) -> float:
        """
        Calculate self-adaptive exploration bonus using Beta distribution.

        This implements the production pattern where exploration is adjusted
        based on success/failure history using Beta distribution sampling.

        Args:
            throughput: Current throughput
            latency: Current latency

        Returns:
            Exploration bonus weighted by Beta distribution coefficient
        """
        if not self.beta_exploration_enable:
            return 0.0

        # Performance score for exploration weighting
        normalized_latency = 1.0 / (latency + 1e-8)
        performance_score = normalized_latency * throughput

        # Sample exploration coefficient from Beta distribution
        exploration_coef = np.random.beta(self.success_alpha, self.failure_beta)

        # Weight exploration bonus by performance and Beta coefficient
        bonus = exploration_coef * performance_score * self.beta_shaping_weight

        return float(bonus)

    def calculate_temporal_performance_trend(
        self, throughput: float, latency: float
    ) -> float:
        """
        Calculate temporal performance trend bonus using gradient analysis.

        This tracks performance improvements over time and provides bonus
        for sustained positive trends (Google's production approach).

        Args:
            throughput: Current throughput
            latency: Current latency

        Returns:
            Temporal trend bonus (positive for improving performance)
        """
        if not self.temporal_tracking_enable:
            return 0.0

        # Calculate performance score
        normalized_latency = 1.0 / (latency + 1e-8)
        performance_score = normalized_latency * throughput

        # Add to history
        self.performance_history.append(performance_score)

        # Calculate trend if we have sufficient history
        if len(self.performance_history) < 3:
            return 0.0

        # Use numpy gradient to calculate performance trend
        performance_array = np.array(list(self.performance_history))
        trend_gradient = np.gradient(performance_array)

        # Recent trend (last few points)
        recent_trend = float(np.mean(trend_gradient[-3:]))

        # AWS GameServer scheduling efficiency pattern
        scheduling_efficiency = min(1.0, throughput / (self.throughput_target + 1e-8))
        self.scheduling_efficiency_history.append(scheduling_efficiency)

        # Combine trend with scheduling efficiency improvement
        efficiency_trend = 0.0
        if len(self.scheduling_efficiency_history) >= 2:
            efficiency_diff = (self.scheduling_efficiency_history[-1] -
                             self.scheduling_efficiency_history[-2])
            efficiency_trend = float(efficiency_diff)

        # Weight temporal components (0.1 for performance trend, 0.2 for efficiency)
        temporal_bonus = 0.1 * recent_trend + 0.2 * efficiency_trend

        return temporal_bonus

    def reset_state(self) -> None:
        """Reset internal state tracking including Phase 2 components."""
        self.last_throughput = 0.0
        self.last_latency = 0.0
        self.throughput_ema = 0.0
        self.throughput_var = 0.0
        self.latency_ema = 0.0
        self.latency_var = 0.0
        self.reward_breakdown = {}
        self.step_count = 0

        # Reset Phase 2 components
        self.success_alpha = 1.0
        self.failure_beta = 1.0
        self.performance_history.clear()
        self.scheduling_efficiency_history.clear()

    def _adaptive_reward_scaling(self, raw_reward: float, scale_factor: float = 1.5) -> float:
        """
        Adaptive reward scaling that removes hard clipping ceiling.

        CRITICAL: This replaces the hard clipping that was preventing the agent
        from learning to outperform Round Robin and Random baselines.

        Args:
            raw_reward: Unscaled reward value
            scale_factor: Base scaling factor

        Returns:
            Scaled reward without artificial ceiling
        """
        # Apply base scaling
        scaled_reward = raw_reward / scale_factor

        # For moderate values, use linear scaling (no ceiling)
        if abs(scaled_reward) <= 4.0:
            return float(scaled_reward)

        # For extreme values, use soft compression instead of hard clipping
        # This preserves gradient information while handling outliers
        sign = 1.0 if scaled_reward >= 0 else -1.0
        abs_scaled = abs(scaled_reward)

        # Logarithmic compression for values > 4.0 (no hard ceiling)
        compressed = 4.0 + np.log(1.0 + abs_scaled - 4.0)

        return float(sign * compressed)