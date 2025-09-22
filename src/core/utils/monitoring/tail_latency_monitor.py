"""
Tail Latency Monitoring System for PPO Training.

This module implements comprehensive tail latency tracking with P90/P95/P99
percentiles monitoring and alert mechanisms as recommended in the PDF.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time


class TailLatencyMonitor:
    """
    Monitors tail latency metrics with percentile tracking and alerting.

    Tracks latency distributions with configurable percentiles and provides
    alert mechanisms for tail latency degradation detection.
    """

    def __init__(
        self,
        percentiles: List[float] = None,
        window_size: int = 1000,
        alert_threshold_p99: float = 5.0,
        enable_alerts: bool = True,
        enable_tracking: bool = True
    ):
        """
        Initialize tail latency monitor.

        Args:
            percentiles: List of percentiles to track (default: [90, 95, 99])
            window_size: Size of sliding window for latency samples
            alert_threshold_p99: P99 latency threshold for alerts (seconds)
            enable_alerts: Whether to generate alerts for threshold violations
            enable_tracking: Whether tracking is enabled
        """
        self.percentiles = percentiles or [90, 95, 99]
        self.window_size = window_size
        self.alert_threshold_p99 = alert_threshold_p99
        self.enable_alerts = enable_alerts
        self.enable_tracking = enable_tracking

        # Sliding window for latency samples
        self.latency_samples: deque = deque(maxlen=window_size)

        # Current percentile values
        self.current_percentiles: Dict[float, float] = {}

        # Alert tracking
        self.alert_count = 0
        self.last_alert_time = 0.0
        self.alert_cooldown = 30.0  # seconds

        # Statistics tracking
        self.total_samples = 0
        self.violation_count = 0

    def update(self, latency: float) -> bool:
        """
        Update monitor with new latency sample.

        Args:
            latency: Latency value in seconds

        Returns:
            bool: True if alert was triggered, False otherwise
        """
        if not self.enable_tracking:
            return False

        # Add sample to sliding window
        self.latency_samples.append(latency)
        self.total_samples += 1

        # Calculate current percentiles
        self._calculate_percentiles()

        # Check for alerts
        alert_triggered = self._check_alerts(latency)

        return alert_triggered

    def _calculate_percentiles(self) -> None:
        """Calculate current percentile values from sliding window."""
        if len(self.latency_samples) == 0:
            return

        samples_array = np.array(self.latency_samples)

        for percentile in self.percentiles:
            self.current_percentiles[percentile] = float(
                np.percentile(samples_array, percentile)
            )

    def _check_alerts(self, current_latency: float) -> bool:
        """
        Check if current latency violates alert thresholds.

        Args:
            current_latency: Current latency sample

        Returns:
            bool: True if alert was triggered
        """
        if not self.enable_alerts:
            return False

        # Check P99 threshold violation
        p99_latency = self.current_percentiles.get(99, 0.0)
        current_time = time.time()

        if p99_latency > self.alert_threshold_p99:
            self.violation_count += 1

            # Check alert cooldown to avoid spam
            if current_time - self.last_alert_time > self.alert_cooldown:
                self.alert_count += 1
                self.last_alert_time = current_time
                return True

        return False

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current tail latency metrics.

        Returns:
            Dictionary with percentile values and statistics
        """
        metrics = {}

        # Add percentile metrics
        for percentile in self.percentiles:
            value = self.current_percentiles.get(percentile, 0.0)
            metrics[f"latency_p{int(percentile)}"] = value

        # Add monitoring statistics
        metrics.update({
            "tail_latency_samples": len(self.latency_samples),
            "tail_latency_alerts": self.alert_count,
            "tail_latency_violations": self.violation_count,
            "tail_latency_violation_rate": (
                self.violation_count / max(1, self.total_samples)
            ),
        })

        return metrics

    def get_current_percentiles(self) -> Dict[float, float]:
        """
        Get current percentile values.

        Returns:
            Dictionary mapping percentile to current value
        """
        return self.current_percentiles.copy()

    def reset(self) -> None:
        """Reset all tracking state."""
        self.latency_samples.clear()
        self.current_percentiles.clear()
        self.alert_count = 0
        self.violation_count = 0
        self.total_samples = 0
        self.last_alert_time = 0.0

    def get_alert_summary(self) -> Dict[str, any]:
        """
        Get summary of alert activity.

        Returns:
            Dictionary with alert statistics and status
        """
        return {
            "total_alerts": self.alert_count,
            "total_violations": self.violation_count,
            "violation_rate": self.violation_count / max(1, self.total_samples),
            "p99_threshold": self.alert_threshold_p99,
            "current_p99": self.current_percentiles.get(99, 0.0),
            "threshold_exceeded": (
                self.current_percentiles.get(99, 0.0) > self.alert_threshold_p99
            ),
            "samples_tracked": len(self.latency_samples),
            "window_size": self.window_size,
        }

    def configure_alerts(
        self,
        threshold_p99: Optional[float] = None,
        cooldown: Optional[float] = None,
        enable: Optional[bool] = None
    ) -> None:
        """
        Configure alert parameters.

        Args:
            threshold_p99: New P99 alert threshold
            cooldown: New alert cooldown period
            enable: Enable/disable alerts
        """
        if threshold_p99 is not None:
            self.alert_threshold_p99 = threshold_p99

        if cooldown is not None:
            self.alert_cooldown = cooldown

        if enable is not None:
            self.enable_alerts = enable

    def is_healthy(self) -> Tuple[bool, str]:
        """
        Check if current tail latency is healthy.

        Returns:
            Tuple of (is_healthy, status_message)
        """
        if not self.enable_tracking:
            return True, "Monitoring disabled"

        if len(self.latency_samples) < 10:
            return True, "Insufficient samples"

        p99 = self.current_percentiles.get(99, 0.0)
        if p99 > self.alert_threshold_p99:
            return False, f"P99 latency {p99:.3f}s exceeds threshold {self.alert_threshold_p99}s"

        return True, "Healthy"


class TailLatencyAggregator:
    """
    Aggregates tail latency metrics across multiple replicas or time periods.
    """

    def __init__(self):
        """Initialize aggregator."""
        self.replica_monitors: Dict[int, TailLatencyMonitor] = {}
        self.global_monitor = TailLatencyMonitor()

    def add_replica_monitor(self, replica_id: int, monitor: TailLatencyMonitor) -> None:
        """Add monitor for specific replica."""
        self.replica_monitors[replica_id] = monitor

    def update_global(self, latency: float) -> None:
        """Update global latency tracking."""
        self.global_monitor.update(latency)

    def get_aggregated_metrics(self) -> Dict[str, float]:
        """
        Get aggregated metrics across all replicas.

        Returns:
            Dictionary with global and per-replica metrics
        """
        metrics = {}

        # Global metrics
        global_metrics = self.global_monitor.get_metrics()
        for key, value in global_metrics.items():
            metrics[f"global_{key}"] = value

        # Per-replica metrics
        for replica_id, monitor in self.replica_monitors.items():
            replica_metrics = monitor.get_metrics()
            for key, value in replica_metrics.items():
                metrics[f"replica_{replica_id}_{key}"] = value

        # Aggregated statistics
        all_p99_values = []
        all_p95_values = []
        all_p90_values = []

        for monitor in self.replica_monitors.values():
            percentiles = monitor.get_current_percentiles()
            if 99 in percentiles:
                all_p99_values.append(percentiles[99])
            if 95 in percentiles:
                all_p95_values.append(percentiles[95])
            if 90 in percentiles:
                all_p90_values.append(percentiles[90])

        if all_p99_values:
            metrics["aggregated_p99_max"] = max(all_p99_values)
            metrics["aggregated_p99_mean"] = np.mean(all_p99_values)
            metrics["aggregated_p99_std"] = np.std(all_p99_values)

        if all_p95_values:
            metrics["aggregated_p95_max"] = max(all_p95_values)
            metrics["aggregated_p95_mean"] = np.mean(all_p95_values)

        if all_p90_values:
            metrics["aggregated_p90_max"] = max(all_p90_values)
            metrics["aggregated_p90_mean"] = np.mean(all_p90_values)

        return metrics