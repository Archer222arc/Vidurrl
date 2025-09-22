"""
Dynamic temperature controller for PPO action selection.

This module provides adaptive temperature scaling based on system
load and QPS pressure to balance exploration and exploitation.
"""

from typing import Optional


class TemperatureController:
    """
    Dynamic temperature controller for policy action selection.

    Adjusts exploration temperature based on system pressure indicators
    like QPS load and latency stress to encourage appropriate exploration.
    """

    def __init__(
        self,
        base_temperature: float = 1.0,
        min_temperature: float = 0.5,
        max_temperature: float = 2.0,
        qps_sensitivity: float = 0.1,
        latency_sensitivity: float = 0.2,
        pressure_ema_alpha: float = 0.1,
        enable_pulse: bool = True,
        pulse_interval: int = 8,  # Much more frequent pulses for continuous exploration
        pulse_magnitude: float = 2.0,  # Stronger pulse boost
        stagnation_threshold: float = 0.005,  # More sensitive stagnation detection
        stagnation_memory: int = 5,  # Shorter memory for faster response
    ):
        """
        Initialize dynamic temperature controller.

        Args:
            base_temperature: Base temperature value
            min_temperature: Minimum allowed temperature
            max_temperature: Maximum allowed temperature
            qps_sensitivity: Sensitivity to QPS pressure changes
            latency_sensitivity: Sensitivity to latency pressure changes
            pressure_ema_alpha: EMA smoothing factor for pressure tracking
            enable_pulse: Enable periodic temperature pulses for exploration
            pulse_interval: Steps between temperature pulses
            pulse_magnitude: Multiplier for pulse temperature boost
            stagnation_threshold: Threshold for detecting stagnant metrics
            stagnation_memory: Number of steps to track for stagnation detection
        """
        self.base_temperature = base_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.qps_sensitivity = qps_sensitivity
        self.latency_sensitivity = latency_sensitivity
        self.pressure_ema_alpha = pressure_ema_alpha

        # Pulse and stagnation detection parameters
        self.enable_pulse = enable_pulse
        self.pulse_interval = pulse_interval
        self.pulse_magnitude = pulse_magnitude
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_memory = stagnation_memory

        # Pressure tracking
        self.qps_pressure_ema = 0.0
        self.latency_pressure_ema = 0.0
        self.current_temperature = base_temperature

        # Pulse and stagnation tracking
        self.step_count = 0
        self.last_pulse_step = 0
        self.is_pulsing = False
        self.pulse_remaining = 0
        self.delta_history = []  # Track recent deltas for stagnation detection

    def compute_temperature(
        self,
        current_qps: float,
        target_qps: float,
        current_latency: float,
        target_latency: float,
        system_load_balance: Optional[float] = None,
        delta_throughput: Optional[float] = None,
        delta_latency: Optional[float] = None,
    ) -> float:
        """
        Compute dynamic temperature based on system pressure indicators.

        Args:
            current_qps: Current QPS
            target_qps: Target/expected QPS
            current_latency: Current average latency
            target_latency: Target latency threshold
            system_load_balance: Load balance score (0-1, higher is better)
            delta_throughput: Change in throughput for stagnation detection
            delta_latency: Change in latency for stagnation detection

        Returns:
            Computed temperature value
        """
        # Calculate QPS pressure (normalized deviation from target)
        qps_pressure = 0.0
        if target_qps > 0:
            qps_pressure = (current_qps - target_qps) / target_qps

        # Calculate latency pressure (normalized deviation from target)
        latency_pressure = 0.0
        if target_latency > 0:
            latency_pressure = max(0.0, (current_latency - target_latency) / target_latency)

        # Update EMA tracking
        self.qps_pressure_ema = (
            (1 - self.pressure_ema_alpha) * self.qps_pressure_ema +
            self.pressure_ema_alpha * qps_pressure
        )
        self.latency_pressure_ema = (
            (1 - self.pressure_ema_alpha) * self.latency_pressure_ema +
            self.pressure_ema_alpha * latency_pressure
        )

        # Compute temperature adjustment
        temperature_delta = 0.0

        # High QPS pressure -> reduce exploration (lower temperature)
        temperature_delta -= self.qps_sensitivity * self.qps_pressure_ema

        # High latency pressure -> reduce exploration (lower temperature)
        temperature_delta -= self.latency_sensitivity * self.latency_pressure_ema

        # Poor load balance -> increase exploration (higher temperature)
        if system_load_balance is not None:
            balance_pressure = 1.0 - system_load_balance  # Convert to pressure metric
            temperature_delta += 0.1 * balance_pressure

        # Track stagnation for pulse triggering
        if delta_throughput is not None and delta_latency is not None:
            # Calculate stagnation metric (how much things are changing)
            delta_magnitude = abs(delta_throughput) + abs(delta_latency)
            self.delta_history.append(delta_magnitude)

            # Keep only recent history
            if len(self.delta_history) > self.stagnation_memory:
                self.delta_history.pop(0)

        # Pulse logic for breaking out of stagnation
        self.step_count += 1
        pulse_boost = 0.0

        if self.enable_pulse:
            # Check if we should trigger a pulse
            should_pulse = False

            # Periodic pulse
            if self.step_count - self.last_pulse_step >= self.pulse_interval:
                should_pulse = True

            # Stagnation-triggered pulse
            if (len(self.delta_history) >= self.stagnation_memory and
                max(self.delta_history) < self.stagnation_threshold):
                should_pulse = True

            # Start pulse
            if should_pulse and not self.is_pulsing:
                self.is_pulsing = True
                self.pulse_remaining = 3  # Pulse lasts for 3 steps
                self.last_pulse_step = self.step_count

            # Apply pulse boost
            if self.is_pulsing:
                pulse_boost = self.pulse_magnitude * self.base_temperature
                self.pulse_remaining -= 1
                if self.pulse_remaining <= 0:
                    self.is_pulsing = False

        # Apply adjustment to base temperature with pulse boost
        self.current_temperature = self.base_temperature + temperature_delta + pulse_boost

        # Clamp to bounds (allow pulse to exceed max_temperature for exploration)
        if self.is_pulsing:
            # During pulse, allow higher temperatures
            pulse_max = self.max_temperature * 2.0
            self.current_temperature = max(self.min_temperature,
                                         min(pulse_max, self.current_temperature))
        else:
            # Normal bounds
            self.current_temperature = max(
                self.min_temperature,
                min(self.max_temperature, self.current_temperature)
            )

        return self.current_temperature

    def get_current_temperature(self) -> float:
        """Get the current temperature value."""
        return self.current_temperature

    def get_pressure_metrics(self) -> dict:
        """
        Get current pressure metrics for monitoring.

        Returns:
            Dictionary with pressure tracking metrics
        """
        return {
            "qps_pressure_ema": self.qps_pressure_ema,
            "latency_pressure_ema": self.latency_pressure_ema,
            "current_temperature": self.current_temperature,
            "base_temperature": self.base_temperature,
            "is_pulsing": self.is_pulsing,
            "pulse_remaining": self.pulse_remaining,
            "step_count": self.step_count,
            "stagnation_level": max(self.delta_history) if self.delta_history else 0.0,
        }

    def reset(self) -> None:
        """Reset pressure tracking and temperature to base value."""
        self.qps_pressure_ema = 0.0
        self.latency_pressure_ema = 0.0
        self.current_temperature = self.base_temperature
        # Reset pulse and stagnation tracking
        self.step_count = 0
        self.last_pulse_step = 0
        self.is_pulsing = False
        self.pulse_remaining = 0
        self.delta_history = []