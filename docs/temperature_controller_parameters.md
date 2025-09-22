# Temperature Controller Guide

This guide explains the dynamic temperature controller used by the PPO scheduler. It covers both the meaning of each parameter and how the controller operates inside the training pipeline.

## 1. Where the Controller Fits in Training
- **Configuration stage**: Values in `configs/ppo_warmstart.json:103-110` are passed through `src/config/training_config.py:67-94`, producing CLI flags consumed by `vidur.main`.
- **Scheduler creation**: `vidur/scheduler/global_scheduler/ppo_scheduler_modular.py:158-199` instantiates `TemperatureController` when `enable_dynamic_temperature` is true, injecting the configured parameters.
- **During rollouts**: Every scheduling step, the PPO actor samples actions. The controller adjusts the softmax temperature before sampling, balancing exploration vs exploitation based on live metrics.
- **Feedback signals**: The scheduler supplies current QPS, latency, load-balance scores, and recent throughput/latency deltas (`ppo_scheduler_modular.py:616-689`). These feed into `compute_temperature()` to determine the next temperature.
- **Logging/monitoring**: `get_pressure_metrics()` outputs temperature, EMA pressures, pulse status, etc., which are written to TensorBoard/CSV via the scheduler’s metric hooks (`ppo_scheduler_modular.py:762-775`).
- **Checkpoint/Resume**: The controller’s state is part of the scheduler module, so when checkpoints are restored (`train_ppo_warmstart_optimized.sh` resume flow), temperature resumes from its saved value, ensuring continuity across chunked training.

## 2. Core Temperature Bounds
- `base_temperature`: Default temperature when pressure is neutral. Higher values encourage broader sampling; lower values drive greedier selections.
- `min_temperature`: Hard floor preserving minimal exploration even under high pressure.
- `max_temperature`: Ceiling during normal operation, preventing excessive randomness from low-pressure conditions or load-balance incentives.

## 3. Pressure Sensitivities
- `qps_sensitivity`: How strongly the controller reacts to request-rate pressure (`current_qps` vs `target_qps`). Larger values lower the temperature quickly to stabilise throughput when overloaded.
- `latency_sensitivity`: Response strength to latency overruns (`current_latency` vs `target_latency`). Higher values prioritise latency recovery by suppressing exploration.
- `pressure_ema_alpha`: Exponential moving-average weight for the pressure trackers. Larger `alpha` makes the controller nimble; smaller `alpha` smooths noise but delays reactions.

## 4. Pulse-Based Exploration
- `enable_pulse`: Toggles periodic/stagnation-triggered pulses that temporarily increase exploration.
- `pulse_interval`: Controller steps between automatically scheduled pulses. Short intervals insert frequent exploratory bursts; long intervals reserve pulses for prolonged calm periods.
- `pulse_magnitude`: Multiplier applied to `base_temperature` during a pulse. While pulsing, the controller can exceed `max_temperature` (up to twice) to explore aggressively.

## 5. Stagnation Detection
- `stagnation_threshold`: Maximum combined change in throughput/latency deltas considered “stagnant.” Staying below this signals a plateau and triggers a pulse.
- `stagnation_memory`: Number of recent delta measurements tracked. Smaller windows react faster; larger windows require longer stagnation before pulsing.

## 6. Runtime Metrics
`get_pressure_metrics()` exposes diagnostic values that the scheduler forwards to logs/TensorBoard:
- `qps_pressure_ema` / `latency_pressure_ema`: Smoothed pressure indicators currently influencing temperature adjustments.
- `current_temperature` and `base_temperature`: Current sampling temperature and its neutral baseline.
- `is_pulsing`, `pulse_remaining`, `step_count`: Pulse state and controller progress.
- `stagnation_level`: Maximum delta magnitude in the recent history window, useful for understanding pulse triggers.

## 7. Practical Tuning Tips
- Start with modest sensitivities (`qps_sensitivity`, `latency_sensitivity` around 0.1–0.2) and monitor how quickly temperature collapses under load. Excessively high values can freeze exploration.
- If the policy gets stuck in local minima despite stable latency, enable pulses or shorten `pulse_interval` to inject exploration bursts.
- For workloads with noisy metrics, reduce `pressure_ema_alpha` or raise `stagnation_threshold` so the controller doesn’t overreact to transient spikes.
- Inspect TensorBoard traces of `current_temperature`, pressures, and latency to verify the controller is pushing exploration during calm periods and suppressing it under stress.

With these mechanisms, the dynamic temperature controller helps PPO adjust exploration intensity in response to real-time system conditions, improving stability and convergence compared to fixed-temperature sampling.
