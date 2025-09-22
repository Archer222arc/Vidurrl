# Training Monitoring Guide

This guide lists the key telemetry exposed by the PPO training stack and where to find it. Use it together with the reward, temperature, and KL guides to interpret the logs.

## 1. Logging Outputs
- **Console / script log**: `OUTPUT_DIR/ppo_training.log`. Includes rollout buffers, PPO losses, KL/entropy, and reward breakdown per update (`vidur/scheduler/global_scheduler/ppo_scheduler_modular.py:1000-1060`).
- **TensorBoard**: Enabled via `monitoring` config (`configs/ppo_warmstart.json:154-171`). The CLI flags point to `<OUTPUT_DIR>/tensorboard`. Launch with:
  ```bash
  tensorboard --logdir <OUTPUT_DIR>/tensorboard --port 6006
  ```
- **CSV exports**: Stored under `<OUTPUT_DIR>/metrics/` if `metrics_export_enabled` is true. Each file contains per-step statistics such as `approx_kl`, `clipfrac`, `entropy`, `temperature`.

## 2. Must-Watch Scalars
| Metric | Source | Healthy Range | Notes |
| --- | --- | --- | --- |
| `approx_kl` | PPOTrainer (`stats['approx_kl']`) | < `target_kl` (config 0.02) | Rising above target → increase `kl_coef` or reduce PPO epochs. |
| `entropy` | PPOTrainer | stays above `entropy_min` (0.5) | Sustained drop indicates over-constraint; check temperature pulses. |
| `clipfrac` | PPOTrainer | 0.1 – 0.3 | Values >0.3 mean many updates are clipped; consider lowering LR or `clip_ratio`. |
| `pg_grad_norm` | PPOTrainer | < `max_grad_norm` | Sudden spikes trigger adaptive batch size / LR adjustments. |
| `current_temperature` | Temperature controller | 0.5 – 2.0 (non-pulse) | Pulses may temporarily exceed max; ensure pulses coincide with stagnation. |
| `balance_penalty` | Reward breakdown | Near zero once balanced | Persistently high → raise load-balance penalties. |

## 3. Action Distribution Insight
- TensorBoard histogram `Actions/replica_*` (if enabled) or CSV column `action_replica_n` shows per-replica selection ratios.
- Goal: distributions converge near uniform unless workload dictates otherwise. Monitor early chunk behavior to ensure warmstart succeeded.

## 4. Temperature & Pressure
- `docs/temperature_controller_parameters.md` covers parameters.
- Monitoring exports include: `temp_current_temperature`, `temp_qps_pressure_ema`, `temp_latency_pressure_ema`, `temp_is_pulsing`.
- Use these to correlate latency spikes with temperature suppression.

## 5. Reward Breakdown
CSV columns: `reward`, `reward_mean`, `absolute_score`, `delta_score`, `balance_penalty`. If the total reward is negative while absolute score stays positive, the delta component may be penalizing regressions too heavily—tune `beta/gamma`.

## 6. Resume Metadata
When chunk mode or resume runs, `training_progress.json` inside the run directory tracks `completed_chunks`, `requests_done`, `latest_checkpoint`. Inspect it to confirm progress after interruptions.

## 7. Troubleshooting Signals
- **Entropy collapse + low temperature** → Temperature controller overly suppressive; lower `qps_sensitivity` or raise `min_temperature`.
- **High `approx_kl` but low reward** → Policy diverging; tighten KL or revisit reward weights.
- **Clipfrac near zero + slow learning** → Increase learning rate or decrease `kl_coef` to allow larger updates.
- **Reward spikes without throughput improvement** → Inspect `reward_breakdown` to ensure latency penalties aren’t being masked.

## 8. Related Docs
- `docs/reward_config_guide.md`
- `docs/temperature_controller_parameters.md`
- `docs/kl_regularization_guide.md`
- `docs/statistics_stabilization_guide.md`
