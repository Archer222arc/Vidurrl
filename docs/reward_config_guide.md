# Reward Configuration Guide

This document explains how the PPO scheduler constructs rewards, where each parameter lives, and how to tune them. Use it alongside `configs/ppo_warmstart.json:70-83` and `src/core/algorithms/rewards/reward_calculator.py:16-213`.

## 1. Where Reward Logic Runs
- **Config → CLI bridge**: `src/config/training_config.py:58-75` converts the `reward_config` block into CLI flags for `vidur.main`.
- **Scheduler wiring**: `vidur/scheduler/global_scheduler/ppo_scheduler_modular.py:96-146` reads those values and instantiates `RewardCalculator`.
- **During rollouts**: Each time the scheduler executes an action, it calls `RewardCalculator.compute_reward(...)` to score the transition. The detailed breakdown is stored in `reward_info` and forwarded to logs/metrics.

## 2. Reward Modes
`RewardCalculator` supports three modes (`mode` in config):
- `instant`: reward depends on the current latency/throughput snapshot.
- `delta`: reward focuses on improvement between consecutive steps. Uses EMA tracking to compute deltas.
- `hybrid`: combines absolute and delta components (default in the optimized config).

Switch modes by setting `reward_mode` in `vidur/config/config.py:588-647` or via `--p_p_o_global_scheduler_modular_config_reward_mode`.

## 3. Legacy Terms (Compatibility Layer)
These parameters exist for backward compatibility with older rewards:
- `latency_weight`: Multiplies the soft-threshold latency penalty when the latency term is enabled.
- `balance_penalty_weight`: Global multiplier for load-balance penalties.
- `latency_threshold`: Soft threshold (seconds). Above it, penalties grow linearly with `latency_penalty_scale`.
- `latency_penalty_scale`: Slope of the latency penalty when latency exceeds the threshold.
- `load_balance_penalty`: Weight applied to replica-imbalance scores generated inside the scheduler.

In the new reward design the absolute/delta structure dominates, but these terms still apply if you enable the legacy penalty path.

## 4. Restructured Reward Components
The optimized reward uses a two-part mixture:

### 4.1 Absolute Score (`absolute_weight`)
`absolute_score = alpha * throughput_term - (1 - alpha) * latency_term - logistic_penalty`
- `throughput_target`: Normalizes throughput into [0, 1]. When throughput equals the target, the sigmoid term outputs ~0.5.
- `alpha`: Balances throughput vs latency inside the absolute score. Higher `alpha` favors throughput.
- `kappa`: Weight of the logistic latency penalty (smooth alternative to hard thresholds).
- `sigma`: Controls the slope of the logistic penalty. Larger values produce smoother penalties.

### 4.2 Delta Score (`delta_weight`)
`delta_score = beta * normalized_throughput_delta - gamma * normalized_latency_delta`
- `beta`: Scales change in throughput between steps (positive is good).
- `gamma`: Scales change in latency between steps (negative drift is penalized).
- EMAs controlled by `ema_alpha` provide the normalization baseline and variance estimates.

The total reward is `absolute_weight * absolute_score + delta_weight * delta_score` plus any legacy penalties.

## 5. Load-Balance Penalties
Two signals penalize skewed replica usage:
- **Legacy penalty**: `balance_penalty_weight * load_balance_penalty`. Uses variance of per-replica queue lengths.
- **Enhanced penalty**: Inside the scheduler we log `reward_info["balance_penalty"]`, which is already multiplied by `load_balance_penalty`. Adjusting `load_balance_penalty` and `balance_penalty_weight` increases the emphasis on spreading load.

See `docs/load_balance_optimization.md` for case studies.

## 6. Monitoring Breakdown
`RewardCalculator` records the intermediate components in `self.reward_breakdown`. During training, `ppo_scheduler_modular.py:1000-1045` pushes them into the metrics exporter.

Useful fields:
- `absolute_score`
- `delta_score`
- `throughput_term`
- `latency_term`
- `logistic_penalty`
- `balance_penalty`

Track these in CSV exports (`runs/ppo_training/exports/...`) or TensorBoard to verify the reward behaves as expected.

## 7. Tuning Checklist
1. **Reward always negative** → Lower `throughput_target`, raise `alpha`, or reduce `kappa`.
2. **Latency dominates updates** → Increase `alpha` or reduce `gamma`.
3. **Load imbalance persists** → Raise `load_balance_penalty` and `balance_penalty_weight`.
4. **Reward oscillates wildly** → Reduce `beta`/`gamma` or lower `ema_alpha` to smooth deltas.
5. **Strategy ignores improvements** → Increase `delta_weight` and ensure `ema_alpha` is not too small.

Adjust one axis at a time and observe `reward_info` along with queue balance and latency histograms.

## 8. Related Files
- `vidur/scheduler/global_scheduler/ppo_scheduler_modular.py:586-669` (reward logging)
- `docs/ppo_optimization_2025_09_18.md` (historical tuning notes)
- `docs/load_balance_optimization.md` (penalty-focused adjustments)
