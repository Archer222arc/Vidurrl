# Statistics Stabilization Guide

Statistics stabilization is a pre-training phase that collects reliable normalization statistics before PPO optimization begins. This updated guide reflects the optimized warmstart workflow.

## 1. Purpose
- Seed state/reward normalizers so PPO inputs have consistent scale.
- Reduce early-step reward variance.
- Align metrics when loading external or behavior-cloned checkpoints.

## 2. Configuration
`configs/ppo_warmstart.json:93-102` controls the feature set:
```json
"statistics_stabilization": {
  "enable_statistics_stabilization": true,
  "stabilization_steps": 200,
  "stabilization_policy": "random",
  "collect_baseline_stats": true,
  "freeze_normalizers_during_stabilization": false,
  "enable_stabilization_logging": true,
  "stabilization_action_distribution": "uniform"
}
```
Key points:
- Increase `stabilization_steps` when the workload has high variance.
- Set `freeze_normalizers_during_stabilization=true` to lock statistics when experimenting with non-random policies.

## 3. Execution Flow
1. `train_ppo_warmstart_optimized.sh` launches PPO; the scheduler checks `_enable_statistics_stabilization` (`vidur/scheduler/global_scheduler/ppo_scheduler_modular.py:1008-1072`).
2. During the stabilization loop, the scheduler:
   - Builds states with the final `StateBuilder`.
   - Applies random/uniform actions (no PPO updates).
   - Feeds metrics into `RunningNormalizer` and reward EMAs.
3. Once `stabilization_steps` complete, PPO updates are unlocked and training proceeds with calibrated statistics.

## 4. Using the Script
- Default: stabilization runs automatically when enabled.
- Disable from CLI for ablation: `--no-p_p_o_global_scheduler_modular_config_enable_statistics_stabilization`.
- Adjust steps: `--p_p_o_global_scheduler_modular_config_statistics_stabilization_steps 300`.
- Dedicated tests:
  - `python scripts/quick_stabilization_test.py` (smoke test).
  - `bash scripts/test_statistics_stabilization.sh` (A/B comparison).

## 5. Monitoring
- Look for `[STATS_STABILIZATION]` messages in `ppo_training.log`.
- Inspect TensorBoard / CSV exports for `normalizer_mean` and `normalizer_std` metrics (stable after the phase).
- Monitor reward variance: a sharp drop after stabilization indicates success.

## 6. Interaction with Warm Start
- Stabilization runs **after** warmstart BC/external checkpoint loading but **before** PPO updates.
- Recommended when loading external models with different training distributions; otherwise, the first PPO steps might explode due to mismatched statistics.

## 7. Troubleshooting
- **Phase skipped unexpectedly** → Ensure CLI flags do not disable it and that `stabilization_steps` > 0.
- **Latencies spike during phase** → Increase `stabilization_action_distribution` diversity or use a heuristic policy instead of random actions.
- **Normalizers still drift early** → Extend `stabilization_steps` or enable `collect_baseline_stats` if disabled.

## 8. Related Docs
- `docs/ppo_warm_start_guide.md`
- `docs/reward_config_guide.md`
- `docs/temperature_controller_parameters.md`
