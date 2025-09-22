# Advanced Optimization Features

This note explains the auxiliary optimization features enabled in `configs/ppo_warmstart.json:127-145` and how they map into the training runtime.

## 1. Learning-Rate Schedule
- **Config**: `advanced_optimization.learning_rate_schedule`
  - `type`: `cosine_annealing_with_warmup`
  - `warmup_steps`: Number of steps spent linearly ramping the LR from zero to `lr`.
  - `min_lr_ratio`: Lower bound fraction applied at the end of each cosine cycle.
  - `restart_period`: Steps between cosine restarts.
- **Runtime**: `vidur/scheduler/global_scheduler/ppo_scheduler_modular.py:846-910` plugs the schedule into the PPO optimizer. The scheduler updates LR each PPO mini-batch.
- **Monitoring**: Check TensorBoard scalar `Training/lr` exported via `PPOTrainer.update()`.

## 2. Adaptive Batch Size
- **Config**: `advanced_optimization.adaptive_batch_size`
  - `enable`: Toggle.
  - `min_batch_size` / `max_batch_size`: Bounds on the PPO mini-batch.
  - `adaptation_rate`: How fast the batch size responds to observed gradient norms.
- **Runtime**: The scheduler inspects `pg_grad_norm` from the previous update (see `ppo_scheduler_modular.py:1020-1050`) and adjusts the next update’s `minibatch_size`. Higher gradients shrink the batch, low gradients enlarge it.
- **Guidance**: Keep the range narrow (e.g., 32-128) to avoid pathological memory usage jumps.

## 3. Gradient Optimization Helpers
- **Gradient checkpointing**: Enabled via `gradient_optimization.use_gradient_checkpointing`. Activates PyTorch checkpointing inside the actor-critic forward pass to save memory at the cost of extra compute.
- **Accumulation steps**: `gradient_optimization.accumulation_steps` controls how many mini-batches accumulate before applying an optimizer step—useful when per-device batch size is limited.
- **Global-norm clipping**: `gradient_optimization.clip_by_global_norm` ensures the per-update gradient norm respects `max_grad_norm` even when accumulation is enabled.
- **Runtime Path**: These features are wired in `vidur/scheduler/global_scheduler/ppo_scheduler_modular.py:845-980` when building the PPO optimizer and in `src/core/algorithms/ppo_trainer.py:214-299` during updates.

## 4. Stabilization Steps
While not strictly part of the advanced block, `ppo_config.stabilization_steps` (e.g., 1500) ties into these features by deferring optimizer updates until the state/reward normalizers settle. The scheduler tracks the stabilization window before enabling gradient application (`ppo_scheduler_modular.py:1008-1072`).

## 5. Debugging Checklist
- LR stuck at zero → Verify warmup has completed and `restart_period` is not too large.
- Batch size swings wildly → Lower `adaptation_rate` or tighten `min/max` bounds.
- Gradient explosion despite clipping → Ensure `clip_by_global_norm=true` and check `pg_grad_norm` exports.
- Training slower than expected → Disable checkpointing/accumulation to confirm the baseline.

## 6. Related Docs
- `docs/statistics_stabilization_guide.md`
- `docs/kl_regularization_guide.md`
- `docs/reward_config_guide.md`
