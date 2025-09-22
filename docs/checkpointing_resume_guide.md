# Checkpointing & Resume Guide

This guide documents how PPO training checkpoints are created, what state they contain, and how the training scripts orchestrate resume flows.

## 1. Where Checkpoints Come From
- `configs/ppo_warmstart.json:146-153` enables checkpoints (`enable_checkpoints=true`, interval=128, `save_optimizer_state=true`, `incremental_checkpoints=true`).
- `src/config/training_config.py:94-113` translates those settings into CLI flags for `vidur.main`.
- `vidur/scheduler/global_scheduler/ppo_scheduler_modular.py:470-520` handles save/load events. The scheduler stores model weights, optimizer state, normalizer statistics, temperature controller state, and step counters.
- Default files live under `./outputs/checkpoints/` with a symlink `latest.pt` pointing to the most recent checkpoint.

## 2. What Is Saved
Each checkpoint contains:
- Actor-Critic weights (`state_dict`).
- AdamW optimizer state (requires `save_optimizer_state=true`).
- Rollout buffer progress metadata (so buffers refill cleanly after resume).
- Normalizer statistics (means/variances) for state and reward.
- Temperature controller metrics (`current_temperature`, EMAs, pulse flags).
- Step counters used for KL decay, entropy warmup, and learning-rate schedules.

## 3. Saving During Training
- The scheduler triggers saves every `checkpoint_interval` PPO steps.
- Additional checkpoints are written at the end of each chunk when `--chunk-mode` is active.
- Incremental checkpoints rotate (keep at most `max_checkpoints`). Older files are deleted automatically.

## 4. Resume Scenarios
`train_ppo_warmstart_optimized.sh:170-280` prompts the operator when both external models and existing checkpoints are present. Options:
1. **Resume from checkpoint** (skip warmstart). Adds `--p_p_o_global_scheduler_modular_config_load_checkpoint ./outputs/checkpoints/latest.pt` to the PPO command.
2. **Warmstart again** with fresh demos (ignores checkpoint).
3. **Use external pretrain only** (checkpoint is ignored).

When chunk mode metadata (`training_progress.json`) exists, the script auto-detects completed chunks and resumes the next slice without user prompts.

## 5. Verifying a Resume
- Logs: `OUTPUT_DIR/ppo_training.log` shows `[Resume] PPO训练恢复` with the checkpoint path.
- Metrics: Step counters continue from the saved values (check TensorBoard `global_step`).
- Checkpoint timestamp: `stat ./outputs/checkpoints/latest.pt` should match the last run.

If resuming fails (e.g., shape mismatch), the scheduler logs a warning and continues from scratch—watch for `[PPO:checkpoint] Continuing with fresh training state`.

## 6. Best Practices
- Keep `save_optimizer_state=true` so momentum carries between chunks/resumes.
- Align `checkpoint_interval` with `rollout_len` multiples to capture whole PPO updates.
- Back up `training_progress.json` alongside checkpoints when running long chunked sessions.
- For experiments, clear the checkpoint directory (`rm ./outputs/checkpoints/*.pt`) before starting a fresh run to avoid accidental resumes.

## 7. Related Files
- `scripts/train_ppo_warmstart_optimized.sh`
- `vidur/scheduler/global_scheduler/ppo_scheduler_modular.py`
- `configs/ppo_warmstart.json`
