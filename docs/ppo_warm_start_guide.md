# PPO Warm-Start Guide

This guide explains the complete warm-start pipeline now used by `train_ppo_warmstart_optimized.sh`, including demonstration collection, behavior cloning, external pretraining, KL alignment, and chunked PPO resumes.

## 1. Why Warm Start?
Cold-start PPO suffers from:
- Highly skewed initial action distribution (single replica overload).
- Noisy rewards before normalizers converge.
- Slow convergence because the policy must rediscover basic load-balancing rules.

Warm start fixes this by seeding the actor with expert behavior and constraining PPO with KL penalties until it learns safely.

## 2. End-to-End Workflow
```
collect demos  →  behavior cloning / external model  →  PPO with KL + temperature control
                                               ↘  optional chunked resumes (5k requests)
```

### 2.1 Scripts
- `scripts/collect_demo.py`: Single-policy heuristic data (RoundRobin/LOR/Random).
- `scripts/collect_demo_mixed.py`: Mixed-policy batches + imbalance scenarios.
- `scripts/standalone_pretrain.py`: High-capacity offline pretraining (enhanced mode).
- `scripts/train_ppo_warmstart_optimized.sh`: Main orchestration script (supports external models, resume, chunked training).
- `scripts/train_ppo_with_external_pretrain.sh`: Thin wrapper that validates an external checkpoint then forwards to the optimized script with `--skip-bc-training`.

### 2.2 Optimized Training Script Highlights (`scripts/train_ppo_warmstart_optimized.sh:51-446`)
- CLI options for replicas, QPS, request budget, demo steps, external checkpoints, and forced warmstart.
- Interactive resume decision tree when checkpoint + external model both exist.
- Warmstart stage runs only when needed (BC or external checkpoint).
- PPO stage launches via `python -m vidur.main` with config-derived arguments and warmstart flags.
- Writes logs to `<OUTPUT_DIR>/ppo_training.log` and persists checkpoints to `./outputs/checkpoints/latest.pt`.

### 2.3 Chunked Training Mode
To avoid huge rollouts (memory pressure), run multiple 5k-request chunks:
```
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode --chunk-size 5000 --total-requests 20000 \
  --external-pretrain ./outputs/unified_pretrain/high_quality_model.pt
```
Each chunk restores `./outputs/checkpoints/latest.pt`, preserves optimizer state (requires `save_optimizer_state=true` in config), and appends to `training_progress.json` inside the run directory.

## 3. Demonstration Collection
1. **Mixed policies**: `round_robin`, `lor`, `random` by default (`train_ppo_warmstart_optimized.sh:329-344`).
2. **State builder parity**: `collect_demo.py:150-183` matches PPO state features (history window, queue size, enhanced features).
3. **Imbalance scenarios**: `collect_demo_mixed.py` optionally samples high/low QPS segments to teach recovery behavior.
4. **Outputs**: Pickle file storing `demo_data` (state/action pairs) plus distribution stats.

## 4. Behavior Cloning & External Models
- `scripts/pretrain_actor.py:364-372` trains the warmstart actor (GRU-based, configurable epochs/batch size).
- `src/pretraining/unified_trainer.py:36-204` supports both standard BC and enhanced standalone pretraining.
- When `--external-pretrain PATH` is provided, the script validates the checkpoint (`model_validator`) and copies it into the run directory (`train_ppo_warmstart_optimized.sh:156-168`).
- `--skip-bc-training` uses the external weights directly; otherwise BC fine-tunes on the fresh demos.

## 5. KL Regularization During PPO
- Warmstart clones the actor into a frozen reference model (`ppo_scheduler_modular.py:500-535`).
- PPO updates include:
  - **Instant KL penalty** (`kl_coef`) against the rollout policy.
  - **Reference KL penalty** with linear decay (`kl_ref_coef_initial/final/decay_steps`).
  - **Target KL monitoring** for operator-triggered early stop.
- See `docs/kl_regularization_guide.md` for full details.

## 6. Temperature and Stabilization
- Temperature control (`docs/temperature_controller_parameters.md`) keeps exploration in check, especially after warmstart.
- Statistics stabilization (`docs/statistics_stabilization_guide.md`) runs before PPO updates to pre-train normalizers using random scheduling.

## 7. Resume & Checkpoints
- Default checkpoint path: `./outputs/checkpoints/latest.pt` with rotating history.
- `save_optimizer_state=true` ensures AdamW momentum carries across warmstart, resumes, and chunked runs.
- Resume options in the script:
  1. Use external pretrain (skip warmstart).
  2. Run warmstart again on new demos.
  3. Load the checkpoint and continue PPO.
- Check `docs/checkpointing_resume_guide.md` for deeper coverage.

## 8. Recommended Command Recipes
### One-shot warmstart + PPO
```
bash scripts/train_ppo_warmstart_optimized.sh \
  --num-replicas 4 --qps 3.5 --ppo-requests 12000
```

### External checkpoint without BC
```
bash scripts/train_ppo_with_external_pretrain.sh \
  ./outputs/unified_pretrain/high_quality_model.pt \
  --num-replicas 8 --qps 5.0 --force-warmstart
```

### Mixed demos + BC fine-tune
```
bash scripts/train_ppo_warmstart_optimized.sh \
  --demo-steps 1000 --bc-epochs 20 \
  --external-pretrain ./outputs/unified_pretrain/specialized_model.pt
```

## 9. Troubleshooting
- **BC accuracy low** → increase epochs or balance demo dataset.
- **KL spikes** → reduce `kl_ref_coef_initial` or extend `kl_ref_decay_steps`.
- **Resume ignores checkpoint** → ensure `save_optimizer_state=true` and that the script detects `latest.pt` before chunk 2 runs.
- **Memory usage high** → switch to chunk mode or reduce `rollout_len` in config.

## 10. Related Reading
- `docs/kl_regularization_guide.md`
- `docs/reward_config_guide.md`
- `docs/temperature_controller_parameters.md`
- `docs/statistics_stabilization_guide.md`
- `docs/load_balance_optimization.md`
