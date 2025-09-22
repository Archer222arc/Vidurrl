# KL Regularization Guide

This note explains how KL-based constraints are wired into the PPO training loop, what each knob controls, and how they interact with warm starts.

## 1. Flow Through the Training Pipeline
- **Config stage**: `configs/ppo_warmstart.json:85-91` defines `target_kl`, `entropy_min`, `kl_coef`, and the KL reference decay schedule. These values are turned into CLI flags by `src/config/training_config.py:58-75`.
- **Scheduler wiring**: When `vidur.main` launches the PPO scheduler, the values reach `vidur/scheduler/global_scheduler/ppo_scheduler_modular.py:120-208`. Here the scheduler stores `self._target_kl` and `self._kl_coef` and passes them to `PPOTrainer` (`src/core/algorithms/ppo_trainer.py:28-85`).
- **Warm start hook**: During warmstart, `_apply_warm_start()` clones the freshly loaded actor and calls `self._ppo.set_reference_policy(...)` (`vidur/scheduler/global_scheduler/ppo_scheduler_modular.py:512-535`). This enables reference-KL regularization against the pretrained behaviour.
- **Training loop**: Every PPO update goes through `PPOTrainer.update()` (`src/core/algorithms/ppo_trainer.py:181-308`), where KL penalties are computed, logged, and mixed into the total loss.

## 2. Instantaneous KL Penalty (`kl_coef`)
- **What it is**: A constant penalty that discourages the new policy from straying far from the behaviour that generated the rollout. Implemented as `kl_penalty = kl_coef * mean(logp_old - new_logp).clamp(min=0)` (`src/core/algorithms/ppo_trainer.py:233-248`).
- **Effect**: Large `kl_coef` values tightly anchor updates to the previous policy (more conservative but slower learning). Small values allow more aggressive policy changes (faster, riskier updates). Start around 0.1–0.3 and adjust based on `approx_kl` in the logs.

## 3. Target KL (`target_kl`)
- **Purpose**: A monitoring threshold meant to catch runaway updates. The scheduler forwards it for logging (`vidur/scheduler/global_scheduler/ppo_scheduler_modular.py:915-919`), and `PPOTrainer` stores it (`src/core/algorithms/ppo_trainer.py:72`).
- **Current behaviour**: The trainer reports `approx_kl` in every update (`src/core/algorithms/ppo_trainer.py:286-307`). Operators watch this metric and compare it to `target_kl`; exceeding the target suggests increasing `kl_coef`, reducing the learning rate, or truncating additional epochs for that update. (Automatic early-stopping based on `target_kl` can be added on top if desired.)

## 4. Reference KL Schedule (`kl_ref_coef_*`)
- **Initialization**: When warmstart loads a pretrained actor, a frozen copy becomes the reference policy (`vidur/scheduler/global_scheduler/ppo_scheduler_modular.py:512-535`).
- **Usage**: Each minibatch computes `KL(current || reference)` and scales it by a decaying coefficient (`src/core/algorithms/ppo_trainer.py:249-261`).
- **Parameters**:
  - `kl_ref_coef_initial`: Initial strength while the PPO policy should stay close to the pretrained behaviour.
  - `kl_ref_coef_final`: Target strength after decay (often 0.0 so the reference pressure disappears).
  - `kl_ref_decay_steps`: Number of minibatch steps (not environment steps) over which the coefficient linearly decays (`src/core/algorithms/ppo_trainer.py:102-119`). Longer schedules keep the policy near the reference for more updates.

## 5. Entropy Safeguards and KL
Although not KL terms, `entropy_min` and `entropy_warmup_coef` interact with KL constraints by preventing the policy from collapsing when KL penalties are strong. They live in the same config block (`configs/ppo_warmstart.json:85-91`) and are applied in `PPOTrainer.update()` (`src/core/algorithms/ppo_trainer.py:242-247`).

## 6. Practical Tuning Tips
- Watch `approx_kl` in TensorBoard/CSV exports; if it sits well below `target_kl`, you can lower `kl_coef` to learn faster. If it spikes above the target, raise `kl_coef` or trim PPO epochs.
- During warmstart runs, keep `kl_ref_coef_initial` reasonably high (0.4–0.8) and decay slowly (`kl_ref_decay_steps` ≥ 3000) so the policy does not immediately forget the pretrained behaviour.
- When running purely online (no warmstart), set `kl_ref_coef_initial=0` to avoid needless reference penalties.
- KL penalties work best together with smaller `clip_ratio` and sensible learning rates; if updates still diverge, revisit those before increasing KL strength excessively.

With these knobs you can control how tightly PPO adheres to prior behaviour—either the immediately previous rollout (`kl_coef`) or an explicit reference model (`kl_ref_*`)—while `target_kl` and entropy safeguards provide monitoring and stability levers during training.
