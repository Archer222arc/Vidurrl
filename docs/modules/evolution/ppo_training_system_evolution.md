# PPO训练系统演进史

## 📚 历史文档索引
(暂无历史归档)

## 当前活跃版本 (2025年)

---

## [2025-10-01] - 参数优化: 修复训练失败问题

### 变更概述
- **改动原因**: v3.0.0训练完全失败 (Entropy=1.14保持random, ExplainedVar=0.12, 性能比RoundRobin差5-22%)
- **影响范围**:
  - `configs/revolutionary_collapse_prevention.json` (直接修改主配置)
  - 所有PPO hyperparameters和collapse prevention系统参数
- **向后兼容性**: 直接修改现有配置,保持架构不变,仅调整参数强度

### 根本原因分析

#### 问题1: Policy始终保持Random (Entropy=1.14, 82% randomness)
**原因**: 5个独立系统协同强制policy保持random:
1. Expert Guidance (weight=1.5) 强制round-robin
2. Action Balance (weight=0.8) 强制均匀分布
3. History Shortcut (weight=0.95) bypass状态学习
4. Enhanced Collapse Detection (boost=20x) 过度干预
5. Context-Aware Entropy 持续强制高熵

**影响**: Policy永远学不到"根据状态选择最优replica"的能力

#### 问题2: Value Function失败 (ExplainedVariance=0.12)
**原因**:
- Epochs不足 (8) - VF更新次数不够
- Reward信号不稳定
- Value coefficient可能过高

#### 问题3: Policy更新过激 (ClipFraction=0.88)
**原因**:
- Clip ratio太小 (0.2) - Trust region过窄
- 88%的更新被clip → 学习效率极低

#### 问题4: Learning Rate衰减到零
**原因**:
- Cooldown enabled + Final LR ratio=0.1
- 训练后期LR接近0,无法继续学习

#### 问题5: Gradient Norm爆炸 (300+)
**原因**:
- Max grad norm=0.5太小
- Gradient经常超出数百倍被压缩

### 具体变更

#### 1. 降低强制Random机制强度

**Expert Guidance**:
```json
// 修改前
"expert_guidance_weight": 1.5,

// 修改后
"expert_guidance_weight": 0.5,  // -67%
```
- **文件**: `configs/revolutionary_collapse_prevention.json:125-126`
- **原因**: 降低KL penalty,允许policy偏离round-robin去优化latency

**Action Balance**:
```json
// 修改前
"action_balance_weight": 0.8,

// 修改后
"action_balance_weight": 0.3,  // -62%
```
- **文件**: `configs/revolutionary_collapse_prevention.json:175-176`
- **原因**: 允许非均匀action distribution

**History Shortcut Weights**:
```json
// 修改前
"history_shortcut_weights": {
  "exploration_phase": 0.95,
  "balance_phase": 0.6,
  "convergence_phase": 0.3
}

// 修改后
"history_shortcut_weights": {
  "exploration_phase": 0.5,   // -47%
  "balance_phase": 0.3,       // -50%
  "convergence_phase": 0.1    // -67%
}
```
- **文件**: `configs/revolutionary_collapse_prevention.json:32-41`
- **原因**: 允许state-based learning,不仅仅复制round-robin

#### 2. 修复PPO Hyperparameters

**Clip Ratio**:
```json
"clip_ratio": 0.2 → 0.3,  // +50%
```
- **预期**: ClipFraction从88%降到30-50%

**Epochs**:
```json
"epochs": 8 → 12,  // +50%
```
- **预期**: ExplainedVariance从0.12提升到>0.6

**Minibatch Size**:
```json
"minibatch_size": 64 → 128,  // 2x
```
- **预期**: Gradient estimates更稳定

**Max Gradient Norm**:
```json
"max_grad_norm": 0.5 → 0.8,  // +60%
```
- **预期**: Gradient norm从300+稳定到<50

#### 3. 修复Learning Rate Schedule

```json
// 修改前
"lr": 0.0003,
"initial_lr_ratio": 0.1,      // 起始LR=3e-5,太低
"schedule_type": "linear",
"cooldown_enable": true,       // 后期继续衰减到接近0

// 修改后
"lr": 0.0004,                  // +33%
"initial_lr_ratio": 0.4,       // 起始LR=1.6e-4,提高5.3x
"schedule_type": "cosine",     // 更平滑
"cooldown_enable": false       // 禁用,防止LR→0
```
- **文件**: `configs/revolutionary_collapse_prevention.json:82-94`

#### 4. 放宽Collapse Detection Thresholds

```json
// CV thresholds
"cv_warning_threshold": 0.3 → 0.4,      // +33%
"cv_emergency_threshold": 0.8 → 1.0,    // +25%

// Intervention strength
"emergency_entropy_boost": 20.0 → 8.0,  // -60%
"intervention_cooldown": 10 → 20,       // 2x
```
- **原因**: 减少false alarms和过度干预

#### 5. 放宽Entropy Bounds

```json
"entropy_bounds": {
  "floor": 0.01 → 0.05,      // 允许稍高最低熵
  "ceiling": 0.25 → 0.6,     // 允许更高初始探索
  "target": 0.12 → 0.25      // 提高目标熵
}
```
- **原因**: 允许更宽的entropy range,自然收敛

#### 6. 增强Latency目标权重

```json
"simplified_reward_latency_weight": 1.0 → 1.5,        // +50%
"simplified_reward_performance_weight": 0.1 → 0.3,    // +200% (from lower baseline)
"simplified_reward_imbalance_weight": 0.6 → 0.3,      // -50%

"objective_weights": {
  "latency": 1.5 → 2.0,      // +33%, 最高权重
  "balance": 3.0 → 1.5       // -50%
}
```
- **文件**: `configs/revolutionary_collapse_prevention.json:128-133, 256-263`
- **原因**: 强化核心优化目标,降低balance penalty

### 预期效果

#### Training Metrics
| 指标 | v3 (失败) | v4 (预期) |
|------|-----------|----------|
| Entropy | 1.14 (82% random) | 0.6→0.3→0.15 |
| ExplainedVariance | 0.12 | >0.6 |
| ClipFraction | 0.88 | 0.3-0.5 |
| GradientNorm | 300+ | 5-50 |
| LearningRate | →0 | 保持1e-4到4e-4 |

#### Performance Metrics
| 调度器 | 延迟 | 目标 |
|--------|------|------|
| RoundRobin (baseline) | 7.5-8.5s | - |
| PPO v3 (失败) | 8.7-9.2s | - |
| PPO v4 (预期) | <8.0s | Beat baseline |

### 测试验证

**监控Checklist**:
- [ ] 前1000 steps: Entropy开始下降, ExplainedVar>0.2
- [ ] 3000 steps: Entropy<0.5, ExplainedVar>0.4
- [ ] 10000 steps: Entropy<0.3, ExplainedVar>0.6
- [ ] 20000 steps: 性能测试 vs RoundRobin

**如果失败 - Debug路径**:
1. Entropy不下降 → 进一步降低expert_guidance_weight到0.3
2. ExplainedVar不提高 → 增加epochs到15-20
3. ClipFraction仍高 → 增加clip_ratio到0.4

### 相关文件
- **修改配置**: `configs/revolutionary_collapse_prevention.json` (v3.0.0 → v3.1.0优化)
- **演进文档**: `docs/modules/evolution/ppo_training_system_evolution.md`

### 使用方法

```bash
# 1. 清除旧checkpoint
rm outputs/checkpoints/latest.pt

# 2. 启动TensorBoard监控
tensorboard --logdir outputs/revolutionary_training/tensorboard --port 6006 &

# 3. 开始训练 (使用优化后的配置)
python vidur/simulator.py \
  --config configs/revolutionary_collapse_prevention.json \
  --training_steps 20000

# 4. 性能测试 (训练完成后)
bash scripts/scheduler_comparison.sh
```

---

## [2025-09-30] - v3.0.0: Revolutionary Collapse Prevention System

### 变更概述
- **改动原因**: 防止训练中的policy collapse和action distribution collapse
- **影响范围**: 整个训练系统架构
- **向后兼容性**: 完全重构,不兼容旧版本

### 主要功能
1. Enhanced Collapse Detection - CV-based早期预警
2. Staged Entropy Control - 分阶段熵控制
3. Expert Guidance - Round-robin expert policy
4. History Augmentation - Dual-stream architecture with shortcut
5. Dynamic Reward System - 自适应reward scaling

### 训练结果
❌ **失败**:
- Entropy保持1.14 (82% random),policy未收敛
- ExplainedVariance=0.12,value function未学习
- 性能比RoundRobin差5-22%

### 废弃原因
配置过于aggressive,多个系统协同阻止policy收敛

### 替代方案
使用v4 Incremental Fix (降低各系统强度)

---

## 文档维护

**文档状态**:
- 当前版本数: 2
- 文档行数: ~180行
- 最后更新: 2025-10-01

**下次归档触发**:
- 版本数达到10个,或
- 文档行数超过800行
