# PPO超参数调参记录

## 文档说明
记录PPO训练的超参数配置、实验结果和失败案例，避免重复错误。

---

## 调参历史

### ❌ 实验1: 学习率过小 (2025-09-30 早期)

**配置**:
```json
{
  "lr": 1e-6,
  "initial_lr_ratio": 0.0001,  // warmup从1e-10开始
  "clip_ratio": 0.3,
  "epochs": 2,
  "value_coef": 1.0
}
```

**实际学习率范围**: 1e-10 (warmup) → 1e-6 (peak) → 1e-10 (cooldown)

**结果**:
- ❌ **Explained_var = 0**: Value网络完全不学习
- ❌ **Clipfraction居高不下**: 0.6-0.85（理想值0.1-0.3）
- ⚠️ Entropy稳定在1.0-1.1

**问题分析**:
1. 学习率太小（1e-6），value网络权重几乎不更新
2. Value预测始终接近0（由于gain=0.01初始化）
3. Advantages估计有巨大偏差 → Policy更新方向错误 → 高clipfraction
4. Warmup从1e-10开始更是雪上加霜

**教训**:
- PPO学习率不应低于1e-5
- Warmup起点不应低于peak lr的10%
- Value head用gain=0.01初始化时，需要足够大的lr才能学习

---

### ❌ 实验2: 学习率灾难性过大 (2025-09-30 中期)

**配置**:
```json
{
  "lr": 0.1,  // 1e-1
  "initial_lr_ratio": 0.2,  // warmup从2e-2开始
  "clip_ratio": 0.2,
  "epochs": 4,
  "value_coef": 1.0,
  "rollout_length": 64
}
```

**实际学习率范围**: 2e-2 (warmup) → 0.1 (peak)

**结果**:
- ❌ **Entropy飙升到完全随机**: 接近log(num_actions)=1.386
- ❌ **训练完全崩溃**: 网络权重爆炸
- ❌ **Policy变成uniform分布**: 完全失去学习能力

**问题分析**:
1. lr=0.1是标准PPO(3e-4)的**333倍**
2. 每次更新权重变化巨大 → 梯度爆炸
3. Policy logits崩溃 → Categorical分布变uniform
4. Rollout length=64也太小，GAE估计不准

**教训**:
- ⚠️ **PPO学习率绝对不能超过1e-3**
- 标准范围: 1e-4 到 3e-4
- lr=0.1会导致不可恢复的训练崩溃
- Rollout length最少128，建议256

---

### ✅ 实验3: 保守配置初步改善 (2025-09-30 run_20250930_094101)

**配置**:
```json
{
  "lr": 1e-4,
  "initial_lr_ratio": 0.2,  // warmup从2e-5开始
  "clip_ratio": 0.2,
  "epochs": 4,
  "value_coef": 1.0,
  "rollout_length": 256
}
```

**实际学习率**: 2e-5 (warmup) → 1e-4 (peak) → 2e-5 (cooldown)

**结果** (6次PPO updates):
- ✅ **Explained_var恢复**: 0.25-0.61（平均0.42）
- ⚠️ **Clipfraction仍偏高**: 0.51-0.85（平均0.62）
- ✅ **VF_loss下降**: 38.76 → 38.29
- ✅ **Entropy稳定**: 1.0-1.14

**问题分析**:
1. Value网络开始学习，但还不够稳定
2. Explained_var波动大（0.61→0.26→0.40）说明预测不准
3. Clipfraction高是因为value预测不准导致advantages有偏差
4. 需要更强的value学习

**改进方向**:
- 增加value_coef让value学得更快
- 降低clip_ratio限制policy更新幅度
- 增加epochs给value更多学习机会

---

### ❌ 实验4: lr=1e-2 训练严重不稳定 (2025-09-30 run_20250930_102008)

**配置**:
```json
{
  "lr": 1e-2,  // 0.01, 是标准值的33倍
  "initial_lr_ratio": 0.001,  // warmup从1e-5开始
  "clip_ratio": 0.15,
  "epochs": 6,
  "value_coef": 1.5,
  "rollout_length": 256
}
```

**实际学习率**: 1e-5 (warmup) → 1e-2 (peak) → 1e-5 (cooldown)

**结果** (PPO update分析):
- ❌ **Entropy**: 1.0788 (接近完全随机1.386)
- ❌ **Clipfraction**: 0.78125 (极度危险！应该≤0.3)
- ❌ **Gradient Norm**: 352.48 (梯度爆炸)
- ❌ **Explained_var**: 5.2e-08 ≈ 0 (value完全不学习)
- ❌ **Approx_KL**: 0.5304 (远超0.05危险阈值)

**问题分析**:
1. lr=0.01太大 → 权重更新幅度巨大
2. 梯度爆炸 (352) → Policy输出剧烈震荡
3. KL散度过大 (0.53) → 违反trust region约束
4. 78%的policy更新被clip → 训练极度低效
5. Entropy单调上升 → Policy趋向随机化
6. Value网络无法跟上policy变化 → EV≈0

**教训**:
- lr=1e-2 (0.01) 对PPO来说仍然**太大**
- 虽然没有像lr=0.1那样立即崩溃，但训练严重不稳定
- **安全上限应该是1e-3**，超过就会出现严重问题
- Gradient norm=352是明确的梯度爆炸信号

---

### 🔄 实验5: 标准PPO配置 (2025-09-30 最新)

**配置**:
```json
{
  "lr": 3e-4,  // 标准PPO学习率
  "initial_lr_ratio": 0.1,  // warmup从3e-5开始
  "clip_ratio": 0.15,
  "epochs": 6,
  "value_coef": 1.5,
  "rollout_length": 256
}
```

**实际学习率**: 3e-5 (warmup) → 3e-4 (peak) → 3e-5 (cooldown)

**等待实验结果...**

---

## 关键经验总结

### 学习率 (lr)
- ✅ **推荐范围**: 1e-4 到 3e-4
- ⚠️ **保守值**: 1e-4 (适合early training)
- ⚠️ **激进值**: 5e-4 (需要密切监控)
- ❌ **禁区**:
  - < 1e-5: Value网络不学习
  - > 1e-3: 训练不稳定 (clipfrac>0.5, gradient norm>100)
  - 1e-2: 严重不稳定 (clipfrac>0.7, entropy飙升, EV≈0)
  - > 0.1: 灾难性崩溃 (entropy→完全随机)

### Warmup配置
- ✅ **initial_lr_ratio**: 0.1-0.2 (peak lr的10-20%)
- ❌ **不要**: < 0.01 (太慢，浪费时间)
- ❌ **不要**: > 0.5 (失去warmup意义)

### Clip Ratio
- ✅ **标准值**: 0.2
- ✅ **保守值**: 0.15 (降低policy震荡)
- ❌ **不要**: > 0.3 (policy更新太激进)

### Value Coefficient
- ✅ **标准值**: 0.5-1.0
- ✅ **增强值**: 1.5 (当value学习困难时)
- ⚠️ **Trade-off**: 太高会影响policy学习

### Epochs
- ✅ **标准值**: 4
- ✅ **增强值**: 6-8 (给value更多学习机会)
- ❌ **不要**: < 3 (样本效率太低)
- ❌ **不要**: > 10 (过拟合风险)

### Rollout Length
- ✅ **推荐值**: 256
- ⚠️ **最小值**: 128 (GAE估计勉强准确)
- ❌ **不要**: < 64 (GAE估计严重不准)

---

## Value网络特殊问题

### Gain=0.01初始化的影响
当前代码中value head使用`gain=0.01`初始化（actor_critic.py:342, 356）:

**问题**:
- Value predictions初始接近0
- 当rewards在±2范围时，value需要很长时间学习
- Explained_var公式: `1 - var(returns - values) / var(returns)`
  - 如果values≈0，var(returns - values) ≈ var(returns)
  - 导致explained_var ≈ 0

**解决方案**:
1. **提高学习率**: 让value网络能快速学习（但不能太高）
2. **增加value_coef**: 加大value loss的权重
3. **增加epochs**: 给value更多学习机会
4. **耐心等待**: Early training阶段explained_var低是正常的

---

## 监控指标理想范围

| 指标 | 理想范围 | 警告范围 | 危险信号 |
|------|----------|----------|----------|
| **Explained_Var** | 0.6-0.9 | 0.3-0.6 | < 0.3 或 < 0 |
| **ClipFraction** | 0.1-0.3 | 0.3-0.5 | > 0.5 |
| **Entropy** | 0.5-1.0 | 0.3-0.5 或 1.0-1.2 | < 0.1 或 > 1.3 |
| **Approx_KL** | < 0.02 | 0.02-0.05 | > 0.05 |
| **VF_Loss** | 下降趋势 | 波动但整体下降 | 上升或爆炸 |
| **Policy_Loss** | -10 到 10 | -50 到 50 | 绝对值 > 100 |

---

## 调参流程建议

### 第1步: 确保基本稳定
```json
{
  "lr": 1e-4,
  "clip_ratio": 0.2,
  "epochs": 4,
  "value_coef": 1.0
}
```
**目标**: Explained_var > 0.3, Entropy稳定

### 第2步: 提升value学习
```json
{
  "lr": 1e-4,
  "clip_ratio": 0.15,
  "epochs": 6,
  "value_coef": 1.5
}
```
**目标**: Explained_var > 0.6, Clipfraction < 0.4

### 第3步: 加速收敛
```json
{
  "lr": 3e-4,
  "clip_ratio": 0.15,
  "epochs": 6,
  "value_coef": 1.5
}
```
**目标**: Explained_var > 0.8, 训练稳定收敛

---

## 紧急修复方案

### 如果Entropy崩溃到0
```json
{
  "entropy_coef": 0.05,  // 提高5倍
  "clip_ratio": 0.3,     // 放宽clip
  "lr": 减半
}
```

### 如果Entropy飙升到完全随机
```json
{
  "lr": 减少到1/10,     // 立即降低lr
  "entropy_coef": 0.001, // 降低entropy bonus
  "max_grad_norm": 0.1   // 严格梯度裁剪
}
```
或者直接**停止训练，恢复checkpoint**

### 如果Clipfraction持续>0.8
```json
{
  "clip_ratio": 0.1,     // 严格限制
  "value_coef": 2.0,     // 强化value学习
  "epochs": 8            // 更多学习机会
}
```

---

## 配置文件版本管理

建议为每次重要实验创建配置副本:
```bash
cp configs/revolutionary_collapse_prevention.json \
   experiments/config_YYYYMMDD_HHMMSS_description.json
```

这样可以：
1. 追溯历史配置
2. 复现成功实验
3. 对比不同配置的效果

---

## 最新调参记录

### ❌ 实验6: v3.0.0 Revolutionary配置训练完全失败 (2025-09-30)

**配置**:
```json
{
  "lr": 0.0003,
  "initial_lr_ratio": 0.1,  // warmup从3e-5开始
  "clip_ratio": 0.2,
  "epochs": 8,
  "minibatch_size": 64,
  "max_grad_norm": 0.5,
  "value_coef": 0.5,
  "schedule_type": "linear",
  "cooldown_enable": true,

  // Forced-random mechanisms
  "expert_guidance_weight": 1.5,
  "action_balance_weight": 0.8,
  "history_shortcut_weights": {
    "exploration_phase": 0.95,
    "balance_phase": 0.6,
    "convergence_phase": 0.3
  },

  // Collapse detection
  "cv_warning_threshold": 0.3,
  "cv_emergency_threshold": 0.8,
  "emergency_entropy_boost": 20.0,
  "intervention_cooldown": 10
}
```

**结果** (训练完全失败):
- ❌ **Entropy = 1.14**: 保持82%随机性，policy未收敛
- ❌ **ExplainedVariance = 0.12**: Value function完全失败
- ❌ **ClipFraction = 0.88**: 88%更新被clip，学习效率极低
- ❌ **Gradient Norm = 300+**: 梯度爆炸，被压缩到0.5
- ❌ **Learning Rate → 0**: Cooldown导致后期LR接近0
- ❌ **Performance**: 比RoundRobin差5-22% (8.7-9.2s vs 7.5-8.5s)

**根本原因分析**:
1. **5个系统协同强制policy保持random**:
   - Expert Guidance (1.5) + Action Balance (0.8) 强制均匀分布
   - History Shortcut (0.95) bypass状态学习
   - Enhanced Collapse Detection (boost=20x) 过度干预
   - Context-Aware Entropy 持续强制高熵

2. **PPO超参数问题**:
   - clip_ratio=0.2太小 → 88% clipfrac
   - epochs=8不足 → ExplainedVar=0.12
   - max_grad_norm=0.5太小 → 300+梯度被压缩

3. **Learning Rate衰减问题**:
   - initial_lr_ratio=0.1 → 起始LR=3e-5太低
   - cooldown_enable=true → 后期LR→0
   - linear schedule → 衰减过快

4. **Collapse Detection过度干预**:
   - cv_warning=0.3太低 → 频繁false alarm
   - emergency_boost=20x → 过度强制探索

**教训**:
- ⚠️ **不要同时使用多个forced-random机制**: 会阻止policy收敛
- ⚠️ **History shortcut > 0.7会bypass学习**: 应≤0.5
- ⚠️ **Collapse detection要保守**: emergency_boost不超过10x
- ⚠️ **Learning rate不能衰减到0**: 禁用cooldown或提高final_lr_ratio

---

### ✅ 实验7: v3.1.0 优化配置 (2025-10-01)

**配置**:
```json
{
  // PPO Hyperparameters优化
  "lr": 0.0004,  // +33%
  "initial_lr_ratio": 0.4,  // 4x，起始LR=1.6e-4
  "clip_ratio": 0.3,  // +50%
  "epochs": 12,  // +50%
  "minibatch_size": 128,  // 2x
  "max_grad_norm": 0.8,  // +60%
  "value_coef": 0.5,
  "schedule_type": "cosine",  // 更平滑
  "cooldown_enable": false,  // 禁用！

  // 降低Forced-random机制强度
  "expert_guidance_weight": 0.5,  // -67%
  "action_balance_weight": 0.3,  // -62%
  "history_shortcut_weights": {
    "exploration_phase": 0.5,  // -47%
    "balance_phase": 0.3,  // -50%
    "convergence_phase": 0.1  // -67%
  },

  // 放宽Collapse detection
  "cv_warning_threshold": 0.4,  // +33%
  "cv_emergency_threshold": 1.0,  // +25%
  "emergency_entropy_boost": 8.0,  // -60%
  "intervention_cooldown": 20,  // 2x

  // 调整Entropy bounds
  "entropy_bounds": {
    "floor": 0.05,  // +5x
    "ceiling": 0.6,  // +2.4x
    "target": 0.25  // +2x
  },

  // 增强Latency目标
  "simplified_reward_latency_weight": 1.5,  // +50%
  "simplified_reward_imbalance_weight": 0.3,  // -50%
  "objective_weights": {
    "latency": 2.0,  // +33%
    "balance": 1.5  // -50%
  }
}
```

**预期效果**:
- ✅ **Entropy**: 0.6→0.3→0.15 (自然收敛)
- ✅ **ExplainedVariance**: >0.6
- ✅ **ClipFraction**: 0.3-0.5
- ✅ **Gradient Norm**: 5-50
- ✅ **Learning Rate**: 保持1e-4到4e-4
- ✅ **Performance**: <8.0s (beat RoundRobin)

**关键改进**:
1. **允许policy自主学习**: 降低所有forced-random权重
2. **修复PPO超参数**: 提高clip_ratio, epochs, minibatch_size
3. **防止LR衰减**: 禁用cooldown, 提高initial_lr_ratio, 使用cosine schedule
4. **减少过度干预**: 放宽collapse detection阈值，降低boost强度

**待验证** (需要训练测试):
- Entropy能否在前3000步开始下降
- ExplainedVar能否在10000步达到>0.6
- 性能能否beat RoundRobin baseline

---

最后更新: 2025-10-01