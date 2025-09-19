# PPO调度器优化方案 - 2025年9月18日

## 概述

本文档记录了针对PPO强化学习调度器的系统性优化方案，解决了动作分布极度倾斜和奖励信号缺陷的核心问题。

## 问题诊断

### 核心问题

基于训练数据 `ppo_metrics_20250918_024328.csv` 的分析，发现以下关键问题：

1. **动作分布极度倾斜**
   - 73-77%的动作集中到单一副本
   - 策略快速收敛到次优解

2. **奖励信号结构性缺陷**
   - `absolute_score ≈ -0.05` (固定负值)
   - `logistic_penalty ≈ 0.059` (固定正值)
   - `total_reward ≈ -0.094` (持续负值)
   - `delta_score = 0.0` (无变化信号)

3. **探索机制失效**
   - 温度脉冲触发率极低 (`temp_is_pulsing=False`)
   - 熵系数不足以维持探索
   - 缺乏KL散度监控

4. **训练过程问题**
   - `clipfrac` 偏高，大量梯度被削减
   - 策略在负奖励区域固化

### 根本原因

策略网络只能学习"在哪个副本亏得最少"，而非"如何获得正向收益"，形成负反馈循环。

## 优化方案

### 1. 奖励信号结构重构

#### 目标
确保在当前性能水平下能够获得正向奖励信号，打破负反馈循环。

#### 实施细节

**配置文件修改** (`vidur/config/config.py`):

```python
# 降低throughput目标，使当前性能可获正奖励
throughput_target: float = field(
    default=0.05,  # 原值: 1.0
    metadata={"help": "Target throughput for normalization in absolute score calculation (requests/second)."},
)

# 大幅减少延迟惩罚权重
alpha: float = field(
    default=0.1,  # 原值: 0.5
    metadata={"help": "Balance factor in absolute score (throughput vs latency weight)."},
)

# 降低logistic惩罚强度
kappa: float = field(
    default=0.05,  # 原值: 0.3
    metadata={"help": "Weight for logistic latency penalty (smooth replacement for threshold penalty)."},
)

# 增加惩罚平滑度
sigma: float = field(
    default=2.0,  # 原值: 1.0
    metadata={"help": "Scale parameter for logistic penalty smoothness."},
)

# 调整奖励分量权重
absolute_weight: float = field(
    default=0.8,  # 原值: 0.7
    metadata={"help": "Weight for absolute score component (w_abs) - primary reward signal."},
)

delta_weight: float = field(
    default=0.2,  # 原值: 0.3
    metadata={"help": "Weight for delta score component (w_delta) - improvement signal."},
)
```

#### 预期效果
- `absolute_score`: -0.05 → +1.99 (实际数据计算)
- `total_reward`: -0.094 → 正值区间

### 2. 探索与正则化增强

#### 目标
防止策略过早收敛，维持多样化探索，添加KL散度监控。

#### 实施细节

**基础探索参数调整**:

```python
# 进一步提升熵系数
entropy_coef: float = 0.25  # 原值: 0.15
```

**新增KL散度监控** (`src/rl_components/ppo_trainer.py`):

```python
# 构造函数新增参数
def __init__(
    self,
    # ... 现有参数
    target_kl: float = 0.01,        # KL散度监控阈值
    entropy_min: float = 0.5,       # 熵下限保护
    kl_coef: float = 0.2,          # KL正则化系数
):

# 损失函数增强
def update(self, ...):
    # KL散度正则化
    kl_div = torch.mean(blogp - new_logp).clamp(min=0)
    kl_penalty = self.kl_coef * kl_div

    # 熵下限保护
    entropy_penalty = 0.0
    if entropy.item() < self.entropy_min:
        entropy_penalty = 0.1 * (self.entropy_min - entropy.item())

    # 总损失
    loss = pi_loss + self.value_coef * vf_loss + kl_penalty - entropy_bonus + entropy_penalty
```

**配置参数新增**:

```python
target_kl: float = field(
    default=0.01,
    metadata={"help": "Target KL divergence for early stopping to prevent policy collapse."},
)
entropy_min: float = field(
    default=0.5,
    metadata={"help": "Minimum entropy threshold to maintain exploration."},
)
kl_coef: float = field(
    default=0.2,
    metadata={"help": "Coefficient for KL regularization loss."},
)
```

#### 关键决策
- **拒绝早停机制**: 根据用户建议，不实施基于KL阈值的早停，允许策略充分学习

### 3. 温度脉冲机制优化

#### 目标
大幅提高温度脉冲触发频率，增强探索刺激。

#### 实施细节

**温度控制器参数调整** (`src/rl_components/temperature_controller.py`):

```python
def __init__(
    self,
    # ... 现有参数
    pulse_interval: int = 8,        # 原值: 50 - 大幅提高脉冲频率
    pulse_magnitude: float = 2.0,   # 原值: 1.5 - 增强脉冲强度
    stagnation_threshold: float = 0.005,  # 原值: 0.01 - 更敏感检测
    stagnation_memory: int = 5,     # 原值: 10 - 更快响应
):
```

#### 预期效果
- 温度脉冲从每50步 → 每8步触发
- `temp_is_pulsing` 从 False → 定期激活
- 防止策略在任何区域过早固化

### 4. 网络架构增强

#### 目标
提升模型处理高维噪声信号的能力，减少价值估计偏差。

#### 实施细节

**Actor-Critic解耦架构** (`src/rl_components/actor_critic.py`):

```python
# 默认启用解耦架构
enable_decoupled_ac: bool = field(
    default=True,
    metadata={"help": "Enable decoupled Actor-Critic architecture for better learning."},
)

# 特征投影层
self.feature_proj = nn.Sequential(
    nn.LayerNorm(state_dim),
    nn.Linear(state_dim, self.feature_projection_dim),
    nn.GELU(),
    nn.Linear(self.feature_projection_dim, hidden_size),
)

# 独立Actor分支
self.actor_branch = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.LayerNorm(hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.LayerNorm(hidden_size),
    nn.ReLU(),
)

# 独立Critic分支
self.critic_branch = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.LayerNorm(hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.LayerNorm(hidden_size),
    nn.ReLU(),
)

# Critic专用GRU
self.critic_gru = nn.GRU(
    input_size=hidden_size,
    hidden_size=hidden_size,
    num_layers=1,
    batch_first=False,
)
```

#### 技术特性
- **多尺度特征处理**: LayerNorm + GELU 投影层
- **解耦学习**: Actor和Critic独立优化
- **正交初始化**: 稳定训练启动
- **梯度裁剪**: 防止训练发散

### 5. 配置参数系统调整

#### 综合平衡配置

```python
# 奖励系统参数
latency_threshold: 6.0          # 原值: 2.0 - 放宽延迟容忍
latency_penalty_scale: 0.5      # 原值: 5.0 - 减少延迟惩罚
load_balance_penalty: 0.15      # 原值: 0.03 - 强化负载均衡

# 训练超参数保持
clip_ratio: 0.15                # 适度裁剪
epochs: 8                       # 充分训练
max_grad_norm: 1.0              # 适中梯度裁剪
```

## 实施指南

### 命令行配置

更新后的训练命令包含所有新参数：

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)

python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  --cluster_config_num_replicas 4 \
  --p_p_o_global_scheduler_modular_config_entropy_coef 0.25 \
  --p_p_o_global_scheduler_modular_config_latency_threshold 6.0 \
  --p_p_o_global_scheduler_modular_config_latency_penalty_scale 0.5 \
  --p_p_o_global_scheduler_modular_config_load_balance_penalty 0.15 \
  --p_p_o_global_scheduler_modular_config_throughput_target 0.05 \
  --p_p_o_global_scheduler_modular_config_absolute_weight 0.8 \
  --p_p_o_global_scheduler_modular_config_delta_weight 0.2 \
  --p_p_o_global_scheduler_modular_config_alpha 0.1 \
  --p_p_o_global_scheduler_modular_config_kappa 0.05 \
  --p_p_o_global_scheduler_modular_config_sigma 2.0 \
  --p_p_o_global_scheduler_modular_config_target_kl 0.01 \
  --p_p_o_global_scheduler_modular_config_entropy_min 0.5 \
  --p_p_o_global_scheduler_modular_config_kl_coef 0.2 \
  --p_p_o_global_scheduler_modular_config_enable_dynamic_temperature \
  # ... 其他现有参数
  > run_modular.log
```

### 验证指标

训练完成后检查以下关键指标改善：

1. **动作分布均匀性**
   - `action_replica_0/1/2/3` 应接近 25% 各自分布
   - 不再有单副本占比 >50% 的情况

2. **奖励信号健康度**
   - `absolute_score` 应为正值 (期望 >1.0)
   - `total_reward` 进入正值区间
   - `delta_score` 有非零变化

3. **探索活跃度**
   - `temp_is_pulsing` 定期为 True
   - `entropy` 维持在较高水平 (>0.5)
   - `clipfrac` 降低到合理范围 (<0.3)

## 技术债务和后续改进

### 当前局限性

1. **参数调整**: throughput_target 大幅降低可能影响真实性能评估
2. **网络复杂度**: 解耦架构增加了计算开销
3. **超参敏感性**: 多个新参数需要细致调优

### 后续优化方向

1. **自适应目标**: 根据系统实际性能动态调整 throughput_target
2. **层次化探索**: 实现更智能的探索策略
3. **多目标优化**: 平衡探索-利用与计算效率

## 合规性声明

本优化方案严格遵循 CLAUDE.md 规范：
- ✅ 无fallback模式 - 所有错误直接抛出
- ✅ 参数化配置 - 通过配置文件驱动
- ✅ 模块化设计 - 各组件职责清晰
- ✅ 实验可复现 - 完整记录配置变更

---

**文档版本**: v1.0
**创建日期**: 2025-09-18
**负责人**: Claude Code Assistant
**相关文件**:
- `src/rl_components/ppo_trainer.py`
- `src/rl_components/actor_critic.py`
- `src/rl_components/temperature_controller.py`
- `vidur/config/config.py`
- `vidur/scheduler/global_scheduler/ppo_scheduler_modular.py`