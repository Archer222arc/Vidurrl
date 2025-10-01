# 奖励系统模块完整接口文档

## 模块概述

奖励系统模块负责PPO训练过程中的奖励计算和优化，采用SimplifiedRewardCalculator统一管理所有奖励组件，通过权重配置实现负载均衡策略优化。

## 核心功能

### 主要特性
- **统一奖励计算**：整合性能评分、负载均衡惩罚、延迟惩罚为统一公式
- **自适应调节**：基于性能趋势的探索奖励机制
- **基线稳定化**：避免持续负奖励影响训练稳定性
- **配置化权重**：通过JSON配置灵活调整各组件权重

### 核心类和接口

#### SimplifiedRewardCalculator
```python
class SimplifiedRewardCalculator:
    def __init__(
        self,
        performance_weight: float = 1.0,      # 性能评分权重
        imbalance_weight: float = 1.0,        # 负载失衡惩罚权重
        latency_weight: float = 0.5,          # 延迟惩罚权重
        exploration_bonus: float = 0.1,       # 探索奖励系数
        throughput_target: float = 3.5,       # 吞吐量目标值 (更新为实际scale)
        latency_target: float = 8.0,          # 延迟目标值 (更新为实际scale)
        latency_threshold: float = 10.0,      # 延迟阈值 (更新为实际scale)
        use_delta: bool = True,               # 启用改进信号
        delta_weight: float = 0.3,            # 改进信号权重
        baseline_enable: bool = True,         # 启用基线调整 (默认启用)
        baseline_value: float = 2.0,          # 基线值 (更新为实际值)
        # 新增组件：稳定性奖励
        enable_stability_reward: bool = True, # 启用稳定性奖励
        stability_weight: float = 0.8,        # 稳定性权重
        rollout_len: int = 256,               # 稳定性评估窗口
        stability_threshold: float = 0.2,     # 稳定性阈值
        # 新增组件：专家指导奖励
        enable_expert_guidance: bool = False, # 启用专家指导
        expert_guidance_weight: float = 0.5,  # 专家指导权重
        expert_num_replicas: int = 4,         # 专家副本数量
    )

    def calculate_reward(
        self,
        metric_store: MetricStore,
        current_time: float,
        replica_ids: List[int],
        get_replica_scheduler_fn,
    ) -> Tuple[float, Dict[str, float]]
```

## 完整的类/函数接口说明

### 主要计算方法

#### calculate_performance_score()
计算统一性能评分，整合绝对性能和改进信号
- **输入参数**：throughput（吞吐量）、latency（延迟）
- **返回值**：performance_score（性能评分，0-2范围）
- **计算逻辑**：
  ```
  absolute_perf = throughput_norm + latency_norm
  delta_perf = 基于历史的改进信号
  performance_score = (1-delta_weight) × absolute_perf + delta_weight × delta_perf
  ```

#### calculate_unified_imbalance_penalty()
计算统一负载失衡惩罚，基于副本间队列长度变异系数
- **输入参数**：replica_ids、get_replica_scheduler_fn
- **返回值**：imbalance_penalty（失衡惩罚值，0-10范围）
- **惩罚阶段**：
  - CV < 0.3：轻度线性惩罚
  - CV 0.3-0.7：中度二次方惩罚
  - CV > 0.7：重度指数惩罚

#### calculate_unified_latency_penalty()
计算统一延迟惩罚，结合软阈值和sigmoid函数
- **输入参数**：latency（当前延迟）
- **返回值**：latency_penalty（延迟惩罚值）
- **计算逻辑**：
  ```
  if latency <= threshold:
      penalty = latency / target
  else:
      penalty = base_penalty + sigmoid(excess)
  ```

#### calculate_adaptive_modifier()
计算自适应调节因子，基于性能趋势提供探索奖励
- **输入参数**：throughput、latency
- **返回值**：adaptive_modifier（调节因子，0.85-1.15范围）
- **调节逻辑**：
  - 性能提升 (trend > 5%)：1 + exploration_bonus
  - 性能下降 (trend < -5%)：1 - exploration_bonus × 0.5
  - 性能稳定：1.0

#### calculate_stability_reward() (新增组件)
计算长期稳定性奖励，基于rollout窗口内的性能一致性
- **输入参数**：无(内部维护性能历史)
- **返回值**：stability_reward（稳定性奖励，-0.4到+0.8范围）
- **计算逻辑**：
  ```
  performance_variation = std(performance_history) / rollout_len
  if variation < threshold:
      stability_reward = +stability_weight * consistency_bonus
  else:
      stability_reward = -stability_weight * inconsistency_penalty
  ```

#### set_ppo_action() & calculate_expert_reward() (新增组件)
设置PPO动作并计算专家指导奖励，基于与True Round Robin的一致性
- **输入参数**：ppo_action（PPO选择的replica）、current_state（当前状态）
- **返回值**：expert_reward（专家奖励，±0.5）
- **计算逻辑**：
  ```
  expert_action = request_counter % num_replicas  // Round Robin逻辑
  if ppo_action == expert_action:
      expert_reward = +expert_guidance_weight  // +0.5
  else:
      expert_reward = -expert_guidance_weight  // -0.5
  ```

## 配置参数详细说明

### 核心权重配置
```json
{
  "simplified_reward_performance_weight": 1.0,    // 性能评分权重
  "simplified_reward_imbalance_weight": 1.2,     // 负载失衡惩罚权重
  "simplified_reward_latency_weight": 0.6,       // 延迟惩罚权重
  "simplified_reward_exploration_bonus": 0.15,   // 探索奖励系数
  "simplified_reward_use_delta": true,           // 启用改进信号
  "simplified_reward_delta_weight": 0.4          // 改进信号权重
}
```

### 基线配置 (更新为当前值)
```json
{
  "reward_baseline": {
    "enable": true,              // 启用基线调整
    "baseline_value": 2.0,       // 基线偏移值 (更新值)
    "adaptive_baseline": false,  // 自适应基线
    "baseline_update_rate": 0.01, // 基线更新率
    "target_reward_mean": 0.0    // 目标奖励均值
  }
}
```

### 目标和阈值配置 (更新为实际scale)
```json
{
  "throughput_target": 3.5,      // 吞吐量目标值 (匹配实际~3.0)
  "latency_target": 8.0,         // 延迟目标值 (匹配实际~9.x)
  "latency_threshold": 10.0       // 延迟软阈值 (realistic threshold)
}
```

### 稳定性奖励配置
```json
{
  "stability_reward": {
    "enable_stability_reward": true,
    "rollout_len": 256,
    "stability_weight": 0.8,
    "stability_threshold": 0.2,
    "max_stability_bonus": 1.0,
    "max_stability_penalty": 0.5
  }
}
```

### 专家指导配置 (新增组件)
```json
{
  "expert_guidance": {
    "enable_expert_guidance": true,
    "expert_guidance_weight": 0.5,
    "expert_num_replicas": 4
  }
}
```

## 奖励计算完整公式

```
Final Reward = baseline + soft_scaling(
    adaptive_modifier × (
        performance_weight × performance_score
        - imbalance_weight × imbalance_penalty
        - latency_weight × latency_penalty
    ) + stability_reward + expert_reward
)

其中:
- baseline = 2.0 (默认配置)
- adaptive_modifier = 0.925-1.15 (基于性能趋势)
- stability_reward = ±0.8 (长期稳定性奖励)
- expert_reward = ±0.5 (Round Robin专家指导)
```

### 组件Scale分析 (基于当前配置)

#### 各组件数值范围和实际影响力

| 组件 | 权重 | 典型范围 | 实际Scale | 影响力排序 |
|------|------|----------|-----------|------------|
| **Performance Score** | 1.0 | 0.8-1.2 | 0.8-1.2 | 4 |
| **Imbalance Penalty** | 1.2 | 0.1-2.0 | 0.12-2.4 | **1 (最高)** |
| **Latency Penalty** | 0.6 | 1.0-4.0 | 0.6-2.4 | 3 |
| **Stability Reward** | 0.8 | ±0.5-1.0 | ±0.4-0.8 | 5 |
| **Expert Guidance** | 0.5 | ±0.5 | ±0.5 | 6 |
| **Baseline** | - | +2.0 | +2.0 | **2 (固定偏移)** |
| **Adaptive Modifier** | - | 0.925-1.15 | 7.5%变化 | 7 (乘法效应) |

#### 典型奖励计算示例

**正常运行场景**:
```
performance_score = 0.9 × 1.0 = +0.9
imbalance_penalty = 0.3 × 1.2 = -0.36
latency_penalty = 1.1 × 0.6 = -0.66
adaptive_modifier = 1.05 (轻微改进)
stability_reward = +0.2
expert_reward = +0.5 (与Round Robin一致)

raw_reward = 1.05 × (0.9 - 0.36 - 0.66) + 0.2 + 0.5 = +0.574
final_reward = 0.574 + 2.0 = +2.574
```

**极端不平衡场景**:
```
performance_score = 0.7 × 1.0 = +0.7
imbalance_penalty = 1.5 × 1.2 = -1.8 (高CV惩罚!)
latency_penalty = 2.0 × 0.6 = -1.2
adaptive_modifier = 0.925 (性能下降)
stability_reward = -0.3
expert_reward = -0.5 (与Round Robin不一致)

raw_reward = 0.925 × (0.7 - 1.8 - 1.2) + (-0.3) + (-0.5) = -2.93
final_reward = -2.93 + 2.0 = -0.93
```

## 使用示例和最佳实践

### 基本使用 (更新为当前配置)
```python
# 在PPO scheduler中创建SimplifiedRewardCalculator
reward_calc = SimplifiedRewardCalculator(
    # 核心权重
    performance_weight=1.0,
    imbalance_weight=1.2,
    latency_weight=0.6,
    exploration_bonus=0.15,
    # 更新后的目标值
    throughput_target=3.5,
    latency_target=8.0,
    latency_threshold=10.0,
    # 改进信号
    use_delta=True,
    delta_weight=0.4,
    # 基线配置
    baseline_enable=True,
    baseline_value=2.0,
    # 稳定性奖励
    enable_stability_reward=True,
    stability_weight=0.8,
    rollout_len=256,
    stability_threshold=0.2,
    # 专家指导
    enable_expert_guidance=True,
    expert_guidance_weight=0.5,
    expert_num_replicas=4
)

# 设置PPO action为专家指导(新增接口)
reward_calc.set_ppo_action(action=selected_replica_id, current_state=state_tensor)

# 计算奖励
reward, info = reward_calc.calculate_reward(
    metric_store, current_time, replica_ids, get_replica_scheduler_fn
)
```

### 权重调优最佳实践
1. **负载均衡优先**：imbalance_weight > performance_weight
2. **稳定性重要**：启用baseline避免持续负奖励
3. **探索平衡**：exploration_bonus保持在0.1-0.2范围
4. **改进激励**：use_delta=true，delta_weight=0.3-0.5

### 监控指标 (更新包含新组件)
```python
# 关键监控指标
info_dict = {
    # 核心组件
    "performance_score": float,      # 性能评分
    "imbalance_penalty": float,      # 失衡惩罚
    "latency_penalty": float,        # 延迟惩罚
    "adaptive_modifier": float,      # 自适应调节因子

    # 新增组件
    "stability_reward": float,       # 稳定性奖励 (±0.8)
    "expert_reward": float,          # 专家指导奖励 (±0.5)

    # 综合指标
    "raw_reward": float,            # 原始奖励值
    "scaled_reward": float,         # 软缩放后奖励
    "baseline": float,              # 基线值 (2.0)
    "total_reward": float,          # 最终奖励值

    # 稳定性相关指标
    "stability_performance_variance": float,  # 性能方差
    "stability_rollout_consistency": float,   # rollout一致性

    # 专家指导相关指标
    "expert_ppo_action": int,        # PPO选择的action
    "expert_expert_action": int,     # Round Robin专家action
    "expert_expert_agreement": bool, # 是否一致
    "expert_agreement_rate": float,  # 一致性率(统计)

    # 原有指标
    "throughput": float,            # 当前吞吐量
    "latency": float,               # 当前延迟
    "step_count": int               # 步数计数
}
```

## 常见问题和排查指南

### Q1: 奖励值持续为负，影响训练稳定性
**解决方案**：启用baseline调整
```json
{
  "reward_baseline": {
    "enable": true,
    "baseline_value": 5.0
  }
}
```

### Q2: 负载均衡效果不佳，副本间差异大
**解决方案**：增加imbalance_weight权重
```json
{
  "simplified_reward_imbalance_weight": 1.5  // 从1.2提升到1.5
}
```

### Q3: 延迟控制不理想，频繁超过阈值
**解决方案**：调整延迟相关参数
```json
{
  "simplified_reward_latency_weight": 0.8,  // 增加延迟惩罚权重
  "latency_threshold": 1.2                  // 降低延迟阈值
}
```

### Q4: 训练缺乏探索，策略过于保守
**解决方案**：增加探索奖励
```json
{
  "simplified_reward_exploration_bonus": 0.2  // 从0.15提升到0.2
}
```

### Q5: 性能改进缓慢，缺乏激励机制
**解决方案**：调整改进信号权重
```json
{
  "simplified_reward_use_delta": true,
  "simplified_reward_delta_weight": 0.5    // 从0.4提升到0.5
}
```

### Q6: 训练不稳定，奖励曲线振荡剧烈 (新增)
**解决方案**：启用稳定性奖励组件
```json
{
  "stability_reward": {
    "enable_stability_reward": true,
    "stability_weight": 1.0,        // 增加稳定性权重
    "stability_threshold": 0.15     // 降低稳定性阈值
  }
}
```

### Q7: PPO策略与理想负载均衡差异较大 (新增)
**解决方案**：启用Round Robin专家指导
```json
{
  "expert_guidance": {
    "enable_expert_guidance": true,
    "expert_guidance_weight": 0.7   // 增加专家指导权重
  }
}
```

### Q8: 奖励值过小或过大，影响PPO学习
**解决方案**：调整基线值
```json
{
  "reward_baseline": {
    "enable": true,
    "baseline_value": 1.5,          // 从2.0调整到1.5(降低) 或3.0(提高)
    "adaptive_baseline": true       // 启用自适应基线
  }
}
```

## 集成指南

### 与PPO Scheduler集成
```python
# 在ppo_scheduler_modular.py中的集成模式
if self._enable_simplified_reward:
    self._reward_calc = SimplifiedRewardCalculator(
        performance_weight=float(getattr(gcfg, 'simplified_reward_performance_weight', 1.0)),
        imbalance_weight=float(getattr(gcfg, 'simplified_reward_imbalance_weight', 1.0)),
        latency_weight=float(getattr(gcfg, 'simplified_reward_latency_weight', 0.5)),
        exploration_bonus=float(getattr(gcfg, 'simplified_reward_exploration_bonus', 0.1)),
        throughput_target=self._throughput_target,
        latency_target=self._latency_threshold / 2.0,
        latency_threshold=self._latency_threshold,
        use_delta=bool(getattr(gcfg, 'simplified_reward_use_delta', True)),
        delta_weight=float(getattr(gcfg, 'simplified_reward_delta_weight', 0.3)),
        baseline_enable=bool(getattr(gcfg, 'reward_baseline_enable', False)),
        baseline_value=float(getattr(gcfg, 'reward_baseline_value', 0.0)),
    )
```

## 性能和优化建议

### 计算复杂度
- **时间复杂度**：O(n)，n为副本数量
- **空间复杂度**：O(k)，k为历史窗口大小
- **计算频率**：每个PPO step调用一次

### 优化建议
1. **历史窗口大小**：保持在10-50范围，避免过大影响性能
2. **调试输出频率**：每200步输出一次，减少日志噪声
3. **软缩放函数**：避免硬截断，保持梯度连续性
4. **EMA更新**：使用指数移动平均减少计算开销

## 版本兼容性

- **当前版本**：SimplifiedRewardCalculator v2.0
- **兼容性**：向后兼容RewardCalculator接口
- **迁移路径**：通过enable_simplified_reward配置开关
- **废弃组件**：原RewardCalculator已移至archieve目录

## 相关模块

- **PPO Scheduler**：主要调用方，负责奖励计算集成
- **Metrics Store**：提供throughput和latency指标
- **Memory Manager**：配合奖励计算的内存优化
- **TensorBoard Monitor**：奖励组件可视化监控