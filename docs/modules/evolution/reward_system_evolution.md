# 奖励系统模块演进史

## 📚 历史文档索引
*（未来版本的历史文档将存储在archive/目录中）*

## 当前活跃版本 (2025年)

### [2025-09-29] - 奖励系统完整重构文档更新（最新版本）

#### 变更概述
- **改动原因**：基于实际代码实现更新文档，包含准确的组件scale分析、新增stability reward和expert guidance组件、修正配置参数值
- **影响范围**：`docs/modules/reward_system.md`完整重写、`docs/modules/expert_guidance_evolution.md`更新
- **向后兼容性**：文档更新，代码无变更，完全兼容现有实现

#### 具体变更

1. **Scale分析修正**：
   - **修正目标值**：throughput_target: 10.0→3.5, latency_target: 1.0→8.0, latency_threshold: 1.5→10.0（匹配实际运行scale）
   - **修正基线值**：baseline_value: 5.0→2.0（匹配当前配置）
   - **权重影响力重新计算**：
     - Imbalance Penalty: 1.2×(0.1-2.0) = 0.12-2.4（最高影响力）
     - Baseline: +2.0（第二大固定影响）
     - Latency Penalty: 0.6×(1.0-4.0) = 0.6-2.4
     - Performance Score: 1.0×(0.8-1.2) = 0.8-1.2

2. **新增组件文档**：
   - **Stability Reward组件**：
     - 范围：±0.4-0.8（stability_weight=0.8）
     - 功能：长期rollout一致性奖励
     - 配置：rollout_len=256, stability_threshold=0.2
   - **Expert Guidance组件**：
     - 范围：±0.5（expert_guidance_weight=0.5）
     - 功能：True Round Robin专家指导
     - 配置：enable_expert_guidance=true, expert_num_replicas=4

3. **公式更新**：
   ```
   Final Reward = baseline + soft_scaling(
       adaptive_modifier × (
           performance_weight × performance_score
           - imbalance_weight × imbalance_penalty
           - latency_weight × latency_penalty
       ) + stability_reward + expert_reward
   )
   ```

4. **监控指标扩展**：
   - 新增stability相关指标：performance_variance, rollout_consistency
   - 新增expert指导指标：expert_ppo_action, expert_expert_action, expert_agreement_rate
   - 完整breakdown包含所有组件：raw_reward, scaled_reward, baseline, total_reward

5. **典型场景示例**：
   - **正常运行**：final_reward = +2.574
   - **极端不平衡**：final_reward = -0.93
   - 体现了负载均衡权重主导的设计理念

#### 技术亮点

1. **模块化架构**：
   - Stability Reward：独立的RolloutAlignedStabilityReward组件
   - Expert Guidance：独立的TrueRoundRobinExpertReward组件
   - 接口统一：set_ppo_action()提供状态感知的专家指导

2. **配置驱动**：
   - 所有新组件通过JSON配置启用/禁用
   - 权重参数完全可配置
   - 支持运行时统计信息获取

3. **troubleshooting增强**：
   - 新增Q6-Q8常见问题解决方案
   - 涵盖训练不稳定、专家指导偏差、奖励scale调整

#### 相关文件更新
- `docs/modules/reward_system.md:1-280`：完整模块接口文档重写
- `docs/modules/expert_guidance_evolution.md:1-170`：增加True Round Robin实现记录

---

### [2025-09-29] - 奖励组件权重分析与文档化（初版）

#### 变更概述
- **改动原因**：系统化分析当前奖励系统组件权重配置，解决training过程中reward组件不明确的问题
- **影响范围**：SimplifiedRewardCalculator、PPO Scheduler配置、奖励计算权重分析
- **向后兼容性**：完全兼容，仅增加文档和分析，未修改代码逻辑

#### 具体变更

1. **新增功能**：
   - **奖励组件权重分析**：完整分析当前5个核心组件的权重分布
   - **配置参数映射**：建立JSON配置到reward计算的完整映射关系
   - **权重比例计算**：量化各组件在最终奖励中的相对影响力
   - **文件位置**：`docs/modules/reward_system.md` - 完整模块接口文档

2. **分析结果**：
   - **SimplifiedRewardCalculator组件权重分布**：
     - Performance Score: 36.2% (正向奖励)
     - Imbalance Penalty: 43.5% (负向惩罚，权重最高)
     - Latency Penalty: 21.7% (负向惩罚)
     - Adaptive Modifier: ±15% (动态调节)
     - Baseline: +5.0 (固定偏移)

3. **文档化成果**：
   - **模块接口文档**：`docs/modules/reward_system.md`
   - **配置参数说明**：完整的JSON配置到Python参数映射
   - **使用示例**：最佳实践和常见问题解决方案
   - **权重调优指南**：基于组件分析的调优建议

#### 关键发现

1. **权重配置特点**：
   - **负载均衡优先**：imbalance_weight (1.2) > performance_weight (1.0)
   - **惩罚主导策略**：惩罚权重(65.1%) > 奖励权重(36.2%)
   - **稳定性保障**：baseline (+5.0) 确保训练稳定

2. **奖励计算公式**：
   ```
   Final Reward = 5.0 + soft_scaling(
       adaptive_modifier × (
           1.0 × performance_score
           - 1.2 × imbalance_penalty
           - 0.6 × latency_penalty
       )
   )
   ```

3. **设计哲学分析**（已更新）：
   - **"负载均衡优先"**策略：imbalance_penalty权重(1.2)最高
   - **稳定性保障**：baseline(+2.0)提供训练稳定性
   - **多组件协同**：5个独立组件(performance, imbalance, latency, stability, expert)协同工作
   - **Delta机制(40%)**鼓励持续改进

#### 配置参数完整映射

**JSON配置 → SimplifiedRewardCalculator参数**：
```json
{
  "simplified_reward_performance_weight": 1.0,    → performance_weight
  "simplified_reward_imbalance_weight": 1.2,     → imbalance_weight
  "simplified_reward_latency_weight": 0.6,       → latency_weight
  "simplified_reward_exploration_bonus": 0.15,   → exploration_bonus
  "simplified_reward_use_delta": true,           → use_delta
  "simplified_reward_delta_weight": 0.4,         → delta_weight
  "reward_baseline": {
    "enable": true,                              → baseline_enable
    "baseline_value": 5.0                       → baseline_value
  },
  "throughput_target": 10.0,                     → throughput_target
  "latency_threshold": 1.5                       → latency_threshold
}
```

#### 测试验证
- [x] 权重计算验证通过
- [x] 配置参数映射确认无误
- [x] 文档结构符合CLAUDE.md规范
- [x] 组件分析逻辑验证正确

#### 相关Issue/PR
- 分析需求：用户询问当前reward组件权重分布
- 文档需求：按照CLAUDE.md规范创建系统文档

---

### [历史版本占位符]
*（未来的版本变更将记录在此处，当记录超过10个版本时将移至archive目录）*

## 废弃功能参考

### 已废弃的RewardCalculator
- **废弃时间**：2025年初
- **废弃原因**：复杂度过高，参数冗余，维护困难
- **替代方案**：SimplifiedRewardCalculator统一管理
- **迁移路径**：通过`enable_simplified_reward: true`配置开关
- **保留位置**：`archieve/src/core/algorithms/rewards/reward_calculator.py`

### 多重惩罚系统简化
- **原系统**：3个独立的负载均衡惩罚 + 2个独立的延迟惩罚
- **简化后**：1个统一失衡惩罚 + 1个统一延迟惩罚
- **简化效果**：参数从20+个减少到6个核心参数
- **性能提升**：计算复杂度从O(n²)降至O(n)

### 探索机制整合
- **原系统**：5个独立的探索奖励项
- **整合后**：1个adaptive_modifier统一管理
- **整合优势**：避免探索奖励冲突，提供一致的探索信号

## 技术债务和改进计划

### 当前技术债务
1. **硬编码阈值**：CV惩罚阶段的0.3和0.7阈值应配置化
2. **历史窗口管理**：performance_history大小固定为50，应可配置
3. **软缩放参数**：scale_factor固定为1.0，缺乏自适应机制

### 计划改进
1. **配置化阈值**：将CV阶段阈值加入JSON配置
2. **自适应窗口**：基于训练阶段动态调整历史窗口大小
3. **智能缩放**：基于奖励分布自动调整软缩放参数

### 监控需求
1. **组件权重监控**：实时监控各组件对最终奖励的贡献度
2. **失衡检测**：当imbalance_penalty持续高于阈值时触发告警
3. **探索效率**：监控adaptive_modifier的变化趋势和探索效果