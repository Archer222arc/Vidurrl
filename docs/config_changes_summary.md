# 配置变更摘要 - PPO优化

## 快速参考

本文档提供PPO调度器优化的配置变更快速参考，详细技术说明请参考 `ppo_optimization_2025_09_18.md`。

## 核心配置变更

### 奖励系统参数

| 参数名 | 原值 | 新值 | 说明 |
|--------|------|------|------|
| `throughput_target` | 1.0 | 0.05 | 降低目标使当前性能可获正奖励 |
| `alpha` | 0.5 | 0.1 | 大幅减少延迟惩罚权重 |
| `kappa` | 0.3 | 0.05 | 降低logistic惩罚强度 |
| `sigma` | 1.0 | 2.0 | 增加惩罚平滑度 |
| `absolute_weight` | 0.7 | 0.8 | 增强绝对奖励分量 |
| `delta_weight` | 0.3 | 0.2 | 减少delta分量(常为0) |

### 探索控制参数

| 参数名 | 原值 | 新值 | 说明 |
|--------|------|------|------|
| `entropy_coef` | 0.15 | 0.25 | 进一步提升探索强度 |
| `target_kl` | - | 0.01 | 新增KL散度监控阈值 |
| `entropy_min` | - | 0.5 | 新增熵下限保护 |
| `kl_coef` | - | 0.2 | 新增KL正则化系数 |

### 温度脉冲参数

| 参数名 | 原值 | 新值 | 说明 |
|--------|------|------|------|
| `pulse_interval` | 50 | 8 | 大幅提高脉冲频率 |
| `pulse_magnitude` | 1.5 | 2.0 | 增强脉冲强度 |
| `stagnation_threshold` | 0.01 | 0.005 | 更敏感的停滞检测 |
| `stagnation_memory` | 10 | 5 | 更快响应停滞状态 |

### 网络架构参数

| 参数名 | 原值 | 新值 | 说明 |
|--------|------|------|------|
| `enable_decoupled_ac` | False | True | 启用Actor-Critic解耦架构 |
| `feature_projection_dim` | - | 256 | 特征投影层维度 |

### 其他调整参数

| 参数名 | 原值 | 新值 | 说明 |
|--------|------|------|------|
| `latency_threshold` | 2.0 | 6.0 | 放宽延迟容忍度 |
| `latency_penalty_scale` | 5.0 | 0.5 | 减少延迟惩罚 |
| `load_balance_penalty` | 0.03 | 0.15 | 强化负载均衡 |

## 命令行参数映射

### 新增必需参数

```bash
--p_p_o_global_scheduler_modular_config_throughput_target 0.05
--p_p_o_global_scheduler_modular_config_absolute_weight 0.8
--p_p_o_global_scheduler_modular_config_delta_weight 0.2
--p_p_o_global_scheduler_modular_config_alpha 0.1
--p_p_o_global_scheduler_modular_config_kappa 0.05
--p_p_o_global_scheduler_modular_config_sigma 2.0
--p_p_o_global_scheduler_modular_config_target_kl 0.01
--p_p_o_global_scheduler_modular_config_entropy_min 0.5
--p_p_o_global_scheduler_modular_config_kl_coef 0.2
```

### 需要更新的现有参数

```bash
--p_p_o_global_scheduler_modular_config_entropy_coef 0.25          # 原值: 0.002
--p_p_o_global_scheduler_modular_config_latency_threshold 6.0      # 原值: 2.0
--p_p_o_global_scheduler_modular_config_latency_penalty_scale 0.5  # 原值: 5.0
--p_p_o_global_scheduler_modular_config_load_balance_penalty 0.15  # 原值: 0.03
```

## 预期效果指标

### 训练前后对比

| 指标 | 修复前 | 修复后(期望) |
|------|--------|-------------|
| `absolute_score` | -0.05 | +1.99 |
| `total_reward` | -0.094 | 正值区间 |
| 单副本动作占比 | 73-77% | ~25% |
| `temp_is_pulsing` | False | 定期True |
| `clipfrac` | 偏高 | <0.3 |

### 关键监控指标

1. **动作分布均匀性**: `action_replica_*` 各项应接近25%
2. **奖励健康度**: `absolute_score > 0`, `total_reward > 0`
3. **探索活跃度**: `entropy > 0.5`, 定期温度脉冲
4. **训练稳定性**: `clipfrac < 0.3`, `approx_kl < 0.01`

## 回滚方案

如遇问题，可通过以下步骤快速回滚：

1. **配置回滚**: 将上述参数改回原值
2. **代码回滚**:
   ```bash
   git checkout HEAD~1 -- src/rl_components/ppo_trainer.py
   git checkout HEAD~1 -- src/rl_components/temperature_controller.py
   ```
3. **验证**: 使用原始命令行参数重新训练

---

**更新时间**: 2025-09-18
**相关文档**: `ppo_optimization_2025_09_18.md`