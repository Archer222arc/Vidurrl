# 🔥 负载均衡优化技术文档

## 📋 问题背景

PPO训练初期存在严重的负载不均衡问题：
- **动作分布极度倾斜**: 73-77%的请求被分配到单个副本
- **恢复时间过长**: 从不均衡状态恢复需要很长时间
- **训练不稳定**: 初期随机性导致策略频繁振荡

## 🎯 优化策略

### 1. 压制Warmup期间的随机性

**问题**: `entropy_warmup_coef=0.4` + `entropy_min=0.3` + 高温度基线导致warmup期间策略做出过多随机决策

**解决方案**:
```bash
# 原始参数
entropy_warmup_coef=0.4    # ❌ 增加探索性
min_temperature=0.7        # ❌ 温度过高

# 优化参数
entropy_warmup_coef=0.0    # ✅ 消除warmup随机性
min_temperature=0.5        # ✅ 降低最低温度
```

**原理**: 让warmup期间更接近BC策略的deterministic行为，避免瞬时负载倾斜。

### 2. 强化KL正则化约束

**问题**: `kl_ref_coef_initial=0.4`, `kl_ref_decay_steps=1500` 对拉回示教策略的力度偏弱

**解决方案**:
```bash
# 原始参数
kl_ref_coef_initial=0.4    # ❌ 约束偏弱
kl_ref_coef_final=0.05     # ❌ 终值过低
kl_ref_decay_steps=1500    # ❌ 衰减过快

# 优化参数
kl_ref_coef_initial=0.6    # ✅ 强化初始约束
kl_ref_coef_final=0.1      # ✅ 保持适度终值
kl_ref_decay_steps=3000    # ✅ 延长约束期
```

**原理**: 确保PPO在前几千步几乎"贴着"示教策略走，从而保持负载均衡。

### 3. 加强负载均衡惩罚

**问题**: 负载倾斜惩罚在奖励中占比过小，被throughput项压制

**解决方案**:
```bash
# 原始参数
balance_penalty_weight=0.1    # ❌ 惩罚权重偏低
load_balance_penalty=0.2      # ❌ 负载惩罚偏低
alpha=0.3                     # ❌ throughput权重过高

# 优化参数
balance_penalty_weight=0.3    # ✅ 提高惩罚权重
load_balance_penalty=0.3      # ✅ 加强负载惩罚
alpha=0.2                     # ✅ 降低throughput压制
```

**原理**: 避免throughput项压制均衡惩罚，让模型更重视负载分布。

### 4. 增强示教数据多样性

**问题**: 示教数据只覆盖均匀负载场景，缺乏极端不均衡的纠偏样本

**解决方案**:
```python
# 新增不均衡场景数据收集
imbalanced_scenarios = [
    {"qps": qps * 2.0, "suffix": "high_load"},    # 高负载场景
    {"qps": qps * 0.5, "suffix": "low_load"},     # 低负载场景
]
```

**原理**: 让BC模型预先学会应对极端情况，提高负载纠偏能力。

## 🏗️ 技术实现

### 模块化架构改进

遵循CLAUDE.md规范，将复杂逻辑分离到独立模块：

```python
# src/demo_collection/mixed_collector.py
class MixedDemoDataProcessor:
    def merge_policy_data(self) -> None:
        """合并多策略示教数据"""
        # 处理常规策略数据
        for policy in self.policies:
            # ...

        # 处理不均衡场景数据
        imbalanced_files = self.temp_dir.glob('*_imbalanced_*.pkl')
        # ...
```

### Resume功能实现

智能checkpoint恢复，自动跳过warmstart：

```bash
# 交互式resume控制
if [ -f "${LATEST_CHECKPOINT}" ]; then
    echo "🔄 发现existing checkpoint: ${LATEST_CHECKPOINT}"
    read -p "请选择 [y/n/q]: " choice
    case $choice in
        [Yy]*) SKIP_WARMSTART=true ;;
        [Nn]*) SKIP_WARMSTART=false ;;
        [Qq]*) exit 0 ;;
    esac
fi

# 条件执行warmstart
if [ "$SKIP_WARMSTART" = false ]; then
    # 执行示教数据收集和BC预训练
else
    echo "⏭️ 跳过warmstart阶段 (从checkpoint恢复)"
fi
```

## 📊 效果验证

### 关键监控指标

在TensorBoard中重点观察：

1. **KL散度轨迹**:
   - 前3000步应保持较高值(0.6→0.1)
   - 验证策略是否贴近示教行为

2. **Entropy变化**:
   - Warmup期间应保持低值
   - 避免过度探索导致倾斜

3. **队列方差**:
   - 负载均衡的直接指标
   - 观察收敛速度和稳定性

4. **动作分布**:
   - 各副本选择频率应趋于均匀
   - 避免单副本主导现象

### 预期改进效果

- ✅ **Warmup阶段稳定性**: 减少随机决策导致的负载倾斜
- ✅ **更快收敛**: 强化KL约束确保策略不偏离示教行为
- ✅ **更强均衡意识**: 提升的惩罚权重让模型重视负载分布
- ✅ **场景泛化能力**: 不均衡场景数据提高极端情况应对能力

## 🔧 参数调优建议

如果优化效果仍不理想，可进一步调整：

### 进一步压制随机性
```bash
entropy_warmup_coef=-0.1     # 负值进一步压制
entropy_min=0.2              # 更低的最小熵值
```

### 更强KL约束
```bash
kl_ref_coef_initial=0.8      # 更强的初始约束
kl_ref_decay_steps=5000      # 更长的约束期
```

### 更激进的负载惩罚
```bash
balance_penalty_weight=0.5   # 更高的惩罚权重
alpha=0.15                   # 进一步降低throughput权重
```

### 添加稳定化期
```python
# 可在PPO主循环中添加
if self._step < stabilization_steps:
    # 前N步只做inference，不更新梯度
    # 直接使用预训练actor输出
```

## 📚 相关文档

- [训练脚本使用说明](../scripts/train_ppo_warmstart_optimized.sh)
- [调度器对比测试](../scripts/scheduler_comparison.sh)
- [项目规范文档](../.claude/CLAUDE.md)
- [CheckpointManager实现](../src/rl_components/checkpoint_manager.py)