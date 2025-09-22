# PPO参数优化日志 - 延迟敏感度增强

## 优化目标
将PPO策略性能向RoundRobin基线靠拢，重点解决PPO倾向于牺牲延迟换取吞吐量的问题。

## 核心调整 (configs/ppo_warmstart.json)

### 1. 奖励函数 - 加强延迟敏感度
```json
"reward_config": {
  "latency_weight": 1.2 → 1.7,           // 提升延迟权重接近RoundRobin曲线
  "latency_penalty_scale": 0.5 → 0.85,   // 加强尾延迟惩罚
  "throughput_target": 2.0 → 1.85        // 降低吞吐量目标，更贴近最佳基线1.918
}
```

**理由**: 原配置允许策略为高吞吐量容忍更高延迟。新配置明确延迟优先级，强制策略将尾延迟控制作为首要目标。

### 2. 探索控制 - 收紧策略搜索
```json
"ppo_config": {
  "entropy_coef": 0.2 → 0.05,           // 大幅降低探索强度
  "entropy_warmup_coef": 0.1 → 0.02     // 相应降低预热阶段探索
}
```

**理由**: 外部预训练已提供良好初值，过高的熵系数会让策略在训练早期重新散开，偏离最优区域。

### 3. KL约束 - 防止过度偏移
```json
"kl_regularization": {
  "target_kl": 0.01 → 0.02,             // 放宽KL约束避免过早停止
  "entropy_min": 0.5 → 0.3,             // 降低最小熵要求
  "kl_coef": 0.2 → 0.15,                // 减弱KL惩罚强度
  "kl_ref_decay_steps": 2000 → 5000     // 延长KL衰减周期
}
```

**理由**: 原target_kl=0.01过于严格，导致训练早期即触发early stopping。新配置允许策略在更大空间内优化，然后逐步收紧。

### 4. 温度控制 - 降低过热风险
```json
"temperature_control": {
  "base_temperature": 1.5 → 1.0,        // 降低基础温度
  "max_temperature": 3.0 → 2.0,         // 限制最大温度
  "qps_sensitivity": 0.1 → 0.05,        // 降低QPS敏感度
  "latency_sensitivity": 0.2 → 0.1      // 降低延迟敏感度
}
```

**理由**: 防止动态温度控制让策略过热，导致不稳定的探索行为。

### 5. 训练规模 - 延长学习周期
```json
"training": {
  "ppo_requests": 2000 → 20000          // 大幅增加训练数据量
}

// scripts/train_ppo_warmstart.sh 默认值同步更新
PPO_REQUESTS=8000 → 20000
QPS=3.0 → 2.5
```

**理由**: 更多训练数据有助于策略收敛到稳定的最优解，初始QPS降低以渐进式增加负载压力。

## 预期效果

### 短期改善
- **延迟分布收紧**: P95/P99延迟应明显降低
- **策略稳定性**: 减少动作分布的剧烈波动
- **收敛速度**: KL约束放宽后训练更充分

### 长期目标
- **追平RoundRobin**: 在相同QPS下实现相近的延迟性能
- **负载适应**: 能在更高QPS下维持合理的延迟水平
- **策略鲁棒**: 对不同负载模式表现一致

## 监控重点

### 关键指标
1. **Advantage统计**: 均值应趋于0，方差逐步收敛
2. **Value Loss**: 避免爆炸，保持在合理范围
3. **Policy Distribution**: 监控是否出现长期偏向某个replica
4. **KL/Entropy曲线**: 确认在新约束下的演化轨迹

### 异常检测
- Value loss突然升高 → 考虑降低critic hidden size
- 策略分布极度不均 → 检查load balance penalty
- KL频繁超限 → 进一步放宽target_kl到0.025

## 后续实验

### 阶段1: 验证基础效果
```bash
bash scripts/train_ppo_warmstart.sh --disable-stats-stabilization  # 对照组
bash scripts/train_ppo_warmstart.sh                               # 优化组
```

### 阶段2: Curriculum Learning (如需要)
逐步调整QPS: 2.5 → 3.5 → 4.5，观察策略在增压下的适应能力。

### 阶段3: 示教数据增强 (如效果仍不足)
```bash
python scripts/collect_demo.py --qps 2.5 --policies "round_robin,lor_hybrid"
# 然后取消 --skip-bc-training 进行精细微调
```

## 配置文件完整性

所有相关配置已同步更新：
- ✅ `configs/ppo_warmstart.json`: 核心参数优化
- ✅ `configs/standalone_pretrain.json`: 保持预训练一致性
- ✅ `scripts/train_ppo_warmstart.sh`: 默认值同步

## 回退方案

如优化效果不佳，可使用以下命令恢复保守配置：
```bash
git checkout HEAD~1 configs/ppo_warmstart.json  # 恢复原配置
# 或手动调整关键参数：
# latency_weight: 1.7 → 1.4
# entropy_coef: 0.05 → 0.1
# target_kl: 0.02 → 0.015
```