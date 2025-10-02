# Toy Model: PPO路由策略验证方案

## 1. 问题设定 (Problem Setup)

### 1.1 系统配置
- **Replica数量**: 2个 (replica 1, replica 2)
- **请求类型**: 2种 (type A, type B)
- **队列模型**: M/M/1 queuing system per replica

### 1.2 处理时间分布
服务时间遵循指数分布 Exponential(μ),其中μ为处理速率(rate parameter):

| Request Type | Replica 1 | Replica 2 |
|--------------|-----------|-----------|
| Type A       | Exp(μ₁)   | Exp(μ₂)   |
| Type B       | Exp(μ₂)   | Exp(μ₁)   |

**约束条件**: μ₁ > μ₂ (μ₁处理更快)

**最优路由逻辑**:
- Type A → Replica 1 (更快)
- Type B → Replica 2 (更快)

### 1.3 到达过程
- Type A: Poisson arrival with rate λ_A
- Type B: Poisson arrival with rate λ_B
- 独立到达,互不影响

### 1.4 验证目标
**核心问题**: PPO能否学习到接近或优于显式最优策略的路由决策?

**预期结果**:
- PPO学习到的策略应收敛到"Type A→Replica 1, Type B→Replica 2"
- 性能指标应优于random和round-robin baseline

---

## 2. 评估指标 (Evaluation Metrics)

参考Vidur现有指标体系,关注以下核心指标:

### 2.1 延迟指标
- **平均端到端延迟 (Mean E2E Latency)**
  - 从请求到达到完成的总时间
  - 包含排队时间 + 服务时间

- **分位数延迟 (P50/P90/P99 Latency)**
  - 评估尾延迟表现

- **平均排队时间 (Mean Queuing Time)**
  - 仅统计在队列中的等待时间

### 2.2 吞吐量指标
- **系统吞吐量 (System Throughput)**
  - 单位时间完成的请求数

- **利用率 (Utilization)**
  - 每个replica的平均负载率
  - ρ = λ/μ (M/M/1队列稳定性要求 ρ < 1)

### 2.3 路由质量指标
- **路由准确率 (Routing Accuracy)**
  - Type A正确路由到Replica 1的比例
  - Type B正确路由到Replica 2的比例

- **负载均衡度 (Load Balance)**
  - 两个replica处理请求数的方差
  - 评估是否存在严重负载倾斜

### 2.4 训练指标
- **Reward曲线**: PPO训练过程的累积奖励
- **Loss曲线**: Actor loss, Critic loss, Entropy
- **收敛速度**: 达到稳定策略的训练步数

---

## 3. Baseline对比策略

### 3.1 Oracle (理论最优)
- **策略**: Type A → Replica 1, Type B → Replica 2
- **作用**: 理论性能上界,验证PPO是否接近最优

### 3.2 Random Routing
- **策略**: 每个请求随机分配到replica 1或replica 2 (50%概率)
- **作用**: 最简单的无状态策略baseline

### 3.3 Round-Robin
- **策略**: 按到达顺序轮流分配到replica 1和replica 2
- **作用**: 常见的负载均衡策略

### 3.4 Type-Agnostic PPO
- **策略**: PPO训练时隐藏请求类型信息,仅基于队列状态决策
- **作用**: 验证请求类型特征的重要性

---

## 4. 实验设计

### 4.1 参数配置

#### 系统参数
```yaml
replicas: 2
request_types: 2

# 处理速率 (requests/second)
service_rates:
  replica_1:
    type_A: 10.0  # μ₁
    type_B: 5.0   # μ₂
  replica_2:
    type_A: 5.0   # μ₂
    type_B: 10.0  # μ₁

# 到达速率 (requests/second)
arrival_rates:
  type_A: 6.0   # λ_A
  type_B: 6.0   # λ_B
```

#### PPO超参数
```yaml
learning_rate: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
value_coef: 0.5
entropy_coef: 0.01
batch_size: 64
epochs_per_update: 10
training_steps: 100000
```

### 4.2 状态空间设计

```python
state = {
    'replica_1_queue_length': int,      # replica 1当前队列长度
    'replica_2_queue_length': int,      # replica 2当前队列长度
    'replica_1_utilization': float,     # replica 1利用率 [0,1]
    'replica_2_utilization': float,     # replica 2利用率 [0,1]
    'current_request_type': int,        # 当前请求类型 {0: A, 1: B}
    'time_since_last_arrival': float,   # 距离上次到达的时间间隔
}
```

### 4.3 动作空间设计

```python
action = {
    0: 'route_to_replica_1',  # 路由到replica 1
    1: 'route_to_replica_2',  # 路由到replica 2
}
```

### 4.4 奖励函数设计

```python
# 基于完成时间的负奖励
reward = -completion_time

# 或基于排队时间和服务时间的加权组合
reward = -(α * queuing_time + β * service_time)

# 可选: 添加路由准确率bonus
if (request_type == A and action == replica_1) or \
   (request_type == B and action == replica_2):
    reward += routing_bonus
```

### 4.5 实验场景

#### Scenario 1: 均衡负载 (Balanced Load)
- λ_A = 6.0, λ_B = 6.0
- 总到达率 = 12.0 < μ₁ + μ₂ = 15.0 (系统稳定)
- 验证PPO在均衡负载下的学习能力

#### Scenario 2: 不均衡负载 (Imbalanced Load)
- λ_A = 8.0, λ_B = 4.0
- 测试PPO对不同到达率的适应性

#### Scenario 3: 高负载 (High Load)
- λ_A = 7.0, λ_B = 7.0
- 总到达率 = 14.0 接近系统容量
- 验证PPO在高负载下的鲁棒性

#### Scenario 4: 动态负载 (Dynamic Load)
- λ_A, λ_B 在训练过程中周期性变化
- 测试PPO的泛化能力

---

## 5. 实现计划

### 5.1 目录结构

```
Vidur_toymodel/
├── src/
│   ├── toymodel/
│   │   ├── __init__.py
│   │   ├── environment.py          # 仿真环境 (M/M/1 queue)
│   │   ├── request_generator.py    # Poisson arrival process
│   │   ├── replica.py              # Replica处理逻辑
│   │   ├── scheduler.py            # 路由策略 (PPO/Baseline)
│   │   └── metrics_collector.py    # 指标收集
│   ├── training/
│   │   ├── __init__.py
│   │   ├── ppo_trainer.py          # PPO训练器
│   │   └── baseline_trainer.py     # Baseline测试器
│   └── utils/
│       ├── __init__.py
│       └── visualization.py        # 结果可视化
├── configs/
│   ├── toymodel_balanced.yaml      # 均衡负载配置
│   ├── toymodel_imbalanced.yaml    # 不均衡负载配置
│   └── toymodel_high_load.yaml     # 高负载配置
├── scripts/
│   ├── train_toymodel_ppo.sh       # 训练PPO
│   ├── eval_baselines.sh           # 评估baseline
│   └── compare_results.sh          # 对比结果
├── experiments/
│   └── toymodel_ppo_routing.yaml   # 实验记录
├── outputs/
│   └── toymodel/
│       ├── checkpoints/            # 模型checkpoint
│       ├── metrics/                # 指标CSV
│       └── tensorboard/            # TensorBoard日志
└── docs/
    └── toymodel_ppo_routing_design.md  # 本文档
```

### 5.2 开发步骤

#### Phase 1: 环境搭建 (Week 1)
- [ ] 实现M/M/1 queue仿真环境
- [ ] 实现Poisson arrival请求生成器
- [ ] 实现基础replica处理逻辑
- [ ] 单元测试: 验证队列稳定性理论值

#### Phase 2: Baseline实现 (Week 1-2)
- [ ] 实现Oracle最优策略
- [ ] 实现Random routing
- [ ] 实现Round-robin
- [ ] 收集baseline性能指标

#### Phase 3: PPO集成 (Week 2-3)
- [ ] 设计状态空间和动作空间
- [ ] 实现奖励函数
- [ ] 集成现有PPO trainer
- [ ] 训练和调试

#### Phase 4: 实验评估 (Week 3-4)
- [ ] 运行所有场景实验
- [ ] 收集和分析指标数据
- [ ] 可视化对比结果
- [ ] 撰写实验报告

---

## 6. 预期结果

### 6.1 成功标准

**定量指标**:
- PPO平均延迟 ≤ Oracle延迟 × 1.1 (10%容忍度)
- PPO路由准确率 ≥ 90%
- PPO收敛步数 < 50000 steps

**定性指标**:
- PPO学习曲线平滑收敛
- 策略在不同负载场景下保持鲁棒
- 显著优于random和round-robin baseline

### 6.2 失败分析预案

**如果PPO未能收敛**:
- 检查奖励函数设计 (reward shaping)
- 调整超参数 (learning rate, clip epsilon)
- 增加状态特征 (如历史统计信息)

**如果PPO性能低于Oracle**:
- 分析路由决策分布
- 检查exploration vs exploitation平衡
- 考虑增加训练步数或调整entropy coefficient

**如果PPO仅在特定场景有效**:
- 增加场景多样性训练 (curriculum learning)
- 实现domain randomization
- 评估state representation的泛化性

---

## 7. 后续扩展方向

### 7.1 复杂化场景
- 增加replica数量 (3-5个)
- 增加请求类型 (多种SLO要求)
- 非指数分布服务时间 (Long-tail distribution)

### 7.2 高级策略
- Multi-agent PPO (每个replica一个agent)
- Hierarchical RL (全局+局部调度)
- Offline RL (从Vidur真实trace学习)

### 7.3 工程优化
- 批量路由决策 (batched routing)
- 在线学习和适应 (online adaptation)
- 集成到Vidur主框架

---

## 8. 参考文献

- [PPO算法] Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
- [M/M/1队列] Kleinrock, L. "Queueing Systems: Volume 1" (1975)
- [Load Balancing] Mitzenmacher, M. "The Power of Two Choices in Randomized Load Balancing" (2001)
- [Vidur项目] 现有Vidur codebase和文档

---

**文档版本**: v1.0
**创建日期**: 2025-10-01
**作者**: Claude Code + User
**状态**: 设计阶段
