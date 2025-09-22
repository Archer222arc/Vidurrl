# Vidur智能调度器 - 训练架构完整指南

## 概述

本文档详细描述了Vidur智能调度器项目中的训练系统架构，包括所有组件接口、状态定义、动作空间和训练流程。该系统采用强化学习(PPO)方法来优化LLM推理服务的请求调度策略。

## 🏗️ 核心架构组件

### 1. 训练组件层次结构

```
src/core/
├── models/                          # 神经网络模型
│   ├── actor_critic.py             # Actor-Critic网络架构
│   └── state_builder.py            # 状态构建器
├── algorithms/                      # 训练算法
│   ├── ppo_trainer.py              # PPO训练器
│   ├── rollout_buffer.py           # 经验回放缓冲区
│   └── rewards/
│       └── reward_calculator.py    # 奖励计算器
└── utils/                          # 工具组件
    ├── normalizers.py              # 状态标准化
    ├── temperature_controller.py   # 温度控制
    └── monitoring/                 # 监控系统
        ├── tensorboard_monitor.py  # TensorBoard集成
        └── metrics_exporter.py     # 指标导出
```

### 2. 调度器集成点

```
vidur/scheduler/global_scheduler/
└── ppo_scheduler_modular.py        # PPO调度器主入口
```

## 🧠 神经网络架构

### ActorCritic 模型 (`src/core/models/actor_critic.py`)

**功能**: 实现共享特征提取和分离的Actor-Critic网络架构

#### 网络组件
```python
# 特征投影层 (多维状态标准化)
feature_proj: nn.Sequential([
    nn.LayerNorm(state_dim),           # 输入标准化
    nn.Linear(state_dim, proj_dim),    # 投影到高维特征空间
    nn.GELU(),                         # 平滑激活函数
    nn.Linear(proj_dim, hidden_size)   # 降维到隐藏维度
])

# 共享MLP编码器 (可配置层数)
shared_mlp: nn.Sequential([
    nn.Linear(hidden_size, hidden_size),
    nn.LayerNorm(hidden_size),
    nn.ReLU()
]) * (1 + layer_N)

# GRU递归层 (时序建模)
gru: nn.GRU(
    input_size=hidden_size,
    hidden_size=hidden_size,
    num_layers=gru_layers,
    batch_first=False  # (T,N,H)格式
)

# 解耦的Actor分支 (策略网络)
actor_branch: nn.Sequential([
    nn.Linear(hidden_size, hidden_size) * 2,  # 2层全连接
    nn.LayerNorm + nn.ReLU,                   # 标准化和激活
    nn.Linear(hidden_size, action_dim),        # 输出动作logits
    nn.LayerNorm + nn.Tanh                    # 有界输出
])

# 解耦的Critic分支 (价值网络)
critic_branch: nn.Sequential([
    nn.Linear(hidden_size, hidden_size) * 2,  # 2层全连接
    nn.LayerNorm + nn.ReLU,                   # 标准化和激活
    nn.GRU(separate),                         # 独立GRU
    nn.Linear(hidden_size, 1)                 # 状态价值输出
])
```

#### 关键接口方法

**1. 动作采样接口** (`act_value`)
```python
def act_value(
    s: torch.Tensor,        # 状态 (N, state_dim)
    hxs: torch.Tensor,      # 隐藏状态 (layers, N, hidden_size)
    masks: torch.Tensor,    # 重置掩码 (N, 1)
    temperature: float = 1.0 # 温度参数 (探索控制)
) -> Tuple[action, log_prob, value, new_hxs]
```

**2. 动作评估接口** (`evaluate_actions`)
```python
def evaluate_actions(
    s: torch.Tensor,        # 状态批次
    hxs: torch.Tensor,      # 隐藏状态
    masks: torch.Tensor,    # 重置掩码
    a: torch.Tensor         # 待评估动作
) -> Tuple[log_prob, entropy, value, new_hxs]
```

### StateBuilder 状态构建器 (`src/core/models/state_builder.py`)

**功能**: 将复杂的调度器状态转换为神经网络可处理的特征向量

#### 状态构建流程

**1. 单副本状态特征** (`build_replica_state`)
```python
# 基础资源特征 (11维)
base_features = [
    queue_length,          # 队列长度
    allocated_blocks,      # 已分配内存块
    total_blocks,         # 总内存块数
    utilization_fraction, # 利用率
    available_fraction,   # 可用率
    running_batches,      # 运行中批次数
    preempted_requests,   # 被抢占请求数
    allocation_map_size,  # 分配映射大小
    batch_capacity,       # 批次容量
    block_size,          # 内存块大小
    num_stages           # 流水线阶段数
]

# 增强时序特征 (8维) - 可选
enhanced_features = [
    queue_delta,          # 队列长度变化
    allocation_delta,     # 分配变化
    load_ema,            # 负载指数移动平均
    queue_trend,         # 队列趋势(斜率)
    queue_variance,      # 队列方差
    historical_peak,     # 历史峰值
    historical_low,      # 历史低点
    time_since_peak     # 距离峰值时间
]

# 请求特征 (K个请求 × 7维/请求)
request_features = [
    age,                 # 请求年龄
    num_prefill_tokens, # 预填充令牌数
    num_processed_tokens, # 已处理令牌数
    remaining_prefill,   # 剩余预填充
    completed,          # 完成标志
    num_decode_tokens,  # 解码令牌数
    priority           # 请求优先级
] * max_queue_requests
```

**2. 全局状态构建** (`build_global_state`)
```python
# 状态组成 = 所有副本状态 + 全局特征 + 增强全局特征
global_state = [
    *replica_states,      # 每个副本的完整状态
    global_queue_length,  # 全局队列长度
    system_throughput,    # 系统吞吐量
    average_latency,      # 平均延迟

    # 增强全局特征 (7维) - 可选
    qps_current,         # 当前QPS
    qps_ema,            # QPS指数移动平均
    qps_trend,          # QPS趋势
    qps_variance,       # QPS方差
    system_load_balance, # 系统负载均衡度
    global_queue_delta,  # 全局队列变化
    completion_rate     # 请求完成率
]
```

**3. 状态维度计算**
```python
def get_state_dimension(num_replicas: int) -> int:
    replica_base = 11                                    # 基础特征
    enhanced_replica = 8 if enable_enhanced else 0      # 增强特征
    request_features = max_queue_requests * 7            # 请求特征

    per_replica_dim = replica_base + enhanced_replica + request_features
    global_base = 3                                      # 基础全局特征
    enhanced_global = 7 if enable_enhanced else 0       # 增强全局特征

    return num_replicas * per_replica_dim + global_base + enhanced_global
```

## 🎯 动作空间定义

### 动作表示
- **动作类型**: 离散动作空间
- **动作维度**: `len(replica_ids)` (等于副本数量)
- **动作含义**: 选择哪个副本来处理当前请求
- **动作范围**: `[0, num_replicas-1]`

### 动作选择流程
```python
# 1. 状态输入 -> Actor网络 -> 动作logits
logits = actor_network(state)  # (batch_size, num_replicas)

# 2. 温度缩放 (探索控制)
scaled_logits = logits / temperature

# 3. 概率分布构造
distribution = Categorical(logits=scaled_logits)

# 4. 动作采样
action = distribution.sample()  # 范围: [0, num_replicas-1]

# 5. 请求分配
selected_replica_id = replica_ids[action]
scheduler.assign_request(request, selected_replica_id)
```

### 温度控制机制
```python
# 动态温度计算 (TemperatureController)
temperature = base_temp * pressure_factor

pressure_factor = f(
    qps_pressure,      # QPS压力
    latency_pressure,  # 延迟压力
    load_balance,      # 负载均衡度
    delta_signals      # 变化信号
)

# 温度范围: [min_temp=0.8, max_temp=3.0]
# 高温度 -> 更多探索
# 低温度 -> 更多利用
```

## 🏆 奖励系统

### RewardCalculator (`src/core/algorithms/rewards/reward_calculator.py`)

**功能**: 根据系统性能指标计算强化学习奖励信号

#### 三种奖励模式

**1. Delta模式** (`mode="delta"`)
```python
# 基于指标变化的奖励
reward = (
    + delta_throughput              # 鼓励吞吐量提升
    - latency_weight * delta_latency # 惩罚延迟增加
    - balance_penalty               # 负载不均衡惩罚
    - threshold_penalty             # 软延迟阈值惩罚
)
```

**2. Instant模式** (`mode="instant"`)
```python
# 基于当前指标值的奖励
reward = (
    + throughput                    # 当前吞吐量奖励
    - latency_weight * latency      # 当前延迟惩罚
    - balance_penalty               # 负载不均衡惩罚
    - threshold_penalty             # 软延迟阈值惩罚
)
```

**3. Hybrid模式** (`mode="hybrid"`) - **推荐**
```python
# 结构化奖励系统
absolute_score = (throughput/target) - alpha*(latency/threshold)
delta_score = beta*norm_delta_tp - gamma*norm_delta_lat
logistic_penalty = kappa/(1 + exp(-(lat-threshold)/sigma))

reward = (
    + absolute_weight * absolute_score    # 绝对性能分数
    + delta_weight * delta_score         # 改进信号
    - load_balance_penalty               # 负载均衡惩罚
    - logistic_penalty                   # 平滑延迟惩罚
)
```

#### 奖励组件详解

**负载均衡惩罚**
```python
def calculate_load_balance_penalty(replica_ids, get_scheduler_fn):
    queue_lengths = [len(get_scheduler_fn(rid)._request_queue)
                    for rid in replica_ids]
    mean_queue = sum(queue_lengths) / len(queue_lengths)

    if mean_queue == 0:
        return 0.0  # 完美均衡

    # 变异系数 = 标准差/均值
    cv = std(queue_lengths) / mean_queue
    return cv  # 越高越不均衡
```

**软延迟阈值惩罚**
```python
def calculate_latency_threshold_penalty(latency):
    if latency <= threshold:
        return 0.0

    # 指数软惩罚
    excess = latency - threshold
    penalty = scale * (1.0 - exp(-excess))
    return penalty
```

## 🔄 PPO训练算法

### PPOTrainer (`src/core/algorithms/ppo_trainer.py`)

**功能**: 实现Proximal Policy Optimization算法

#### 核心超参数
```python
lr: float = 3e-4           # 学习率
clip_ratio: float = 0.2    # PPO裁剪比例
entropy_coef: float = 0.01 # 熵系数(探索)
value_coef: float = 0.5    # 价值损失权重
epochs: int = 4            # 每次更新的训练轮数
minibatch_size: int = 64   # 小批次大小
max_grad_norm: float = 0.5 # 梯度裁剪
target_kl: float = 0.01    # 目标KL散度
```

#### 损失函数组成
```python
# 1. PPO策略损失 (带裁剪)
ratio = exp(new_logp - old_logp)
surr1 = ratio * advantages
surr2 = clip(ratio, 1-clip_ratio, 1+clip_ratio) * advantages
policy_loss = -min(surr1, surr2).mean()

# 2. 价值函数损失 (带裁剪)
value_clipped = old_values + clip(new_values - old_values,
                                 -clip_ratio, clip_ratio)
value_loss = max((new_values - returns)^2,
                (value_clipped - returns)^2).mean()

# 3. 熵奖励 (鼓励探索)
entropy_bonus = entropy_coef * entropy.mean()

# 4. KL正则化 (防止过大更新)
kl_penalty = kl_coef * kl_divergence

# 5. 总损失
total_loss = policy_loss + value_coef*value_loss + kl_penalty - entropy_bonus
```

#### 预热启动和KL正则化
```python
# 设置参考策略
def set_reference_policy(reference_policy):
    self.reference_policy = reference_policy.eval()

# KL参考损失
def compute_kl_reference_loss(states, hxs, masks):
    current_logits = self.policy.get_logits(states, hxs, masks)
    ref_logits = self.reference_policy.get_logits(states, hxs, masks)

    kl_div = KL(softmax(current_logits) || softmax(ref_logits))
    return kl_div.mean()

# 动态KL系数衰减
kl_ref_coef = initial_coef * (1 - step/decay_steps) + final_coef * (step/decay_steps)
```

### RolloutBuffer (`src/core/algorithms/rollout_buffer.py`)

**功能**: 存储和处理PPO经验序列

#### 数据存储
```python
class RolloutBuffer:
    s: List[torch.Tensor]     # 状态序列
    a: List[torch.Tensor]     # 动作序列
    logp: List[torch.Tensor]  # 对数概率序列
    v: List[torch.Tensor]     # 价值估计序列
    r: List[float]            # 奖励序列
    masks: List[float]        # 终止掩码序列
```

#### GAE优势估计
```python
def compute_gae(last_value):
    # 逆向计算GAE优势
    advantages = torch.zeros(rollout_len)
    last_gae = 0

    for t in reversed(range(rollout_len)):
        if t == rollout_len - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        # TD误差
        delta = rewards[t] + gamma * next_value * masks[t] - values[t]

        # GAE递推
        last_gae = delta + gamma * gae_lambda * masks[t] * last_gae
        advantages[t] = last_gae

    returns = advantages + values

    # 优势标准化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return states, actions, log_probs, values, returns, advantages
```

## ⚙️ 配置系统

### PPOGlobalSchedulerModularConfig

**核心配置组**
```python
# 网络架构
hidden_size: int = 128           # 隐藏层维度
layer_N: int = 2                # MLP层数
gru_layers: int = 2             # GRU层数
enable_decoupled_ac: bool = True # 解耦Actor-Critic

# PPO超参数
lr: float = 5e-3                # 学习率
gamma: float = 0.95             # 折扣因子
gae_lambda: float = 0.95        # GAE lambda
clip_ratio: float = 0.15        # PPO裁剪比例
entropy_coef: float = 0.25      # 熵系数
value_coef: float = 0.5         # 价值权重
epochs: int = 8                 # 训练轮数
rollout_len: int = 32           # 回合长度
minibatch_size: int = 64        # 小批次大小

# 奖励系统
reward_mode: RewardMode = "hybrid"     # 奖励模式
throughput_target: float = 0.05        # 目标吞吐量
absolute_weight: float = 0.8           # 绝对分数权重
delta_weight: float = 0.2              # 差分分数权重
alpha: float = 0.1                     # 延迟惩罚系数
latency_threshold: float = 6.0         # 延迟阈值

# 状态构建
enable_enhanced_features: bool = True   # 增强特征
max_queue_requests_per_replica: int = 4 # 每副本最大队列请求数
state_history_window: int = 5          # 历史窗口
qps_window: int = 10                   # QPS计算窗口

# 温度控制
enable_dynamic_temperature: bool = True # 动态温度
base_temperature: float = 1.5          # 基础温度
min_temperature: float = 0.8           # 最小温度
max_temperature: float = 3.0           # 最大温度

# 监控和检查点
enable_tensorboard: bool = True         # TensorBoard监控
enable_checkpoints: bool = True         # 检查点保存
checkpoint_interval: int = 100         # 保存间隔
metrics_export_enabled: bool = False   # 指标导出
```

### 奖励模式枚举
```python
class RewardMode(str, Enum):
    delta = "delta"      # 差分奖励模式
    instant = "instant"  # 即时奖励模式
    hybrid = "hybrid"    # 混合奖励模式(推荐)
```

## 🔧 工具组件

### 状态标准化 (`RunningNormalizer`)
```python
class RunningNormalizer:
    def update(self, x):
        # 在线更新均值和方差
        self.count += len(x)
        delta = x - self.mean
        self.mean += delta.sum() / self.count
        self.m2 += (delta * (x - self.mean)).sum()

    def normalize(self, x):
        var = self.m2 / max(self.count - 1, 1)
        std = sqrt(var + self.eps)
        return clip((x - self.mean) / std, -self.clip, self.clip)
```

### 温度控制器 (`TemperatureController`)
```python
class TemperatureController:
    def compute_temperature(self, current_qps, target_qps,
                          current_latency, target_latency,
                          system_load_balance, **kwargs):
        # QPS压力计算
        qps_pressure = (target_qps - current_qps) / target_qps

        # 延迟压力计算
        latency_pressure = (current_latency - target_latency) / target_latency

        # 综合压力
        total_pressure = (
            self.qps_sensitivity * qps_pressure +
            self.latency_sensitivity * latency_pressure
        )

        # 温度调整
        temp = self.base_temperature * (1 + total_pressure)
        return clip(temp, self.min_temperature, self.max_temperature)
```

## 📊 监控系统

### TensorBoard集成
```python
class TensorBoardMonitor:
    def log_training_metrics(self, stats, step):
        # PPO训练指标
        self.writer.add_scalar('PPO/PolicyLoss', stats['pi_loss'], step)
        self.writer.add_scalar('PPO/ValueLoss', stats['vf_loss'], step)
        self.writer.add_scalar('PPO/Entropy', stats['entropy'], step)

    def log_reward_metrics(self, reward_info, reward, step):
        # 奖励组件分解
        self.writer.add_scalar('Reward/Total', reward, step)
        self.writer.add_scalar('Reward/Throughput', reward_info['throughput'], step)
        self.writer.add_scalar('Reward/Latency', reward_info['latency'], step)
```

### 指标导出
```python
class MetricsExporter:
    def append_training_metrics(self, step, metrics, metadata):
        # 导出训练指标到CSV/Parquet
        record = {
            'step': step,
            'timestamp': time.time(),
            **metrics,
            **metadata
        }
        self.training_buffer.append(record)
```

## 🚀 训练流程

### 主训练循环 (`schedule()` 方法)

```python
def schedule() -> List[Tuple[int, Request]]:
    # 1. 状态构建和标准化
    state = state_builder.build_global_state(replicas, schedulers, time, metrics)
    normalized_state = normalizer.normalize(state)

    # 2. 奖励计算
    reward, reward_info = reward_calculator.calculate_reward(
        metrics, time, replica_ids, schedulers
    )

    # 3. 动作选择 (带温度控制)
    temperature = temperature_controller.compute_temperature(...)
    action, log_prob, value, new_hxs = actor_critic.act_value(
        normalized_state, hidden_states, masks, temperature
    )

    # 4. 经验存储
    rollout_buffer.add_step(
        state, action, log_prob, value, reward, masks
    )

    # 5. PPO更新 (当缓冲区满时)
    if rollout_buffer.is_full():
        # 计算GAE优势
        states, actions, log_probs, values, returns, advantages = \
            rollout_buffer.compute_gae(bootstrap_value)

        # PPO策略更新
        stats = ppo_trainer.update(
            states, actions, log_probs, values, returns, advantages,
            masks, hidden_states
        )

        # 日志记录
        tensorboard.log_training_metrics(stats, step)
        metrics_exporter.append_training_metrics(stats)

        # 检查点保存
        if should_save_checkpoint:
            checkpoint_manager.save_checkpoint(...)

    # 6. 请求分配
    selected_replica = replica_ids[action]
    request = request_queue.pop(0)
    return [(selected_replica, request)]
```

### 统计稳定化阶段

在PPO训练开始前执行统计收集阶段:
```python
def statistics_stabilization_step():
    # 1. 构建状态 (更新标准化统计)
    state = state_builder.build_global_state(...)
    normalizer.update(state)

    # 2. 计算奖励 (更新奖励统计)
    reward, _ = reward_calculator.calculate_reward(...)

    # 3. 随机动作选择 (不使用策略网络)
    action = random.randint(0, num_replicas-1)

    # 4. 请求分配
    return [(replica_ids[action], request)]
```

## 📈 性能监控指标

### 训练指标
- **Policy Loss**: 策略网络损失
- **Value Loss**: 价值网络损失
- **Entropy**: 策略熵 (探索度)
- **KL Divergence**: KL散度 (更新幅度)
- **Clip Fraction**: 裁剪比例
- **Explained Variance**: 价值函数解释方差
- **Gradient Norm**: 梯度范数

### 系统指标
- **Throughput**: 系统吞吐量 (req/s)
- **Latency**: 平均响应延迟 (s)
- **Queue Length**: 全局队列长度
- **Load Balance**: 副本间负载均衡度
- **Temperature**: 当前探索温度
- **Reward Components**: 奖励组件分解

### 状态统计
- **State Mean/Std**: 状态均值和标准差
- **Utilization**: 副本利用率分布
- **Action Distribution**: 动作选择分布
- **QPS Metrics**: QPS趋势和方差

## 🔄 检查点和推理

### 检查点保存
```python
checkpoint_data = {
    'step': current_step,
    'model_state_dict': actor_critic.state_dict(),
    'normalizer_state': normalizer.get_state(),
    'training_state': {'step': step, 'rollouts': rollouts},
    'metadata': {
        'reward_mode': mode,
        'state_builder_config': {...},
        'ppo_config': {...},
        'reward_config': {...}
    }
}
```

### 推理模式
```python
# 加载检查点进入推理模式
scheduler = PPOGlobalSchedulerModular(
    config_with_inference_only=True,
    load_checkpoint="path/to/checkpoint.pt"
)

# 推理时行为
with torch.inference_mode():
    action, _, _, _ = actor_critic.act_value(state, hxs, masks, temperature)
    # 跳过所有训练相关操作
```

---

**★ Insight ─────────────────────────────────────**
1. **模块化设计**: 系统采用清晰的模块分离，每个组件都有明确的职责和接口，便于测试和维护
2. **状态工程**: 通过StateBuilder将复杂的调度器状态转换为结构化特征，支持时序建模和趋势分析
3. **奖励塑造**: 混合奖励模式结合绝对性能和改进信号，避免了传统delta奖励的静止状态问题
`─────────────────────────────────────────────────`

该架构通过强化学习方法学习最优的请求调度策略，在保证系统性能的同时实现负载均衡和资源利用率优化。