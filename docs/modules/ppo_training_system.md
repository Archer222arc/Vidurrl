# PPO训练系统模块接口文档

## 模块概述
PPO (Proximal Policy Optimization) 训练系统，用于训练调度器策略网络。

## 核心组件

### 1. 配置文件
- **位置**: `configs/revolutionary_collapse_prevention*.json`
- **功能**: 定义训练hyperparameters、reward system、entropy control等

### 2. 训练入口
- **文件**: `vidur/simulator.py`
- **主要参数**:
  - `--config`: 配置文件路径
  - `--training_steps`: 训练步数

### 3. 关键模块
- **PPO Scheduler**: `vidur/scheduler/global_scheduler/ppo_scheduler_modular.py`
- **Actor-Critic网络**: `src/core/models/actor_critic.py`
- **PPO Trainer**: `src/core/algorithms/ppo_trainer.py`

## 当前配置版本

### revolutionary_collapse_prevention_v4_incremental_fix.json

**关键参数**:
```json
{
  "ppo_config": {
    "lr": 0.0004,
    "clip_ratio": 0.3,
    "epochs": 12,
    "minibatch_size": 128
  },
  "cluster_config": {
    "global_scheduler_config": {
      "expert_guidance_weight": 0.5,
      "action_balance_weight": 0.3,
      "simplified_reward_latency_weight": 1.5
    }
  }
}
```

## 使用示例

### 训练新模型
```bash
python vidur/simulator.py \
  --config configs/revolutionary_collapse_prevention_v4_incremental_fix.json \
  --training_steps 20000
```

### 监控训练
```bash
tensorboard --logdir outputs/revolutionary_training_v4_incremental/tensorboard
```

## 常见问题

### Q: Entropy不下降怎么办?
A: 降低`expert_guidance_weight`和`action_balance_weight`

### Q: ExplainedVariance太低?
A: 增加`epochs`和`minibatch_size`

## 相关文档
- [演进历史](./evolution/ppo_training_system_evolution.md) - 版本变更记录
- [废弃功能](../deprecated/deprecated_ppo_training.md) - 已移除的功能
