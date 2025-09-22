# 独立预训练系统使用指南

## 概述

独立预训练系统允许你进行长步骤、高质量的预训练，生成可重复使用的预训练模型，为PPO warmstart提供更好的初始策略。

## 架构设计

```
configs/
├── standalone_pretrain.json          # 预训练配置文件

src/pretraining/
├── __init__.py
├── standalone_trainer.py             # 核心训练逻辑
└── model_validator.py               # 模型验证工具

scripts/
├── standalone_pretrain.sh           # 独立预训练脚本
├── train_ppo_with_external_pretrain.sh  # 使用外部预训练的PPO训练
└── train_ppo_warmstart_optimized.sh  # 增强版warmstart脚本 (支持外部预训练)
```

## 使用流程

### 1. 独立预训练

```bash
# 使用默认配置
bash scripts/standalone_pretrain.sh

# 使用自定义配置
bash scripts/standalone_pretrain.sh configs/my_pretrain_config.json
```

### 2. 验证预训练模型

```bash
# 验证模型
python -m src.pretraining.model_validator ./outputs/standalone_pretrain/best_model.pt
```

### 3. 使用外部预训练模型进行PPO训练

```bash
# 方式1: 使用专用脚本 (推荐)
bash scripts/train_ppo_with_external_pretrain.sh ./outputs/standalone_pretrain/best_model.pt

# 方式2: 直接使用增强版warmstart脚本
bash scripts/train_ppo_warmstart_optimized.sh \
    --external-pretrain ./outputs/standalone_pretrain/best_model.pt \
    --skip-bc-training
```

## 配置说明

### 预训练配置 (`configs/standalone_pretrain.json`)

```json
{
  "state_dim": 64,              // 状态维度
  "action_dim": 4,              // 动作维度
  "hidden_size": 256,           // 隐藏层大小
  "layer_N": 3,                 // MLP层数
  "gru_layers": 2,              // GRU层数
  "learning_rate": 1e-4,        // 学习率
  "weight_decay": 1e-5,         // 权重衰减
  "batch_size": 512,            // 批大小
  "epochs": 100,                // 训练轮数
  "validation_split": 0.2,      // 验证集比例
  "early_stopping_patience": 20, // 早停patience
  "save_interval": 10,          // 保存间隔
  "device": "cpu",              // 训练设备
  "output_dir": "./outputs/standalone_pretrain"
}
```

## 优势对比

### 标准预训练模式 vs 独立预训练模式

| 特性 | 标准模式 | 独立预训练模式 |
|------|----------|----------------|
| **训练规模** | 受单次训练限制 | 可进行大规模长期训练 |
| **模型复用** | 每次重新训练 | 一次训练，多次使用 |
| **训练时间** | 较短 (30-50 epochs) | 较长 (100+ epochs) |
| **数据利用** | 单一收集策略 | 可集成多种数据源 |
| **质量保证** | 基础质量 | 高质量，充分收敛 |
| **实验效率** | 每次都需要预训练 | 跳过预训练，直接PPO |

## 最佳实践

### 1. 预训练策略

- **长期训练**: 使用100+ epochs充分收敛
- **多策略数据**: 收集多种调度策略的示教数据
- **数据增强**: 启用噪声注入增强泛化能力
- **早停机制**: 避免过拟合

### 2. 模型使用

- **验证优先**: 使用前务必验证模型兼容性
- **跳过BC**: 使用高质量外部预训练时跳过BC训练
- **实验对比**: 对比标准模式和外部预训练模式的效果

### 3. 性能优化

- **设备选择**: 有GPU时使用GPU训练
- **批大小**: 根据内存调整批大小
- **学习率**: 使用学习率调度器自适应调整

## 故障排除

### 常见问题

1. **模型验证失败**
   - 检查模型文件是否存在
   - 确认模型格式正确
   - 验证配置参数匹配

2. **训练内存不足**
   - 减少batch_size
   - 降低模型规模参数
   - 使用CPU训练

3. **收敛速度慢**
   - 调整学习率
   - 检查数据质量
   - 增加训练轮数

## 输出文件说明

```
outputs/standalone_pretrain/
├── best_model.pt              # 最佳模型 (用于PPO warmstart)
├── final_model.pt             # 最终模型
├── checkpoint_epoch_*.pt      # 定期checkpoint
├── training_history.json      # 训练历史
└── tensorboard/              # TensorBoard日志
```

## 扩展功能

系统设计支持未来扩展：

- **多模态数据**: 支持不同类型的示教数据
- **分布式训练**: 支持多GPU/多节点训练
- **模型集成**: 支持多个预训练模型的集成
- **在线学习**: 支持在线更新预训练模型