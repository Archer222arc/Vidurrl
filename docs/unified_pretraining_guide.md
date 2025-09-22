# 统一预训练系统使用指南

## 概述

统一预训练系统整合了原始warmup training和独立预训练的所有功能，提供一致的接口和配置管理。

## 重构后的架构

```
src/pretraining/
├── __init__.py
├── behavior_cloning_trainer.py    # BC训练核心逻辑
├── model_validator.py             # 模型验证工具
├── unified_trainer.py             # 统一预训练管理器
└── standalone_trainer.py          # 独立预训练实现 (被unified_trainer调用)

scripts/
├── pretrain_actor.py              # 兼容原始接口的脚本
├── standalone_pretrain.sh         # 独立预训练脚本
├── train_ppo_with_external_pretrain.sh  # 使用外部预训练的PPO训练
└── train_ppo_warmstart_optimized.sh     # 增强版warmstart脚本

configs/
├── standalone_pretrain.json       # 独立预训练配置
└── unified_pretrain.json         # 统一预训练配置
```

## 使用方式

### 1. 兼容原始接口 (推荐用于现有脚本)

```bash
# 原始命令行接口，完全兼容
python scripts/pretrain_actor.py \
  --demo demo_data.pkl \
  --epochs 30 \
  --batch_size 256 \
  --lr 5e-4 \
  --hidden_size 128 \
  --output pretrained_actor.pt

# 微调模式 (新功能)
python scripts/pretrain_actor.py \
  --demo demo_data.pkl \
  --epochs 10 \
  --resume ./outputs/external_pretrain/best_model.pt \
  --output fine_tuned_actor.pt
```

### 2. 统一预训练管理器 (推荐用于新开发)

```bash
# 标准模式
python -m src.pretraining.unified_trainer \
  --demo-files demo1.pkl demo2.pkl \
  --mode standard \
  --state-dim 64 \
  --action-dim 4 \
  --epochs 30 \
  --output standard_model.pt

# 增强模式
python -m src.pretraining.unified_trainer \
  --config configs/unified_pretrain.json \
  --demo-files demo1.pkl demo2.pkl \
  --mode enhanced \
  --output enhanced_model.pt

# 微调模式
python -m src.pretraining.unified_trainer \
  --demo-files new_demo.pkl \
  --base-model ./outputs/pretrain/best_model.pt \
  --fine-tune-epochs 10 \
  --output fine_tuned_model.pt
```

### 3. PPO训练集成

```bash
# 使用统一预训练模型的PPO训练
bash scripts/train_ppo_with_external_pretrain.sh \
  ./outputs/unified_pretrain/enhanced_model.pt \
  --num-replicas 8 \
  --qps 5.0

# 或者直接在warmstart脚本中使用
bash scripts/train_ppo_warmstart_optimized.sh \
  --external-pretrain ./outputs/unified_pretrain/enhanced_model.pt \
  --skip-bc-training \
  --num-replicas 8
```

## 训练模式对比

### Standard模式 (兼容原始)
- **用途**: 兼容现有脚本和流程
- **特点**: 简单快速，基础功能
- **配置**: 较少参数，快速启动
- **适用**: 日常开发和测试

### Enhanced模式 (新功能)
- **用途**: 高质量预训练模型生成
- **特点**: 长期训练，早停，TensorBoard监控
- **配置**: 丰富参数，完整功能
- **适用**: 生产环境和重要实验

## 配置文件说明

### 标准配置 (`configs/standalone_pretrain.json`)
```json
{
  "training_mode": "standard",
  "state_dim": 64,
  "action_dim": 4,
  "hidden_size": 128,
  "epochs": 30,
  "batch_size": 256,
  "learning_rate": 1e-3,
  "device": "cpu"
}
```

### 增强配置 (`configs/unified_pretrain.json`)
```json
{
  "training_mode": "enhanced",
  "state_dim": 64,
  "action_dim": 4,
  "hidden_size": 256,
  "layer_N": 3,
  "epochs": 100,
  "early_stopping_patience": 20,
  "use_tensorboard": true,
  "augment_data": true
}
```

## 迁移指南

### 从原始 pretrain_actor.py 迁移

**之前:**
```bash
python scripts/pretrain_actor.py --demo demo.pkl --epochs 30 --output model.pt
```

**现在:**
```bash
# 选择1: 保持原始接口 (推荐，无需修改)
python scripts/pretrain_actor.py --demo demo.pkl --epochs 30 --output model.pt

# 选择2: 使用统一接口 (新功能)
python -m src.pretraining.unified_trainer \
  --demo-files demo.pkl \
  --mode standard \
  --epochs 30 \
  --output model.pt
```

### 从独立预训练迁移

**之前:**
```bash
# 需要多个步骤和脚本
```

**现在:**
```bash
# 一个统一的命令
python -m src.pretraining.unified_trainer \
  --config configs/unified_pretrain.json \
  --demo-files demo1.pkl demo2.pkl \
  --mode enhanced
```

## 优势总结

### 1. 统一性
- **接口统一**: 所有预训练功能使用相同的核心组件
- **配置统一**: 一致的参数格式和验证
- **输出统一**: 标准化的模型保存格式

### 2. 兼容性
- **向后兼容**: 原始脚本无需修改即可使用
- **接口兼容**: 保持原有的命令行参数格式
- **格式兼容**: 支持多种模型保存格式

### 3. 可扩展性
- **模块化设计**: 核心逻辑可独立复用
- **配置驱动**: 新功能通过配置文件控制
- **插件化架构**: 易于添加新的训练模式

### 4. 维护性
- **代码复用**: 消除重复代码
- **统一测试**: 一套测试覆盖所有功能
- **文档一致**: 统一的使用说明和配置

## 最佳实践

### 1. 选择合适的模式
- **开发阶段**: 使用standard模式快速迭代
- **实验阶段**: 使用enhanced模式获得最佳效果
- **生产环境**: 使用enhanced模式训练，standard模式微调

### 2. 配置管理
- **版本控制**: 将配置文件纳入版本控制
- **环境隔离**: 不同环境使用不同配置
- **参数验证**: 使用模型验证器检查兼容性

### 3. 性能优化
- **批大小调优**: 根据内存情况调整batch_size
- **设备选择**: 有GPU时使用GPU训练
- **早停策略**: 使用早停避免过拟合

### 4. 故障排除
- **模型验证**: 训练前后都进行模型验证
- **日志监控**: 使用TensorBoard监控训练过程
- **增量调试**: 从小规模开始逐步扩大