# 测试命令参考 - PPO优化版本

## 推理测试命令 (使用最新优化参数)

### 完整推理测试命令

```bash
python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  --cluster_config_num_replicas 4 \
  --p_p_o_global_scheduler_modular_config_debug_dump_global_state \
  --p_p_o_global_scheduler_modular_config_max_queue_requests_per_replica 8 \
  --p_p_o_global_scheduler_modular_config_lr 0.0003 \
  --p_p_o_global_scheduler_modular_config_gamma 0.95 \
  --p_p_o_global_scheduler_modular_config_reward_latency_weight 1.5 \
  --p_p_o_global_scheduler_modular_config_balance_penalty_weight 0 \
  --p_p_o_global_scheduler_modular_config_entropy_coef 0.25 \
  --synthetic_request_generator_config_num_requests 500 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 2 \
  --metrics_config_subsamples 200000 \
  --p_p_o_global_scheduler_modular_config_inference_only \
  --p_p_o_global_scheduler_modular_config_load_checkpoint /Users/ruicheng/Documents/GitHub/Vidur/Vidur_arc2/outputs/checkpoints/latest.pt \
  --no-p_p_o_global_scheduler_modular_config_enable_tensorboard \
  --no-p_p_o_global_scheduler_modular_config_enable_checkpoints \
  --p_p_o_global_scheduler_modular_config_tensorboard_log_dir ./runs/ppo_training/run_$(date +%Y%m%d_%H%M%S) \
  --p_p_o_global_scheduler_modular_config_latency_threshold 6.0 \
  --p_p_o_global_scheduler_modular_config_latency_penalty_scale 0.5 \
  --p_p_o_global_scheduler_modular_config_load_balance_penalty 0.15 \
  --p_p_o_global_scheduler_modular_config_throughput_target 0.05 \
  --p_p_o_global_scheduler_modular_config_absolute_weight 0.8 \
  --p_p_o_global_scheduler_modular_config_delta_weight 0.2 \
  --p_p_o_global_scheduler_modular_config_alpha 0.1 \
  --p_p_o_global_scheduler_modular_config_kappa 0.05 \
  --p_p_o_global_scheduler_modular_config_sigma 2.0 \
  --p_p_o_global_scheduler_modular_config_target_kl 0.01 \
  --p_p_o_global_scheduler_modular_config_entropy_min 0.5 \
  --p_p_o_global_scheduler_modular_config_kl_coef 0.2 \
  --p_p_o_global_scheduler_modular_config_enable_dynamic_temperature \
  --p_p_o_global_scheduler_modular_config_base_temperature 1.0 \
  --p_p_o_global_scheduler_modular_config_min_temperature 0.5 \
  --p_p_o_global_scheduler_modular_config_max_temperature 2.0 \
  > run_modular.log
```

### 推理测试命令 (自动架构兼容)

✅ **已实现自动架构匹配！** 推理模式下会自动从checkpoint重建正确的网络架构。

**推荐命令 (完全自动化)**:

```bash
python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  --cluster_config_num_replicas 4 \
  --p_p_o_global_scheduler_modular_config_inference_only \
  --p_p_o_global_scheduler_modular_config_load_checkpoint /Users/ruicheng/Documents/GitHub/Vidur/Vidur_arc2/outputs/checkpoints/latest.pt \
  --no-p_p_o_global_scheduler_modular_config_enable_tensorboard \
  --no-p_p_o_global_scheduler_modular_config_enable_checkpoints \
  --synthetic_request_generator_config_num_requests 500 \
  --poisson_request_interval_generator_config_qps 2 \
  > run_modular.log
```

**完全匹配训练配置的推理命令** (适用于您的旧checkpoint):

```bash
python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  --cluster_config_num_replicas 4 \
  --p_p_o_global_scheduler_modular_config_inference_only \
  --p_p_o_global_scheduler_modular_config_load_checkpoint /Users/ruicheng/Documents/GitHub/Vidur/Vidur_arc2/outputs/checkpoints/latest.pt \
  --p_p_o_global_scheduler_modular_config_max_queue_requests_per_replica 8 \
  --p_p_o_global_scheduler_modular_config_lr 0.0003 \
  --p_p_o_global_scheduler_modular_config_gamma 0.95 \
  --p_p_o_global_scheduler_modular_config_reward_latency_weight 1.5 \
  --p_p_o_global_scheduler_modular_config_balance_penalty_weight 0 \
  --p_p_o_global_scheduler_modular_config_entropy_coef 0.25 \
  --synthetic_request_generator_config_num_requests 500 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 2 \
  --metrics_config_subsamples 200000 \
  --no-p_p_o_global_scheduler_modular_config_enable_tensorboard \
  --no-p_p_o_global_scheduler_modular_config_enable_checkpoints \
  --p_p_o_global_scheduler_modular_config_enable_enhanced_features \
  --p_p_o_global_scheduler_modular_config_state_history_window 5 \
  --p_p_o_global_scheduler_modular_config_qps_window 10 \
  --p_p_o_global_scheduler_modular_config_latency_threshold 6.0 \
  --p_p_o_global_scheduler_modular_config_latency_penalty_scale 0.5 \
  --p_p_o_global_scheduler_modular_config_load_balance_penalty 0.15 \
  --p_p_o_global_scheduler_modular_config_throughput_target 0.05 \
  --p_p_o_global_scheduler_modular_config_absolute_weight 0.8 \
  --p_p_o_global_scheduler_modular_config_delta_weight 0.2 \
  --p_p_o_global_scheduler_modular_config_alpha 0.1 \
  --p_p_o_global_scheduler_modular_config_kappa 0.05 \
  --p_p_o_global_scheduler_modular_config_sigma 2.0 \
  --p_p_o_global_scheduler_modular_config_target_kl 0.01 \
  --p_p_o_global_scheduler_modular_config_entropy_min 0.5 \
  --p_p_o_global_scheduler_modular_config_kl_coef 0.2 \
  --p_p_o_global_scheduler_modular_config_enable_dynamic_temperature \
  --p_p_o_global_scheduler_modular_config_base_temperature 1.0 \
  --p_p_o_global_scheduler_modular_config_min_temperature 0.5 \
  --p_p_o_global_scheduler_modular_config_max_temperature 2.0 \
  > run_modular.log
```

**StateBuilder配置需要匹配** (最小化版本):

```bash
python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  --cluster_config_num_replicas 4 \
  --p_p_o_global_scheduler_modular_config_inference_only \
  --p_p_o_global_scheduler_modular_config_load_checkpoint /Users/ruicheng/Documents/GitHub/Vidur/Vidur_arc2/outputs/checkpoints/latest.pt \
  --no-p_p_o_global_scheduler_modular_config_enable_tensorboard \
  --no-p_p_o_global_scheduler_modular_config_enable_checkpoints \
  --synthetic_request_generator_config_num_requests 500 \
  --poisson_request_interval_generator_config_qps 2 \
  --p_p_o_global_scheduler_modular_config_max_queue_requests_per_replica 8 \
  --p_p_o_global_scheduler_modular_config_enable_enhanced_features \
  --p_p_o_global_scheduler_modular_config_state_history_window 5 \
  --p_p_o_global_scheduler_modular_config_qps_window 10 \
  > run_modular.log
```

### 备用推理测试命令 (如果checkpoint配置不完整)

```bash
python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  --cluster_config_num_replicas 4 \
  --p_p_o_global_scheduler_modular_config_inference_only \
  --p_p_o_global_scheduler_modular_config_load_checkpoint /Users/ruicheng/Documents/GitHub/Vidur/Vidur_arc2/outputs/checkpoints/latest.pt \
  --no-p_p_o_global_scheduler_modular_config_enable_tensorboard \
  --no-p_p_o_global_scheduler_modular_config_enable_checkpoints \
  --synthetic_request_generator_config_num_requests 500 \
  --poisson_request_interval_generator_config_qps 2 \
  --p_p_o_global_scheduler_modular_config_entropy_coef 0.25 \
  --p_p_o_global_scheduler_modular_config_throughput_target 0.05 \
  --p_p_o_global_scheduler_modular_config_alpha 0.1 \
  --p_p_o_global_scheduler_modular_config_kappa 0.05 \
  > run_modular.log
```

## 训练测试命令 (使用最新优化参数)

### 短期训练测试

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)

python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  --cluster_config_num_replicas 4 \
  --p_p_o_global_scheduler_modular_config_max_queue_requests_per_replica 8 \
  --p_p_o_global_scheduler_modular_config_lr 0.0003 \
  --synthetic_request_generator_config_num_requests 100 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 2 \
  --metrics_config_subsamples 200000 \
  --p_p_o_global_scheduler_modular_config_tensorboard_log_dir ./outputs/runs/ppo_training/run_${RUN_ID} \
  --p_p_o_global_scheduler_modular_config_tensorboard_force_kill \
  --p_p_o_global_scheduler_modular_config_tensorboard_port 6006 \
  --p_p_o_global_scheduler_modular_config_tensorboard_auto_start \
  --p_p_o_global_scheduler_modular_config_metrics_export_enabled \
  --p_p_o_global_scheduler_modular_config_metrics_export_format csv \
  --p_p_o_global_scheduler_modular_config_metrics_export_path ./outputs/runs/ppo_training/exports/run_${RUN_ID} \
  --p_p_o_global_scheduler_modular_config_metrics_export_interval 25 \
  --p_p_o_global_scheduler_modular_config_entropy_coef 0.25 \
  --p_p_o_global_scheduler_modular_config_latency_threshold 6.0 \
  --p_p_o_global_scheduler_modular_config_latency_penalty_scale 0.5 \
  --p_p_o_global_scheduler_modular_config_load_balance_penalty 0.15 \
  --p_p_o_global_scheduler_modular_config_throughput_target 0.05 \
  --p_p_o_global_scheduler_modular_config_absolute_weight 0.8 \
  --p_p_o_global_scheduler_modular_config_delta_weight 0.2 \
  --p_p_o_global_scheduler_modular_config_alpha 0.1 \
  --p_p_o_global_scheduler_modular_config_kappa 0.05 \
  --p_p_o_global_scheduler_modular_config_sigma 2.0 \
  --p_p_o_global_scheduler_modular_config_target_kl 0.01 \
  --p_p_o_global_scheduler_modular_config_entropy_min 0.5 \
  --p_p_o_global_scheduler_modular_config_kl_coef 0.2 \
  --p_p_o_global_scheduler_modular_config_enable_dynamic_temperature \
  --p_p_o_global_scheduler_modular_config_base_temperature 1.0 \
  --p_p_o_global_scheduler_modular_config_min_temperature 0.5 \
  --p_p_o_global_scheduler_modular_config_max_temperature 2.0 \
  > run_modular_test.log
```

## 参数变更说明

### 更新的参数

| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| `entropy_coef` | 0.002 | 0.25 | 大幅提升探索强度 |
| `latency_threshold` | - | 6.0 | 放宽延迟容忍度 |
| `latency_penalty_scale` | - | 0.5 | 减少延迟惩罚 |
| `load_balance_penalty` | - | 0.15 | 强化负载均衡 |

### 新增的关键参数

```bash
# 奖励结构优化
--p_p_o_global_scheduler_modular_config_throughput_target 0.05
--p_p_o_global_scheduler_modular_config_absolute_weight 0.8
--p_p_o_global_scheduler_modular_config_delta_weight 0.2
--p_p_o_global_scheduler_modular_config_alpha 0.1
--p_p_o_global_scheduler_modular_config_kappa 0.05
--p_p_o_global_scheduler_modular_config_sigma 2.0

# 探索控制
--p_p_o_global_scheduler_modular_config_target_kl 0.01
--p_p_o_global_scheduler_modular_config_entropy_min 0.5
--p_p_o_global_scheduler_modular_config_kl_coef 0.2
```

## 快速执行脚本

### 推理测试脚本

创建 `scripts/test_inference_simple.sh`:

```bash
#!/bin/bash

# 最简推理测试脚本 - 只依赖checkpoint配置
echo "开始执行PPO推理测试 (从checkpoint读取所有配置)..."

python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  --cluster_config_num_replicas 4 \
  --p_p_o_global_scheduler_modular_config_inference_only \
  --p_p_o_global_scheduler_modular_config_load_checkpoint /Users/ruicheng/Documents/GitHub/Vidur/Vidur_arc2/outputs/checkpoints/latest.pt \
  --no-p_p_o_global_scheduler_modular_config_enable_tensorboard \
  --no-p_p_o_global_scheduler_modular_config_enable_checkpoints \
  --synthetic_request_generator_config_num_requests 500 \
  --poisson_request_interval_generator_config_qps 2 \
  > run_modular_inference.log

echo "推理测试完成，日志保存在 run_modular_inference.log"
echo "如果遇到配置问题，请使用备用命令 (包含关键参数)"
```

创建 `scripts/test_inference_with_params.sh` (备用):

```bash
#!/bin/bash

# 备用推理测试脚本 - 包含关键参数以防checkpoint不完整
echo "开始执行PPO推理测试 (包含优化参数)..."

python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  --cluster_config_num_replicas 4 \
  --p_p_o_global_scheduler_modular_config_inference_only \
  --p_p_o_global_scheduler_modular_config_load_checkpoint /Users/ruicheng/Documents/GitHub/Vidur/Vidur_arc2/outputs/checkpoints/latest.pt \
  --no-p_p_o_global_scheduler_modular_config_enable_tensorboard \
  --synthetic_request_generator_config_num_requests 500 \
  --poisson_request_interval_generator_config_qps 2 \
  --p_p_o_global_scheduler_modular_config_entropy_coef 0.25 \
  --p_p_o_global_scheduler_modular_config_throughput_target 0.05 \
  --p_p_o_global_scheduler_modular_config_alpha 0.1 \
  --p_p_o_global_scheduler_modular_config_kappa 0.05 \
  > run_modular_inference.log

echo "推理测试完成，日志保存在 run_modular_inference.log"
```

### 训练测试脚本

创建 `scripts/test_training_optimized.sh`:

```bash
#!/bin/bash

# 优化后的训练测试脚本
RUN_ID=$(date +%Y%m%d_%H%M%S)
echo "开始执行优化后的PPO训练测试... Run ID: ${RUN_ID}"

python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  --cluster_config_num_replicas 4 \
  --synthetic_request_generator_config_num_requests 100 \
  --poisson_request_interval_generator_config_qps 2 \
  --p_p_o_global_scheduler_modular_config_tensorboard_log_dir ./outputs/runs/ppo_training/run_${RUN_ID} \
  --p_p_o_global_scheduler_modular_config_metrics_export_enabled \
  --p_p_o_global_scheduler_modular_config_metrics_export_path ./outputs/runs/ppo_training/exports/run_${RUN_ID} \
  --p_p_o_global_scheduler_modular_config_entropy_coef 0.25 \
  --p_p_o_global_scheduler_modular_config_throughput_target 0.05 \
  --p_p_o_global_scheduler_modular_config_alpha 0.1 \
  --p_p_o_global_scheduler_modular_config_kappa 0.05 \
  --p_p_o_global_scheduler_modular_config_target_kl 0.01 \
  --p_p_o_global_scheduler_modular_config_entropy_min 0.5 \
  --p_p_o_global_scheduler_modular_config_kl_coef 0.2 \
  > run_modular_training_${RUN_ID}.log

echo "训练测试完成，日志保存在 run_modular_training_${RUN_ID}.log"
echo "TensorBoard访问: http://localhost:6006"
echo "CSV导出路径: ./outputs/runs/ppo_training/exports/run_${RUN_ID}/"
```

## 期望验证结果

运行优化后的测试命令，应该观察到：

1. **动作分布改善**: 不再有单副本占比 >50% 的情况
2. **奖励信号健康**: `absolute_score > 0`, `total_reward > 0`
3. **探索活跃**: 定期温度脉冲，`entropy > 0.5`
4. **训练稳定**: `clipfrac < 0.3`, 梯度更新平衡

---

**注意事项**:
- 确保checkpoint文件路径正确
- 根据实际需要调整 `num_requests` 参数
- 监控日志输出以验证参数生效
- 使用TensorBoard观察实时训练指标

---

## 预训练系统命令 - 统一架构

### 1. 兼容原始接口的预训练命令

#### 标准BC预训练 (兼容原始脚本)
```bash
# 基础BC训练 (保持原始接口)
python scripts/pretrain_actor.py \
  --demo ./data/demo_data.pkl \
  --epochs 30 \
  --batch_size 256 \
  --lr 5e-4 \
  --hidden_size 128 \
  --layer_N 2 \
  --gru_layers 2 \
  --output ./outputs/pretrained_actor.pt \
  --device cpu

# 微调模式 (基于外部预训练模型)
python scripts/pretrain_actor.py \
  --demo ./data/new_demo_data.pkl \
  --epochs 10 \
  --resume ./outputs/external_pretrain/best_model.pt \
  --output ./outputs/fine_tuned_actor.pt \
  --lr 1e-4
```

### 2. 统一预训练管理器命令

#### 标准模式预训练
```bash
# 单个示教文件
python -m src.pretraining.unified_trainer \
  --demo-files ./data/demo_data.pkl \
  --mode standard \
  --state-dim 64 \
  --action-dim 4 \
  --hidden-size 128 \
  --epochs 30 \
  --device cpu \
  --output standard_model.pt

# 多个示教文件
python -m src.pretraining.unified_trainer \
  --demo-files ./data/demo1.pkl ./data/demo2.pkl ./data/demo3.pkl \
  --mode standard \
  --state-dim 64 \
  --action-dim 4 \
  --epochs 30 \
  --output multi_demo_model.pt
```

#### 增强模式预训练
```bash
# 使用配置文件
python -m src.pretraining.unified_trainer \
  --config configs/unified_pretrain.json \
  --demo-files ./data/demo1.pkl ./data/demo2.pkl \
  --mode enhanced \
  --output enhanced_model.pt

# 命令行参数
python -m src.pretraining.unified_trainer \
  --demo-files ./data/demo_data.pkl \
  --mode enhanced \
  --state-dim 64 \
  --action-dim 4 \
  --hidden-size 256 \
  --layer-N 3 \
  --epochs 100 \
  --device cpu \
  --output enhanced_model.pt
```

#### 微调模式
```bash
# 基于预训练模型微调
python -m src.pretraining.unified_trainer \
  --demo-files ./data/new_demo.pkl \
  --base-model ./outputs/unified_pretrain/enhanced_model.pt \
  --fine-tune-epochs 10 \
  --output fine_tuned_model.pt
```

### 3. 预训练数据管理命令

#### 数据集管理
```bash
# 查看现有数据集
bash scripts/manage_pretrain_data.sh info

# 收集标准预训练数据集 (一次收集，多次使用)
bash scripts/manage_pretrain_data.sh collect

# 收集大规模数据集 (高质量预训练)
bash scripts/manage_pretrain_data.sh collect-large

# 收集自定义数据集
bash scripts/manage_pretrain_data.sh collect-custom

# 列出所有数据集
bash scripts/manage_pretrain_data.sh list

# 清理临时和旧数据
bash scripts/manage_pretrain_data.sh clean
```

### 4. 独立预训练脚本命令

#### 使用预管理的数据集
```bash
# 简单启动 (自动使用标准数据集)
bash scripts/standalone_pretrain.sh

# 使用自定义配置
bash scripts/standalone_pretrain.sh configs/custom_pretrain.json
```

### 5. 模型验证命令

#### 验证预训练模型
```bash
# 基础验证
python -m src.pretraining.model_validator ./outputs/pretrained_actor.pt

# 带兼容性检查的验证
python -c "
from src.pretraining.model_validator import validate_pretrained_model
target_config = {
    'state_dim': 64,
    'action_dim': 4,
    'hidden_size': 128,
    'layer_N': 2,
    'gru_layers': 2
}
success = validate_pretrained_model('./outputs/pretrained_actor.pt', target_config)
print('验证通过' if success else '验证失败')
"
```

### 5. PPO训练集成命令

#### 使用外部预训练模型的PPO训练
```bash
# 专用脚本 (推荐)
bash scripts/train_ppo_with_external_pretrain.sh \
  ./outputs/unified_pretrain/enhanced_model.pt \
  --num-replicas 8 \
  --qps 5.0 \
  --ppo-requests 10000

# 增强版warmstart脚本
bash scripts/train_ppo_warmstart_optimized.sh \
  --external-pretrain ./outputs/unified_pretrain/enhanced_model.pt \
  --skip-bc-training \
  --num-replicas 8 \
  --qps 5.0 \
  --ppo-requests 10000

# 外部预训练 + BC微调
bash scripts/train_ppo_warmstart_optimized.sh \
  --external-pretrain ./outputs/unified_pretrain/enhanced_model.pt \
  --bc-epochs 10 \
  --num-replicas 4 \
  --qps 3.0 \
  --demo-steps 500
```

#### 标准warmstart训练 (无外部预训练)
```bash
# 标准模式
bash scripts/train_ppo_warmstart_optimized.sh \
  --num-replicas 4 \
  --qps 3.0 \
  --ppo-requests 8000 \
  --bc-epochs 30 \
  --demo-steps 700

# 强制重新训练
bash scripts/train_ppo_warmstart_optimized.sh \
  --force-warmstart \
  --num-replicas 8 \
  --bc-epochs 50 \
  --demo-steps 1000
```

### 6. 完整训练流程示例

#### 方案1: 传统流程 (兼容现有)
```bash
# 步骤1: 收集示教数据 (假设已有)
# 步骤2: BC预训练
python scripts/pretrain_actor.py \
  --demo ./data/demo_data.pkl \
  --epochs 30 \
  --output ./outputs/pretrained_actor.pt

# 步骤3: PPO训练
bash scripts/train_ppo_warmstart_optimized.sh \
  --num-replicas 4 \
  --bc-epochs 30
```

#### 方案2: 增强流程 (推荐)
```bash
# 步骤1: 高质量独立预训练
python -m src.pretraining.unified_trainer \
  --config configs/unified_pretrain.json \
  --demo-files ./data/demo1.pkl ./data/demo2.pkl \
  --mode enhanced \
  --output high_quality_model.pt

# 步骤2: 验证模型
python -m src.pretraining.model_validator ./outputs/unified_pretrain/high_quality_model.pt

# 步骤3: PPO训练 (跳过BC)
bash scripts/train_ppo_with_external_pretrain.sh \
  ./outputs/unified_pretrain/high_quality_model.pt \
  --num-replicas 8 \
  --qps 5.0
```

#### 方案3: 混合流程 (最佳实践)
```bash
# 步骤1: 长期独立预训练 (周期性执行)
python -m src.pretraining.unified_trainer \
  --config configs/unified_pretrain.json \
  --demo-files ./data/large_demo_dataset/*.pkl \
  --mode enhanced \
  --epochs 200 \
  --output foundation_model.pt

# 步骤2: 针对性微调 (每次实验)
python -m src.pretraining.unified_trainer \
  --demo-files ./data/specific_demo.pkl \
  --base-model ./outputs/unified_pretrain/foundation_model.pt \
  --fine-tune-epochs 5 \
  --output specialized_model.pt

# 步骤3: PPO训练
bash scripts/train_ppo_with_external_pretrain.sh \
  ./outputs/unified_pretrain/specialized_model.pt
```

### 7. 配置文件管理

#### 创建标准配置
```bash
# 生成标准配置文件
python -c "
from src.pretraining.unified_trainer import UnifiedPretrainer
import json
trainer = UnifiedPretrainer({})
config = trainer.create_standard_config(state_dim=64, action_dim=4, epochs=30)
with open('configs/my_standard_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('标准配置已保存到 configs/my_standard_config.json')
"

# 生成增强配置文件
python -c "
from src.pretraining.unified_trainer import UnifiedPretrainer
import json
trainer = UnifiedPretrainer({})
config = trainer.create_enhanced_config(state_dim=64, action_dim=4, epochs=100)
with open('configs/my_enhanced_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('增强配置已保存到 configs/my_enhanced_config.json')
"
```

### 8. 监控和调试命令

#### TensorBoard监控
```bash
# 启动TensorBoard (预训练)
tensorboard --logdir ./outputs/unified_pretrain/tensorboard --port 6007

# 启动TensorBoard (PPO训练)
tensorboard --logdir ./outputs/runs/ppo_training --port 6006
```

#### 日志查看
```bash
# 实时查看训练日志
tail -f ./outputs/warmstart_external/run_*/ppo_training.log

# 查看预训练日志
tail -f ./outputs/unified_pretrain/tensorboard/events*
```

### 9. 故障排除命令

#### 诊断模型兼容性
```bash
# 详细模型诊断
python -c "
from src.pretraining.model_validator import PretrainedModelValidator
validator = PretrainedModelValidator()
is_valid, info = validator.validate_model('./outputs/pretrained_actor.pt')
validator.print_model_info(info)
"

# 检查配置兼容性
python -c "
from src.pretraining.model_validator import PretrainedModelValidator
import torch

model_path = './outputs/pretrained_actor.pt'
checkpoint = torch.load(model_path, map_location='cpu')
model_config = checkpoint['model_config']

target_config = {'state_dim': 64, 'action_dim': 4}
validator = PretrainedModelValidator()
is_compatible, message = validator.check_compatibility(model_config, target_config)
print(f'兼容性: {message}')
"
```

#### 清理和重置
```bash
# 清理临时文件
rm -rf ./outputs/temp_*
rm -rf ./outputs/*/temp_demo

# 重置checkpoints
rm -f ./outputs/checkpoints/latest.pt
rm -f ./outputs/checkpoints/*.pt

# 清理TensorBoard日志
rm -rf ./outputs/runs/ppo_training/run_*
rm -rf ./outputs/unified_pretrain/tensorboard
```

---

## 分块训练系统 - 模块化大规模训练

### 分块训练概述

分块训练系统将大型PPO训练任务分解为多个较小的"块"，每个块处理有限数量的请求。这种方法具有以下优势：

- **内存效率**: 避免单次处理大量请求导致的内存溢出
- **检查点管理**: 自动保存和恢复，支持中断后继续训练
- **进度跟踪**: 实时监控训练进度和ETA估算
- **容错性**: 单个块失败不影响整体训练

### 1. 基本分块训练命令

#### 标准分块训练
```bash
# 基础分块训练 - 将20000个请求分成4块，每块5000个请求
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode \
  --chunk-size 5000 \
  --total-requests 20000 \
  --num-replicas 4 \
  --qps 3.5

# 小规模测试分块训练
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode \
  --chunk-size 1000 \
  --total-requests 4000 \
  --num-replicas 2 \
  --qps 2.0
```

#### 高性能分块训练
```bash
# 大规模分块训练 - 8副本高QPS
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode \
  --chunk-size 10000 \
  --total-requests 100000 \
  --num-replicas 8 \
  --qps 5.0 \
  --config configs/ppo_warmstart.json

# 外部预训练 + 分块训练
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode \
  --chunk-size 5000 \
  --total-requests 30000 \
  --external-pretrain ./outputs/unified_pretrain/enhanced_model.pt \
  --skip-bc-training \
  --num-replicas 6 \
  --qps 4.0
```

### 2. 分块训练监控命令

#### 实时进度监控
```bash
# 自动查找最新进度文件并显示状态
bash scripts/monitor_training.sh

# 显示详细进度信息
bash scripts/monitor_training.sh -v

# 持续监控模式 - 每10秒刷新一次
bash scripts/monitor_training.sh -w 10

# 指定进度文件进行监控
bash scripts/monitor_training.sh ./outputs/warmstart_training_optimized/run_20250921_143012/training_progress.json
```

#### 使用模块化监控器
```bash
# 直接调用模块化监控器
python3 -m src.monitoring -v

# JSON格式输出（便于脚本处理）
python3 -m src.monitoring --json

# 持续监控特定文件
python3 -m src.monitoring \
  ./outputs/warmstart_training_optimized/run_*/training_progress.json \
  -w 15 -v
```

### 3. 分块训练恢复和管理

#### 训练恢复
```bash
# 自动恢复中断的分块训练（脚本会自动检测）
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode \
  --chunk-size 5000 \
  --total-requests 20000 \
  --output-dir ./outputs/warmstart_training_optimized/run_20250921_143012

# 强制重新开始分块训练
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode \
  --chunk-size 5000 \
  --total-requests 20000 \
  --force-warmstart
```

#### 直接使用模块化训练器
```bash
# 使用模块化分块训练器 - 更多控制选项
python3 -m src.training \
  --output-dir "./outputs/custom_chunk_training/$(date +%Y%m%d_%H%M%S)" \
  --total-requests 15000 \
  --chunk-size 3000 \
  --config-file configs/ppo_warmstart.json \
  --num-replicas 4 \
  --qps 3.0 \
  --skip-warmstart false
```

### 4. 分块训练配置优化

#### 针对不同硬件的配置建议

**高内存环境** (32GB+ RAM):
```bash
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode \
  --chunk-size 15000 \
  --total-requests 100000 \
  --num-replicas 8 \
  --qps 6.0
```

**中等内存环境** (16GB RAM):
```bash
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode \
  --chunk-size 8000 \
  --total-requests 50000 \
  --num-replicas 4 \
  --qps 4.0
```

**低内存环境** (8GB RAM):
```bash
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode \
  --chunk-size 3000 \
  --total-requests 20000 \
  --num-replicas 2 \
  --qps 2.5
```

### 5. 分块训练故障排除

#### 常见问题诊断
```bash
# 检查进度文件状态
python3 -c "
import json
with open('./outputs/warmstart_training_optimized/run_*/training_progress.json', 'r') as f:
    progress = json.load(f)
print(f'状态: {progress[\"status\"]}')
print(f'完成块数: {progress[\"completed_chunks\"]}/{progress[\"total_chunks\"]}')
print(f'最新检查点: {progress[\"latest_checkpoint\"]}')
"

# 查看块日志文件
tail -n 50 ./outputs/warmstart_training_optimized/run_*/chunk_*.log

# 检查检查点文件
ls -la ./outputs/checkpoints/
python3 -c "
import torch
try:
    checkpoint = torch.load('./outputs/checkpoints/latest.pt', map_location='cpu')
    print('检查点文件有效')
    print(f'模型配置: {checkpoint.get(\"model_config\", \"无\")}')
except Exception as e:
    print(f'检查点文件问题: {e}')
"
```

#### 手动清理和重置
```bash
# 清理失败的分块训练
rm -rf ./outputs/warmstart_training_optimized/run_*/chunk_*.log
rm -f ./outputs/warmstart_training_optimized/run_*/training_progress.json

# 重置检查点（谨慎使用）
rm -f ./outputs/checkpoints/latest.pt
rm -f ./outputs/checkpoints/*.pt
```

### 6. 高级分块训练配置

#### 自定义检查点策略
分块训练会自动使用配置文件中的检查点设置：

- `save_optimizer_state: true` - 保存优化器状态，确保学习率调度连续性
- `incremental_checkpoints: true` - 启用增量检查点，保留训练历史

#### 监控集成
```bash
# 同时启动训练和监控
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode \
  --chunk-size 5000 \
  --total-requests 20000 &

# 在另一个终端启动监控
bash scripts/monitor_training.sh -w 30 -v
```

### 7. 分块训练最佳实践

#### 推荐工作流程
```bash
# 1. 小规模测试（验证配置）
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode --chunk-size 1000 --total-requests 3000

# 2. 中等规模验证（确认稳定性）
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode --chunk-size 5000 --total-requests 15000

# 3. 大规模生产训练
bash scripts/train_ppo_warmstart_optimized.sh \
  --chunk-mode --chunk-size 10000 --total-requests 100000
```

#### 性能调优建议
- **块大小**: 根据可用内存选择，一般为 1000-15000 请求
- **总请求数**: 确保足够的训练样本，建议 20000+ 用于收敛
- **QPS**: 平衡训练速度和系统负载，建议从低值开始调试
- **副本数**: 与块大小配合，避免内存压力

### 8. 分块训练输出文件

分块训练会产生以下文件结构：
```
outputs/warmstart_training_optimized/run_YYYYMMDD_HHMMSS/
├── training_progress.json          # 进度跟踪文件
├── chunk_1.log                     # 第1块训练日志
├── chunk_2.log                     # 第2块训练日志
├── ...
├── tensorboard/                    # TensorBoard日志（连续）
│   └── events.out.tfevents.*
└── metrics/                        # CSV导出指标
    └── training_metrics.csv

outputs/checkpoints/
├── latest.pt                       # 最新检查点（软链接）
├── checkpoint_epoch_*.pt           # 历史检查点
└── optimizer_*.pt                  # 优化器状态文件
```

---

**相关文档**:
- `ppo_optimization_2025_09_18.md` - 详细技术方案
- `config_changes_summary.md` - 参数变更对照
- `experiments/ppo_optimization_v1.json` - 实验配置
- `unified_pretraining_guide.md` - 统一预训练系统指南
- `standalone_pretraining_guide.md` - 独立预训练系统指南

**新增模块化组件**:
- `src/training/` - 分块训练核心模块
- `src/monitoring/` - 进度监控模块
- `scripts/monitor_training.sh` - 简化监控脚本