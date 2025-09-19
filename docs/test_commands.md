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

**相关文档**:
- `ppo_optimization_2025_09_18.md` - 详细技术方案
- `config_changes_summary.md` - 参数变更对照
- `experiments/ppo_optimization_v1.json` - 实验配置