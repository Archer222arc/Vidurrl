#!/bin/bash

# =============================================================================
# PPO训练脚本 - 包含完整checkpoint管理
#
# 基于最新优化参数，支持StateBuilder配置保存和自动兼容推理模式
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"

RUN_ID=$(date +%Y%m%d_%H%M%S)

echo "🚀 开始PPO训练 - Run ID: ${RUN_ID}"
echo "📋 配置:"
echo "   - 请求数量: 5000"
echo "   - QPS: 2"
echo "   - 副本数: 4"
echo "   - 输出目录: ./outputs/runs/ppo_training/run_${RUN_ID}"
echo ""

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
  --synthetic_request_generator_config_num_requests 5000 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 3 \
  --metrics_config_subsamples 200000 \
  --p_p_o_global_scheduler_modular_config_tensorboard_log_dir ./outputs/runs/ppo_training/run_${RUN_ID} \
  --p_p_o_global_scheduler_modular_config_tensorboard_force_kill \
  --p_p_o_global_scheduler_modular_config_tensorboard_port 6006 \
  --p_p_o_global_scheduler_modular_config_tensorboard_auto_start \
  --p_p_o_global_scheduler_modular_config_metrics_export_enabled \
  --p_p_o_global_scheduler_modular_config_metrics_export_format csv \
  --p_p_o_global_scheduler_modular_config_metrics_export_path ./outputs/runs/ppo_training/exports/run_${RUN_ID} \
  --p_p_o_global_scheduler_modular_config_metrics_export_interval 50 \
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
  --p_p_o_global_scheduler_modular_config_enable_checkpoints \
  --p_p_o_global_scheduler_modular_config_checkpoint_dir ./outputs/checkpoints \
  --p_p_o_global_scheduler_modular_config_checkpoint_interval 128 \
  --p_p_o_global_scheduler_modular_config_max_checkpoints 5 \
  > run_modular_${RUN_ID}.log 2>&1

echo ""
echo "🎉 训练完成！"
echo "📂 日志文件: run_modular_${RUN_ID}.log"
echo "📊 TensorBoard: http://localhost:6006"
echo "📝 指标导出: ./outputs/runs/ppo_training/exports/run_${RUN_ID}"
echo "💾 Checkpoint: ./outputs/checkpoints/latest.pt"
echo ""
echo "🔗 快速测试推理:"
echo "   bash scripts/test_inference.sh"
