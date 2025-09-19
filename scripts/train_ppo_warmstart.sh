#!/bin/bash

# =============================================================================
# PPO热身启动训练脚本 - 完整的两阶段训练流程
#
# 阶段1: 收集示教数据并进行行为克隆预训练
# 阶段2: 使用KL正则化的PPO训练
# =============================================================================

set -e

RUN_ID=$(date +%Y%m%d_%H%M%S)
echo "🚀 开始PPO热身启动训练 - Run ID: ${RUN_ID}"

# 配置参数
DEMO_POLICY="round_robin"
DEMO_STEPS=1000
BC_EPOCHS=20
PPO_REQUESTS=5000
QPS=3
NUM_REPLICAS=4

# 输出目录
OUTPUT_DIR="./outputs/warmstart_training/run_${RUN_ID}"
DEMO_DATA_PATH="${OUTPUT_DIR}/demo_data.pkl"
PRETRAINED_ACTOR_PATH="${OUTPUT_DIR}/pretrained_actor.pt"

mkdir -p "${OUTPUT_DIR}"

echo "📋 训练配置:"
echo "   - 示教策略: ${DEMO_POLICY}"
echo "   - 示教步数: ${DEMO_STEPS}"
echo "   - BC轮数: ${BC_EPOCHS}"
echo "   - PPO请求数: ${PPO_REQUESTS}"
echo "   - QPS: ${QPS}"
echo "   - 副本数: ${NUM_REPLICAS}"
echo "   - 输出目录: ${OUTPUT_DIR}"
echo ""

# =============================================================================
# 阶段1: 收集示教数据
# =============================================================================
echo "📊 [阶段1] 收集示教数据..."

python scripts/collect_demo.py \
  --policy "${DEMO_POLICY}" \
  --steps "${DEMO_STEPS}" \
  --replicas "${NUM_REPLICAS}" \
  --qps "${QPS}" \
  --output "${DEMO_DATA_PATH}"

if [ $? -eq 0 ]; then
    echo "✅ 示教数据收集完成"
else
    echo "❌ 示教数据收集失败"
    exit 1
fi

# =============================================================================
# 阶段2: 行为克隆预训练
# =============================================================================
echo "🤖 [阶段2] Actor预训练 (行为克隆)..."

python scripts/pretrain_actor.py \
  --demo "${DEMO_DATA_PATH}" \
  --epochs "${BC_EPOCHS}" \
  --batch_size 256 \
  --lr 1e-3 \
  --hidden_size 128 \
  --layer_N 2 \
  --gru_layers 2 \
  --output "${PRETRAINED_ACTOR_PATH}"

if [ $? -eq 0 ]; then
    echo "✅ Actor预训练完成"
else
    echo "❌ Actor预训练失败"
    exit 1
fi

# =============================================================================
# 阶段3: PPO训练 (带KL正则化)
# =============================================================================
echo "🚀 [阶段3] PPO训练 (热身启动 + KL正则化)..."

python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  --cluster_config_num_replicas "${NUM_REPLICAS}" \
  --p_p_o_global_scheduler_modular_config_max_queue_requests_per_replica 8 \
  --p_p_o_global_scheduler_modular_config_lr 0.0003 \
  --p_p_o_global_scheduler_modular_config_gamma 0.95 \
  --p_p_o_global_scheduler_modular_config_reward_latency_weight 1.5 \
  --p_p_o_global_scheduler_modular_config_balance_penalty_weight 0 \
  --p_p_o_global_scheduler_modular_config_entropy_coef 0.25 \
  --synthetic_request_generator_config_num_requests "${PPO_REQUESTS}" \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps "${QPS}" \
  --metrics_config_subsamples 200000 \
  --p_p_o_global_scheduler_modular_config_tensorboard_log_dir "${OUTPUT_DIR}/tensorboard" \
  --p_p_o_global_scheduler_modular_config_tensorboard_force_kill \
  --p_p_o_global_scheduler_modular_config_tensorboard_port 6006 \
  --p_p_o_global_scheduler_modular_config_tensorboard_auto_start \
  --p_p_o_global_scheduler_modular_config_metrics_export_enabled \
  --p_p_o_global_scheduler_modular_config_metrics_export_format csv \
  --p_p_o_global_scheduler_modular_config_metrics_export_path "${OUTPUT_DIR}/metrics" \
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
  --p_p_o_global_scheduler_modular_config_enable_warm_start \
  --p_p_o_global_scheduler_modular_config_pretrained_actor_path "${PRETRAINED_ACTOR_PATH}" \
  --p_p_o_global_scheduler_modular_config_kl_ref_coef_initial 0.5 \
  --p_p_o_global_scheduler_modular_config_kl_ref_coef_final 0.0 \
  --p_p_o_global_scheduler_modular_config_kl_ref_decay_steps 1000 \
  --p_p_o_global_scheduler_modular_config_warmup_steps 500 \
  --p_p_o_global_scheduler_modular_config_entropy_warmup_coef 0.5 \
  > "${OUTPUT_DIR}/ppo_training.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ PPO训练完成"
else
    echo "❌ PPO训练失败，请检查日志: ${OUTPUT_DIR}/ppo_training.log"
fi

# =============================================================================
# 结果汇总
# =============================================================================
echo ""
echo "🎉 PPO热身启动训练完成！"
echo "📂 输出目录: ${OUTPUT_DIR}"
echo "📊 训练日志: ${OUTPUT_DIR}/ppo_training.log"
echo "📈 TensorBoard: http://localhost:6006"
echo "💾 最新Checkpoint: ./outputs/checkpoints/latest.pt"
echo ""
echo "🔗 快速测试推理:"
echo "   bash scripts/scheduler_comparison.sh"
echo ""
echo "📋 输出文件说明:"
echo "   - ${DEMO_DATA_PATH}: 示教数据"
echo "   - ${PRETRAINED_ACTOR_PATH}: 预训练Actor模型"
echo "   - ${OUTPUT_DIR}/tensorboard/: TensorBoard日志"
echo "   - ${OUTPUT_DIR}/metrics/: 训练指标CSV"
echo "   - ./outputs/checkpoints/: PPO模型checkpoint"