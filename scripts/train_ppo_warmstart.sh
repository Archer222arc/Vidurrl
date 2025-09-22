#!/bin/bash

# =============================================================================
# PPO热身启动训练脚本 - 优化版本
#
# 使用方法：
#   bash scripts/train_ppo_warmstart_optimized.sh [选项]
#
# 选项：
#   --config FILE              配置文件路径 (默认: configs/ppo_warmstart.json)
#   --num-replicas N           副本数量 (默认: 4)
#   --qps RATE                 QPS速率 (默认: 3.0)
#   --ppo-requests N           PPO训练请求数 (默认: 8000)
#   --bc-epochs N              BC训练轮数 (默认: 30)
#   --demo-steps N             每策略示教步数 (默认: 700)
#   --output-dir DIR           输出目录 (默认: auto-generated)
#   --external-pretrain PATH   使用外部预训练模型路径
#   --skip-bc-training         跳过BC预训练 (配合--external-pretrain使用)
#   --force-warmstart          强制执行warmstart (忽略checkpoint)
#   --help                     显示帮助信息
# =============================================================================

set -e

# 显示帮助信息
show_help() {
    echo "PPO热身启动训练脚本 - 优化版本"
    echo ""
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --config FILE              配置文件路径 (默认: configs/ppo_warmstart.json)"
    echo "  --num-replicas N           副本数量 (默认: 4)"
    echo "  --qps RATE                 QPS速率 (默认: 3.0)"
    echo "  --ppo-requests N           PPO训练请求数 (默认: 8000)"
    echo "  --bc-epochs N              BC训练轮数 (默认: 30)"
    echo "  --demo-steps N             每策略示教步数 (默认: 700)"
    echo "  --output-dir DIR           输出目录 (默认: auto-generated)"
    echo "  --external-pretrain PATH   使用外部预训练模型路径"
    echo "  --skip-bc-training         跳过BC预训练 (配合--external-pretrain使用)"
    echo "  --force-warmstart          强制执行warmstart (忽略checkpoint)"
    echo "  --disable-stats-stabilization 禁用统计量稳定化 (默认启用)"
    echo "  --help                     显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认配置"
    echo "  $0 --num-replicas 8 --qps 5.0       # 自定义副本数和QPS"
    echo "  $0 --config configs/my_config.json   # 使用自定义配置"
    echo "  $0 --force-warmstart                 # 强制重新训练"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"

# 默认配置参数
CONFIG_FILE=""
DEMO_POLICIES="round_robin lor random"
DEMO_STEPS_PER_POLICY=700
BC_EPOCHS=30
PPO_REQUESTS=20000
QPS=2.5
NUM_REPLICAS=4
OUTPUT_DIR=""
EXTERNAL_PRETRAIN=""
SKIP_BC_TRAINING=false
FORCE_WARMSTART=false
DISABLE_STATS_STABILIZATION=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --num-replicas)
            NUM_REPLICAS="$2"
            shift 2
            ;;
        --qps)
            QPS="$2"
            shift 2
            ;;
        --ppo-requests)
            PPO_REQUESTS="$2"
            shift 2
            ;;
        --bc-epochs)
            BC_EPOCHS="$2"
            shift 2
            ;;
        --demo-steps)
            DEMO_STEPS_PER_POLICY="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --external-pretrain)
            EXTERNAL_PRETRAIN="$2"
            shift 2
            ;;
        --skip-bc-training)
            SKIP_BC_TRAINING=true
            shift
            ;;
        --force-warmstart)
            FORCE_WARMSTART=true
            shift
            ;;
        --disable-stats-stabilization)
            DISABLE_STATS_STABILIZATION=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 参数验证
if [[ "$SKIP_BC_TRAINING" == "true" && -z "$EXTERNAL_PRETRAIN" ]]; then
    echo "❌ 错误: --skip-bc-training 必须配合 --external-pretrain 使用"
    exit 1
fi

if [[ -n "$EXTERNAL_PRETRAIN" && ! -f "$EXTERNAL_PRETRAIN" ]]; then
    echo "❌ 错误: 外部预训练模型文件不存在: $EXTERNAL_PRETRAIN"
    exit 1
fi

# 加载配置文件 (如果指定)
if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
    echo "📄 加载配置文件: $CONFIG_FILE"
    # 这里可以添加JSON配置文件解析逻辑
fi

RUN_ID=$(date +%Y%m%d_%H%M%S)
echo "🚀 开始PPO优化热身启动训练 - Run ID: ${RUN_ID}"

# 设置输出目录
if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ -n "$EXTERNAL_PRETRAIN" ]]; then
        OUTPUT_DIR="./outputs/warmstart_external/run_${RUN_ID}"
    else
        OUTPUT_DIR="./outputs/warmstart_training_optimized/run_${RUN_ID}"
    fi
fi
DEMO_DATA_PATH="${OUTPUT_DIR}/demo_data.pkl"
PRETRAINED_ACTOR_PATH="${OUTPUT_DIR}/pretrained_actor.pt"

mkdir -p "${OUTPUT_DIR}"

# 外部预训练模型验证和处理
if [[ -n "$EXTERNAL_PRETRAIN" ]]; then
    echo "🔍 验证外部预训练模型..."
    python -m src.pretraining.model_validator "$EXTERNAL_PRETRAIN"

    if [ $? -ne 0 ]; then
        echo "❌ 外部预训练模型验证失败"
        exit 1
    fi

    # 复制外部预训练模型到输出目录
    cp "$EXTERNAL_PRETRAIN" "$PRETRAINED_ACTOR_PATH"
    echo "📂 外部预训练模型已复制到: $PRETRAINED_ACTOR_PATH"
fi

# =============================================================================
# Resume功能检查 - 交互式控制
# =============================================================================
LATEST_CHECKPOINT="./outputs/checkpoints/latest.pt"
RESUME_ARGS=""
SKIP_WARMSTART=false

# 检查是否强制warmstart
if [[ "$FORCE_WARMSTART" == "true" ]]; then
    echo "🔥 --force-warmstart 选项已启用，强制执行完整warmstart训练"
    SKIP_WARMSTART=false
elif [ -f "${LATEST_CHECKPOINT}" ]; then
    echo "🔄 发现existing checkpoint: ${LATEST_CHECKPOINT}"
    echo "📊 Checkpoint信息:"
    echo "   - 文件: $(readlink ${LATEST_CHECKPOINT} 2>/dev/null || echo ${LATEST_CHECKPOINT})"
    echo "   - 大小: $(du -h ${LATEST_CHECKPOINT} | cut -f1)"
    echo "   - 修改时间: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" ${LATEST_CHECKPOINT})"
    echo ""
    echo "🤔 是否从checkpoint恢复训练? (这将跳过warmstart阶段)"
    echo "   y/Y) 从checkpoint恢复"
    echo "   n/N) 重新开始完整训练"
    echo "   q/Q) 退出脚本"

    while true; do
        read -p "请选择 [y/n/q]: " choice
        case $choice in
            [Yy]* )
                echo "✅ 选择从checkpoint恢复训练"
                RESUME_ARGS="--p_p_o_global_scheduler_modular_config_load_checkpoint ${LATEST_CHECKPOINT}"
                SKIP_WARMSTART=true
                break;;
            [Nn]* )
                echo "🆕 选择重新开始完整训练"
                SKIP_WARMSTART=false
                break;;
            [Qq]* )
                echo "👋 退出脚本"
                exit 0;;
            * ) echo "❌ 请输入 y, n 或 q";;
        esac
    done
else
    echo "🆕 未发现checkpoint，执行完整warmstart训练流程"
    SKIP_WARMSTART=false
fi

echo "📋 优化训练配置:"
if [ "$SKIP_WARMSTART" = true ]; then
    echo "   - Resume模式: 从 ${LATEST_CHECKPOINT} 恢复"
    echo "   - 跳过: 示教数据收集 + BC预训练"
elif [[ -n "$EXTERNAL_PRETRAIN" ]]; then
    echo "   - 🌟 外部预训练模式: 使用高质量预训练模型"
    echo "   - 预训练模型: ${EXTERNAL_PRETRAIN}"
    if [[ "$SKIP_BC_TRAINING" == "true" ]]; then
        echo "   - BC训练: 跳过 (直接使用外部模型)"
    else
        echo "   - BC训练: ${BC_EPOCHS} epochs (在外部模型基础上微调)"
        echo "   - 示教策略: ${DEMO_POLICIES}"
        echo "   - 每策略步数: ${DEMO_STEPS_PER_POLICY}"
    fi
else
    echo "   - 标准预训练模式: 从零开始BC训练"
    echo "   - 示教策略: ${DEMO_POLICIES} (混合策略)"
    echo "   - 每策略步数: ${DEMO_STEPS_PER_POLICY} (总计$(($(echo ${DEMO_POLICIES} | wc -w) * ${DEMO_STEPS_PER_POLICY}))步)"
    echo "   - BC轮数: ${BC_EPOCHS} (增加)"
fi
echo "   - PPO请求数: ${PPO_REQUESTS} (增加)"
echo "   - QPS: ${QPS}"
echo "   - 副本数: ${NUM_REPLICAS}"
echo "   - 输出目录: ${OUTPUT_DIR}"
echo ""

# =============================================================================
# 阶段1&2: Warmstart数据收集和预训练 (条件执行)
# =============================================================================
if [ "$SKIP_WARMSTART" = false ]; then

    # 阶段1: 示教数据收集 (如果需要)
    if [[ "$SKIP_BC_TRAINING" == "false" ]]; then
        echo "📊 [阶段1] 收集混合策略示教数据..."

    # 收集混合策略示教数据 - 使用模块化方法
    collect_mixed_demo() {
        local output_path="$1"
        local temp_dir="${OUTPUT_DIR}/temp_demo"

        python -m src.demo_collection.mixed_collector \
          --output "${output_path}" \
          --policies ${DEMO_POLICIES} \
          --steps_per_policy "${DEMO_STEPS_PER_POLICY}" \
          --num_replicas "${NUM_REPLICAS}" \
          --qps "${QPS}" \
          --temp_dir "${temp_dir}" \
          --include_imbalanced
    }

    collect_mixed_demo "${DEMO_DATA_PATH}"

        if [ $? -eq 0 ]; then
            echo "✅ 示教数据收集完成"
        else
            echo "❌ 示教数据收集失败"
            exit 1
        fi
    else
        echo "⏭️ [阶段1] 跳过示教数据收集 (使用外部预训练模型)"
    fi

    # 阶段2: Actor预训练
    if [[ "$SKIP_BC_TRAINING" == "false" ]]; then
        if [[ -n "$EXTERNAL_PRETRAIN" ]]; then
            echo "🔧 [阶段2] Actor微调 (基于外部预训练模型)..."
        else
            echo "🤖 [阶段2] Actor预训练 (行为克隆) - 增强版..."
        fi

    python scripts/pretrain_actor.py \
      --demo "${DEMO_DATA_PATH}" \
      --epochs "${BC_EPOCHS}" \
      --batch_size 256 \
      --lr 5e-4 \
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
    else
        echo "⏭️ [阶段2] 跳过BC预训练 (使用外部预训练模型)"
    fi
else
    echo "⏭️ 跳过warmstart阶段 (从checkpoint恢复)"
fi

# =============================================================================
# 阶段3: PPO训练 (优化参数)
# =============================================================================
if [ "$SKIP_WARMSTART" = true ]; then
    echo "🚀 [Resume] PPO训练恢复 (从checkpoint: ${LATEST_CHECKPOINT})..."
else
    echo "🚀 [阶段3] PPO训练 (优化版 - 平衡exploration/exploitation)..."
fi

# 构建训练命令
echo "🔧 构建训练参数..."

# 使用配置文件或默认配置
if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
    echo "📄 使用配置文件: $CONFIG_FILE"
    PPO_ARGS=$(python src/config/training_config.py "$CONFIG_FILE" "$OUTPUT_DIR")
else
    echo "📄 使用默认配置: configs/ppo_warmstart.json"
    PPO_ARGS=$(python src/config/training_config.py "configs/ppo_warmstart.json" "$OUTPUT_DIR")
fi

# 覆盖命令行参数
PPO_ARGS="$PPO_ARGS --cluster_config_num_replicas ${NUM_REPLICAS}"
PPO_ARGS="$PPO_ARGS --poisson_request_interval_generator_config_qps ${QPS}"
PPO_ARGS="$PPO_ARGS --synthetic_request_generator_config_num_requests ${PPO_REQUESTS}"

# 添加额外的固定参数
PPO_ARGS="$PPO_ARGS --p_p_o_global_scheduler_modular_config_max_queue_requests_per_replica 8"

# 统计量稳定化参数
if [ "$DISABLE_STATS_STABILIZATION" = false ]; then
    PPO_ARGS="$PPO_ARGS --p_p_o_global_scheduler_modular_config_enable_statistics_stabilization"
    PPO_ARGS="$PPO_ARGS --p_p_o_global_scheduler_modular_config_enable_stabilization_logging"
    echo "📊 统计量稳定化: 已启用 (前100步使用随机策略收集统计)"
else
    PPO_ARGS="$PPO_ARGS --no-p_p_o_global_scheduler_modular_config_enable_statistics_stabilization"
    echo "📊 统计量稳定化: 已禁用"
fi

echo "🚀 启动PPO训练..."
python -m vidur.main \
  $PPO_ARGS \
  ${RESUME_ARGS} \
  $(if [ "$SKIP_WARMSTART" = false ]; then echo "--p_p_o_global_scheduler_modular_config_enable_warm_start --p_p_o_global_scheduler_modular_config_pretrained_actor_path ${PRETRAINED_ACTOR_PATH}"; fi) \
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
if [ "$SKIP_WARMSTART" = true ]; then
    echo "🎉 PPO训练恢复完成！"
    echo "🔄 从checkpoint恢复: ${LATEST_CHECKPOINT}"
else
    echo "🎉 PPO优化热身启动训练完成！"
fi
echo "📂 输出目录: ${OUTPUT_DIR}"
echo "📊 训练日志: ${OUTPUT_DIR}/ppo_training.log"
echo "📈 TensorBoard: http://localhost:6006"
echo "💾 最新Checkpoint: ./outputs/checkpoints/latest.pt"
echo ""
echo "🔍 负载均衡优化改进点："
echo "   - 混合策略示教: ${DEMO_POLICIES} + 极端不均衡场景"
echo "   - 强化BC预训练: ${BC_EPOCHS} epochs"
echo "   - 🔥 真正的Warmstart: 加载预训练权重 + KL约束生效"
echo "   - 压制warmup随机性: entropy_warmup_coef=0.0, min_temp=0.5"
echo "   - 强化KL约束: initial=0.6, final=0.1, decay=3000 steps"
echo "   - 🛡️ 稳定化期: 前1500步只推理不更新参数"
echo "   - 加强负载惩罚: balance_penalty=0.3, load_balance=0.3"
echo "   - 调整奖励权重: alpha=0.2 (降低throughput压制)"
echo ""
echo "🔗 快速测试推理:"
echo "   bash scripts/scheduler_comparison.sh"
echo ""
echo "📋 优化版输出文件说明:"
echo "   - ${DEMO_DATA_PATH}: 增强示教数据"
echo "   - ${PRETRAINED_ACTOR_PATH}: 强化预训练Actor模型"
echo "   - ${OUTPUT_DIR}/tensorboard/: TensorBoard日志"
echo "   - ${OUTPUT_DIR}/metrics/: 训练指标CSV"
echo "   - ./outputs/checkpoints/: PPO模型checkpoint"