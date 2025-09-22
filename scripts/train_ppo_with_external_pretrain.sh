#!/bin/bash

# =============================================================================
# PPO训练脚本 - 支持外部预训练模型
#
# 使用方法：
#   bash scripts/train_ppo_with_external_pretrain.sh [外部模型路径] [其他选项]
#
# 示例：
#   bash scripts/train_ppo_with_external_pretrain.sh ./outputs/standalone_pretrain/best_model.pt
#   bash scripts/train_ppo_with_external_pretrain.sh ./outputs/standalone_pretrain/best_model.pt --num-replicas 8
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"

# 解析参数
EXTERNAL_PRETRAIN="$1"
shift  # 移除第一个参数，剩余参数传递给原脚本

# 验证外部预训练模型
if [[ -z "$EXTERNAL_PRETRAIN" ]]; then
    echo "❌ 使用方法: $0 <外部预训练模型路径> [其他选项]"
    exit 1
fi

if [[ ! -f "$EXTERNAL_PRETRAIN" ]]; then
    echo "❌ 外部预训练模型文件不存在: $EXTERNAL_PRETRAIN"
    exit 1
fi

echo "🔍 验证外部预训练模型..."
echo "✅ Vidur: 使用核心统一结构"
python -m src.core.algorithms.training.pretraining.model_validator "$EXTERNAL_PRETRAIN"

if [ $? -ne 0 ]; then
    echo "❌ 外部预训练模型验证失败"
    exit 1
fi

echo ""
echo "✅ 外部预训练模型验证通过"
echo "🚀 启动PPO训练 (使用外部预训练模型)"

# 调用增强版warmstart脚本，传递外部预训练模型和其他参数
bash scripts/train_ppo_warmstart_optimized.sh \
    --external-pretrain "$EXTERNAL_PRETRAIN" \
    --skip-bc-training \
    "$@"