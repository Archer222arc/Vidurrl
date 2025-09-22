#!/bin/bash

# =============================================================================
# 独立预训练脚本 - 简洁版本
#
# 使用方法：
#   bash scripts/standalone_pretrain.sh [配置文件]
#
# 示例：
#   bash scripts/standalone_pretrain.sh                              # 使用默认配置
#   bash scripts/standalone_pretrain.sh configs/standalone_pretrain.json  # 使用自定义配置
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"

# 默认配置文件
DEFAULT_CONFIG="configs/standalone_pretrain.json"
CONFIG_FILE="${1:-$DEFAULT_CONFIG}"

echo "🚀 独立预训练开始"
echo "📄 配置文件: $CONFIG_FILE"

# 检查配置文件
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    echo "请创建配置文件或使用默认配置"
    exit 1
fi

echo "📊 配置内容:"
cat "$CONFIG_FILE"
echo ""

# 检查和管理预训练示教数据
DEMO_DIR="./data/pretraining"
LARGE_DEMO_FILE="${DEMO_DIR}/large_demo_dataset.pkl"
STANDARD_DEMO_FILE="${DEMO_DIR}/standard_demo_dataset.pkl"

# 创建预训练数据目录
mkdir -p "$DEMO_DIR"

# 优先使用大规模数据集，其次使用标准数据集
if [[ -f "$LARGE_DEMO_FILE" ]]; then
    DEMO_FILE="$LARGE_DEMO_FILE"
    echo "📂 找到大规模预训练数据集: $LARGE_DEMO_FILE"
    echo "🎯 使用高质量8K样本数据集进行预训练"
elif [[ -f "$STANDARD_DEMO_FILE" ]]; then
    DEMO_FILE="$STANDARD_DEMO_FILE"
    echo "📂 找到标准预训练数据集: $STANDARD_DEMO_FILE"
    echo "♻️  重复使用已收集的数据，节省时间"
else
    DEMO_FILE="$STANDARD_DEMO_FILE"
    echo "📊 未找到预训练数据集，开始收集..."
    echo "💡 此数据集将被重复使用，避免重复收集"

    # 收集标准的预训练数据集（更大规模）
    python -m src.demo_collection.mixed_collector \
        --output "$DEMO_FILE" \
        --policies round_robin lor random \
        --steps_per_policy 1000 \
        --num_replicas 4 \
        --qps 3.0 \
        --temp_dir "${DEMO_DIR}/temp_collection" \
        --include_imbalanced

    if [ $? -eq 0 ]; then
        echo "✅ 标准预训练数据集收集完成"
        echo "📂 保存位置: $DEMO_FILE"
        echo "💾 此数据集可重复用于多次预训练实验"
    else
        echo "❌ 示教数据收集失败"
        exit 1
    fi
fi

# 显示数据集信息
if [[ -f "$DEMO_FILE" ]]; then
    python -c "
import pickle
with open('$DEMO_FILE', 'rb') as f:
    data = pickle.load(f)
stats = data.get('stats', {})
print(f'📊 数据集信息: {stats.get(\"total_samples\", \"unknown\")} 样本')
print(f'🎯 策略分布: {list(stats.get(\"policy_distribution\", {}).keys())}')
"
fi

# 调用统一预训练管理器
python -m src.pretraining.unified_trainer --config "$CONFIG_FILE" --demo-files "$DEMO_FILE"

if [ $? -eq 0 ]; then
    echo "✅ 独立预训练完成"
    echo "📂 查看输出目录中的 best_model.pt 文件"
else
    echo "❌ 独立预训练失败"
    exit 1
fi