#!/bin/bash

# =============================================================================
# Checkpoint兼容性修复脚本
#
# 解决模型架构变更导致的checkpoint加载错误：
# "Unexpected key(s) in state_dict: temporal_back_projection.weight"
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${REPO_ROOT}/outputs/checkpoints"

echo "🔧 Checkpoint兼容性修复脚本"
echo "=" * 50

# 检查checkpoint目录
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

echo "📂 Checkpoint目录: $CHECKPOINT_DIR"
echo "📋 当前checkpoint文件:"
ls -la "$CHECKPOINT_DIR"

echo ""
echo "⚠️  检测到模型架构不兼容问题："
echo "   - 旧checkpoint包含temporal_back_projection层"
echo "   - 当前代码架构已更新，不再包含这些层"
echo "   - 需要清理不兼容的checkpoint文件"

echo ""
echo "🤔 请选择修复方案："
echo "   1) 备份并清理所有checkpoint (推荐 - 重新开始训练)"
echo "   2) 仅删除latest.pt链接 (保留历史checkpoint，但从头训练)"
echo "   3) 查看checkpoint详细信息后决定"
echo "   q) 退出不做任何修改"

while true; do
    read -p "请选择 [1/2/3/q]: " choice
    case $choice in
        1 )
            echo "✅ 选择方案1: 备份并清理所有checkpoint"

            # 创建备份目录
            BACKUP_DIR="${CHECKPOINT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
            echo "📦 创建备份目录: $BACKUP_DIR"
            cp -r "$CHECKPOINT_DIR" "$BACKUP_DIR"

            # 清理checkpoint目录
            echo "🧹 清理checkpoint目录..."
            rm -f "$CHECKPOINT_DIR"/*.pt

            echo "✅ 清理完成!"
            echo "📁 备份位置: $BACKUP_DIR"
            echo "🚀 现在可以重新开始训练"
            break;;

        2 )
            echo "✅ 选择方案2: 仅删除latest.pt链接"

            if [ -L "$CHECKPOINT_DIR/latest.pt" ]; then
                echo "🔗 删除latest.pt链接..."
                rm "$CHECKPOINT_DIR/latest.pt"
                echo "✅ latest.pt已删除"
                echo "📁 历史checkpoint文件保留"
                echo "🚀 训练将从头开始，但历史文件可供分析"
            else
                echo "ℹ️  latest.pt不存在或不是链接"
            fi
            break;;

        3 )
            echo "🔍 查看checkpoint详细信息..."

            # 检查最新checkpoint的内容
            if [ -f "$CHECKPOINT_DIR/latest.pt" ]; then
                echo "📊 Latest checkpoint信息:"
                python3 -c "
import torch
import sys
try:
    checkpoint = torch.load('$CHECKPOINT_DIR/latest.pt', map_location='cpu')
    print(f'Checkpoint keys: {list(checkpoint.keys())}')
    if 'model_state_dict' in checkpoint:
        model_keys = list(checkpoint['model_state_dict'].keys())
        temporal_keys = [k for k in model_keys if 'temporal' in k]
        print(f'Total model parameters: {len(model_keys)}')
        print(f'Temporal-related keys: {temporal_keys}')
        if 'training_step' in checkpoint:
            print(f'Training step: {checkpoint[\"training_step\"]}')
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            print(f'Model config: {config}')
except Exception as e:
    print(f'Error loading checkpoint: {e}')
    sys.exit(1)
"
            else
                echo "❌ 找不到latest.pt文件"
            fi

            echo ""
            echo "🤔 查看信息后，请重新选择修复方案:"
            continue;;

        [Qq]* )
            echo "👋 退出脚本，未进行任何修改"
            exit 0;;

        * )
            echo "❌ 请输入 1, 2, 3 或 q";;
    esac
done

echo ""
echo "✅ Checkpoint兼容性修复完成"
echo "🚀 现在可以重新运行训练脚本:"
echo "   bash scripts/train_ppo_warmstart_optimized.sh --force-warmstart"