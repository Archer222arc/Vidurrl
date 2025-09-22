#!/bin/bash

# =============================================================================
# 预训练数据管理脚本
#
# 使用方法：
#   bash scripts/manage_pretrain_data.sh [命令]
#
# 命令：
#   collect        收集新的标准预训练数据集
#   collect-large  收集大规模预训练数据集
#   collect-custom 收集自定义配置的数据集
#   info           显示现有数据集信息
#   clean          清理临时和旧数据
#   list           列出所有可用数据集
#   help           显示帮助信息
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"

DEMO_DIR="./data/pretraining"
STANDARD_DEMO_FILE="${DEMO_DIR}/standard_demo_dataset.pkl"
LARGE_DEMO_FILE="${DEMO_DIR}/large_demo_dataset.pkl"

# 显示帮助信息
show_help() {
    echo "预训练数据管理脚本"
    echo ""
    echo "使用方法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  collect        收集标准预训练数据集 (4策略 x 1000步)"
    echo "  collect-large  收集大规模预训练数据集 (4策略 x 2000步)"
    echo "  collect-custom 收集自定义配置的数据集"
    echo "  info           显示现有数据集信息"
    echo "  clean          清理临时和旧数据"
    echo "  list           列出所有可用数据集"
    echo "  help           显示此帮助信息"
    echo ""
    echo "数据集说明:"
    echo "  - 标准数据集: 适用于快速预训练实验"
    echo "  - 大规模数据集: 适用于高质量预训练模型"
    echo "  - 自定义数据集: 根据特定需求收集"
}

# 收集标准数据集
collect_standard() {
    echo "📊 收集标准预训练数据集..."
    mkdir -p "$DEMO_DIR"

    python -m src.demo_collection.mixed_collector \
        --output "$STANDARD_DEMO_FILE" \
        --policies round_robin lor random \
        --steps_per_policy 1000 \
        --num_replicas 4 \
        --qps 3.0 \
        --temp_dir "${DEMO_DIR}/temp_standard" \
        --include_imbalanced

    if [ $? -eq 0 ]; then
        echo "✅ 标准数据集收集完成: $STANDARD_DEMO_FILE"
    else
        echo "❌ 标准数据集收集失败"
        exit 1
    fi
}

# 收集大规模数据集
collect_large() {
    echo "📊 收集大规模预训练数据集..."
    mkdir -p "$DEMO_DIR"

    python -m src.demo_collection.mixed_collector \
        --output "$LARGE_DEMO_FILE" \
        --policies round_robin lor random \
        --steps_per_policy 2000 \
        --num_replicas 8 \
        --qps 5.0 \
        --temp_dir "${DEMO_DIR}/temp_large" \
        --include_imbalanced

    if [ $? -eq 0 ]; then
        echo "✅ 大规模数据集收集完成: $LARGE_DEMO_FILE"
    else
        echo "❌ 大规模数据集收集失败"
        exit 1
    fi
}

# 收集自定义数据集
collect_custom() {
    echo "📊 收集自定义预训练数据集..."
    echo "请输入参数 (直接回车使用默认值):"

    read -p "策略列表 (默认: round_robin lor random): " policies
    policies=${policies:-"round_robin lor random"}

    read -p "每策略步数 (默认: 1500): " steps
    steps=${steps:-1500}

    read -p "副本数量 (默认: 4): " replicas
    replicas=${replicas:-4}

    read -p "QPS (默认: 3.0): " qps
    qps=${qps:-3.0}

    read -p "输出文件名 (默认: custom_demo_dataset.pkl): " filename
    filename=${filename:-"custom_demo_dataset.pkl"}

    custom_file="${DEMO_DIR}/${filename}"

    echo "🔧 收集配置:"
    echo "   - 策略: $policies"
    echo "   - 每策略步数: $steps"
    echo "   - 副本数: $replicas"
    echo "   - QPS: $qps"
    echo "   - 输出: $custom_file"

    read -p "确认开始收集? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi

    mkdir -p "$DEMO_DIR"

    python -m src.demo_collection.mixed_collector \
        --output "$custom_file" \
        --policies $policies \
        --steps_per_policy "$steps" \
        --num_replicas "$replicas" \
        --qps "$qps" \
        --temp_dir "${DEMO_DIR}/temp_custom" \
        --include_imbalanced

    if [ $? -eq 0 ]; then
        echo "✅ 自定义数据集收集完成: $custom_file"
    else
        echo "❌ 自定义数据集收集失败"
        exit 1
    fi
}

# 显示数据集信息
show_info() {
    echo "📊 预训练数据集信息:"
    echo ""

    for dataset in "$DEMO_DIR"/*.pkl; do
        if [[ -f "$dataset" ]]; then
            echo "📂 $(basename "$dataset"):"
            python -c "
import pickle
try:
    with open('$dataset', 'rb') as f:
        data = pickle.load(f)
    stats = data.get('stats', {})
    metadata = data.get('metadata', {})

    print(f'   - 样本总数: {stats.get(\"total_samples\", \"unknown\")}')
    print(f'   - 状态维度: {stats.get(\"state_dim\", \"unknown\")}')
    print(f'   - 策略分布: {stats.get(\"policy_distribution\", {})}')
    print(f'   - 收集时间: {metadata.get(\"collection_time\", \"unknown\")}')
    import os
    file_size_mb = os.path.getsize('$dataset') / 1024 / 1024
    print(f'   - 文件大小: {file_size_mb:.1f} MB')
except Exception as e:
    print(f'   ❌ 无法读取: {e}')
"
            echo ""
        fi
    done

    if [[ ! -f "$DEMO_DIR"/*.pkl ]]; then
        echo "📂 未找到任何预训练数据集"
        echo "💡 使用 'collect' 命令收集数据"
    fi
}

# 列出数据集
list_datasets() {
    echo "📋 可用的预训练数据集:"
    echo ""

    if [[ -d "$DEMO_DIR" ]]; then
        for dataset in "$DEMO_DIR"/*.pkl; do
            if [[ -f "$dataset" ]]; then
                filename=$(basename "$dataset")
                size=$(du -h "$dataset" | cut -f1)
                mtime=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$dataset")
                echo "  📄 $filename ($size, $mtime)"
            fi
        done
    fi

    if [[ -z "$(find "$DEMO_DIR" -name "*.pkl" 2>/dev/null)" ]]; then
        echo "  📂 无可用数据集"
    fi
}

# 清理数据
clean_data() {
    echo "🧹 清理预训练数据..."

    # 清理临时目录
    if [[ -d "${DEMO_DIR}/temp_"* ]]; then
        rm -rf "${DEMO_DIR}"/temp_*
        echo "✅ 已清理临时收集目录"
    fi

    # 清理模拟器临时输出目录
    if [[ -d "${DEMO_DIR}/simulator_temp" ]]; then
        rm -rf "${DEMO_DIR}/simulator_temp"
        echo "✅ 已清理模拟器临时输出目录"
    fi

    # 列出可删除的文件
    echo ""
    echo "📂 可清理的数据文件:"
    for dataset in "$DEMO_DIR"/*.pkl; do
        if [[ -f "$dataset" ]]; then
            size=$(du -h "$dataset" | cut -f1)
            echo "  📄 $(basename "$dataset") ($size)"
        fi
    done

    echo ""
    read -p "是否删除所有数据文件? [y/N]: " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -f "$DEMO_DIR"/*.pkl
        echo "✅ 已清理所有数据文件"
    else
        echo "已取消清理操作"
    fi
}

# 主逻辑
case "${1:-help}" in
    collect)
        collect_standard
        ;;
    collect-large)
        collect_large
        ;;
    collect-custom)
        collect_custom
        ;;
    info)
        show_info
        ;;
    list)
        list_datasets
        ;;
    clean)
        clean_data
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ 未知命令: $1"
        echo "使用 --help 查看帮助"
        exit 1
        ;;
esac