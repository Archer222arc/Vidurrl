#!/bin/bash

# =============================================================================
# 训练状态监控脚本 - 调用模块化监控组件
# 符合CLAUDE.md规范的简化脚本设计
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}"

# 显示帮助信息
show_help() {
    echo "训练状态监控脚本"
    echo ""
    echo "使用方法: $0 [选项] [进度文件]"
    echo ""
    echo "选项:"
    echo "  -v, --verbose     显示详细信息"
    echo "  -w, --watch N     监控模式，每N秒刷新"
    echo "  -j, --json        JSON格式输出"
    echo "  -h, --help        显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                           # 自动查找最新进度文件"
    echo "  $0 -v                        # 显示详细信息"
    echo "  $0 -w 5                      # 每5秒刷新一次"
    echo "  $0 progress.json             # 指定进度文件"
}

# 解析参数并调用模块化组件
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# 调用模块化监控器
python3 -m src.monitoring "${ARGS[@]}"