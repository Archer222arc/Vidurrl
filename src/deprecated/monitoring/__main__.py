#!/usr/bin/env python3
"""
进度监控器命令行接口
提供CLI入口调用模块化的监控功能
"""

import argparse
import sys
from pathlib import Path

from .progress_monitor import ProgressMonitor, find_latest_progress_file


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='监控分块训练进度')
    parser.add_argument('progress_file', nargs='?',
                       default='./outputs/*/training_progress.json',
                       help='进度文件路径 (默认: 自动搜索最新)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='显示详细信息')
    parser.add_argument('-w', '--watch', type=int, metavar='SECONDS',
                       help='持续监控模式，每N秒刷新一次')
    parser.add_argument('--json', action='store_true',
                       help='以JSON格式输出')

    args = parser.parse_args()

    # 处理文件路径
    if '*' in args.progress_file:
        progress_file = find_latest_progress_file(args.progress_file)
        if not progress_file:
            print("❌ 未找到进度文件", file=sys.stderr)
            sys.exit(1)
    else:
        progress_file = Path(args.progress_file)

    try:
        monitor = ProgressMonitor(progress_file)

        if args.watch:
            # 持续监控模式
            monitor.monitor_continuously(args.watch, args.verbose)
        elif args.json:
            # JSON输出模式
            print(monitor.get_json_output())
        else:
            # 单次显示模式
            monitor.print_progress_summary(args.verbose)

    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 监控错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()