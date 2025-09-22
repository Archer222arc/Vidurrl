#!/usr/bin/env python3
"""
训练进度监控器 - 监控分块训练进度，提供ETA和状态报告
符合CLAUDE.md规范的模块化设计
"""

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


class ProgressMonitor:
    """训练进度监控器"""

    def __init__(self, progress_file: Union[str, Path]):
        """初始化监控器

        Args:
            progress_file: 进度文件路径
        """
        self.progress_file = Path(progress_file)

    def load_progress(self) -> Dict:
        """加载训练进度文件"""
        if not self.progress_file.exists():
            raise FileNotFoundError(f"进度文件不存在: {self.progress_file}")

        with open(self.progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_progress_summary(self) -> Dict:
        """获取进度摘要"""
        progress = self.load_progress()

        completed_chunks = progress.get('completed_chunks', 0)
        total_chunks = progress.get('total_chunks', 0)
        requests_done = progress.get('requests_done', 0)
        total_requests = progress.get('total_requests', 0)

        summary = {
            "completed_chunks": completed_chunks,
            "total_chunks": total_chunks,
            "requests_done": requests_done,
            "total_requests": total_requests,
            "progress_percent": (completed_chunks / total_chunks * 100) if total_chunks > 0 else 0,
            "requests_percent": (requests_done / total_requests * 100) if total_requests > 0 else 0,
            "status": progress.get("status", "unknown"),
            "remaining_chunks": total_chunks - completed_chunks,
            "chunk_size": progress.get("chunk_size", 0),
            "latest_checkpoint": progress.get("latest_checkpoint", ""),
            "start_time": progress.get("start_time", ""),
            "last_update": progress.get("last_update", "")
        }

        # 计算ETA
        eta_str, eta_seconds = self._calculate_eta(progress)
        summary["eta_string"] = eta_str
        summary["eta_seconds"] = eta_seconds

        return summary

    def print_progress_summary(self, verbose: bool = False) -> None:
        """打印进度摘要"""
        try:
            progress = self.load_progress()
            summary = self.get_progress_summary()

            print("🧩 分块训练进度监控")
            print("=" * 50)

            # 状态信息
            status_icons = {
                'completed': '✅',
                'failed': '❌',
                'running': '🏃',
                'unknown': '❓'
            }
            status = summary["status"]
            print(f"{status_icons.get(status, '❓')} 状态: {status}")

            # 进度条和统计
            if summary["total_chunks"] > 0:
                progress_percent = summary["progress_percent"]
                bar_length = 30
                filled_length = int(bar_length * summary["completed_chunks"] / summary["total_chunks"])
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"📊 块进度: [{bar}] {progress_percent:.1f}%")
                print(f"📦 已完成块: {summary['completed_chunks']}/{summary['total_chunks']}")

            if summary["total_requests"] > 0:
                request_percent = summary["requests_percent"]
                print(f"📈 请求进度: {summary['requests_done']:,}/{summary['total_requests']:,} ({request_percent:.1f}%)")

            # ETA信息
            if status == 'running':
                print(f"⏰ 预计完成时间: {summary['eta_string']}")
                print(f"📋 剩余块数: {summary['remaining_chunks']}")

                if summary["eta_seconds"] > 0:
                    estimated_completion = datetime.now() + timedelta(seconds=summary["eta_seconds"])
                    print(f"🎯 预计完成: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")

            # 详细信息
            if verbose:
                self._print_verbose_info(progress, summary)

        except Exception as e:
            print(f"❌ 监控错误: {e}", file=sys.stderr)

    def monitor_continuously(self, interval: int, verbose: bool = False) -> None:
        """持续监控模式"""
        print(f"🔄 监控模式启动，每 {interval} 秒刷新一次...")
        print("按 Ctrl+C 退出\n")

        try:
            while True:
                # 清屏
                print("\033[2J\033[H", end="")

                print(f"📄 进度文件: {self.progress_file}")
                print(f"⏰ 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print()

                self.print_progress_summary(verbose)

                # 检查是否完成
                summary = self.get_progress_summary()
                if summary["status"] in ['completed', 'failed']:
                    print(f"\n🏁 训练已{summary['status']}，退出监控")
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n👋 退出监控")

    def get_json_output(self) -> str:
        """获取JSON格式的输出"""
        summary = self.get_progress_summary()
        progress = self.load_progress()

        output = {
            **progress,
            **summary
        }

        return json.dumps(output, indent=2, ensure_ascii=False)

    def _calculate_eta(self, progress: Dict) -> Tuple[str, int]:
        """计算预计完成时间"""
        completed_chunks = progress.get('completed_chunks', 0)
        total_chunks = progress.get('total_chunks', 0)

        if completed_chunks == 0:
            return "未知", 0

        start_time_str = progress.get('start_time', '')
        if not start_time_str:
            return "未知", 0

        try:
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            elapsed_seconds = int((current_time - start_time).total_seconds())

            if elapsed_seconds <= 0:
                return "计算中...", 0

            avg_time_per_chunk = elapsed_seconds / completed_chunks
            remaining_chunks = total_chunks - completed_chunks
            eta_seconds = int(avg_time_per_chunk * remaining_chunks)

            return self._format_duration(eta_seconds), eta_seconds
        except Exception as e:
            return f"计算错误: {e}", 0

    def _format_duration(self, seconds: int) -> str:
        """格式化时间持续时间"""
        if seconds < 60:
            return f"{seconds}秒"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{minutes}分{seconds}秒"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}小时{minutes}分钟"

    def _print_verbose_info(self, progress: Dict, summary: Dict) -> None:
        """打印详细信息"""
        print("\n📄 详细信息:")

        config_info = [
            ("配置文件", progress.get('config_file', 'N/A')),
            ("输出目录", progress.get('output_dir', 'N/A')),
            ("每块大小", f"{summary.get('chunk_size', 'N/A'):,}"),
            ("开始时间", progress.get('start_time', 'N/A')),
            ("最后更新", progress.get('last_update', 'N/A'))
        ]

        for label, value in config_info:
            print(f"   • {label}: {value}")

        if summary["latest_checkpoint"]:
            print(f"   • 最新检查点: {summary['latest_checkpoint']}")

        external_pretrain = progress.get('external_pretrain', '')
        if external_pretrain:
            print(f"   • 外部预训练: {external_pretrain}")

        # 块详细信息
        chunks = progress.get('chunks', [])
        if chunks:
            print(f"\n📊 块执行详情:")
            recent_chunks = chunks[-5:]  # 显示最近5个块

            for chunk in recent_chunks:
                chunk_id = chunk.get('chunk_id', 'N/A')
                chunk_status = chunk.get('status', 'N/A')
                chunk_requests = chunk.get('requests', 'N/A')
                chunk_timestamp = chunk.get('timestamp', 'N/A')

                status_emoji = {
                    'completed': '✅',
                    'running': '🏃',
                    'failed': '❌'
                }.get(chunk_status, '❓')

                print(f"   {status_emoji} 块 {chunk_id}: {chunk_requests:,} 请求 - {chunk_status} ({chunk_timestamp})")


# 模块入口函数
def create_progress_monitor(progress_file: Union[str, Path]) -> ProgressMonitor:
    """创建进度监控器实例"""
    return ProgressMonitor(progress_file)


def find_latest_progress_file(search_pattern: str = "./outputs/*/training_progress.json") -> Optional[Path]:
    """查找最新的进度文件"""
    import glob

    files = glob.glob(search_pattern)
    if not files:
        return None

    # 选择最新修改的文件
    latest_file = max(files, key=lambda f: Path(f).stat().st_mtime)
    return Path(latest_file)