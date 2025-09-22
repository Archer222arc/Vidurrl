#!/usr/bin/env python3
"""
训练进度管理器 - 负责进度跟踪和状态管理
符合CLAUDE.md规范的模块化设计
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ProgressManager:
    """训练进度管理器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.progress_file = self.output_dir / "training_progress.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def init_progress(self, total_chunks: int, chunk_size: int, total_requests: int, config: Optional[Dict] = None) -> None:
        """初始化进度跟踪"""
        progress_data = {
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
            "total_requests": total_requests,
            "completed_chunks": 0,
            "requests_done": 0,
            "latest_checkpoint": "",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "last_update": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "output_dir": str(self.output_dir),
            "chunks": []
        }

        # 添加配置信息
        if config:
            progress_data.update({
                "config_file": config.get("config_file", ""),
                "external_pretrain": config.get("external_pretrain", ""),
                "num_replicas": config.get("num_replicas", 4),
                "qps": config.get("qps", 3.5)
            })

        self._save_progress(progress_data)
        print(f"📋 初始化进度跟踪: {self.progress_file}")

    def load_progress(self) -> Optional[Dict]:
        """加载进度数据"""
        if not self.progress_file.exists():
            return None

        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"❌ 进度文件损坏: {e}")
            return None

    def update_chunk_progress(self, chunk_id: int, status: str,
                            checkpoint_path: str = "", requests: int = 0) -> None:
        """更新单个chunk的进度"""
        progress = self.load_progress()
        if not progress:
            raise RuntimeError("进度文件不存在，需要先初始化")

        # 更新基本信息
        progress["last_update"] = datetime.now(timezone.utc).isoformat()

        if status == "completed":
            progress["completed_chunks"] += 1
            progress["requests_done"] += requests
            if checkpoint_path:
                progress["latest_checkpoint"] = checkpoint_path

        # 更新或添加chunk信息
        chunk_info = {
            "chunk_id": chunk_id,
            "status": status,
            "requests": requests,
            "checkpoint": checkpoint_path,
            "log_file": str(self.output_dir / f"chunk_{chunk_id}.log"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # 查找并更新现有chunk记录
        chunk_found = False
        for i, chunk in enumerate(progress["chunks"]):
            if chunk["chunk_id"] == chunk_id:
                progress["chunks"][i] = chunk_info
                chunk_found = True
                break

        if not chunk_found:
            progress["chunks"].append(chunk_info)

        # 检查是否全部完成
        if progress["completed_chunks"] >= progress["total_chunks"]:
            progress["status"] = "completed"
            progress["end_time"] = datetime.now(timezone.utc).isoformat()

        self._save_progress(progress)
        print(f"✅ 进度更新: Chunk {chunk_id} - {status}")

    def mark_failed(self, chunk_id: int, error_message: str = "") -> None:
        """标记训练失败"""
        progress = self.load_progress()
        if progress:
            progress["status"] = "failed"
            progress["last_update"] = datetime.now(timezone.utc).isoformat()
            if error_message:
                progress["error_message"] = error_message

            # 更新当前chunk状态
            self.update_chunk_progress(chunk_id, "failed")

    def can_resume(self) -> Tuple[bool, int, str]:
        """检查是否可以恢复训练

        Returns:
            (can_resume, next_chunk_id, latest_checkpoint)
        """
        progress = self.load_progress()
        if not progress:
            return False, 1, ""

        completed_chunks = progress.get("completed_chunks", 0)
        total_chunks = progress.get("total_chunks", 0)
        status = progress.get("status", "running")

        if status == "completed":
            print("✅ 训练已完成")
            return False, 0, ""
        elif status == "failed":
            print("❌ 训练之前失败，可以从失败点恢复")
            return True, completed_chunks + 1, progress.get("latest_checkpoint", "")
        elif completed_chunks < total_chunks:
            print(f"⏸️ 可以从第 {completed_chunks + 1} 块恢复训练")
            return True, completed_chunks + 1, progress.get("latest_checkpoint", "")
        else:
            return False, 0, ""

    def get_progress_summary(self) -> Dict:
        """获取进度摘要"""
        progress = self.load_progress()
        if not progress:
            return {}

        completed_chunks = progress.get("completed_chunks", 0)
        total_chunks = progress.get("total_chunks", 0)
        requests_done = progress.get("requests_done", 0)
        total_requests = progress.get("total_requests", 0)

        summary = {
            "completed_chunks": completed_chunks,
            "total_chunks": total_chunks,
            "requests_done": requests_done,
            "total_requests": total_requests,
            "progress_percent": (completed_chunks / total_chunks * 100) if total_chunks > 0 else 0,
            "requests_percent": (requests_done / total_requests * 100) if total_requests > 0 else 0,
            "status": progress.get("status", "unknown"),
            "remaining_chunks": total_chunks - completed_chunks
        }

        # 计算ETA
        eta_str, eta_seconds = self._calculate_eta(progress)
        summary["eta_string"] = eta_str
        summary["eta_seconds"] = eta_seconds

        return summary

    def _calculate_eta(self, progress: Dict) -> Tuple[str, int]:
        """计算预计完成时间"""
        completed_chunks = progress.get("completed_chunks", 0)
        total_chunks = progress.get("total_chunks", 0)

        if completed_chunks == 0:
            return "未知", 0

        start_time_str = progress.get("start_time", "")
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

    def _save_progress(self, progress_data: Dict) -> None:
        """保存进度数据"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)


# 兼容性函数 - 为脚本提供简单接口
def create_progress_manager(output_dir: str) -> ProgressManager:
    """创建进度管理器实例"""
    return ProgressManager(output_dir)