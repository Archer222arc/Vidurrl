#!/usr/bin/env python3
"""
分块训练器 - 核心分块训练逻辑
符合CLAUDE.md规范的模块化设计
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .progress_manager import ProgressManager


class ChunkTrainer:
    """分块训练器 - 管理分块训练的执行逻辑"""

    def __init__(self, config: Dict):
        """初始化分块训练器

        Args:
            config: 训练配置字典，包含所有必要参数
        """
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.progress_manager = ProgressManager(str(self.output_dir))

        # 训练参数
        self.total_requests = config["total_requests"]
        self.chunk_size = config["chunk_size"]
        self.total_chunks = (self.total_requests + self.chunk_size - 1) // self.chunk_size

        # 路径配置
        self.repo_root = Path(config.get("repo_root", ".")).resolve()
        self.config_file = config.get("config_file", "configs/ppo_warmstart.json")
        self.training_config_script = self.repo_root / "src/config/training_config.py"

        print(f"🧩 分块训练器初始化完成")
        print(f"   - 总请求数: {self.total_requests:,}")
        print(f"   - 块大小: {self.chunk_size:,}")
        print(f"   - 总块数: {self.total_chunks}")
        print(f"   - 输出目录: {self.output_dir}")

    def run(self) -> bool:
        """执行分块训练

        Returns:
            bool: 训练是否成功完成
        """
        # 检查是否可以恢复
        can_resume, start_chunk, resume_checkpoint = self.progress_manager.can_resume()

        if not can_resume and start_chunk == 0:
            print("✅ 训练已完成")
            return True

        if not can_resume:
            # 初始化新的训练
            self.progress_manager.init_progress(
                self.total_chunks, self.chunk_size, self.total_requests, self.config
            )
            start_chunk = 1
            resume_checkpoint = ""

        # 执行分块训练循环
        for chunk_id in range(start_chunk, self.total_chunks + 1):
            remaining_requests = self.total_requests - (chunk_id - 1) * self.chunk_size
            current_requests = min(remaining_requests, self.chunk_size)

            print(f"\n╭─────────────────────────────────────────────────────────────╮")
            print(f"│ Chunk {chunk_id}/{self.total_chunks} - {current_requests:,} requests")
            print(f"╰─────────────────────────────────────────────────────────────╯")

            is_first_chunk = (chunk_id == 1)
            success = self._execute_chunk(chunk_id, current_requests, is_first_chunk, resume_checkpoint)

            if not success:
                print(f"❌ Chunk {chunk_id} 训练失败")
                self.progress_manager.mark_failed(chunk_id, "训练执行失败")
                return False

            # 清除resume_checkpoint，后续chunk都从latest.pt恢复
            resume_checkpoint = ""

        print(f"\n🎉 所有分块训练完成！")
        self._print_final_summary()
        return True

    def _execute_chunk(self, chunk_id: int, requests: int, is_first_chunk: bool, resume_checkpoint: str) -> bool:
        """执行单个chunk的训练"""
        print(f"🚀 [Chunk {chunk_id}] 开始训练...")

        # 更新进度状态为运行中
        self.progress_manager.update_chunk_progress(chunk_id, "running", "", requests)

        # 构建训练命令
        cmd = self._build_training_command(chunk_id, requests, is_first_chunk, resume_checkpoint)
        log_file = self.output_dir / f"chunk_{chunk_id}.log"

        print(f"📝 [Chunk {chunk_id}] 日志文件: {log_file}")

        try:
            # 执行训练命令
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=self.repo_root,
                    env=self._get_training_env()
                )

            if result.returncode == 0:
                print(f"✅ [Chunk {chunk_id}] 训练完成")

                # 查找最新checkpoint
                latest_checkpoint = self._find_latest_checkpoint()
                self.progress_manager.update_chunk_progress(
                    chunk_id, "completed", latest_checkpoint, requests
                )
                return True
            else:
                print(f"❌ [Chunk {chunk_id}] 训练失败 (exit code: {result.returncode})")
                self.progress_manager.update_chunk_progress(chunk_id, "failed", "", requests)
                return False

        except Exception as e:
            print(f"❌ [Chunk {chunk_id}] 训练执行异常: {e}")
            self.progress_manager.update_chunk_progress(chunk_id, "failed", "", requests)
            return False

    def _build_training_command(self, chunk_id: int, requests: int,
                               is_first_chunk: bool, resume_checkpoint: str) -> List[str]:
        """构建训练命令"""
        # 获取基础训练参数
        try:
            base_args = subprocess.check_output([
                sys.executable, str(self.training_config_script),
                self.config_file, str(self.output_dir)
            ], text=True, cwd=self.repo_root).strip().split()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"构建训练参数失败: {e}")

        # 覆盖关键参数
        cmd = [sys.executable, "-m", "vidur.main"] + base_args

        # 更新请求数和基本配置
        cmd = self._update_args(cmd, {
            "--cluster_config_num_replicas": str(self.config["num_replicas"]),
            "--poisson_request_interval_generator_config_qps": str(self.config["qps"]),
            "--synthetic_request_generator_config_num_requests": str(requests),
            "--p_p_o_global_scheduler_modular_config_max_queue_requests_per_replica": "8",
            "--p_p_o_global_scheduler_modular_config_tensorboard_log_dir": str(self.output_dir / "tensorboard")
        })

        # 处理warmstart和resume参数
        if is_first_chunk:
            # 第一个chunk: 根据配置设置warmstart
            if not self.config.get("skip_warmstart", True):
                cmd.extend([
                    "--p_p_o_global_scheduler_modular_config_enable_warm_start",
                    "--p_p_o_global_scheduler_modular_config_pretrained_actor_path",
                    self.config.get("pretrained_actor_path", "")
                ])
            elif self.config.get("external_pretrain") and not resume_checkpoint:
                cmd.extend([
                    "--p_p_o_global_scheduler_modular_config_pretrained_actor_path",
                    self.config.get("pretrained_actor_path", "")
                ])

            # 如果有恢复checkpoint，使用它
            if resume_checkpoint:
                cmd.extend([
                    "--p_p_o_global_scheduler_modular_config_load_checkpoint",
                    resume_checkpoint
                ])
        else:
            # 后续chunks: 总是从最新checkpoint恢复
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                cmd.extend([
                    "--p_p_o_global_scheduler_modular_config_load_checkpoint",
                    latest_checkpoint
                ])
                print(f"📂 [Chunk {chunk_id}] 从checkpoint恢复: {latest_checkpoint}")
            else:
                print(f"⚠️ [Chunk {chunk_id}] 警告: 未找到checkpoint文件")

        return cmd

    def _update_args(self, cmd: List[str], updates: Dict[str, str]) -> List[str]:
        """更新命令行参数"""
        result = []
        i = 0
        while i < len(cmd):
            arg = cmd[i]
            if arg in updates:
                result.extend([arg, updates[arg]])
                i += 2  # 跳过旧值
            else:
                result.append(arg)
                i += 1

        # 添加新参数（如果不存在）
        for key, value in updates.items():
            if key not in cmd:
                result.extend([key, value])

        return result

    def _find_latest_checkpoint(self) -> str:
        """查找最新的checkpoint"""
        checkpoint_patterns = [
            "./outputs/checkpoints/latest.pt",
            str(self.output_dir.parent / "checkpoints/latest.pt"),
            str(self.repo_root / "outputs/checkpoints/latest.pt")
        ]

        for pattern in checkpoint_patterns:
            checkpoint_path = Path(pattern)
            if checkpoint_path.exists():
                return str(checkpoint_path.resolve())

        return ""

    def _get_training_env(self) -> Dict[str, str]:
        """获取训练环境变量"""
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root)
        return env

    def _print_final_summary(self) -> None:
        """打印最终摘要"""
        summary = self.progress_manager.get_progress_summary()

        print(f"📊 训练统计:")
        print(f"   - 总块数: {summary['total_chunks']}")
        print(f"   - 完成块数: {summary['completed_chunks']}")
        print(f"   - 总请求数: {summary['total_requests']:,}")
        print(f"   - 完成请求数: {summary['requests_done']:,}")
        print(f"   - 进度文件: {self.progress_manager.progress_file}")
        print(f"   - TensorBoard: http://localhost:6006")

        latest_checkpoint = self._find_latest_checkpoint()
        if latest_checkpoint:
            print(f"   - 最新checkpoint: {latest_checkpoint}")


# 模块入口函数
def run_chunk_training(config: Dict) -> bool:
    """运行分块训练的入口函数

    Args:
        config: 训练配置字典

    Returns:
        bool: 训练是否成功
    """
    trainer = ChunkTrainer(config)
    return trainer.run()