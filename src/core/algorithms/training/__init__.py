"""
训练模块 - 模块化训练功能
包含分块训练、进度管理、预训练等核心训练逻辑
"""

# 延迟导入以避免模块循环引用
# from .chunk_trainer import ChunkTrainer
from .progress_manager import ProgressManager
from . import pretraining

__all__ = ['ProgressManager', 'pretraining']