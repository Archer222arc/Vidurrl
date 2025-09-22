"""
核心模块 - Vidur训练系统的核心组件
包含模型、算法和工具函数
"""

from . import models
from . import algorithms
from . import utils

# 便捷导入
from .models import ActorCritic, StateBuilder
from .algorithms import PPOTrainer, RolloutBuffer
from .utils import RunningNormalizer, TemperatureController

# 新增组件导入
try:
    from .algorithms.rewards import RewardCalculator
    from .algorithms.training import ChunkTrainer, ProgressManager
    from .utils.monitoring import TensorBoardMonitor, MetricsExporter, ProgressMonitor
    from .utils.infrastructure.checkpoints import CheckpointManager
    from .utils.infrastructure.config import TrainingConfig
except ImportError as e:
    # 部分组件可能还未完全迁移，暂时忽略导入错误
    pass

__all__ = [
    'models', 'algorithms', 'utils',
    'ActorCritic', 'StateBuilder', 'PPOTrainer', 'RolloutBuffer',
    'RunningNormalizer', 'TemperatureController'
]