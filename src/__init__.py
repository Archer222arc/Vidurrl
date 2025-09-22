"""
Vidur训练系统 - 新模块化架构
提供清晰的模块组织和向后兼容性
"""

# 版本信息
__version__ = "2.0.0-restructured"

# 新结构的便捷导入 (所有组件已迁移到core)
try:
    # 核心组件
    from .core.models import ActorCritic, StateBuilder
    from .core.algorithms import PPOTrainer, RolloutBuffer
    from .core.utils import RunningNormalizer, TemperatureController

    # 迁移到core的专业模块
    from .core.algorithms.training import ProgressManager
    from .core.algorithms.training.chunk_trainer import ChunkTrainer
    from .core.utils.monitoring import ProgressMonitor, TensorBoardMonitor, MetricsExporter
    from .core.algorithms.rewards import RewardCalculator
    from .core.utils.infrastructure.checkpoints import CheckpointManager
    from .core.utils.infrastructure.config import load_config, build_ppo_args

    _NEW_STRUCTURE_AVAILABLE = True
    print("✅ Vidur: 使用核心统一结构")

except ImportError as e:
    # 如果新结构不可用，提供兼容性访问
    print(f"⚠️  Vidur: 新结构不完整，使用兼容模式: {e}")
    _NEW_STRUCTURE_AVAILABLE = False

# 便捷的模块访问
class VidurModules:
    """提供便捷的模块访问接口"""

    @property
    def core(self):
        if _NEW_STRUCTURE_AVAILABLE:
            from . import core
            return core
        else:
            return None

    @property
    def training(self):
        from . import training
        return training

    @property
    def monitoring(self):
        from . import monitoring
        return monitoring

# 全局模块访问器
modules = VidurModules()

# 迁移指南
def show_migration_guide():
    """显示模块重构迁移指南"""
    print("""
🔄 Vidur模块结构重构指南:

新的模块组织 (推荐):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
├── src.core/               # 核心组件
│   ├── models/            # 神经网络模型 (ActorCritic, StateBuilder)
│   ├── algorithms/        # 算法实现 (PPOTrainer, RolloutBuffer)
│   └── utils/             # 工具函数 (Normalizers, TemperatureController)
├── src.training/          # 训练管理 (ChunkTrainer, ProgressManager)
├── src.monitoring/        # 监控系统 (ProgressMonitor, TensorBoard)
├── src.rewards/           # 奖励计算 (RewardCalculator)
├── src.data/             # 数据处理 (collection, preprocessing)
└── src.infrastructure/   # 基础设施 (config, checkpoints, scheduling)

旧导入 -> 新导入示例:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from src.rl_components.actor_critic import ActorCritic
  ↓
from src.core.models.actor_critic import ActorCritic

注意: 旧导入仍然可用（兼容性重定向），但建议迁移到新结构。
    """)

__all__ = [
    'modules', 'show_migration_guide',
]

if _NEW_STRUCTURE_AVAILABLE:
    __all__.extend([
        'ActorCritic', 'StateBuilder', 'PPOTrainer', 'RolloutBuffer',
        'RunningNormalizer', 'TemperatureController',
        'ChunkTrainer', 'ProgressManager', 'ProgressMonitor',
        'TensorBoardMonitor', 'MetricsExporter',
        'RewardCalculator', 'CheckpointManager', 'load_config', 'build_ppo_args'
    ])