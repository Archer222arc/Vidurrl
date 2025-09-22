"""
兼容性重定向模块 - rl_components已迁移到新结构
⚠️ 此模块已被重构，请更新导入路径
"""
import warnings

# 发出迁移警告
warnings.warn(
    "src.rl_components is deprecated and has been restructured. "
    "Please update imports to use the new modular structure:\n"
    "- Actor/Critic models: src.core.models\n"
    "- PPO algorithms: src.core.algorithms\n"
    "- Monitoring: src.core.utils.monitoring\n"
    "- Infrastructure: src.core.utils.infrastructure",
    DeprecationWarning,
    stacklevel=2
)

# 重定向导入到新位置
try:
    # 尝试从新位置导入
    from ..core.models.actor_critic import ActorCritic
    from ..core.algorithms.ppo_trainer import PPOTrainer
    from ..core.algorithms.rollout_buffer import RolloutBuffer
    from ..core.models.state_builder import StateBuilder
    from ..core.utils.normalizers import RunningNormalizer
    from ..core.utils.temperature_controller import TemperatureController
    from ..core.utils.monitoring.tensorboard_monitor import TensorBoardMonitor
    from ..core.utils.monitoring.metrics_exporter import MetricsExporter
    from ..core.utils.infrastructure.checkpoints.checkpoint_manager import CheckpointManager
    from ..core.algorithms.rewards.reward_calculator import RewardCalculator

    # 保持向后兼容的接口
    __all__ = [
        'ActorCritic', 'PPOTrainer', 'RolloutBuffer', 'StateBuilder',
        'RunningNormalizer', 'TemperatureController', 'TensorBoardMonitor',
        'MetricsExporter', 'CheckpointManager', 'RewardCalculator'
    ]

    print("🔄 使用新模块结构 (通过兼容性重定向)")

except ImportError as e:
    print(f"❌ 无法从新结构导入: {e}")
    raise ImportError(
        "rl_components has been restructured. Please update your imports to use the new structure."
    ) from e