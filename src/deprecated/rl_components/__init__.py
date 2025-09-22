"""
RL components兼容性模块 - 重定向到新结构
⚠️  此模块已被重组，请更新导入路径

新的模块结构：
- Actor/Critic models: src.core.models
- PPO algorithms: src.core.algorithms
- Monitoring: src.monitoring
- Checkpoints: src.infrastructure.checkpoints
- Rewards: src.rewards
"""
import warnings

# 发出弃用警告
warnings.warn(
    "src.rl_components is deprecated and will be removed in future versions. "
    "Please update imports to use the new modular structure:\n"
    "- Actor/Critic models: src.core.models\n"
    "- PPO algorithms: src.core.algorithms\n"
    "- Monitoring: src.monitoring\n"
    "- Checkpoints: src.infrastructure.checkpoints\n"
    "- Rewards: src.rewards",
    DeprecationWarning,
    stacklevel=2
)

# 重定向导入 - 优先尝试新位置，失败则使用deprecated位置
try:
    # 从新位置导入
    from ..core.models.actor_critic import ActorCritic, init_layer
    from ..core.utils.normalizers import RunningNormalizer
    from ..core.algorithms.ppo_trainer import PPOTrainer
    from ..rewards.reward_calculator import RewardCalculator
    from ..core.algorithms.rollout_buffer import RolloutBuffer
    from ..core.models.state_builder import StateBuilder
    from ..monitoring.tensorboard_monitor import TensorBoardMonitor, PPOTrainingDetector
    from ..infrastructure.checkpoints.checkpoint_manager import CheckpointManager, InferenceMode
    from ..monitoring.metrics_exporter import MetricsExporter
    from ..core.utils.temperature_controller import TemperatureController

    _USING_NEW_STRUCTURE = True
    print("✅ rl_components: 使用新模块结构 (重定向)")

except ImportError as e:
    # 如果新结构不可用，从deprecated位置导入
    try:
        from ..deprecated.rl_components.actor_critic import ActorCritic, init_layer
        from ..deprecated.rl_components.normalizers import RunningNormalizer
        from ..deprecated.rl_components.ppo_trainer import PPOTrainer
        from ..deprecated.rl_components.reward_calculator import RewardCalculator
        from ..deprecated.rl_components.rollout_buffer import RolloutBuffer
        from ..deprecated.rl_components.state_builder import StateBuilder
        from ..deprecated.rl_components.tensorboard_monitor import TensorBoardMonitor, PPOTrainingDetector
        from ..deprecated.rl_components.checkpoint_manager import CheckpointManager, InferenceMode
        from ..deprecated.rl_components.metrics_exporter import MetricsExporter
        from ..deprecated.rl_components.temperature_controller import TemperatureController

        _USING_NEW_STRUCTURE = False
        print("⚠️  rl_components: 使用deprecated结构")

    except ImportError as fallback_error:
        print(f"❌ rl_components导入失败: 新结构错误={e}, deprecated错误={fallback_error}")
        raise

# 保持相同的公共接口
__all__ = [
    "ActorCritic",
    "init_layer",
    "RunningNormalizer",
    "PPOTrainer",
    "RewardCalculator",
    "RolloutBuffer",
    "StateBuilder",
    "TensorBoardMonitor",
    "PPOTrainingDetector",
    "CheckpointManager",
    "InferenceMode",
    "MetricsExporter",
    "TemperatureController",
]