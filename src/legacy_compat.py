#!/usr/bin/env python3
"""
遗留兼容性重定向
为src/目录重构提供无缝的向后兼容性
"""

# 这样使用：在任何可能使用旧导入的地方，先导入这个模块
# import src.legacy_compat  # 启用重定向

# 现有代码可以继续使用：
# from src.rl_components.actor_critic import ActorCritic  # 仍然工作


def create_rl_components_redirect():
    """为rl_components创建重定向内容"""
    return '''"""
rl_components兼容性模块 - 重定向到新结构
⚠️  此模块已被重组，请更新导入路径
"""
import warnings

# 发出弃用警告
warnings.warn(
    "src.rl_components is deprecated and will be removed in future versions. "
    "Please update imports to use the new modular structure:\\n"
    "- Actor/Critic models: src.core.models\\n"
    "- PPO algorithms: src.core.algorithms\\n"
    "- Monitoring: src.monitoring\\n"
    "- Checkpoints: src.infrastructure.checkpoints",
    DeprecationWarning,
    stacklevel=2
)

# 重定向导入 (当新结构就位时启用)
try:
    # 尝试从新位置导入
    from ..core.models.actor_critic import *
    from ..core.algorithms.ppo_trainer import *
    from ..monitoring.tensorboard_monitor import *
    from ..infrastructure.checkpoints.checkpoint_manager import *

    # 如果成功，提供旧接口
    print("✅ 使用新模块结构")

except ImportError:
    # 如果新结构不存在，从当前位置导入
    try:
        from .actor_critic import *
        from .ppo_trainer import *
        from .tensorboard_monitor import *
        from .checkpoint_manager import *
        print("⚠️  使用遗留结构，建议尽快迁移")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        raise
'''


def create_updated_src_init():
    """创建更新的src/__init__.py"""
    return '''"""
Vidur训练系统 - 统一模块接口
提供清晰的模块组织和向后兼容性
"""

# 版本信息
__version__ = "2.0.0"  # 重构版本

# 兼容性导入 - 逐步弃用
try:
    # 新结构可用时的便捷导入
    from .core.models import ActorCritic
    from .core.algorithms import PPOTrainer
    from .training import ChunkTrainer, ProgressManager
    from .monitoring import ProgressMonitor

    # 标记新结构可用
    _NEW_STRUCTURE_AVAILABLE = True

except ImportError:
    # 回退到旧结构
    _NEW_STRUCTURE_AVAILABLE = False
    import warnings
    warnings.warn("Using legacy structure", DeprecationWarning)


# 提供兼容性接口
class VidurCompat:
    """兼容性接口类"""

    @staticmethod
    def get_available_modules():
        """返回可用的模块列表"""
        if _NEW_STRUCTURE_AVAILABLE:
            return {
                'core': ['models', 'algorithms', 'utils'],
                'training': ['chunk_trainer', 'progress_manager'],
                'monitoring': ['progress_monitor', 'tensorboard_monitor'],
                'infrastructure': ['config', 'checkpoints']
            }
        else:
            return {
                'legacy': ['rl_components', 'training', 'monitoring']
            }

    @staticmethod
    def migration_guide():
        """打印迁移指南"""
        print("""
🔄 Vidur模块结构迁移指南:

旧导入 -> 新导入:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from src.rl_components.actor_critic import ActorCritic
  ↓
from src.core.models.actor_critic import ActorCritic

from src.rl_components.ppo_trainer import PPOTrainer
  ↓
from src.core.algorithms.ppo_trainer import PPOTrainer

from src.rl_components.tensorboard_monitor import TensorBoardMonitor
  ↓
from src.monitoring.tensorboard_monitor import TensorBoardMonitor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

新结构提供更清晰的模块分离和更好的可维护性。
        """)


# 全局兼容性实例
compat = VidurCompat()

# 便捷函数
def show_migration_guide():
    """显示迁移指南"""
    compat.migration_guide()

def check_structure():
    """检查当前使用的结构"""
    if _NEW_STRUCTURE_AVAILABLE:
        print("✅ 使用新模块结构")
    else:
        print("⚠️  使用遗留结构")

    print("可用模块:", compat.get_available_modules())


# 自动设置
if __name__ == "__main__":
    check_structure()
    show_migration_guide()
'''


if __name__ == "__main__":
    print("🔧 生成兼容性重定向文件...")

    # 打印rl_components重定向内容
    print("\n📁 src/rl_components/__init__.py 内容:")
    print(create_rl_components_redirect())

    print("\n" + "="*60)

    # 打印更新的src init内容
    print("\n📁 src/__init__.py 更新内容:")
    print(create_updated_src_init())