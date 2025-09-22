#!/usr/bin/env python3
"""
Vidur训练系统兼容性层
提供旧导入路径的重定向，支持无痛重构
"""

import warnings
import sys
from typing import Any


class CompatibilityRedirector:
    """兼容性重定向器 - 将旧导入重定向到新位置"""

    def __init__(self):
        self._redirect_map = {
            # 旧路径 -> 新路径的映射 (所有组件已迁移到core)
            'src.rl_components.actor_critic': 'src.core.models.actor_critic',
            'src.rl_components.ppo_trainer': 'src.core.algorithms.ppo_trainer',
            'src.rl_components.rollout_buffer': 'src.core.algorithms.rollout_buffer',
            'src.rl_components.state_builder': 'src.core.models.state_builder',
            'src.rl_components.normalizers': 'src.core.utils.normalizers',
            'src.rl_components.temperature_controller': 'src.core.utils.temperature_controller',

            # 基础设施组件重定向
            'src.infrastructure.checkpoints.checkpoint_manager': 'src.core.utils.infrastructure.checkpoints.checkpoint_manager',
            'src.infrastructure.config.training_config': 'src.core.utils.infrastructure.config.training_config',

            # 监控组件重定向
            'src.monitoring.tensorboard_monitor': 'src.core.utils.monitoring.tensorboard_monitor',
            'src.monitoring.metrics_exporter': 'src.core.utils.monitoring.metrics_exporter',
            'src.monitoring.progress_monitor': 'src.core.utils.monitoring.progress_monitor',

            # 奖励系统重定向
            'src.rewards.reward_calculator': 'src.core.algorithms.rewards.reward_calculator',

            # 训练组件重定向
            'src.training.chunk_trainer': 'src.core.algorithms.training.chunk_trainer',
            'src.training.progress_manager': 'src.core.algorithms.training.progress_manager',
        }

    def redirect_import(self, old_module_path: str) -> Any:
        """重定向导入到新模块"""
        if old_module_path in self._redirect_map:
            new_path = self._redirect_map[old_module_path]

            # 发出弃用警告
            warnings.warn(
                f"Importing from '{old_module_path}' is deprecated. "
                f"Please use '{new_path}' instead.",
                DeprecationWarning,
                stacklevel=3
            )

            # 动态导入新模块
            try:
                return __import__(new_path, fromlist=[''])
            except ImportError:
                # 如果新位置不存在，尝试旧位置（向后兼容）
                return __import__(old_module_path, fromlist=[''])

        # 如果没有重定向映射，正常导入
        return __import__(old_module_path, fromlist=[''])


# 全局重定向器实例
_redirector = CompatibilityRedirector()


def setup_import_hooks():
    """设置导入钩子，自动处理重定向"""
    import importlib.util
    import importlib.machinery

    class RedirectFinder(importlib.machinery.PathFinder):
        def find_spec(self, fullname, path, target=None):
            # 检查是否需要重定向
            redirect_prefixes = [
                'src.rl_components.',
                'src.infrastructure.',
                'src.monitoring.',
                'src.rewards.',
                'src.training.'
            ]

            for prefix in redirect_prefixes:
                if fullname.startswith(prefix):
                    new_name = _redirector._redirect_map.get(fullname)
                    if new_name:
                        warnings.warn(
                            f"Auto-redirecting import from '{fullname}' to '{new_name}'",
                            DeprecationWarning,
                            stacklevel=4
                        )
                        return super().find_spec(new_name, path, target)
                    break

            return super().find_spec(fullname, path, target)

    # 注册自定义finder
    if RedirectFinder not in sys.meta_path:
        sys.meta_path.insert(0, RedirectFinder())


# 便捷函数
def enable_compatibility_mode():
    """启用兼容性模式 - 自动重定向旧导入"""
    setup_import_hooks()
    print("✅ Vidur兼容性模式已启用 - 旧导入将自动重定向")


def create_legacy_module(old_path: str, new_imports: dict):
    """创建遗留模块文件，重定向到新位置

    Args:
        old_path: 旧模块路径 (如 'src/rl_components/__init__.py')
        new_imports: 新导入映射 {'ActorCritic': 'src.core.models.actor_critic'}
    """
    import_lines = []
    for name, new_module in new_imports.items():
        import_lines.append(f"from {new_module} import {name}")

    content = f'''"""
兼容性模块 - 自动生成的重定向
原模块路径: {old_path}
"""
import warnings

warnings.warn(
    "This module has been moved. Please update your imports to use the new structure.",
    DeprecationWarning,
    stacklevel=2
)

# 重定向导入
{chr(10).join(import_lines)}

# 保持向后兼容
__all__ = {list(new_imports.keys())}
'''

    return content


# 使用示例
if __name__ == '__main__':
    # 测试兼容性重定向
    enable_compatibility_mode()

    # 现在可以安全地重构代码结构，旧导入仍然工作
    print("兼容性层测试完成")