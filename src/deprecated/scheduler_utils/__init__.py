"""
调度工具兼容性重定向 - 已移至 src.infrastructure.scheduling
"""
import warnings

warnings.warn(
    "src.scheduler_utils is deprecated. Please use src.infrastructure.scheduling",
    DeprecationWarning,
    stacklevel=2
)

# 重定向到新位置
try:
    from ..infrastructure.scheduling import *
    print("✅ scheduler_utils: 使用新模块结构 (重定向)")
except ImportError:
    print("⚠️  scheduler_utils: 新位置暂无内容，保持空模块")