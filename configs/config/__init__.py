"""
配置模块兼容性重定向 - 已移至 src.infrastructure.config
"""
import warnings

warnings.warn(
    "src.config is deprecated. Please use src.infrastructure.config",
    DeprecationWarning,
    stacklevel=2
)

# 重定向到新位置
try:
    from ..infrastructure.config import *
    print("✅ config: 使用新模块结构 (重定向)")
except ImportError:
    # 从deprecated位置导入
    try:
        from ..deprecated.config.training_config import *
        print("⚠️  config: 使用deprecated结构")
    except ImportError as e:
        print(f"❌ config导入失败: {e}")
        raise