"""
预训练模块兼容性重定向 - 已移至 src.training.pretraining
"""
import warnings

warnings.warn(
    "src.pretraining is deprecated. Please use src.training.pretraining",
    DeprecationWarning,
    stacklevel=2
)

# 重定向到新位置
try:
    from ..training.pretraining import *
    print("✅ pretraining: 使用新模块结构 (重定向)")
except ImportError:
    # 如果新位置不可用，从deprecated位置导入
    try:
        from ..deprecated.pretraining.behavior_cloning_trainer import *
        from ..deprecated.pretraining.model_validator import *
        from ..deprecated.pretraining.standalone_trainer import *
        from ..deprecated.pretraining.unified_trainer import *
        print("⚠️  pretraining: 使用deprecated结构")
    except ImportError as e:
        print(f"❌ pretraining导入失败: {e}")
        raise