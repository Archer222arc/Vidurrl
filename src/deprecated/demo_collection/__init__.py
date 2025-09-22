"""
示教数据收集模块兼容性重定向 - 已移至 src.data.collection
"""
import warnings

warnings.warn(
    "src.demo_collection is deprecated. Please use src.data.collection",
    DeprecationWarning,
    stacklevel=2
)

# 重定向到新位置
try:
    from ..data.collection import *
    print("✅ demo_collection: 使用新模块结构 (重定向)")
except ImportError:
    # 从deprecated位置导入
    try:
        from ..deprecated.demo_collection.mixed_collector import *
        print("⚠️  demo_collection: 使用deprecated结构")
    except ImportError as e:
        print(f"❌ demo_collection导入失败: {e}")
        raise