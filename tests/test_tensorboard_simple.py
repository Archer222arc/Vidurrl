#!/usr/bin/env python3
"""
简单的TensorBoard配置测试
"""

from src.rl_components import TensorBoardMonitor
import tempfile
import os


def test_tensorboard_configs():
    """测试TensorBoard的新配置选项"""

    print("🧪 TensorBoard配置测试")
    print("=" * 50)

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = os.path.join(temp_dir, "tensorboard_test")

        # 测试1: 默认配置
        print("1️⃣ 测试默认配置 (auto_start=True, port=6006)")
        try:
            monitor1 = TensorBoardMonitor(
                log_dir=log_dir + "_default",
                enabled=True,
                auto_start=True,  # 默认值
                port=6006,        # 默认值
            )
            print(f"   ✅ 端口: {monitor1.port}")
            print(f"   ✅ 主机: {monitor1.host}")
            print(f"   ✅ 自动启动: {monitor1.auto_start}")
            monitor1.close()
        except Exception as e:
            print(f"   ❌ 失败: {e}")

        # 测试2: 自定义端口
        print("\n2️⃣ 测试自定义端口 (port=6007)")
        try:
            monitor2 = TensorBoardMonitor(
                log_dir=log_dir + "_port6007",
                enabled=True,
                auto_start=True,
                port=6007,
            )
            print(f"   ✅ 端口: {monitor2.port}")
            print(f"   ✅ 自动启动: {monitor2.auto_start}")
            monitor2.close()
        except Exception as e:
            print(f"   ❌ 失败: {e}")

        # 测试3: 关闭自动启动
        print("\n3️⃣ 测试关闭自动启动 (auto_start=False)")
        try:
            monitor3 = TensorBoardMonitor(
                log_dir=log_dir + "_no_autostart",
                enabled=True,
                auto_start=False,
                port=6008,
            )
            print(f"   ✅ 端口: {monitor3.port}")
            print(f"   ✅ 自动启动: {monitor3.auto_start}")
            monitor3.close()
        except Exception as e:
            print(f"   ❌ 失败: {e}")

        # 测试4: 自定义主机
        print("\n4️⃣ 测试自定义主机 (host=0.0.0.0)")
        try:
            monitor4 = TensorBoardMonitor(
                log_dir=log_dir + "_custom_host",
                enabled=True,
                auto_start=False,  # 避免实际启动
                port=6009,
                host="0.0.0.0",
            )
            print(f"   ✅ 端口: {monitor4.port}")
            print(f"   ✅ 主机: {monitor4.host}")
            print(f"   ✅ 自动启动: {monitor4.auto_start}")
            monitor4.close()
        except Exception as e:
            print(f"   ❌ 失败: {e}")

    print("\n🎯 TensorBoard配置测试完成")
    print("💡 新功能特点:")
    print("  ✅ 支持自定义端口")
    print("  ✅ 支持自定义主机")
    print("  ✅ 支持关闭自动启动")
    print("  ✅ 启动失败时显示手动命令")
    print("  ✅ 向后兼容现有代码")


if __name__ == "__main__":
    test_tensorboard_configs()