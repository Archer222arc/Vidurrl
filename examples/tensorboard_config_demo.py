#!/usr/bin/env python3
"""
TensorBoard配置选项演示

展示如何使用新的TensorBoard控制选项：
- tensorboard_auto_start: 控制是否自动启动TensorBoard服务器
- tensorboard_port: 自定义TensorBoard端口
"""

import subprocess
import time
from pathlib import Path


def demo_default_config():
    """演示默认配置 (auto_start=True, port=6006)"""

    print("🎯 演示1: 默认TensorBoard配置")
    print("=" * 50)
    print("📋 配置:")
    print("  - auto_start: True (默认)")
    print("  - port: 6006 (默认)")
    print("  - 自动启动TensorBoard服务器")
    print("")

    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--cluster_config_num_replicas", "2",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "10",
        "--poisson_request_interval_generator_config_qps", "5.0",
        "--metrics_config_output_dir", "./outputs/simulator_output/demo_default",
    ]

    print("🚀 命令:")
    print(" ".join(cmd))
    print("")
    print("⏱️  运行5秒演示...")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # 读取前几行输出来显示TensorBoard启动信息
        lines_read = 0
        for line in process.stdout:
            if "TensorBoard" in line or "PPO:init" in line:
                print(f"📊 {line.strip()}")
            lines_read += 1
            if lines_read > 20:  # 读取足够的行数后停止
                break

        process.terminate()
        process.wait(timeout=3)
        print("✅ 默认配置演示完成")
        print("💡 TensorBoard应该在 http://localhost:6006 自动启动")

    except Exception as e:
        print(f"❌ 演示失败: {e}")


def demo_custom_port():
    """演示自定义端口配置"""

    print("\n🎯 演示2: 自定义端口配置")
    print("=" * 50)
    print("📋 配置:")
    print("  - auto_start: True")
    print("  - port: 6007 (自定义)")
    print("  - TensorBoard在6007端口启动")
    print("")

    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--cluster_config_num_replicas", "2",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "8",
        "--poisson_request_interval_generator_config_qps", "6.0",
        "--p_p_o_global_scheduler_modular_config_tensorboard_port", "6007",
        "--metrics_config_output_dir", "./outputs/simulator_output/demo_port6007",
    ]

    print("🚀 命令:")
    print(" ".join(cmd))
    print("")
    print("⏱️  运行5秒演示...")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        lines_read = 0
        for line in process.stdout:
            if "TensorBoard" in line or "6007" in line or "PPO:init" in line:
                print(f"📊 {line.strip()}")
            lines_read += 1
            if lines_read > 20:
                break

        process.terminate()
        process.wait(timeout=3)
        print("✅ 自定义端口演示完成")
        print("💡 TensorBoard应该在 http://localhost:6007 启动")

    except Exception as e:
        print(f"❌ 演示失败: {e}")


def demo_no_auto_start():
    """演示关闭自动启动"""

    print("\n🎯 演示3: 关闭自动启动")
    print("=" * 50)
    print("📋 配置:")
    print("  - auto_start: False")
    print("  - port: 6008")
    print("  - 只记录日志，不启动服务器")
    print("")

    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--cluster_config_num_replicas", "2",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "6",
        "--poisson_request_interval_generator_config_qps", "8.0",
        "--no-p_p_o_global_scheduler_modular_config_tensorboard_auto_start",
        "--p_p_o_global_scheduler_modular_config_tensorboard_port", "6008",
        "--metrics_config_output_dir", "./outputs/simulator_output/demo_no_autostart",
    ]

    print("🚀 命令:")
    print(" ".join(cmd))
    print("")
    print("⏱️  运行5秒演示...")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        lines_read = 0
        for line in process.stdout:
            if "TensorBoard" in line or "手动启动" in line or "PPO:init" in line:
                print(f"📊 {line.strip()}")
            lines_read += 1
            if lines_read > 20:
                break

        process.terminate()
        process.wait(timeout=3)
        print("✅ 关闭自动启动演示完成")
        print("💡 应该显示手动启动命令而不是自动启动服务器")

    except Exception as e:
        print(f"❌ 演示失败: {e}")


def main():
    """运行TensorBoard配置演示"""

    print("🎉 TensorBoard配置选项演示")
    print("=" * 60)
    print("📝 新功能特性:")
    print("  ✨ tensorboard_auto_start: 控制自动启动")
    print("  ✨ tensorboard_port: 自定义端口")
    print("  ✨ 向后兼容: 现有脚本无需修改")
    print("  ✨ 错误处理: 启动失败时提供手动命令")
    print("=" * 60)

    # 演示1: 默认配置
    demo_default_config()

    # 演示2: 自定义端口
    demo_custom_port()

    # 演示3: 关闭自动启动
    demo_no_auto_start()

    print("\n" + "=" * 60)
    print("🎊 TensorBoard配置演示完成！")
    print("")
    print("📚 使用指南:")
    print("  🔸 默认使用: 无需额外参数，自动在6006端口启动")
    print("  🔸 自定义端口: --p_p_o_global_scheduler_modular_config_tensorboard_port 6007")
    print("  🔸 关闭自动启动: --no-p_p_o_global_scheduler_modular_config_tensorboard_auto_start")
    print("  🔸 两者组合: 可以同时设置端口和关闭自动启动")
    print("")
    print("🔗 CLI参数格式:")
    print("  --p_p_o_global_scheduler_modular_config_tensorboard_port <端口>")
    print("  --p_p_o_global_scheduler_modular_config_tensorboard_auto_start")
    print("  --no-p_p_o_global_scheduler_modular_config_tensorboard_auto_start")


if __name__ == "__main__":
    main()