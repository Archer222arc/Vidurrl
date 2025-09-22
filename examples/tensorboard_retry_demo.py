#!/usr/bin/env python3
"""
TensorBoard重试机制演示

展示新的TensorBoard重试功能：
- tensorboard_start_retries: 重试次数控制
- tensorboard_retry_delay: 重试延迟控制
- 自动处理端口冲突和启动失败
"""

import subprocess
import time
import socket
import threading
from pathlib import Path


def demo_retry_configuration():
    """演示重试配置选项"""

    print("🎯 TensorBoard重试机制演示")
    print("=" * 60)
    print("🆕 新功能特性:")
    print("  ✨ tensorboard_start_retries: 控制重试次数 (默认: 3)")
    print("  ✨ tensorboard_retry_delay: 控制重试延迟 (默认: 5.0秒)")
    print("  ✨ 智能错误处理: 端口冲突、命令未找到、启动失败")
    print("  ✨ URL规范化: 自动转换通配符host为用户友好的URL")
    print("  ✨ 噪音控制: 多次失败后自动禁用重试")
    print("=" * 60)

    print("\n📋 CLI参数格式:")
    print("  --p_p_o_global_scheduler_modular_config_tensorboard_start_retries <次数>")
    print("  --p_p_o_global_scheduler_modular_config_tensorboard_retry_delay <秒>")


def demo_port_conflict_handling():
    """演示端口冲突处理"""

    print("\n🎯 演示1: 端口冲突处理")
    print("=" * 50)

    # 创建虚拟服务器占用端口
    def occupy_port(port, duration=8):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('localhost', port))
            sock.listen(1)
            print(f"🔒 虚拟服务器占用端口 {port} ({duration}秒)")
            time.sleep(duration)
            sock.close()
            print(f"🔓 端口 {port} 已释放")
        except Exception as e:
            print(f"虚拟服务器错误: {e}")

    # 启动虚拟服务器占用6020端口
    port = 6020
    server_thread = threading.Thread(target=occupy_port, args=(port, 8), daemon=True)
    server_thread.start()

    time.sleep(1)  # 确保服务器启动

    print(f"🚀 尝试在被占用的端口 {port} 启动TensorBoard...")
    print("配置: start_retries=2, retry_delay=3.0")

    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--cluster_config_num_replicas", "2",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "5",
        "--poisson_request_interval_generator_config_qps", "10.0",
        f"--p_p_o_global_scheduler_modular_config_tensorboard_port", str(port),
        "--p_p_o_global_scheduler_modular_config_tensorboard_start_retries", "2",
        "--p_p_o_global_scheduler_modular_config_tensorboard_retry_delay", "3.0",
        "--metrics_config_output_dir", "./outputs/simulator_output/retry_demo",
    ]

    print("💡 预期行为:")
    print("  1. 首次启动失败 (端口被占用)")
    print("  2. 等待3秒后重试")
    print("  3. 第二次重试也失败")
    print("  4. 显示手动启动命令")
    print("")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # 读取前几行来展示重试行为
        lines_read = 0
        for line in process.stdout:
            if "TensorBoard" in line or "重试" in line or "失败" in line or "手动启动" in line:
                print(f"📊 {line.strip()}")
            lines_read += 1
            if lines_read > 30:  # 读取足够行数后停止
                break

        process.terminate()
        process.wait(timeout=3)
        print("✅ 端口冲突演示完成")

    except Exception as e:
        print(f"❌ 演示失败: {e}")


def demo_custom_retry_settings():
    """演示自定义重试设置"""

    print("\n🎯 演示2: 自定义重试设置")
    print("=" * 50)
    print("配置: start_retries=1, retry_delay=1.0 (快速失败)")

    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--cluster_config_num_replicas", "2",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "3",
        "--poisson_request_interval_generator_config_qps", "15.0",
        "--p_p_o_global_scheduler_modular_config_tensorboard_port", "6021",
        "--p_p_o_global_scheduler_modular_config_tensorboard_start_retries", "1",  # 只尝试1次
        "--p_p_o_global_scheduler_modular_config_tensorboard_retry_delay", "1.0",   # 快速重试
        "--metrics_config_output_dir", "./outputs/simulator_output/retry_demo2",
    ]

    print("💡 预期行为:")
    print("  1. 首次启动成功 (6021端口空闲)")
    print("  2. 无需重试")
    print("")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        lines_read = 0
        for line in process.stdout:
            if "TensorBoard" in line or "成功" in line or "PPO:init" in line:
                print(f"📊 {line.strip()}")
            lines_read += 1
            if lines_read > 20:
                break

        process.terminate()
        process.wait(timeout=3)
        print("✅ 自定义重试设置演示完成")

    except Exception as e:
        print(f"❌ 演示失败: {e}")


def demo_backwards_compatibility():
    """演示向后兼容性"""

    print("\n🎯 演示3: 向后兼容性")
    print("=" * 50)
    print("配置: 无重试相关参数 (使用默认值)")

    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--cluster_config_num_replicas", "2",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "3",
        "--poisson_request_interval_generator_config_qps", "20.0",
        "--p_p_o_global_scheduler_modular_config_tensorboard_port", "6022",
        "--metrics_config_output_dir", "./outputs/simulator_output/retry_demo3",
    ]

    print("💡 预期行为:")
    print("  1. 使用默认重试设置 (3次重试, 5秒延迟)")
    print("  2. 现有脚本无需修改")
    print("")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        lines_read = 0
        for line in process.stdout:
            if "TensorBoard" in line or "PPO:init" in line:
                print(f"📊 {line.strip()}")
            lines_read += 1
            if lines_read > 20:
                break

        process.terminate()
        process.wait(timeout=3)
        print("✅ 向后兼容性演示完成")

    except Exception as e:
        print(f"❌ 演示失败: {e}")


def main():
    """运行完整的重试机制演示"""

    demo_retry_configuration()
    demo_port_conflict_handling()
    demo_custom_retry_settings()
    demo_backwards_compatibility()

    print("\n" + "=" * 60)
    print("🎊 TensorBoard重试机制演示完成！")
    print("")
    print("📚 重试功能总结:")
    print("  🔸 智能重试: 自动处理端口冲突和启动失败")
    print("  🔸 可配置性: 用户可控制重试次数和延迟")
    print("  🔸 错误容错: 多种错误场景的优雅处理")
    print("  🔸 噪音控制: 重复失败后自动静默")
    print("  🔸 向后兼容: 现有代码无需修改")
    print("")
    print("🛠️  常见使用场景:")
    print("  • 端口被占用: 自动重试直到端口可用或达到最大次数")
    print("  • TensorBoard未安装: 提供清晰的安装提示")
    print("  • 临时网络问题: 通过重试机制提高成功率")
    print("  • 多实例运行: 通过自定义端口避免冲突")


if __name__ == "__main__":
    main()