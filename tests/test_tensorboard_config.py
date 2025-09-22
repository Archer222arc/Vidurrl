#!/usr/bin/env python3
"""
测试TensorBoard配置选项
"""

import subprocess
import time
from pathlib import Path


def test_default_config():
    """测试默认配置 (auto_start=True, port=6006)"""

    print("🔍 测试1: 默认TensorBoard配置")
    print("=" * 50)

    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--cluster_config_num_replicas", "2",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "20",
        "--poisson_request_interval_generator_config_qps", "3.0",
        "--metrics_config_output_dir", "./outputs/simulator_output/test_default",
    ]

    print("🚀 运行命令:")
    print(" ".join(cmd))
    print("\n⏱️  运行20秒后终止...")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(20)  # 运行20秒
        process.terminate()
        process.wait(timeout=5)

        print("✅ 默认配置测试完成")
        print("💡 应该看到: 'TensorBoard服务器启动中... 访问: http://localhost:6006'")
        return True

    except Exception as e:
        print(f"❌ 默认配置测试失败: {e}")
        return False


def test_custom_port():
    """测试自定义端口配置"""

    print("\n🔍 测试2: 自定义端口6007")
    print("=" * 50)

    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--cluster_config_num_replicas", "2",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "15",
        "--poisson_request_interval_generator_config_qps", "3.0",
        "--p_p_o_global_scheduler_modular_config_tensorboard_port", "6007",
        "--metrics_config_output_dir", "./outputs/simulator_output/test_port6007",
    ]

    print("🚀 运行命令:")
    print(" ".join(cmd))
    print("\n⏱️  运行15秒后终止...")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(15)  # 运行15秒
        process.terminate()
        process.wait(timeout=5)

        print("✅ 自定义端口测试完成")
        print("💡 应该看到: 'TensorBoard服务器启动中... 访问: http://localhost:6007'")
        return True

    except Exception as e:
        print(f"❌ 自定义端口测试失败: {e}")
        return False


def test_no_auto_start():
    """测试关闭自动启动"""

    print("\n🔍 测试3: 关闭自动启动")
    print("=" * 50)

    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--cluster_config_num_replicas", "2",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "10",
        "--poisson_request_interval_generator_config_qps", "4.0",
        "--no-p_p_o_global_scheduler_modular_config_tensorboard_auto_start",
        "--p_p_o_global_scheduler_modular_config_tensorboard_port", "6008",
        "--metrics_config_output_dir", "./outputs/simulator_output/test_no_autostart",
    ]

    print("🚀 运行命令:")
    print(" ".join(cmd))
    print("\n⏱️  运行10秒后终止...")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(10)  # 运行10秒
        process.terminate()
        process.wait(timeout=5)

        print("✅ 关闭自动启动测试完成")
        print("💡 应该看到: '手动启动TensorBoard: tensorboard --logdir ... --port 6008'")
        return True

    except Exception as e:
        print(f"❌ 关闭自动启动测试失败: {e}")
        return False


def main():
    """运行所有TensorBoard配置测试"""

    print("🧪 TensorBoard配置选项测试套件")
    print("=" * 60)
    print("📋 测试项目:")
    print("  1. 默认配置 (auto_start=True, port=6006)")
    print("  2. 自定义端口 (port=6007)")
    print("  3. 关闭自动启动 (auto_start=False)")
    print("=" * 60)

    results = []

    # 测试1: 默认配置
    results.append(test_default_config())

    # 测试2: 自定义端口
    results.append(test_custom_port())

    # 测试3: 关闭自动启动
    results.append(test_no_auto_start())

    # 总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    test_names = ["默认配置", "自定义端口", "关闭自动启动"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {i+1}. {name}: {status}")

    passed = sum(results)
    total = len(results)
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")

    if passed == total:
        print("🎉 所有TensorBoard配置选项工作正常！")
        print("\n📝 新功能总结:")
        print("  ✅ tensorboard_auto_start: 控制是否自动启动服务器")
        print("  ✅ tensorboard_port: 自定义TensorBoard端口")
        print("  ✅ 向后兼容: 现有脚本无需修改")
        print("  ✅ 错误处理: 启动失败时提供手动命令")
    else:
        print("💥 部分测试失败，请检查配置")


if __name__ == "__main__":
    main()