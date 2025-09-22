#!/usr/bin/env python3
"""
PPO 推理模式演示

展示如何使用训练好的模块化 PPO 调度器进行推理。
"""

import subprocess
import os
from pathlib import Path


def run_ppo_inference():
    """运行PPO推理模式演示"""

    print("🔍 启动PPO推理模式演示")
    print("=" * 60)

    # 检查checkpoint
    checkpoint_dir = Path("./outputs/checkpoints_demo")
    if not checkpoint_dir.exists() or not list(checkpoint_dir.glob("*.pt")):
        print("❌ 没有找到训练好的checkpoint")
        print("💡 请先运行训练:")
        print("   python examples/ppo_tensorboard_demo.py")
        return False

    # 找最新的checkpoint
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

    print(f"📁 使用checkpoint: {latest_checkpoint}")
    print("🎯 推理配置: 2副本, 50请求, 无训练")
    print("📊 TensorBoard: 已禁用 (推理模式)")
    print("-" * 60)

    # PPO推理命令
    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--cluster_config_num_replicas", "2",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "50",
        "--poisson_request_interval_generator_config_qps", "2.0",
        # 推理模式配置
        "--p_p_o_global_scheduler_modular_config_inference_only",
        "--p_p_o_global_scheduler_modular_config_load_checkpoint", str(latest_checkpoint),
        "--no-p_p_o_global_scheduler_modular_config_enable_tensorboard",
        # 输出配置
        "--metrics_config_output_dir", "./outputs/simulator_output/inference_demo",
    ]

    print("🚀 执行命令:")
    print(" ".join(cmd))
    print("\n📋 运行中...")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print("✅ 推理模式运行成功!")
            print("\n📊 输出摘要:")
            # 提取关键信息
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if any(keyword in line for keyword in ["INFO", "推理模式", "inference", "completed"]):
                    print(f"   {line}")

            print(f"\n📁 详细结果保存在: ./outputs/simulator_output/inference_demo")
            return True
        else:
            print("❌ 推理模式运行失败")
            print("错误输出:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("⏰ 推理运行超时 (120秒)")
        return False
    except Exception as e:
        print(f"❌ 运行异常: {e}")
        return False


def main():
    """主函数"""
    success = run_ppo_inference()

    if success:
        print("\n" + "=" * 60)
        print("🎉 PPO推理模式演示完成!")
        print("\n📝 关键特点:")
        print("  ✅ 加载预训练checkpoint")
        print("  ✅ 无梯度计算 (inference_mode)")
        print("  ✅ 无TensorBoard监控")
        print("  ✅ 快速调度决策")
        print("\n💡 与训练模式对比:")
        print("  训练模式: 梯度计算 + 模型更新 + 监控")
        print("  推理模式: 仅前向传播 + 快速决策")
    else:
        print("\n💥 推理模式演示失败")
        print("🔧 请检查:")
        print("  1. 是否有训练好的checkpoint")
        print("  2. 系统环境是否正常")


if __name__ == "__main__":
    main()