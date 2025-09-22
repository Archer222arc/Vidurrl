#!/usr/bin/env python3
"""
PPO TensorBoard 监控演示

展示如何使用带 TensorBoard 监控的模块化 PPO 调度器进行训练。
"""

import subprocess
import time
import webbrowser
from pathlib import Path

def run_ppo_with_tensorboard():
    """运行带TensorBoard监控的PPO训练演示"""

    print("🚀 启动PPO训练 + TensorBoard监控演示")
    print("=" * 60)

    # 创建日志目录
    log_dir = Path("./outputs/runs/ppo_demo")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 日志目录: {log_dir.absolute()}")
    print("🎯 训练配置: 2副本, 100请求, 1QPS")
    print("📊 TensorBoard: http://localhost:6006")
    print("-" * 60)

    # PPO训练命令
    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--p_p_o_global_scheduler_modular_config_enable_tensorboard",
        "--p_p_o_global_scheduler_modular_config_tensorboard_log_dir", str(log_dir),
        "--p_p_o_global_scheduler_modular_config_rollout_len", "8",
        "--p_p_o_global_scheduler_modular_config_lr", "0.001",
        "--p_p_o_global_scheduler_modular_config_entropy_coef", "0.02",
        "--cluster_config_num_replicas", "2",
        "--synthetic_request_generator_config_num_requests", "100",
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "1",
        "--metrics_config_subsamples", "5000"
    ]

    print("💡 提示: 训练启动后自动打开 TensorBoard")
    print("📈 可观察指标:")
    print("  - Training/PolicyLoss: 策略损失")
    print("  - Training/ValueLoss: 价值函数损失")
    print("  - Training/Entropy: 策略熵")
    print("  - Reward/Total: 总奖励")
    print("  - Reward/Throughput: 吞吐量")
    print("  - Reward/Latency: 延迟")
    print("  - System/BufferProgress: 缓冲区进度")
    print("-" * 60)

    try:
        # 延时后打开浏览器
        def open_tensorboard():
            time.sleep(3)  # 等待TensorBoard启动
            try:
                webbrowser.open("http://localhost:6006")
            except:
                pass

        import threading
        browser_thread = threading.Thread(target=open_tensorboard, daemon=True)
        browser_thread.start()

        # 运行训练
        print("🎬 开始训练...")
        result = subprocess.run(cmd, timeout=120)  # 2分钟超时

        if result.returncode == 0:
            print("✅ 训练完成!")
        else:
            print("⚠️  训练提前结束")

    except subprocess.TimeoutExpired:
        print("⏰ 演示训练完成 (2分钟)")
    except KeyboardInterrupt:
        print("🛑 用户中断训练")
    except Exception as e:
        print(f"❌ 训练出错: {e}")

    print("-" * 60)
    print("📊 TensorBoard仍在运行，可访问查看完整指标")
    print("🔍 日志文件位置:", log_dir.absolute())
    print("=" * 60)


if __name__ == "__main__":
    run_ppo_with_tensorboard()