#!/usr/bin/env python3
import re
import sys
import csv
import pandas as pd
from pathlib import Path

def find_latest_simulator_output(scheduler_name):
    """查找对应调度器的最新simulator输出目录"""
    simulator_output_dir = Path("outputs/simulator_output")
    if not simulator_output_dir.exists():
        return None

    # 获取所有输出目录，按时间排序
    output_dirs = sorted([d for d in simulator_output_dir.iterdir() if d.is_dir()],
                        key=lambda x: x.name, reverse=True)

    # 查找最近4个目录（对应4个调度器的测试）
    return output_dirs[:4] if len(output_dirs) >= 4 else output_dirs

def extract_metrics_from_csv(simulator_dir):
    """从CSV文件中提取真实的性能指标"""
    if not simulator_dir or not simulator_dir.exists():
        return {"status": "❌ 未找到", "latency": "N/A", "throughput": "N/A"}

    try:
        # 读取throughput_latency.csv获取平均指标
        throughput_file = simulator_dir / "throughput_latency.csv"
        request_metrics_file = simulator_dir / "request_metrics.csv"

        if throughput_file.exists():
            df = pd.read_csv(throughput_file)
            # 计算最后阶段的平均吞吐量和延迟（排除启动阶段的0值）
            valid_data = df[(df['throughput'] > 0) & (df['avg_latency'] > 0)]

            if len(valid_data) > 0:
                avg_throughput = valid_data['throughput'].mean()
                avg_latency = valid_data['avg_latency'].mean()
                return {
                    "status": "✅ 成功",
                    "latency": f"{avg_latency:.3f}",
                    "throughput": f"{avg_throughput:.3f}"
                }

        # 备用方案：从request_metrics.csv计算
        if request_metrics_file.exists():
            df = pd.read_csv(request_metrics_file)
            if len(df) > 0:
                avg_latency = df['request_e2e_time'].mean()
                # 估算吞吐量（请求数/总时间）
                total_time = df['request_e2e_time'].sum() / len(df) if len(df) > 0 else 1
                throughput = len(df) / max(total_time, 1)
                return {
                    "status": "✅ 成功",
                    "latency": f"{avg_latency:.3f}",
                    "throughput": f"{throughput:.3f}"
                }

        return {"status": "⚠️ 无数据", "latency": "N/A", "throughput": "N/A"}

    except Exception as e:
        return {"status": "❌ 解析失败", "latency": "N/A", "throughput": "N/A"}

def extract_key_metrics_fallback(log_path):
    """从日志文件提取指标的备用方案"""
    if not log_path.exists():
        return {"status": "❌ 未找到", "latency": "N/A", "throughput": "N/A"}

    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # 检查是否有错误
        if "Traceback" in content or "Error" in content:
            return {"status": "❌ 错误", "latency": "N/A", "throughput": "N/A"}

        return {"status": "⚠️ 运行完成", "latency": "N/A", "throughput": "N/A"}

    except Exception as e:
        return {"status": "❌ 解析失败", "latency": "N/A", "throughput": "N/A"}

def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    schedulers = [
        ("PPO", "ppo_output.log"),
        ("Random", "random_output.log"),
        ("RoundRobin", "round_robin_output.log"),
        ("LOR", "lor_output.log")
    ]

    print(f"{'调度器':<12} {'状态':<10} {'延迟(s)':<10} {'吞吐量(req/s)':<12}")
    print("-" * 52)

    # 获取最新的simulator输出目录
    simulator_dirs = find_latest_simulator_output("all")

    for i, (name, log_file) in enumerate(schedulers):
        # 尝试从CSV文件提取指标
        if simulator_dirs and i < len(simulator_dirs):
            # 反向映射：第i个调度器对应倒数第(i+1)个目录
            # PPO(i=0) -> simulator_dirs[3], Random(i=1) -> simulator_dirs[2], etc.
            reverse_index = len(simulator_dirs) - 1 - i
            metrics = extract_metrics_from_csv(simulator_dirs[reverse_index])
        else:
            # 备用方案：从日志文件提取
            log_path = results_dir / log_file
            metrics = extract_key_metrics_fallback(log_path)

        print(f"{name:<12} {metrics['status']:<10} {metrics['latency']:<10} {metrics['throughput']:<12}")

if __name__ == "__main__":
    main()
