#!/usr/bin/env python3
"""
调度器性能对比结果分析脚本

从各调度器的输出日志中提取性能指标，生成对比报告
"""

import re
import sys
import json
from pathlib import Path
from typing import Dict, Optional, Any

def extract_metrics_from_log(log_path: Path) -> Dict[str, Any]:
    """从日志文件中提取性能指标"""
    if not log_path.exists():
        return {"error": f"Log file not found: {log_path}"}

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {"error": f"Failed to read log: {e}"}

    metrics = {}

    # 提取平均延迟
    latency_patterns = [
        r'Average latency:\s*([\d.]+)',
        r'average_latency[:\s]+([\d.]+)',
        r'latency.*?(\d+\.\d+)',
        r'延迟.*?(\d+\.\d+)',
    ]

    for pattern in latency_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            metrics['average_latency'] = float(match.group(1))
            break

    # 提取吞吐量
    throughput_patterns = [
        r'Throughput:\s*([\d.]+)',
        r'throughput[:\s]+([\d.]+)',
        r'requests/second.*?(\d+\.\d+)',
        r'吞吐量.*?(\d+\.\d+)',
    ]

    for pattern in throughput_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            metrics['throughput'] = float(match.group(1))
            break

    # 提取总处理时间
    duration_patterns = [
        r'Total simulation time:\s*([\d.]+)',
        r'execution time.*?(\d+\.\d+)',
        r'duration.*?(\d+\.\d+)',
    ]

    for pattern in duration_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            metrics['total_duration'] = float(match.group(1))
            break

    # 如果找不到指标，尝试从其他地方提取
    if not metrics:
        # 检查是否有错误信息
        error_patterns = [
            r'Error.*',
            r'Exception.*',
            r'Traceback.*',
            r'Failed.*'
        ]

        for pattern in error_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                metrics['error'] = match.group(0)[:200]  # 限制错误信息长度
                break

    return metrics

def generate_comparison_report(results_dir: Path) -> str:
    """生成性能对比报告"""

    schedulers = {
        'PPO': 'ppo_output.log',
        'Random': 'random_output.log',
        'Round Robin': 'round_robin_output.log',
        'LOR': 'lor_output.log'
    }

    all_metrics = {}

    # 提取各调度器指标
    for scheduler_name, log_file in schedulers.items():
        log_path = results_dir / log_file
        metrics = extract_metrics_from_log(log_path)
        all_metrics[scheduler_name] = metrics

    # 生成报告
    report = []
    report.append("=" * 80)
    report.append("📊 调度器性能对比报告")
    report.append("=" * 80)
    report.append("")

    # 性能指标对比表
    report.append("## 性能指标对比")
    report.append("")
    report.append(f"{'调度器':<15} {'平均延迟(s)':<12} {'吞吐量(req/s)':<15} {'状态':<10}")
    report.append("-" * 60)

    valid_results = {}

    for scheduler_name in ['PPO', 'Random', 'Round Robin', 'LOR']:
        metrics = all_metrics[scheduler_name]

        if 'error' in metrics:
            status = "❌ 失败"
            latency_str = "N/A"
            throughput_str = "N/A"
        else:
            latency = metrics.get('average_latency', 'N/A')
            throughput = metrics.get('throughput', 'N/A')

            if latency != 'N/A' and throughput != 'N/A':
                status = "✅ 成功"
                latency_str = f"{latency:.4f}"
                throughput_str = f"{throughput:.4f}"
                valid_results[scheduler_name] = metrics
            else:
                status = "⚠️ 部分"
                latency_str = f"{latency:.4f}" if latency != 'N/A' else "N/A"
                throughput_str = f"{throughput:.4f}" if throughput != 'N/A' else "N/A"

        report.append(f"{scheduler_name:<15} {latency_str:<12} {throughput_str:<15} {status:<10}")

    report.append("")

    # 性能排名
    if valid_results:
        report.append("## 性能排名")
        report.append("")

        # 延迟排名 (越低越好)
        if any('average_latency' in metrics for metrics in valid_results.values()):
            latency_ranking = sorted(
                [(name, metrics.get('average_latency', float('inf')))
                 for name, metrics in valid_results.items()
                 if 'average_latency' in metrics],
                key=lambda x: x[1]
            )

            report.append("### 平均延迟排名 (越低越好):")
            for i, (name, latency) in enumerate(latency_ranking, 1):
                report.append(f"{i}. {name}: {latency:.4f}s")
            report.append("")

        # 吞吐量排名 (越高越好)
        if any('throughput' in metrics for metrics in valid_results.values()):
            throughput_ranking = sorted(
                [(name, metrics.get('throughput', 0))
                 for name, metrics in valid_results.items()
                 if 'throughput' in metrics],
                key=lambda x: x[1], reverse=True
            )

            report.append("### 吞吐量排名 (越高越好):")
            for i, (name, throughput) in enumerate(throughput_ranking, 1):
                report.append(f"{i}. {name}: {throughput:.4f} req/s")
            report.append("")

    # 详细结果
    report.append("## 详细测试结果")
    report.append("")

    for scheduler_name, metrics in all_metrics.items():
        report.append(f"### {scheduler_name} 调度器")
        if 'error' in metrics:
            report.append(f"❌ 测试失败: {metrics['error']}")
        else:
            for key, value in metrics.items():
                if key != 'error':
                    report.append(f"- {key}: {value}")
        report.append("")

    # 测试环境信息
    report.append("## 测试配置")
    report.append(f"- 请求数量: {NUM_REQUESTS}")
    report.append(f"- QPS: {QPS}")
    report.append(f"- 副本数: {NUM_REPLICAS}")
    report.append(f"- 测试时间: {COMPARISON_ID}")
    report.append("")

    report.append("=" * 80)

    return "\n".join(report)

if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    # 从环境变量获取配置
    import os
    global NUM_REQUESTS, QPS, NUM_REPLICAS, COMPARISON_ID
    NUM_REQUESTS = os.environ.get('NUM_REQUESTS', '500')
    QPS = os.environ.get('QPS', '2')
    NUM_REPLICAS = os.environ.get('NUM_REPLICAS', '4')
    COMPARISON_ID = os.environ.get('COMPARISON_ID', 'unknown')

    report = generate_comparison_report(results_dir)
    print(report)

    # 保存报告到文件
    report_file = results_dir.parent / f"scheduler_comparison_report_{COMPARISON_ID}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📝 详细报告已保存到: {report_file}")
