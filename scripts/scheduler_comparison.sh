#!/bin/bash

# =============================================================================
# 调度器性能对比评测脚本
#
# 对比PPO、Random、Round Robin、LOR调度器在相同配置下的性能
# 输出平均延迟和吞吐量指标，严格遵循CLAUDE.md项目规范
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"

# 配置参数
COMPARISON_ID=$(date +%Y%m%d_%H%M%S)
NUM_REQUESTS=500
QPS=3
NUM_REPLICAS=4
OUTPUT_DIR="./outputs/runs/scheduler_comparison/run_${COMPARISON_ID}"
RESULTS_DIR="${OUTPUT_DIR}/results"
CHECKPOINT_PATH="/Users/ruicheng/Documents/GitHub/Vidur/Vidur_arc2/outputs/checkpoints/latest.pt"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${RESULTS_DIR}"

echo "🚀 开始调度器性能对比评测 - Run ID: ${COMPARISON_ID}"
echo "📋 测试配置:"
echo "   - 请求数量: ${NUM_REQUESTS}"
echo "   - QPS: ${QPS}"
echo "   - 副本数: ${NUM_REPLICAS}"
echo "   - 输出目录: ${OUTPUT_DIR}"
echo ""

# 基础命令参数
BASE_PARAMS="
  --cluster_config_num_replicas ${NUM_REPLICAS}
  --synthetic_request_generator_config_num_requests ${NUM_REQUESTS}
  --interval_generator_config_type poisson
  --poisson_request_interval_generator_config_qps ${QPS}
  --metrics_config_subsamples 200000
"

# =============================================================================
# 测试1: PPO调度器 (使用checkpoint)
# =============================================================================
echo "📊 [1/4] 测试 PPO 调度器..."

python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  ${BASE_PARAMS} \
  --p_p_o_global_scheduler_modular_config_debug_dump_global_state \
  --p_p_o_global_scheduler_modular_config_max_queue_requests_per_replica 8 \
  --p_p_o_global_scheduler_modular_config_inference_only \
  --p_p_o_global_scheduler_modular_config_load_checkpoint "${CHECKPOINT_PATH}" \
  --no-p_p_o_global_scheduler_modular_config_enable_tensorboard \
  --no-p_p_o_global_scheduler_modular_config_enable_checkpoints \
  2>&1 | tee "${RESULTS_DIR}/ppo_output.log"

if [ $? -eq 0 ]; then
    echo "✅ PPO 调度器测试完成"
else
    echo "❌ PPO 调度器测试失败，请检查日志: ${RESULTS_DIR}/ppo_output.log"
fi

# =============================================================================
# 测试2: Random调度器
# =============================================================================
echo "📊 [2/4] 测试 Random 调度器..."

python -m vidur.main \
  --global_scheduler_config_type random \
  ${BASE_PARAMS} \
  2>&1 | tee "${RESULTS_DIR}/random_output.log"

if [ $? -eq 0 ]; then
    echo "✅ Random 调度器测试完成"
else
    echo "❌ Random 调度器测试失败，请检查日志: ${RESULTS_DIR}/random_output.log"
fi

# =============================================================================
# 测试3: Round Robin调度器
# =============================================================================
echo "📊 [3/4] 测试 Round Robin 调度器..."

python -m vidur.main \
  --global_scheduler_config_type round_robin \
  ${BASE_PARAMS} \
  2>&1 | tee "${RESULTS_DIR}/round_robin_output.log"

if [ $? -eq 0 ]; then
    echo "✅ Round Robin 调度器测试完成"
else
    echo "❌ Round Robin 调度器测试失败，请检查日志: ${RESULTS_DIR}/round_robin_output.log"
fi

# =============================================================================
# 测试4: LOR调度器
# =============================================================================
echo "📊 [4/4] 测试 LOR 调度器..."

python -m vidur.main \
  --global_scheduler_config_type lor \
  ${BASE_PARAMS} \
  2>&1 | tee "${RESULTS_DIR}/lor_output.log"

if [ $? -eq 0 ]; then
    echo "✅ LOR 调度器测试完成"
else
    echo "❌ LOR 调度器测试失败，请检查日志: ${RESULTS_DIR}/lor_output.log"
fi

# =============================================================================
# 结果分析和汇总
# =============================================================================
echo ""
echo "🔍 分析测试结果..."

# 创建结果分析脚本
cat > "${OUTPUT_DIR}/analyze_results.py" << 'EOF'
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
EOF

# 设置环境变量并运行分析脚本
export NUM_REQUESTS="${NUM_REQUESTS}"
export QPS="${QPS}"
export NUM_REPLICAS="${NUM_REPLICAS}"
export COMPARISON_ID="${COMPARISON_ID}"

python3 "${OUTPUT_DIR}/analyze_results.py" "${RESULTS_DIR}"

# =============================================================================
# 快速总览 - 直接输出简洁的性能对比表格
# =============================================================================
echo ""
echo "📊 性能对比总览"
echo "=================================================================="

# 创建增强的总览脚本 - 从simulator_output CSV文件提取真实指标
cat > "${OUTPUT_DIR}/quick_summary.py" << 'EOF'
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
EOF

# 运行快速总览
python3 "${OUTPUT_DIR}/quick_summary.py" "${RESULTS_DIR}"

echo ""
echo "🎉 调度器性能对比评测完成！"
echo "📂 结果目录: ${OUTPUT_DIR}"
echo "📊 详细日志: ${RESULTS_DIR}/"
echo "📝 对比报告: ${OUTPUT_DIR}/scheduler_comparison_report_${COMPARISON_ID}.md"
echo ""
echo "🔗 快速查看结果:"
echo "   cat ${OUTPUT_DIR}/scheduler_comparison_report_${COMPARISON_ID}.md"
