#!/bin/bash

# =============================================================================
# 统计量稳定化机制测试脚本
#
# 测试新实现的statistics_stabilization功能，验证：
# 1. 前100步使用随机策略收集统计量
# 2. 统计量稳定后再开始PPO训练
# 3. 对比有无稳定化的训练曲线差异
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"

# 配置参数
TEST_ID=$(date +%Y%m%d_%H%M%S)
NUM_REQUESTS=1000
QPS=2
NUM_REPLICAS=4
OUTPUT_DIR="./outputs/runs/statistics_stabilization_test/run_${TEST_ID}"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

echo "🧪 开始统计量稳定化机制测试 - Test ID: ${TEST_ID}"
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
# 测试1: 启用统计量稳定化的PPO训练
# =============================================================================
echo "📊 [1/2] 测试启用统计量稳定化的PPO训练..."

python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  ${BASE_PARAMS} \
  --p_p_o_global_scheduler_modular_config_enable_statistics_stabilization \
  --p_p_o_global_scheduler_modular_config_statistics_stabilization_steps 100 \
  --p_p_o_global_scheduler_modular_config_stabilization_policy random \
  --p_p_o_global_scheduler_modular_config_enable_stabilization_logging \
  --p_p_o_global_scheduler_modular_config_max_queue_requests_per_replica 8 \
  --p_p_o_global_scheduler_modular_config_enable_tensorboard \
  --p_p_o_global_scheduler_modular_config_tensorboard_log_dir "${OUTPUT_DIR}/tensorboard_with_stabilization" \
  --no-p_p_o_global_scheduler_modular_config_enable_checkpoints \
  2>&1 | tee "${OUTPUT_DIR}/with_stabilization.log"

if [ $? -eq 0 ]; then
    echo "✅ 启用稳定化的测试完成"
else
    echo "❌ 启用稳定化的测试失败"
fi

# =============================================================================
# 测试2: 禁用统计量稳定化的PPO训练（对照组）
# =============================================================================
echo "📊 [2/2] 测试禁用统计量稳定化的PPO训练（对照组）..."

python -m vidur.main \
  --global_scheduler_config_type ppo_modular \
  ${BASE_PARAMS} \
  --no-p_p_o_global_scheduler_modular_config_enable_statistics_stabilization \
  --p_p_o_global_scheduler_modular_config_max_queue_requests_per_replica 8 \
  --p_p_o_global_scheduler_modular_config_enable_tensorboard \
  --p_p_o_global_scheduler_modular_config_tensorboard_log_dir "${OUTPUT_DIR}/tensorboard_without_stabilization" \
  --no-p_p_o_global_scheduler_modular_config_enable_checkpoints \
  2>&1 | tee "${OUTPUT_DIR}/without_stabilization.log"

if [ $? -eq 0 ]; then
    echo "✅ 禁用稳定化的测试完成"
else
    echo "❌ 禁用稳定化的测试失败"
fi

# =============================================================================
# 分析测试结果
# =============================================================================
echo ""
echo "🔍 分析测试结果..."

# 创建结果分析脚本
cat > "${OUTPUT_DIR}/analyze_stabilization_effects.py" << 'EOF'
#!/usr/bin/env python3
"""
统计量稳定化效果分析脚本

对比有无统计量稳定化的训练过程，分析：
1. 前100步的动作选择随机性
2. 奖励曲线的稳定性
3. 损失函数的收敛速度
"""

import re
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def extract_stabilization_logs(log_path: Path) -> Dict:
    """从日志中提取统计量稳定化相关信息"""
    if not log_path.exists():
        return {"error": f"Log file not found: {log_path}"}

    stabilization_logs = []
    ppo_training_start = None

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取稳定化阶段日志
        stabilization_pattern = r'\[STATS_STABILIZATION\] step=(\d+)/(\d+) - collecting baseline statistics'
        for match in re.finditer(stabilization_pattern, content):
            step = int(match.group(1))
            total_steps = int(match.group(2))
            stabilization_logs.append((step, total_steps))

        # 检查是否完成稳定化
        completion_pattern = r'\[STATS_STABILIZATION\] Completed! Normalizer statistics collected over (\d+) steps'
        completion_match = re.search(completion_pattern, content)
        if completion_match:
            completed_steps = int(completion_match.group(1))
        else:
            completed_steps = None

        # 检查PPO训练开始
        if "Transitioning to PPO training mode" in content:
            ppo_training_start = True

        # 提取最终归一化统计
        stats_pattern = r'\[STATS_STABILIZATION\] Final normalizer stats - mean_magnitude=([\d.]+), std_magnitude=([\d.]+)'
        stats_match = re.search(stats_pattern, content)
        final_stats = None
        if stats_match:
            final_stats = {
                "mean_magnitude": float(stats_match.group(1)),
                "std_magnitude": float(stats_match.group(2))
            }

        return {
            "stabilization_logs": stabilization_logs,
            "completed_steps": completed_steps,
            "ppo_training_started": ppo_training_start,
            "final_normalizer_stats": final_stats,
            "total_stabilization_logs": len(stabilization_logs)
        }

    except Exception as e:
        return {"error": f"Failed to parse log: {e}"}

def extract_training_metrics(log_path: Path) -> Dict:
    """提取训练过程的关键指标"""
    if not log_path.exists():
        return {"error": f"Log file not found: {log_path}"}

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取PPO步骤的奖励
        reward_pattern = r'\[PPO:step=(\d+)\].*reward=([-\d.]+)'
        rewards = []
        for match in re.finditer(reward_pattern, content):
            step = int(match.group(1))
            reward = float(match.group(2))
            rewards.append((step, reward))

        # 提取训练统计
        training_pattern = r'\[PPO:train\] .*pg_loss=([-\d.]+).*v_loss=([-\d.]+).*entropy=([-\d.]+)'
        training_stats = []
        for match in re.finditer(training_pattern, content):
            pg_loss = float(match.group(1))
            v_loss = float(match.group(2))
            entropy = float(match.group(3))
            training_stats.append((pg_loss, v_loss, entropy))

        return {
            "rewards": rewards,
            "training_stats": training_stats,
            "total_reward_logs": len(rewards),
            "total_training_logs": len(training_stats)
        }

    except Exception as e:
        return {"error": f"Failed to parse training metrics: {e}"}

def analyze_early_stability(rewards: List[Tuple[int, float]], window_size: int = 50) -> Dict:
    """分析前期奖励的稳定性"""
    if len(rewards) < window_size:
        return {"error": "Insufficient data for stability analysis"}

    early_rewards = [r[1] for r in rewards[:window_size]]

    return {
        "mean": np.mean(early_rewards),
        "std": np.std(early_rewards),
        "min": np.min(early_rewards),
        "max": np.max(early_rewards),
        "coefficient_of_variation": np.std(early_rewards) / abs(np.mean(early_rewards)) if np.mean(early_rewards) != 0 else float('inf')
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_stabilization_effects.py <output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    # 分析两个日志文件
    with_stabilization_log = output_dir / "with_stabilization.log"
    without_stabilization_log = output_dir / "without_stabilization.log"

    print("=" * 80)
    print("📊 统计量稳定化效果分析报告")
    print("=" * 80)
    print()

    # 分析启用稳定化的情况
    print("## 启用统计量稳定化的结果")
    print("-" * 40)

    with_results = extract_stabilization_logs(with_stabilization_log)
    if "error" in with_results:
        print(f"❌ 错误: {with_results['error']}")
    else:
        print(f"✅ 稳定化阶段日志数量: {with_results['total_stabilization_logs']}")
        print(f"✅ 完成稳定化步数: {with_results['completed_steps']}")
        print(f"✅ PPO训练开始: {'是' if with_results['ppo_training_started'] else '否'}")

        if with_results['final_normalizer_stats']:
            stats = with_results['final_normalizer_stats']
            print(f"📈 最终归一化统计 - 均值幅度: {stats['mean_magnitude']:.4f}, 标准差幅度: {stats['std_magnitude']:.4f}")

    print()

    # 分析禁用稳定化的情况（对照组）
    print("## 禁用统计量稳定化的结果（对照组）")
    print("-" * 40)

    without_results = extract_stabilization_logs(without_stabilization_log)
    if "error" in without_results:
        print(f"📝 预期结果: 无稳定化日志（正常）")
    else:
        print(f"⚠️  意外发现稳定化日志: {without_results['total_stabilization_logs']}")

    print()

    # 对比训练指标
    print("## 训练指标对比")
    print("-" * 40)

    with_training = extract_training_metrics(with_stabilization_log)
    without_training = extract_training_metrics(without_stabilization_log)

    if "error" not in with_training and "error" not in without_training:
        # 分析前期稳定性
        with_stability = analyze_early_stability(with_training['rewards'])
        without_stability = analyze_early_stability(without_training['rewards'])

        print("### 前50步奖励稳定性对比:")
        if "error" not in with_stability:
            print(f"启用稳定化 - 均值: {with_stability['mean']:.4f}, 标准差: {with_stability['std']:.4f}, 变异系数: {with_stability['coefficient_of_variation']:.4f}")

        if "error" not in without_stability:
            print(f"禁用稳定化 - 均值: {without_stability['mean']:.4f}, 标准差: {without_stability['std']:.4f}, 变异系数: {without_stability['coefficient_of_variation']:.4f}")

        # 计算改善程度
        if "error" not in with_stability and "error" not in without_stability:
            stability_improvement = (without_stability['coefficient_of_variation'] - with_stability['coefficient_of_variation']) / without_stability['coefficient_of_variation'] * 100
            print(f"📈 稳定性改善: {stability_improvement:.1f}% (变异系数降低)")

    print()
    print("## 结论")
    print("-" * 40)

    if "error" not in with_results and with_results['completed_steps']:
        print("✅ 统计量稳定化机制成功运行")
        print(f"✅ 成功收集了 {with_results['completed_steps']} 步的基线统计")
        print("✅ 平滑过渡到PPO训练模式")
    else:
        print("❌ 统计量稳定化机制存在问题")

    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
EOF

# 运行分析
echo "📈 生成分析报告..."
python3 "${OUTPUT_DIR}/analyze_stabilization_effects.py" "${OUTPUT_DIR}"

echo ""
echo "🎉 统计量稳定化机制测试完成！"
echo "📂 结果目录: ${OUTPUT_DIR}"
echo "📊 TensorBoard对比:"
echo "   - 启用稳定化: ${OUTPUT_DIR}/tensorboard_with_stabilization"
echo "   - 禁用稳定化: ${OUTPUT_DIR}/tensorboard_without_stabilization"
echo ""
echo "🔗 查看详细日志:"
echo "   cat ${OUTPUT_DIR}/with_stabilization.log | grep STATS_STABILIZATION"
echo "   cat ${OUTPUT_DIR}/without_stabilization.log | head -100"