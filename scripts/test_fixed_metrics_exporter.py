#!/usr/bin/env python3
"""
测试修复后的MetricsExporter

验证CSV架构演化功能是否正常工作，模拟真实的PPO训练指标记录场景。
"""

import tempfile
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.utils.monitoring.metrics_exporter import MetricsExporter


def test_csv_schema_evolution():
    """测试CSV架构演化功能"""
    print("🧪 测试CSV架构演化功能...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建MetricsExporter实例
        exporter = MetricsExporter(
            export_path=temp_dir,
            export_format="csv",
            flush_interval=3,  # 小的flush间隔便于测试
            enabled=True
        )

        print(f"📁 测试目录: {temp_dir}")

        # 阶段1: 添加基础训练指标（模拟早期步骤数据）
        print("\n📊 阶段1: 添加基础训练指标...")
        for step in range(5):
            base_metrics = {
                "reward": step * 0.1,
                "action": step % 4,
                "queue_length": 10 + step,
                "latency": 1.5 + step * 0.1,
            }

            exporter.append_training_metrics(
                step=step,
                metrics=base_metrics,
                metadata={
                    "replica_id": step % 4,
                    "data_type": "training"
                }
            )

        print(f"✅ 已添加5条基础指标记录")

        # 阶段2: 添加PPO更新指标（模拟PPO算法更新）
        print("\n📊 阶段2: 添加PPO更新指标...")
        for update_step in [5, 8, 11]:
            ppo_metrics = {
                "pi_loss": -0.01 + update_step * 0.001,
                "vf_loss": 0.05 + update_step * 0.002,
                "entropy": 1.3 - update_step * 0.01,
                "approx_kl": 0.001 + update_step * 0.0001,
                "clipfrac": 0.0,
                "pg_grad_norm": 2.0 + update_step * 0.1,
                "explained_var": 0.2 + update_step * 0.01,
                "lr": 0.0003,
            }

            exporter.append_training_metrics(
                step=update_step,
                metrics=ppo_metrics,
                metadata={
                    "data_type": "ppo_update",
                    "rollout_length": 32,
                    "buffer_size": 32
                }
            )

        print(f"✅ 已添加3条PPO更新指标记录")

        # 阶段3: 继续添加基础指标
        print("\n📊 阶段3: 继续添加基础指标...")
        for step in range(12, 15):
            base_metrics = {
                "reward": step * 0.1,
                "action": step % 4,
                "queue_length": 10 + step,
                "latency": 1.5 + step * 0.1,
                "throughput": 0.8 + step * 0.02,  # 新字段
            }

            exporter.append_training_metrics(
                step=step,
                metrics=base_metrics,
                metadata={
                    "replica_id": step % 4,
                    "data_type": "training"
                }
            )

        print(f"✅ 已添加3条带新字段的基础指标记录")

        # 强制flush确保所有数据写入
        exporter.flush()
        exporter.close()

        # 验证结果
        print("\n🔍 验证结果...")
        csv_files = list(Path(temp_dir).glob("*.csv"))

        if not csv_files:
            print("❌ 没有找到CSV文件!")
            return False

        csv_file = csv_files[0]
        print(f"📄 CSV文件: {csv_file}")

        # 读取并分析CSV内容
        with open(csv_file, 'r') as f:
            lines = f.readlines()

        print(f"📏 总行数: {len(lines)}")

        if len(lines) < 2:
            print("❌ CSV文件内容不足!")
            return False

        # 检查头部
        header = lines[0].strip()
        print(f"📊 CSV头部: {header}")

        # 检查关键字段是否存在
        expected_fields = ['pi_loss', 'vf_loss', 'entropy', 'approx_kl', 'reward', 'action']
        missing_fields = []

        for field in expected_fields:
            if field not in header:
                missing_fields.append(field)

        if missing_fields:
            print(f"❌ 缺失字段: {missing_fields}")
            return False
        else:
            print("✅ 所有关键字段都存在于CSV头部!")

        # 检查数据行
        data_lines = [line.strip() for line in lines[1:] if line.strip()]
        print(f"📊 数据行数: {len(data_lines)}")

        # 检查PPO更新行
        ppo_lines = [line for line in data_lines if 'ppo_update' in line]
        print(f"🎯 PPO更新行数: {len(ppo_lines)}")

        if len(ppo_lines) != 3:
            print(f"❌ 期望3行PPO更新数据，实际得到 {len(ppo_lines)} 行")
            return False

        # 显示示例PPO行
        if ppo_lines:
            print(f"📄 PPO更新示例: {ppo_lines[0][:100]}...")

        print("✅ CSV架构演化测试通过!")
        return True


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n🧪 测试向后兼容性...")

    with tempfile.TemporaryDirectory() as temp_dir:
        exporter = MetricsExporter(
            export_path=temp_dir,
            export_format="csv",
            flush_interval=5,
            enabled=True
        )

        # 只添加单一类型的指标（应该像以前一样工作）
        for step in range(8):
            metrics = {
                "reward": step * 0.1,
                "action": step % 3,
                "loss": 0.1 - step * 0.01,
            }

            exporter.append_training_metrics(
                step=step,
                metrics=metrics
            )

        exporter.close()

        # 检查结果
        csv_files = list(Path(temp_dir).glob("*.csv"))
        if not csv_files:
            print("❌ 向后兼容性测试失败: 没有CSV文件")
            return False

        with open(csv_files[0], 'r') as f:
            content = f.read()

        if 'reward' not in content or 'action' not in content:
            print("❌ 向后兼容性测试失败: 缺少预期字段")
            return False

        print("✅ 向后兼容性测试通过!")
        return True


def main():
    """运行所有测试"""
    print("🚀 测试修复后的MetricsExporter")
    print("=" * 50)

    # 运行测试
    test1_passed = test_csv_schema_evolution()
    test2_passed = test_backward_compatibility()

    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果:")
    print(f"  🎯 CSV架构演化: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"  🔄 向后兼容性: {'✅ 通过' if test2_passed else '❌ 失败'}")

    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过! MetricsExporter修复成功!")
        return 0
    else:
        print("\n💥 有测试失败，需要进一步调试")
        return 1


if __name__ == "__main__":
    exit(main())