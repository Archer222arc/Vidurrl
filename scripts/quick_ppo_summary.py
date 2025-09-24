#!/usr/bin/env python3
"""
快速PPO训练摘要工具

快速显示最新PPO训练指标的摘要，用于训练过程中的快速检查。
"""

import pandas as pd
import sys
from pathlib import Path


def quick_summary(csv_path: str) -> None:
    """
    显示PPO训练的快速摘要

    Args:
        csv_path: CSV文件路径
    """
    try:
        # 加载数据
        df = pd.read_csv(csv_path)
        ppo_updates = df[df['data_type'] == 'ppo_update'].copy()

        if len(ppo_updates) == 0:
            print("❌ 没有找到PPO更新数据")
            return

        # 排序并获取最新数据
        ppo_updates = ppo_updates.sort_values('step')
        latest = ppo_updates.iloc[-1]

        print(f"🚀 PPO训练快速摘要")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📊 训练步骤: {int(latest['step'])}")
        print(f"🔄 PPO更新: {len(ppo_updates)}次")
        print(f"⏰ 最新时间: {latest['datetime']}")
        print()

        # 关键指标
        print("📈 关键指标:")
        metrics = [
            ('策略损失', 'pi_loss', '{:.6f}'),
            ('价值损失', 'vf_loss', '{:.6f}'),
            ('策略熵', 'entropy', '{:.4f}'),
            ('KL散度', 'approx_kl', '{:.6f}'),
            ('梯度范数', 'pg_grad_norm', '{:.3f}'),
            ('解释方差', 'explained_var', '{:.4f}')
        ]

        for name, key, fmt in metrics:
            if key in latest and not pd.isna(latest[key]):
                value = latest[key]
                print(f"  {name}: {fmt.format(value)}")
            else:
                print(f"  {name}: N/A")

        # 训练状态简评
        print()
        print("🎯 状态:")
        if 'entropy' in latest and not pd.isna(latest['entropy']):
            entropy = latest['entropy']
            if entropy > 1.0:
                print("  🎲 探索充分")
            elif entropy > 0.5:
                print("  ⚖️  探索适中")
            else:
                print("  🎯 策略收敛")

        if len(ppo_updates) >= 5:
            recent_losses = ppo_updates.tail(5)['pi_loss'].dropna()
            if len(recent_losses) >= 3:
                trend = recent_losses.diff().mean()
                if trend < -0.001:
                    print("  ✅ 损失下降")
                elif trend > 0.001:
                    print("  ⚠️  损失上升")
                else:
                    print("  📊 损失稳定")

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    except Exception as e:
        print(f"❌ 错误: {e}")


def main():
    if len(sys.argv) != 2:
        print("用法: python quick_ppo_summary.py <csv_file>")
        return 1

    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"❌ 文件不存在: {csv_path}")
        return 1

    quick_summary(csv_path)
    return 0


if __name__ == "__main__":
    exit(main())