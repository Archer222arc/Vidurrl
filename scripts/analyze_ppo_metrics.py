#!/usr/bin/env python3
"""
PPO训练指标分析脚本

该脚本从CSV文件中提取并分析PPO训练的关键指标，
包括策略损失、价值损失、熵等核心训练信息。
"""

import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_ppo_metrics(csv_path: str) -> pd.DataFrame:
    """
    加载并过滤PPO训练指标数据

    Args:
        csv_path: CSV文件路径

    Returns:
        包含PPO更新指标的DataFrame
    """
    print(f"📊 加载指标文件: {csv_path}")

    # 加载CSV文件
    df = pd.read_csv(csv_path)

    print(f"总记录数: {len(df)}")
    print(f"数据类型分布:")
    print(df['data_type'].value_counts())

    # 过滤PPO更新数据
    ppo_updates = df[df['data_type'] == 'ppo_update'].copy()
    print(f"\nPPO更新记录数: {len(ppo_updates)}")

    return ppo_updates


def analyze_ppo_metrics(ppo_df: pd.DataFrame) -> None:
    """
    分析PPO训练指标

    Args:
        ppo_df: PPO更新数据的DataFrame
    """
    print("\n" + "="*60)
    print("PPO训练指标分析")
    print("="*60)

    # 关键指标列表
    key_metrics = [
        'pi_loss',      # 策略损失
        'vf_loss',      # 价值函数损失
        'entropy',      # 策略熵
        'approx_kl',    # 近似KL散度
        'clipfrac',     # 裁剪比例
        'pg_grad_norm', # 策略梯度范数
        'explained_var', # 解释方差
        'lr'            # 学习率
    ]

    print("\n📈 关键指标统计:")
    print("-" * 40)

    for metric in key_metrics:
        if metric in ppo_df.columns:
            values = ppo_df[metric].dropna()
            if len(values) > 0:
                print(f"{metric:15s}: 平均={values.mean():8.6f}, 标准差={values.std():8.6f}, "
                      f"最小={values.min():8.6f}, 最大={values.max():8.6f}")
            else:
                print(f"{metric:15s}: 无数据")
        else:
            print(f"{metric:15s}: 列不存在")

    # 检查数据完整性
    print("\n🔍 数据完整性检查:")
    print("-" * 40)

    for metric in key_metrics:
        if metric in ppo_df.columns:
            non_null_count = ppo_df[metric].count()
            total_count = len(ppo_df)
            completeness = (non_null_count / total_count) * 100 if total_count > 0 else 0
            print(f"{metric:15s}: {non_null_count}/{total_count} ({completeness:5.1f}%)")
        else:
            print(f"{metric:15s}: 列不存在")


def plot_training_progress(ppo_df: pd.DataFrame, output_dir: str = None) -> None:
    """
    绘制训练进度图表

    Args:
        ppo_df: PPO更新数据的DataFrame
        output_dir: 输出目录（可选）
    """
    if len(ppo_df) == 0:
        print("⚠️  没有PPO更新数据可供绘图")
        return

    print("\n📊 生成训练进度图表...")

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PPO训练指标进度', fontsize=16)

    # 指标绘图配置
    plot_configs = [
        ('pi_loss', '策略损失', 'red'),
        ('vf_loss', '价值函数损失', 'blue'),
        ('entropy', '策略熵', 'green'),
        ('approx_kl', '近似KL散度', 'orange'),
        ('clipfrac', '裁剪比例', 'purple'),
        ('explained_var', '解释方差', 'brown')
    ]

    for i, (metric, title, color) in enumerate(plot_configs):
        row, col = i // 3, i % 3
        ax = axes[row, col]

        if metric in ppo_df.columns and ppo_df[metric].count() > 0:
            values = ppo_df[metric].dropna()
            steps = ppo_df.loc[values.index, 'step']

            ax.plot(steps, values, color=color, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel('训练步数')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

            # 添加趋势线
            if len(values) > 1:
                z = np.polyfit(range(len(values)), values, 1)
                p = np.poly1d(z)
                ax.plot(steps, p(range(len(values))), "--", color='gray', alpha=0.8)
        else:
            ax.text(0.5, 0.5, f'无{title}数据', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / 'ppo_training_progress.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 图表已保存: {output_path}")

    plt.show()


def export_clean_metrics(ppo_df: pd.DataFrame, output_path: str) -> None:
    """
    导出清理后的PPO指标数据

    Args:
        ppo_df: PPO更新数据的DataFrame
        output_path: 输出文件路径
    """
    # 选择关键列
    key_columns = [
        'step', 'datetime', 'timestamp',
        'pi_loss', 'vf_loss', 'entropy', 'approx_kl',
        'clipfrac', 'pg_grad_norm', 'explained_var', 'lr',
        'rollout_length', 'buffer_size'
    ]

    # 过滤存在的列
    available_columns = [col for col in key_columns if col in ppo_df.columns]
    clean_df = ppo_df[available_columns].copy()

    # 保存
    clean_df.to_csv(output_path, index=False)
    print(f"📁 清理后的指标已导出: {output_path}")
    print(f"导出列: {', '.join(available_columns)}")


def main():
    parser = argparse.ArgumentParser(description='分析PPO训练指标')
    parser.add_argument('csv_path', help='PPO指标CSV文件路径')
    parser.add_argument('--output-dir', '-o', help='输出目录（用于保存图表和清理数据）')
    parser.add_argument('--plot', action='store_true', help='生成训练进度图表')
    parser.add_argument('--export-clean', action='store_true', help='导出清理后的指标数据')

    args = parser.parse_args()

    # 检查输入文件
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"❌ 文件不存在: {csv_path}")
        return 1

    # 设置输出目录
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 加载数据
        ppo_df = load_ppo_metrics(str(csv_path))

        if len(ppo_df) == 0:
            print("⚠️  没有找到PPO更新数据")
            return 1

        # 分析指标
        analyze_ppo_metrics(ppo_df)

        # 生成图表
        if args.plot:
            import numpy as np  # 延迟导入
            plot_training_progress(ppo_df, str(output_dir))

        # 导出清理数据
        if args.export_clean:
            clean_path = output_dir / f"ppo_metrics_clean_{csv_path.stem}.csv"
            export_clean_metrics(ppo_df, str(clean_path))

        print(f"\n✅ 分析完成！")
        return 0

    except Exception as e:
        print(f"❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())