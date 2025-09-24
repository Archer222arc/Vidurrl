#!/usr/bin/env python3
"""
PPO训练指标仪表板

实时显示PPO训练的关键指标，提供清晰的训练进度可视化。
支持多种输出格式和实时监控功能。
"""

import pandas as pd
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any
import sys


class PPOMetricsDashboard:
    """PPO训练指标仪表板"""

    def __init__(self, csv_path: str, refresh_interval: int = 5):
        """
        初始化仪表板

        Args:
            csv_path: CSV文件路径
            refresh_interval: 刷新间隔（秒）
        """
        self.csv_path = Path(csv_path)
        self.refresh_interval = refresh_interval
        self.last_step = 0

    def load_latest_metrics(self) -> Optional[pd.DataFrame]:
        """
        加载最新的PPO指标数据

        Returns:
            PPO更新数据的DataFrame，如果失败返回None
        """
        try:
            if not self.csv_path.exists():
                return None

            df = pd.read_csv(self.csv_path)
            ppo_updates = df[df['data_type'] == 'ppo_update'].copy()

            if len(ppo_updates) > 0:
                ppo_updates = ppo_updates.sort_values('step')

            return ppo_updates

        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None

    def format_metric(self, value: float, metric_name: str) -> str:
        """
        格式化指标值

        Args:
            value: 指标值
            metric_name: 指标名称

        Returns:
            格式化后的字符串
        """
        if pd.isna(value):
            return "N/A"

        # 根据指标类型使用不同的格式
        if metric_name in ['pi_loss', 'vf_loss']:
            return f"{value:8.6f}"
        elif metric_name in ['entropy', 'approx_kl', 'explained_var']:
            return f"{value:8.4f}"
        elif metric_name in ['clipfrac']:
            return f"{value:8.6f}"
        elif metric_name in ['pg_grad_norm']:
            return f"{value:8.3f}"
        elif metric_name == 'lr':
            return f"{value:.6f}"
        else:
            return f"{value:8.4f}"

    def get_trend_indicator(self, current: float, previous: float) -> str:
        """
        获取趋势指示器

        Args:
            current: 当前值
            previous: 前一个值

        Returns:
            趋势指示符
        """
        if pd.isna(current) or pd.isna(previous):
            return "─"

        diff = current - previous
        if abs(diff) < 1e-8:
            return "─"
        elif diff > 0:
            return "↑"
        else:
            return "↓"

    def display_latest_metrics(self, df: pd.DataFrame) -> None:
        """
        显示最新的指标

        Args:
            df: PPO指标数据
        """
        if len(df) == 0:
            print("⚠️  没有PPO训练数据")
            return

        # 获取最新的记录
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else None

        # 清屏
        print("\033[2J\033[H", end="")

        print("=" * 80)
        print(f"🚀 PPO训练指标仪表板 - 步骤 {int(latest['step'])}")
        print("=" * 80)

        # 时间信息
        print(f"⏰ 最新更新: {latest['datetime']}")
        print(f"📊 总训练步数: {int(latest['step'])}")
        print(f"📈 PPO更新次数: {len(df)}")

        print("\n" + "─" * 80)
        print("📊 核心训练指标")
        print("─" * 80)

        # 指标显示配置
        metrics_config = [
            ('pi_loss', '策略损失', 'Lower is better'),
            ('vf_loss', '价值函数损失', 'Lower is better'),
            ('entropy', '策略熵', 'Balance needed'),
            ('approx_kl', 'KL散度', 'Low values preferred'),
            ('clipfrac', '裁剪比例', 'Low values preferred'),
            ('pg_grad_norm', '梯度范数', 'Moderate values'),
            ('explained_var', '解释方差', 'Higher is better'),
            ('lr', '学习率', 'Fixed/scheduled')
        ]

        for metric, chinese_name, description in metrics_config:
            if metric in latest and not pd.isna(latest[metric]):
                current_val = latest[metric]
                trend = self.get_trend_indicator(
                    current_val,
                    previous[metric] if previous is not None and metric in previous else current_val
                )

                formatted_val = self.format_metric(current_val, metric)
                print(f"{chinese_name:12s} │ {formatted_val} {trend} │ {description}")
            else:
                print(f"{chinese_name:12s} │     N/A     │ {description}")

        # 训练状态评估
        print("\n" + "─" * 80)
        print("🎯 训练状态评估")
        print("─" * 80)

        if len(df) >= 5:
            recent_df = df.tail(5)
            self.assess_training_health(recent_df)
        else:
            print("⏳ 数据不足，需要更多训练步骤进行评估")

        print("\n" + "─" * 80)
        print(f"🔄 下次刷新: {self.refresh_interval}秒后 (Ctrl+C 退出)")
        print("─" * 80)

    def assess_training_health(self, recent_df: pd.DataFrame) -> None:
        """
        评估训练健康状态

        Args:
            recent_df: 最近的训练数据
        """
        assessments = []

        # 策略损失趋势
        if 'pi_loss' in recent_df.columns:
            pi_losses = recent_df['pi_loss'].dropna()
            if len(pi_losses) >= 3:
                recent_trend = pi_losses.iloc[-3:].diff().mean()
                if recent_trend < -0.001:
                    assessments.append("✅ 策略损失持续下降")
                elif recent_trend > 0.001:
                    assessments.append("⚠️  策略损失上升，需要关注")
                else:
                    assessments.append("📊 策略损失稳定")

        # 熵值检查
        if 'entropy' in recent_df.columns:
            entropies = recent_df['entropy'].dropna()
            if len(entropies) > 0:
                avg_entropy = entropies.mean()
                if avg_entropy > 1.0:
                    assessments.append("🎲 熵值较高，探索充分")
                elif avg_entropy > 0.5:
                    assessments.append("⚖️  熵值适中，探索-利用平衡")
                else:
                    assessments.append("🎯 熵值较低，策略趋于确定")

        # KL散度检查
        if 'approx_kl' in recent_df.columns:
            kl_divs = recent_df['approx_kl'].dropna()
            if len(kl_divs) > 0:
                avg_kl = kl_divs.mean()
                if avg_kl > 0.01:
                    assessments.append("⚠️  KL散度较高，策略变化剧烈")
                elif avg_kl > 0.005:
                    assessments.append("📈 KL散度适中，策略稳步优化")
                else:
                    assessments.append("🔒 KL散度很低，策略变化缓慢")

        # 解释方差检查
        if 'explained_var' in recent_df.columns:
            explained_vars = recent_df['explained_var'].dropna()
            if len(explained_vars) > 0:
                avg_ev = explained_vars.mean()
                if avg_ev > 0.5:
                    assessments.append("✅ 价值函数学习良好")
                elif avg_ev > 0.0:
                    assessments.append("📊 价值函数学习进展中")
                else:
                    assessments.append("⚠️  价值函数学习困难")

        # 显示评估结果
        if assessments:
            for assessment in assessments:
                print(f"  {assessment}")
        else:
            print("  📊 数据不足，无法评估")

    def run_dashboard(self, watch_mode: bool = False) -> None:
        """
        运行仪表板

        Args:
            watch_mode: 是否启用监控模式
        """
        if not watch_mode:
            # 单次显示模式
            df = self.load_latest_metrics()
            if df is not None:
                self.display_latest_metrics(df)
            else:
                print("❌ 无法加载指标数据")
            return

        # 监控模式
        print(f"👀 启动PPO训练监控 - 文件: {self.csv_path}")
        print(f"🔄 刷新间隔: {self.refresh_interval}秒")
        print("按 Ctrl+C 退出监控\n")

        try:
            while True:
                df = self.load_latest_metrics()
                if df is not None and len(df) > 0:
                    current_step = df.iloc[-1]['step']
                    if current_step != self.last_step:
                        self.display_latest_metrics(df)
                        self.last_step = current_step
                    else:
                        # 如果没有新数据，只更新时间
                        print(f"\r⏳ 等待新的训练数据... {time.strftime('%H:%M:%S')}", end="", flush=True)
                else:
                    print(f"\r❌ 数据文件不存在或为空... {time.strftime('%H:%M:%S')}", end="", flush=True)

                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print("\n\n👋 监控已停止")


def main():
    parser = argparse.ArgumentParser(description='PPO训练指标仪表板')
    parser.add_argument('csv_path', help='PPO指标CSV文件路径')
    parser.add_argument('--watch', '-w', action='store_true', help='启用实时监控模式')
    parser.add_argument('--interval', '-i', type=int, default=5, help='刷新间隔（秒）')

    args = parser.parse_args()

    # 检查文件是否存在
    csv_path = Path(args.csv_path)
    if not args.watch and not csv_path.exists():
        print(f"❌ 文件不存在: {csv_path}")
        return 1

    # 创建并运行仪表板
    dashboard = PPOMetricsDashboard(str(csv_path), args.interval)
    dashboard.run_dashboard(args.watch)

    return 0


if __name__ == "__main__":
    exit(main())