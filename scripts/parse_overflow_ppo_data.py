#!/usr/bin/env python3
"""
解析溢出PPO数据工具

专门用于解析由于CSV架构演化问题导致的"溢出"PPO数据，
直接从原始数据中提取并显示PPO训练指标。
"""

import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple


class PPODataParser:
    """PPO溢出数据解析器"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.header = []
        self.ppo_rows = []

    def load_data(self) -> bool:
        """加载CSV数据"""
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                self.header = next(reader)

                # 查找PPO更新行
                for row in reader:
                    if 'ppo_update' in row:
                        self.ppo_rows.append(row)

            print(f"📊 加载了 {len(self.ppo_rows)} 行PPO数据")
            return len(self.ppo_rows) > 0

        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False

    def parse_ppo_row(self, row: List[str]) -> Dict[str, float]:
        """
        解析单个PPO行，根据实际观察的数据结构

        根据分析，PPO数据在溢出列中的位置：
        - 溢出列1: approx_kl
        - 溢出列2: buffer_size
        - 溢出列3: clipfrac
        - 溢出列6: entropy
        - 溢出列7: explained_var
        - 溢出列8: lr
        - 溢出列9: pg_grad_norm
        - 溢出列10: pi_loss
        - 溢出列14: vf_loss
        """
        base_cols = len(self.header)
        overflow_data = row[base_cols:] if len(row) > base_cols else []

        ppo_metrics = {}

        # 根据观察到的模式解析
        try:
            if len(overflow_data) >= 1:
                ppo_metrics['approx_kl'] = float(overflow_data[0]) if overflow_data[0] else 0.0
            if len(overflow_data) >= 2:
                ppo_metrics['buffer_size'] = float(overflow_data[1]) if overflow_data[1] else 0.0
            if len(overflow_data) >= 3:
                ppo_metrics['clipfrac'] = float(overflow_data[2]) if overflow_data[2] else 0.0
            if len(overflow_data) >= 6:
                ppo_metrics['entropy'] = float(overflow_data[5]) if overflow_data[5] else 0.0
            if len(overflow_data) >= 7:
                ppo_metrics['explained_var'] = float(overflow_data[6]) if overflow_data[6] else 0.0
            if len(overflow_data) >= 8:
                ppo_metrics['lr'] = float(overflow_data[7]) if overflow_data[7] else 0.0
            if len(overflow_data) >= 9:
                ppo_metrics['pg_grad_norm'] = float(overflow_data[8]) if overflow_data[8] else 0.0
            if len(overflow_data) >= 10:
                ppo_metrics['pi_loss'] = float(overflow_data[9]) if overflow_data[9] else 0.0
            if len(overflow_data) >= 14:
                ppo_metrics['vf_loss'] = float(overflow_data[13]) if overflow_data[13] else 0.0

        except (ValueError, IndexError) as e:
            print(f"⚠️  解析行时出错: {e}")

        return ppo_metrics

    def get_step_from_row(self, row: List[str]) -> int:
        """从行中提取步骤号"""
        # 假设step在标准列中
        if 'step' in self.header:
            step_idx = self.header.index('step')
            if step_idx < len(row):
                try:
                    return int(float(row[step_idx]))
                except (ValueError, IndexError):
                    pass
        return 0

    def analyze_ppo_data(self) -> Dict[str, Any]:
        """分析PPO数据"""
        if not self.ppo_rows:
            return {}

        all_metrics = []
        steps = []

        for row in self.ppo_rows:
            metrics = self.parse_ppo_row(row)
            step = self.get_step_from_row(row)

            if metrics:
                all_metrics.append(metrics)
                steps.append(step)

        if not all_metrics:
            return {}

        # 计算统计信息
        stats = {}
        metric_names = ['pi_loss', 'vf_loss', 'entropy', 'approx_kl', 'clipfrac',
                       'pg_grad_norm', 'explained_var', 'lr']

        for metric in metric_names:
            values = [m.get(metric, 0.0) for m in all_metrics if metric in m]
            if values:
                stats[metric] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1] if values else 0.0
                }

        return {
            'stats': stats,
            'total_updates': len(all_metrics),
            'steps': steps,
            'latest_metrics': all_metrics[-1] if all_metrics else {},
            'first_step': min(steps) if steps else 0,
            'last_step': max(steps) if steps else 0
        }

    def display_summary(self) -> None:
        """显示PPO训练摘要"""
        analysis = self.analyze_ppo_data()

        if not analysis:
            print("❌ 没有可分析的PPO数据")
            return

        print("🚀 PPO训练数据解析结果")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📊 PPO更新次数: {analysis['total_updates']}")
        print(f"📈 训练步骤范围: {analysis['first_step']} - {analysis['last_step']}")
        print(f"⏰ 最新步骤: {analysis['last_step']}")

        print("\n📊 PPO指标统计:")
        print("-" * 50)

        stats = analysis['stats']
        for metric in ['pi_loss', 'vf_loss', 'entropy', 'approx_kl', 'clipfrac', 'pg_grad_norm', 'explained_var']:
            if metric in stats:
                s = stats[metric]
                print(f"{metric:15s}: 平均={s['mean']:8.6f}, 最新={s['latest']:8.6f}, 范围=[{s['min']:6.4f}, {s['max']:6.4f}]")
            else:
                print(f"{metric:15s}: 无数据")

        # 显示最新指标
        latest = analysis['latest_metrics']
        if latest:
            print(f"\n🎯 最新指标详情:")
            print("-" * 30)
            for key, value in latest.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")

        # 状态评估
        print(f"\n🎯 训练状态评估:")
        print("-" * 30)

        if 'entropy' in latest:
            entropy = latest['entropy']
            if entropy > 1.0:
                print("  🎲 探索充分 (熵值高)")
            elif entropy > 0.5:
                print("  ⚖️  探索适中")
            else:
                print("  🎯 策略收敛 (熵值低)")

        if 'pi_loss' in stats and stats['pi_loss']['count'] >= 3:
            recent_pi_loss = stats['pi_loss']['latest']
            if recent_pi_loss < -0.01:
                print("  ✅ 策略损失较低")
            elif recent_pi_loss > 0.01:
                print("  ⚠️  策略损失较高")
            else:
                print("  📊 策略损失正常")

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def export_clean_data(self, output_path: str) -> bool:
        """导出清理后的PPO数据"""
        try:
            analysis = self.analyze_ppo_data()
            if not analysis:
                return False

            # 创建清理后的数据
            clean_data = []
            for i, row in enumerate(self.ppo_rows):
                metrics = self.parse_ppo_row(row)
                step = self.get_step_from_row(row)

                # 获取时间戳信息
                timestamp = ""
                if 'datetime' in self.header:
                    dt_idx = self.header.index('datetime')
                    if dt_idx < len(row):
                        timestamp = row[dt_idx]

                clean_row = {
                    'step': step,
                    'datetime': timestamp,
                    'data_type': 'ppo_update',
                    **metrics
                }
                clean_data.append(clean_row)

            # 写入CSV
            if clean_data:
                fieldnames = ['step', 'datetime', 'data_type'] + list(clean_data[0].keys())[3:]
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(clean_data)

                print(f"📁 已导出清理数据: {output_path}")
                print(f"📊 导出 {len(clean_data)} 行PPO数据")
                return True

        except Exception as e:
            print(f"❌ 导出失败: {e}")

        return False


def main():
    parser = argparse.ArgumentParser(description='解析溢出的PPO CSV数据')
    parser.add_argument('csv_file', help='输入CSV文件路径')
    parser.add_argument('--export', '-e', help='导出清理后数据的路径')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式，只显示关键信息')

    args = parser.parse_args()

    if not Path(args.csv_file).exists():
        print(f"❌ 文件不存在: {args.csv_file}")
        return 1

    # 创建解析器
    parser = PPODataParser(args.csv_file)

    # 加载数据
    if not parser.load_data():
        print("❌ 无法加载PPO数据")
        return 1

    # 显示摘要
    if not args.quiet:
        parser.display_summary()

    # 导出数据
    if args.export:
        success = parser.export_clean_data(args.export)
        if not success:
            return 1

    return 0


if __name__ == "__main__":
    exit(main())