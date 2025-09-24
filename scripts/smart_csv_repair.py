#!/usr/bin/env python3
"""
智能CSV修复工具

分析实际的PPO数据行，推断出正确的列名和数据结构，
然后重建完整的CSV文件。
"""

import csv
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple


def extract_ppo_data_sample(csv_path: str) -> Tuple[List[str], List[List[str]]]:
    """
    提取PPO数据样本

    Args:
        csv_path: CSV文件路径

    Returns:
        (头部列表, PPO数据行列表)
    """
    print(f"🔍 提取PPO数据样本...")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

        # 查找PPO更新行
        ppo_rows = []
        for line_num, row in enumerate(reader, start=2):
            # 在行中查找'ppo_update'文本
            if 'ppo_update' in row:
                ppo_rows.append(row)
                if len(ppo_rows) >= 3:  # 取前3个样本就够了
                    break

    print(f"📊 找到 {len(ppo_rows)} 个PPO数据样本")
    return header, ppo_rows


def infer_missing_columns(header: List[str], ppo_sample: List[str]) -> List[str]:
    """
    从PPO数据样本推断缺失的列名

    Args:
        header: 现有头部
        ppo_sample: PPO数据样本行

    Returns:
        推断的缺失列名
    """
    # PPO指标在MetricsExporter中的标准顺序
    standard_ppo_fields = [
        'pi_loss', 'vf_loss', 'entropy', 'approx_kl',
        'clipfrac', 'pg_grad_norm', 'explained_var', 'lr',
        'rollout_length', 'buffer_size'
    ]

    missing_fields = []
    for field in standard_ppo_fields:
        if field not in header:
            missing_fields.append(field)

    extra_columns_needed = len(ppo_sample) - len(header)
    print(f"📏 PPO行长度: {len(ppo_sample)}, 头部长度: {len(header)}")
    print(f"📈 需要额外列数: {extra_columns_needed}")
    print(f"🎯 标准缺失字段: {missing_fields}")

    # 如果缺失字段数不够，添加通用字段
    if len(missing_fields) < extra_columns_needed:
        for i in range(len(missing_fields), extra_columns_needed):
            missing_fields.append(f"unknown_field_{i+1}")

    return missing_fields[:extra_columns_needed]


def analyze_ppo_values(header: List[str], ppo_rows: List[List[str]]) -> Dict[str, List[float]]:
    """
    分析PPO值的分布，帮助确认字段映射

    Args:
        header: 完整头部
        ppo_rows: PPO数据行

    Returns:
        字段名到值列表的映射
    """
    field_values = {}

    # 找到可能包含PPO指标的列
    potential_ppo_start = len(header) - 10  # 假设PPO字段在末尾

    ppo_field_names = [
        'pi_loss', 'vf_loss', 'entropy', 'approx_kl',
        'clipfrac', 'pg_grad_norm', 'explained_var', 'lr',
        'rollout_length', 'buffer_size'
    ]

    for i, field_name in enumerate(ppo_field_names):
        col_idx = potential_ppo_start + i
        if col_idx < len(header):
            values = []
            for row in ppo_rows:
                if col_idx < len(row):
                    try:
                        val = float(row[col_idx]) if row[col_idx] else 0.0
                        values.append(val)
                    except ValueError:
                        values.append(0.0)
            field_values[field_name] = values

    return field_values


def rebuild_csv_with_proper_headers(csv_path: str, output_path: str) -> bool:
    """
    重建CSV文件，使用正确的头部

    Args:
        csv_path: 输入文件路径
        output_path: 输出文件路径

    Returns:
        是否成功
    """
    try:
        # 提取样本数据
        header, ppo_samples = extract_ppo_data_sample(csv_path)
        if not ppo_samples:
            print("❌ 没有找到PPO数据样本")
            return False

        # 推断缺失列
        missing_columns = infer_missing_columns(header, ppo_samples[0])
        new_header = header + missing_columns

        print(f"📊 原始头部列数: {len(header)}")
        print(f"📈 新头部列数: {len(new_header)}")
        print(f"🆕 新增列: {missing_columns}")

        # 重建文件
        with open(csv_path, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            original_header = next(reader)  # 跳过原始头部

            with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)

                # 写入新头部
                writer.writerow(new_header)

                # 写入数据行，确保每行都有正确的列数
                for row in reader:
                    # 截断或填充到正确长度
                    if len(row) > len(new_header):
                        row = row[:len(new_header)]
                    elif len(row) < len(new_header):
                        row = row + [''] * (len(new_header) - len(row))

                    writer.writerow(row)

        print(f"✅ 重建完成: {output_path}")

        # 验证结果
        return verify_rebuilt_csv(output_path, missing_columns)

    except Exception as e:
        print(f"❌ 重建失败: {e}")
        return False


def verify_rebuilt_csv(csv_path: str, expected_new_columns: List[str]) -> bool:
    """
    验证重建的CSV文件

    Args:
        csv_path: CSV文件路径
        expected_new_columns: 期望的新列

    Returns:
        验证是否成功
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

            # 检查几行数据的一致性
            inconsistent_rows = 0
            for i, row in enumerate(reader):
                if len(row) != len(header):
                    inconsistent_rows += 1
                if i >= 100:  # 只检查前100行
                    break

        if inconsistent_rows > 0:
            print(f"⚠️  发现 {inconsistent_rows} 行长度不一致")
            return False

        # 检查新列是否存在
        missing = [col for col in expected_new_columns if col not in header]
        if missing:
            print(f"⚠️  缺失期望列: {missing}")
            return False

        print("✅ CSV文件验证通过")
        return True

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


def show_ppo_preview(csv_path: str) -> None:
    """
    显示修复后PPO数据的预览

    Args:
        csv_path: CSV文件路径
    """
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        ppo_data = df[df['data_type'] == 'ppo_update']

        if len(ppo_data) == 0:
            print("⚠️  没有找到PPO更新数据")
            return

        print(f"\n📊 PPO数据预览 (共 {len(ppo_data)} 行):")
        print("=" * 60)

        # 显示关键PPO指标
        ppo_columns = ['step', 'pi_loss', 'vf_loss', 'entropy', 'approx_kl', 'lr']
        available_columns = [col for col in ppo_columns if col in ppo_data.columns]

        if available_columns:
            preview = ppo_data[available_columns].head(5)
            print(preview.to_string(index=False))
        else:
            print("⚠️  没有找到标准PPO列")

        # 显示统计信息
        if 'entropy' in ppo_data.columns:
            entropy_values = ppo_data['entropy'].dropna()
            if len(entropy_values) > 0:
                print(f"\n🎲 熵统计: 平均={entropy_values.mean():.4f}, 范围=[{entropy_values.min():.4f}, {entropy_values.max():.4f}]")

    except ImportError:
        print("⚠️  需要pandas来显示预览")
    except Exception as e:
        print(f"⚠️  预览失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='智能修复损坏的PPO CSV文件')
    parser.add_argument('input_csv', help='输入CSV文件')
    parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    parser.add_argument('--preview', action='store_true', help='显示修复后的PPO数据预览')

    args = parser.parse_args()

    if not Path(args.input_csv).exists():
        print(f"❌ 输入文件不存在: {args.input_csv}")
        return 1

    print("🔧 开始智能修复CSV文件...")
    success = rebuild_csv_with_proper_headers(args.input_csv, args.output)

    if success:
        print("\n🎉 修复成功!")

        if args.preview:
            show_ppo_preview(args.output)

        return 0
    else:
        print("\n💥 修复失败")
        return 1


if __name__ == "__main__":
    exit(main())