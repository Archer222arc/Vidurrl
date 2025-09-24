#!/usr/bin/env python3
"""
修复损坏的PPO指标CSV文件

该工具可以修复由于CSV架构演化问题导致的"损坏"CSV文件，
恢复缺失的PPO训练指标列名。
"""

import csv
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any


def analyze_csv_structure(csv_path: str) -> Dict[str, Any]:
    """
    分析CSV文件结构

    Args:
        csv_path: CSV文件路径

    Returns:
        分析结果字典
    """
    print(f"🔍 分析CSV文件: {csv_path}")

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        print(f"📊 列数: {len(header)}")
        print(f"📏 数据行数: {len(rows)}")

        # 检查数据类型分布
        data_types = {}
        ppo_rows = []

        for i, row in enumerate(rows):
            if len(row) > len(header):
                # 找到超出头部长度的行
                try:
                    data_type_idx = header.index('data_type') if 'data_type' in header else None
                    if data_type_idx is not None and len(row) > data_type_idx:
                        data_type = row[data_type_idx]
                        data_types[data_type] = data_types.get(data_type, 0) + 1

                        if data_type == 'ppo_update':
                            ppo_rows.append((i + 1, row))  # +1 for 1-based line numbering

                except (ValueError, IndexError):
                    pass

        return {
            'header': header,
            'rows': rows,
            'data_types': data_types,
            'ppo_rows': ppo_rows,
            'has_overflow': any(len(row) > len(header) for row in rows)
        }

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None


def detect_missing_ppo_columns(analysis: Dict[str, Any]) -> List[str]:
    """
    检测缺失的PPO列

    Args:
        analysis: CSV分析结果

    Returns:
        缺失的PPO列名列表
    """
    header = analysis['header']
    ppo_rows = analysis['ppo_rows']

    # 已知的PPO指标列
    known_ppo_columns = [
        'pi_loss', 'vf_loss', 'entropy', 'approx_kl',
        'clipfrac', 'pg_grad_norm', 'explained_var', 'lr',
        'rollout_length', 'buffer_size'
    ]

    missing_columns = []
    for col in known_ppo_columns:
        if col not in header:
            missing_columns.append(col)

    # 检查是否有超出头部的数据（说明确实有缺失列）
    if ppo_rows and missing_columns:
        # 估算实际缺失的列数
        max_overflow = max(len(row) - len(header) for _, row in ppo_rows)
        print(f"🔍 检测到 {max_overflow} 个超出列，预期缺失列: {len(missing_columns)}")

    return missing_columns


def repair_csv_file(csv_path: str, output_path: str = None) -> bool:
    """
    修复CSV文件

    Args:
        csv_path: 输入CSV文件路径
        output_path: 输出文件路径（如果为None则覆盖原文件）

    Returns:
        修复是否成功
    """
    # 分析文件
    analysis = analyze_csv_structure(csv_path)
    if not analysis:
        return False

    # 检测缺失列
    missing_columns = detect_missing_ppo_columns(analysis)
    if not missing_columns:
        print("✅ 文件看起来没有问题，无需修复")
        return True

    print(f"🔧 检测到缺失的PPO列: {missing_columns}")

    # 创建新的头部
    original_header = analysis['header']
    new_header = original_header + missing_columns

    print(f"📊 原始列数: {len(original_header)}")
    print(f"📈 修复后列数: {len(new_header)}")

    # 准备输出路径
    if output_path is None:
        output_path = csv_path
        backup_path = csv_path + '.backup'
        # 创建备份
        try:
            Path(csv_path).rename(backup_path)
            print(f"💾 已创建备份: {backup_path}")
        except Exception as e:
            print(f"⚠️  创建备份失败: {e}")
            return False
    else:
        backup_path = None

    try:
        # 重写文件
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入新头部
            writer.writerow(new_header)

            # 写入数据行
            for row in analysis['rows']:
                # 确保每行都有足够的列
                padded_row = row + [''] * (len(new_header) - len(row))
                writer.writerow(padded_row)

        print(f"✅ 修复完成，输出到: {output_path}")

        # 验证修复结果
        if verify_repair(output_path, missing_columns):
            print("🎉 修复验证成功!")
            # 删除备份（如果创建了）
            if backup_path and Path(backup_path).exists():
                Path(backup_path).unlink()
                print("🗑️  已删除备份文件")
            return True
        else:
            print("❌ 修复验证失败")
            # 恢复备份（如果存在）
            if backup_path and Path(backup_path).exists():
                Path(backup_path).rename(output_path)
                print("🔄 已恢复备份文件")
            return False

    except Exception as e:
        print(f"❌ 修复过程中出错: {e}")
        # 恢复备份（如果存在）
        if backup_path and Path(backup_path).exists():
            Path(backup_path).rename(output_path)
            print("🔄 已恢复备份文件")
        return False


def verify_repair(csv_path: str, expected_columns: List[str]) -> bool:
    """
    验证修复结果

    Args:
        csv_path: 修复后的CSV文件路径
        expected_columns: 期望的列名

    Returns:
        验证是否成功
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

        missing = [col for col in expected_columns if col not in header]
        if missing:
            print(f"⚠️  验证失败，仍缺失列: {missing}")
            return False

        print("✅ 验证成功，所有预期列都存在")
        return True

    except Exception as e:
        print(f"❌ 验证过程中出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='修复损坏的PPO指标CSV文件')
    parser.add_argument('csv_file', help='输入CSV文件路径')
    parser.add_argument('--output', '-o', help='输出文件路径（默认覆盖原文件）')
    parser.add_argument('--analyze-only', action='store_true', help='仅分析，不修复')

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.csv_file).exists():
        print(f"❌ 文件不存在: {args.csv_file}")
        return 1

    if args.analyze_only:
        # 仅分析模式
        analysis = analyze_csv_structure(args.csv_file)
        if analysis:
            missing = detect_missing_ppo_columns(analysis)
            print(f"\n📊 分析结果:")
            print(f"  数据类型分布: {analysis['data_types']}")
            print(f"  是否有数据溢出: {'是' if analysis['has_overflow'] else '否'}")
            print(f"  缺失的PPO列: {missing if missing else '无'}")
            print(f"  PPO更新行数: {len(analysis['ppo_rows'])}")
        return 0

    # 修复模式
    print("🔧 开始修复CSV文件...")
    success = repair_csv_file(args.csv_file, args.output)

    if success:
        print("\n🎉 修复成功完成!")
        if args.output:
            print(f"📁 修复后的文件: {args.output}")
        else:
            print(f"📁 原文件已更新: {args.csv_file}")
        return 0
    else:
        print("\n💥 修复失败")
        return 1


if __name__ == "__main__":
    exit(main())