#!/usr/bin/env python3
"""
分块训练器命令行接口
提供CLI入口调用模块化的分块训练功能
"""

import argparse
import sys
from pathlib import Path

import importlib
def get_chunk_trainer():
    """延迟导入chunk_trainer模块"""
    chunk_trainer_module = importlib.import_module('src.core.algorithms.training.chunk_trainer')
    return chunk_trainer_module.run_chunk_training


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='分块训练器 - 模块化分块训练功能',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 基本参数
    parser.add_argument('--output-dir', required=True,
                       help='输出目录')
    parser.add_argument('--total-requests', type=int, required=True,
                       help='总训练请求数')
    parser.add_argument('--chunk-size', type=int, required=True,
                       help='每块的请求数')

    # 配置参数
    parser.add_argument('--config-file', default='configs/ppo_warmstart.json',
                       help='配置文件路径')
    parser.add_argument('--num-replicas', type=int, default=4,
                       help='副本数量')
    parser.add_argument('--qps', type=float, default=3.5,
                       help='QPS速率')

    # 训练模式参数
    parser.add_argument('--skip-warmstart', choices=['true', 'false'], default='false',
                       help='是否跳过warmstart')
    parser.add_argument('--external-pretrain', default='',
                       help='外部预训练模型路径')
    parser.add_argument('--pretrained-actor-path', default='',
                       help='预训练actor模型路径')

    args = parser.parse_args()

    # 构建配置字典
    config = {
        'output_dir': args.output_dir,
        'total_requests': args.total_requests,
        'chunk_size': args.chunk_size,
        'config_file': args.config_file,
        'num_replicas': args.num_replicas,
        'qps': args.qps,
        'skip_warmstart': args.skip_warmstart == 'true',
        'external_pretrain': args.external_pretrain,
        'pretrained_actor_path': args.pretrained_actor_path,
        'repo_root': str(Path(__file__).parent.parent.parent.parent.parent)
    }

    # 执行分块训练
    try:
        run_chunk_training = get_chunk_trainer()
        success = run_chunk_training(config)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 分块训练失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()