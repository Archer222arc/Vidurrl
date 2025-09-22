#!/usr/bin/env python3
"""
Standalone预训练demo数据收集脚本

从standalone_pretrain.json配置自动读取参数并收集demo数据。
确保与增强StateBuilder兼容，生成210维状态数据。
"""

import json
import sys
from pathlib import Path


def _bootstrap_repo_path() -> None:
    """Ensure imports resolve to the current repo checkout."""
    repo_root = Path(__file__).resolve().parent.parent
    conflict_root = repo_root.parent / "Vidur"

    def _same_path(a: str, b: Path) -> bool:
        try:
            return Path(a).resolve() == b.resolve()
        except (OSError, RuntimeError):
            return False

    sys.path[:] = [p for p in sys.path if not _same_path(p, conflict_root)]

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_path()

from scripts.collect_demo_mixed import MixedDemoCollector


def collect_standalone_demo(config_path: str = "configs/standalone_pretrain.json"):
    """
    根据standalone_pretrain.json配置收集demo数据

    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)

    demo_config = config["demo_collection"]
    output_dir = config["output_dir"]

    print("🎯 Standalone预训练Demo数据收集")
    print(f"📄 配置文件: {config_path}")
    print(f"📊 策略: {demo_config['strategies']}")
    print(f"📊 每策略步数: {demo_config['steps_per_strategy']}")
    print(f"📊 副本数: {demo_config['num_replicas']}")
    print(f"📊 QPS: {demo_config['qps']}")
    print(f"📂 输出目录: {output_dir}")
    print("=" * 60)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 创建收集器
    collector = MixedDemoCollector(policies=demo_config["strategies"])

    # 设置输出路径
    output_path = Path(output_dir) / "standalone_demo_data.pkl"

    # 收集数据
    collector.collect_mixed_demonstrations(
        steps_per_policy=demo_config["steps_per_strategy"],
        num_replicas=demo_config["num_replicas"],
        qps=demo_config["qps"],
        output_path=str(output_path)
    )

    print(f"\n🎉 Demo数据收集完成!")
    print(f"📁 输出文件: {output_path}")
    print(f"📊 状态维度: 210 (与enhanced StateBuilder兼容)")
    print("")
    print("📋 下一步:")
    print(f"   1. 运行预训练: python scripts/standalone_pretrain.py --demo {output_path}")
    print("   2. 或使用在warmstart训练中")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone预训练Demo数据收集")
    parser.add_argument("--config", type=str, default="configs/standalone_pretrain.json",
                       help="配置文件路径")

    args = parser.parse_args()

    try:
        collect_standalone_demo(args.config)
    except Exception as e:
        print(f"❌ 收集失败: {e}")
        sys.exit(1)