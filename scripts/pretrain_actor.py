#!/usr/bin/env python3
"""
Actor预训练脚本 - 统一接口

使用统一预训练管理器进行Actor预训练，
兼容原始接口的同时提供更强大的功能。
"""

import argparse
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

from src.pretraining.unified_trainer import UnifiedPretrainer


def main():
    """兼容原始接口的主函数"""
    parser = argparse.ArgumentParser(description="Actor预训练 - 行为克隆")
    parser.add_argument("--demo", type=str, required=True, help="示教数据文件路径")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--hidden_size", type=int, default=128, help="隐藏层大小")
    parser.add_argument("--layer_N", type=int, default=2, help="MLP层数")
    parser.add_argument("--gru_layers", type=int, default=2, help="GRU层数")
    parser.add_argument("--output", type=str, default="./outputs/pretrained_actor.pt",
                       help="输出模型路径")
    parser.add_argument("--device", type=str, default="cpu", help="训练设备")
    parser.add_argument("--resume", type=str, help="从预训练模型开始 (微调模式)")

    args = parser.parse_args()

    # 从示教文件中推断配置
    import pickle
    with open(args.demo, 'rb') as f:
        data = pickle.load(f)

    stats = data.get('stats', {})
    state_dim = stats.get('state_dim', 64)  # 默认值
    action_dim = stats.get('action_dim', 4)  # 默认值

    # 创建配置
    config = {
        'training_mode': 'standard',
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_size': args.hidden_size,
        'layer_N': args.layer_N,
        'gru_layers': args.gru_layers,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'validation_split': 0.1,
        'device': args.device,
        'augment_data': False,
        'output_dir': str(Path(args.output).parent),
    }

    print(f"📄 配置信息:")
    print(f"   - 示教文件: {args.demo}")
    print(f"   - 状态维度: {state_dim}")
    print(f"   - 动作维度: {action_dim}")
    print(f"   - 训练轮数: {args.epochs}")
    print(f"   - 输出路径: {args.output}")

    # 创建统一预训练器
    pretrainer = UnifiedPretrainer(config)

    try:
        if args.resume:
            # 微调模式
            print(f"🔧 微调模式: 基于 {args.resume}")
            output_path = pretrainer.load_and_fine_tune(
                base_model_path=args.resume,
                demo_files=[args.demo],
                output_filename=Path(args.output).name,
                fine_tune_epochs=args.epochs
            )
        else:
            # 标准训练模式
            print(f"🤖 标准训练模式")
            output_path = pretrainer.train_from_demo_files(
                demo_files=[args.demo],
                output_filename=Path(args.output).name
            )

        print(f"🎉 预训练完成!")
        print(f"📂 输出模型: {output_path}")

        # 验证模型
        if pretrainer.validate_model(output_path):
            print("✅ 模型验证通过")
        else:
            print("❌ 模型验证失败")

    except Exception as e:
        print(f"❌ 预训练失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()