#!/usr/bin/env python3
"""
统一预训练管理器

整合所有预训练功能，提供统一的接口和配置管理。
支持：
1. 标准BC训练 (原pretrain_actor.py功能)
2. 增强独立预训练 (standalone_trainer.py功能)
3. 模型验证和兼容性检查
4. 灵活的配置管理
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .behavior_cloning_trainer import BehaviorCloningTrainer, DemoDataset, create_bc_trainer_from_config
from .model_validator import validate_pretrained_model


class UnifiedPretrainer:
    """统一预训练管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', './outputs/unified_pretrain'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 训练模式
        self.training_mode = config.get('training_mode', 'standard')  # standard | enhanced

        # TensorBoard
        self.use_tensorboard = config.get('use_tensorboard', False)
        self.writer = None
        if self.use_tensorboard:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")

        print(f"🔧 统一预训练管理器已初始化")
        print(f"   - 训练模式: {self.training_mode}")
        print(f"   - 输出目录: {self.output_dir}")

    def train_from_demo_files(
        self,
        demo_files: List[str],
        output_filename: str = "pretrained_model.pt"
    ) -> str:
        """从示教文件训练预训练模型"""

        if self.training_mode == 'standard':
            return self._train_standard_bc(demo_files, output_filename)
        elif self.training_mode == 'enhanced':
            return self._train_enhanced_bc(demo_files, output_filename)
        else:
            raise ValueError(f"不支持的训练模式: {self.training_mode}")

    def _train_standard_bc(self, demo_files: List[str], output_filename: str) -> str:
        """标准BC训练 - 兼容原始接口"""
        print(f"📚 [标准模式] 开始BC训练...")

        # 加载所有示教数据
        all_demo_data = []
        latest_stats = {}

        for demo_file in demo_files:
            with open(demo_file, 'rb') as f:
                data = pickle.load(f)
                all_demo_data.extend(data['demo_data'])
                if 'stats' in data:
                    latest_stats = data['stats']

        print(f"📂 加载示教数据完成: 共{len(all_demo_data)}个样本")

        # 更新配置
        if latest_stats:
            self.config.update(latest_stats)

        # 创建BC训练器
        trainer = create_bc_trainer_from_config(self.config)

        # 创建并训练归一化器 - 与PPO训练保持一致
        from ....utils.normalizers import RunningNormalizer
        normalizer = RunningNormalizer(eps=1e-6, clip=5.0)

        # 用所有状态数据训练归一化器
        for item in all_demo_data:
            state = np.array(item['state'], dtype=np.float32)
            normalizer.update(state)

        # 计算方差和标准差以显示统计信息
        var = normalizer.m2 / max(normalizer.count - 1, 1) if normalizer.m2 is not None else np.zeros_like(normalizer.mean)
        std = np.sqrt(np.maximum(var, normalizer.eps))
        print(f"🔧 归一化器已训练: 均值范围={normalizer.mean.min():.3f}~{normalizer.mean.max():.3f}, 标准差范围={std.min():.3f}~{std.max():.3f}")

        # 创建数据集
        dataset = DemoDataset(
            all_demo_data,
            normalizer=normalizer,
            augment_data=self.config.get('augment_data', False),
            noise_std=self.config.get('noise_std', 0.01)
        )

        # 训练
        history = trainer.train(
            dataset=dataset,
            epochs=self.config.get('epochs', 30),
            batch_size=self.config.get('batch_size', 256),
            validation_split=self.config.get('validation_split', 0.1)
        )

        # 保存模型
        output_path = self.output_dir / output_filename
        trainer.save_pretrained_model(str(output_path))

        # 保存训练历史
        if self.use_tensorboard and self.writer:
            for epoch, (train_loss, train_acc, val_loss, val_acc) in enumerate(zip(
                history['train_loss'], history['train_acc'],
                history['val_loss'], history['val_acc']
            )):
                self.writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
                self.writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)

        return str(output_path)

    def _train_enhanced_bc(self, demo_files: List[str], output_filename: str) -> str:
        """增强BC训练 - 使用standalone_trainer的功能"""
        print(f"🚀 [增强模式] 开始独立预训练...")

        from .standalone_trainer import StandalonePretrainer

        # 创建增强预训练器
        enhanced_config = self.config.copy()
        enhanced_config['output_dir'] = str(self.output_dir)

        pretrainer = StandalonePretrainer(enhanced_config)

        # 执行训练
        history = pretrainer.train(demo_files)

        # 复制最佳模型到统一输出位置
        best_model_path = self.output_dir / "best_model.pt"
        output_path = self.output_dir / output_filename

        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, output_path)
            print(f"📂 最佳模型已复制到: {output_path}")

        return str(output_path)

    def validate_model(self, model_path: str) -> bool:
        """验证预训练模型"""
        print(f"🔍 验证预训练模型: {model_path}")

        target_config = {
            'state_dim': self.config.get('state_dim'),
            'action_dim': self.config.get('action_dim'),
            'hidden_size': self.config.get('hidden_size'),
            'layer_N': self.config.get('layer_N'),
            'gru_layers': self.config.get('gru_layers'),
        }

        return validate_pretrained_model(model_path, target_config)

    def load_and_fine_tune(
        self,
        base_model_path: str,
        demo_files: List[str],
        output_filename: str = "fine_tuned_model.pt",
        fine_tune_epochs: int = 10
    ) -> str:
        """加载预训练模型并进行微调"""
        print(f"🔧 开始微调预训练模型: {base_model_path}")

        # 验证基础模型
        if not self.validate_model(base_model_path):
            raise ValueError("基础模型验证失败")

        # 加载所有示教数据
        all_demo_data = []
        for demo_file in demo_files:
            with open(demo_file, 'rb') as f:
                data = pickle.load(f)
                all_demo_data.extend(data['demo_data'])

        # 创建BC训练器并加载预训练模型
        trainer = create_bc_trainer_from_config(self.config)
        trainer.load_pretrained_model(base_model_path)

        # 创建数据集（启用数据增强）
        dataset = DemoDataset(
            all_demo_data,
            augment_data=True,
            noise_std=self.config.get('noise_std', 0.01)
        )

        # 微调训练
        print(f"🎯 开始微调训练: {fine_tune_epochs} epochs")
        history = trainer.train(
            dataset=dataset,
            epochs=fine_tune_epochs,
            batch_size=self.config.get('batch_size', 256),
            validation_split=self.config.get('validation_split', 0.1)
        )

        # 保存微调后的模型
        output_path = self.output_dir / output_filename
        trainer.save_pretrained_model(str(output_path))

        return str(output_path)

    def create_standard_config(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        epochs: int = 30
    ) -> Dict[str, Any]:
        """创建标准配置"""
        return {
            'training_mode': 'standard',
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_size': hidden_size,
            'layer_N': 2,
            'gru_layers': 2,
            'epochs': epochs,
            'batch_size': 256,
            'learning_rate': 1e-3,
            'validation_split': 0.1,
            'device': 'cpu',
            'augment_data': False,
        }

    def create_enhanced_config(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        epochs: int = 100
    ) -> Dict[str, Any]:
        """创建增强配置"""
        return {
            'training_mode': 'enhanced',
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_size': hidden_size,
            'layer_N': 3,
            'gru_layers': 2,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'batch_size': 512,
            'epochs': epochs,
            'validation_split': 0.2,
            'early_stopping_patience': 20,
            'save_interval': 10,
            'device': 'cpu',
            'augment_data': True,
            'noise_std': 0.01,
            'use_tensorboard': True,
        }

    def __del__(self):
        """清理资源"""
        if self.writer:
            self.writer.close()


def main():
    """统一预训练主函数"""
    parser = argparse.ArgumentParser(description="统一预训练管理器")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--demo-files", nargs='+', required=True, help="示教数据文件列表")
    parser.add_argument("--output", type=str, default="pretrained_model.pt", help="输出模型文件名")
    parser.add_argument("--mode", choices=['standard', 'enhanced'], default='standard', help="训练模式")
    parser.add_argument("--base-model", type=str, help="基础模型路径（用于微调）")
    parser.add_argument("--fine-tune-epochs", type=int, default=10, help="微调轮数")

    # 快速配置选项
    parser.add_argument("--state-dim", type=int, help="状态维度")
    parser.add_argument("--action-dim", type=int, help="动作维度")
    parser.add_argument("--hidden-size", type=int, help="隐藏层大小")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--device", type=str, default="cpu", help="训练设备")

    args = parser.parse_args()

    # 加载配置
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # 创建默认配置
        config = {}

    # 命令行参数覆盖
    if args.mode:
        config['training_mode'] = args.mode
    if args.state_dim:
        config['state_dim'] = args.state_dim
    if args.action_dim:
        config['action_dim'] = args.action_dim
    if args.hidden_size:
        config['hidden_size'] = args.hidden_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.device:
        config['device'] = args.device

    # 验证必要参数
    required_params = ['state_dim', 'action_dim']
    missing_params = [p for p in required_params if p not in config]
    if missing_params:
        print(f"❌ 缺少必要参数: {missing_params}")
        print("请通过配置文件或命令行参数提供")
        return

    # 创建统一预训练器
    pretrainer = UnifiedPretrainer(config)

    try:
        if args.base_model:
            # 微调模式
            output_path = pretrainer.load_and_fine_tune(
                base_model_path=args.base_model,
                demo_files=args.demo_files,
                output_filename=args.output,
                fine_tune_epochs=args.fine_tune_epochs
            )
            print(f"🎉 微调完成! 输出: {output_path}")
        else:
            # 标准训练模式
            output_path = pretrainer.train_from_demo_files(
                demo_files=args.demo_files,
                output_filename=args.output
            )
            print(f"🎉 训练完成! 输出: {output_path}")

        # 验证输出模型
        if pretrainer.validate_model(output_path):
            print("✅ 输出模型验证通过")
        else:
            print("❌ 输出模型验证失败")

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        raise


if __name__ == "__main__":
    main()