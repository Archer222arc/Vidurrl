#!/usr/bin/env python3
"""
独立预训练脚本 - 增强版行为克隆

支持长步骤、多策略、多阶段的预训练，生成高质量的预训练模型供PPO warmstart使用。

主要特性：
- 支持多轮数据收集和增量训练
- 可配置的训练策略组合
- 模型检查点保存和恢复
- 训练进度监控和早停
- 标准化的模型输出格式
"""

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


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

from src.core.models.actor_critic import ActorCritic
from src.core.utils.normalizers import RunningNormalizer


class EnhancedDemoDataset(Dataset):
    """增强的示教数据集，支持多种数据增强策略"""

    def __init__(
        self,
        demo_data: List[Dict[str, Any]],
        normalizer: Optional[RunningNormalizer] = None,
        augment_data: bool = True,
        noise_std: float = 0.01
    ):
        self.demo_data = demo_data
        self.normalizer = normalizer
        self.augment_data = augment_data
        self.noise_std = noise_std

        # 预处理数据
        self.states = []
        self.actions = []
        self.weights = []  # 样本权重，用于平衡不同策略的数据

        self._preprocess_data()

    def _preprocess_data(self):
        """预处理和统计数据"""
        strategy_counts = {}

        for item in self.demo_data:
            state = np.array(item['state'], dtype=np.float32)
            action = item['action']
            strategy = item.get('strategy', 'unknown')

            # 归一化状态
            if self.normalizer:
                state = self.normalizer.normalize(state)

            self.states.append(state)
            self.actions.append(action)

            # 统计策略分布
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # 计算样本权重（平衡不同策略）
        total_samples = len(self.demo_data)
        num_strategies = len(strategy_counts)

        for item in self.demo_data:
            strategy = item.get('strategy', 'unknown')
            strategy_weight = total_samples / (num_strategies * strategy_counts[strategy])
            self.weights.append(strategy_weight)

        print(f"📊 数据预处理完成:")
        print(f"   - 总样本数: {total_samples}")
        print(f"   - 策略分布: {strategy_counts}")
        print(f"   - 数据增强: {'启用' if self.augment_data else '禁用'}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx].copy()
        action = self.actions[idx]
        weight = self.weights[idx]

        # 数据增强：添加少量噪声
        if self.augment_data and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, state.shape)
            state = state + noise

        return torch.tensor(state, dtype=torch.float32), action, weight


class StandalonePretrainer:
    """独立预训练器 - 支持长期训练和模型管理"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        layer_N: int = 3,
        gru_layers: int = 2,
        learning_rate: float = 1e-4,
        device: str = "cpu",
        output_dir: str = "./outputs/standalone_pretrain"
    ):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 模型配置
        self.model_config = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_size': hidden_size,
            'layer_N': layer_N,
            'gru_layers': gru_layers,
        }

        # 创建Actor-Critic网络
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            layer_N=layer_N,
            gru_layers=gru_layers,
        ).to(self.device)

        # 设置优化器（只训练Actor部分）- 直接访问，缺失属性会自然报错
        actor_params = []
        if self.actor_critic.enable_decoupled:
            # 解耦架构：Actor参数在actor_branch和actor_head中
            actor_params.extend(self.actor_critic.actor_branch.parameters())
            actor_params.extend(self.actor_critic.actor_head.parameters())
        else:
            # 耦合架构：Actor参数在actor中
            actor_params.extend(self.actor_critic.actor.parameters())

        self.optimizer = torch.optim.AdamW(actor_params, lr=learning_rate, weight_decay=1e-5)

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=10, verbose=True
        )

        # 训练状态
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0

        # TensorBoard记录
        self.writer = SummaryWriter(self.output_dir / "tensorboard")

        print(f"🤖 独立预训练器已创建:")
        print(f"   - 模型配置: {self.model_config}")
        print(f"   - 设备: {self.device}")
        print(f"   - 输出目录: {self.output_dir}")

    def collect_extended_demo_data(
        self,
        strategies: List[str],
        steps_per_strategy: int,
        num_replicas: int = 4,
        qps: float = 3.0,
        include_variants: bool = True
    ) -> str:
        """收集扩展的示教数据"""
        print(f"📊 开始收集扩展示教数据...")
        print(f"   - 策略: {strategies}")
        print(f"   - 每策略步数: {steps_per_strategy}")
        print(f"   - 副本数: {num_replicas}")
        print(f"   - QPS: {qps}")

        demo_file = self.output_dir / f"extended_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        # TODO: 这里应该调用数据收集模块
        # 暂时模拟数据收集过程
        demo_data = []
        for strategy in strategies:
            for step in range(steps_per_strategy):
                # 模拟示教数据项
                demo_item = {
                    'state': np.random.randn(64).tolist(),  # 示例状态维度
                    'action': np.random.randint(0, 4),       # 示例动作
                    'strategy': strategy,
                    'step': step,
                    'timestamp': time.time()
                }
                demo_data.append(demo_item)

        # 保存数据
        data_package = {
            'demo_data': demo_data,
            'stats': {
                'total_samples': len(demo_data),
                'strategies': strategies,
                'steps_per_strategy': steps_per_strategy,
                'state_dim': 64,  # 示例
                'action_dim': 4,  # 示例
            },
            'metadata': {
                'collection_time': datetime.now().isoformat(),
                'num_replicas': num_replicas,
                'qps': qps,
                'include_variants': include_variants,
            }
        }

        with open(demo_file, 'wb') as f:
            pickle.dump(data_package, f)

        print(f"✅ 示教数据收集完成: {demo_file}")
        return str(demo_file)

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.actor_critic.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (states, actions, weights) in enumerate(dataloader):
            states = states.to(self.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

            # 前向传播
            with torch.no_grad():
                # 初始化隐藏状态
                batch_size = states.shape[0]
                hxs = torch.zeros(batch_size, self.actor_critic.gru_layers * self.actor_critic.hidden_size,
                                device=self.device)
                mask = torch.ones(batch_size, 1, device=self.device)

            # 获取Actor输出
            action_logits, _, _, _ = self.actor_critic.act_value(states, hxs, mask)

            # 计算加权交叉熵损失
            loss = F.cross_entropy(action_logits, actions, reduction='none')
            weighted_loss = (loss * weights).mean()

            # 反向传播
            self.optimizer.zero_grad()
            weighted_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            total_loss += weighted_loss.item()
            pred = action_logits.argmax(dim=1)
            correct += (pred == actions).sum().item()
            total += actions.size(0)

            # 记录批次指标
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Train/Batch_Loss', weighted_loss.item(),
                                     self.epoch * len(dataloader) + batch_idx)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """验证模型"""
        self.actor_critic.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for states, actions, weights in dataloader:
                states = states.to(self.device)
                actions = torch.tensor(actions, dtype=torch.long).to(self.device)
                weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

                # 初始化隐藏状态
                batch_size = states.shape[0]
                hxs = torch.zeros(batch_size, self.actor_critic.gru_layers * self.actor_critic.hidden_size,
                                device=self.device)
                mask = torch.ones(batch_size, 1, device=self.device)

                # 获取Actor输出
                action_logits, _, _, _ = self.actor_critic.act_value(states, hxs, mask)

                # 计算损失
                loss = F.cross_entropy(action_logits, actions, reduction='none')
                weighted_loss = (loss * weights).mean()

                total_loss += weighted_loss.item()
                pred = action_logits.argmax(dim=1)
                correct += (pred == actions).sum().item()
                total += actions.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def train(
        self,
        demo_files: List[str],
        epochs: int = 100,
        batch_size: int = 512,
        validation_split: float = 0.2,
        early_stopping_patience: int = 20,
        save_interval: int = 10
    ) -> Dict[str, List[float]]:
        """执行完整的训练流程"""
        print(f"🚀 开始独立预训练...")
        print(f"   - 训练epochs: {epochs}")
        print(f"   - 批大小: {batch_size}")
        print(f"   - 验证集比例: {validation_split}")
        print(f"   - 早停patience: {early_stopping_patience}")

        # 加载所有示教数据
        all_demo_data = []
        for demo_file in demo_files:
            with open(demo_file, 'rb') as f:
                data = pickle.load(f)
                all_demo_data.extend(data['demo_data'])

        print(f"📂 加载示教数据完成: 共{len(all_demo_data)}个样本")

        # 数据集分割
        split_idx = int(len(all_demo_data) * (1 - validation_split))
        train_data = all_demo_data[:split_idx]
        val_data = all_demo_data[split_idx:]

        # 创建数据集和数据加载器
        train_dataset = EnhancedDemoDataset(train_data, augment_data=True)
        val_dataset = EnhancedDemoDataset(val_data, augment_data=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        print(f"📊 数据集统计:")
        print(f"   - 训练集: {len(train_data)} 样本")
        print(f"   - 验证集: {len(val_data)} 样本")

        # 训练循环
        for epoch in range(epochs):
            self.epoch = epoch
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(current_lr)

            # TensorBoard记录
            self.writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            # 早停检查
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                # 保存最佳模型
                self.save_model("best_model.pt", include_training_state=True)
            else:
                self.patience_counter += 1

            # 定期保存
            if (epoch + 1) % save_interval == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pt", include_training_state=True)

            # 输出进度
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

            # 早停
            if self.patience_counter >= early_stopping_patience:
                print(f"🛑 早停触发 (patience={early_stopping_patience})")
                break

        # 保存最终模型
        self.save_model("final_model.pt", include_training_state=False)

        # 保存训练历史
        history_file = self.output_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        self.writer.close()
        print(f"🎉 独立预训练完成!")
        print(f"📊 最佳验证损失: {self.best_loss:.4f}")
        print(f"📂 输出目录: {self.output_dir}")

        return history

    def save_model(self, filename: str, include_training_state: bool = False):
        """保存模型"""
        save_path = self.output_dir / filename

        save_dict = {
            'model_state_dict': self.actor_critic.state_dict(),
            'model_config': self.model_config,
            'training_metadata': {
                'epoch': self.epoch,
                'best_loss': self.best_loss,
                'save_time': datetime.now().isoformat(),
                'model_type': 'standalone_pretrained',
                'architecture': 'actor_critic',
            }
        }

        if include_training_state:
            save_dict.update({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'patience_counter': self.patience_counter,
            })

        torch.save(save_dict, save_path)
        print(f"💾 模型已保存: {save_path}")

    def load_model(self, model_path: str, load_training_state: bool = False):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # 加载模型权重
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])

        # 验证模型配置
        saved_config = checkpoint['model_config']
        if saved_config != self.model_config:
            print(f"⚠️ 模型配置不匹配:")
            print(f"   当前: {self.model_config}")
            print(f"   保存: {saved_config}")

        # 加载训练状态
        if load_training_state and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint.get('epoch', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.patience_counter = checkpoint.get('patience_counter', 0)

        metadata = checkpoint.get('training_metadata', {})
        print(f"📂 模型已加载: {model_path}")
        print(f"   - 训练epoch: {metadata.get('epoch', 'unknown')}")
        print(f"   - 最佳损失: {metadata.get('best_loss', 'unknown')}")
        print(f"   - 保存时间: {metadata.get('save_time', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description="独立预训练脚本 - 增强版行为克隆")
    parser.add_argument("--config", type=str, help="配置文件路径 (JSON格式)")
    parser.add_argument("--demo-files", nargs='+', help="示教数据文件路径列表")
    parser.add_argument("--collect-demo", action='store_true', help="是否收集新的示教数据")
    parser.add_argument("--strategies", nargs='+', default=["round_robin", "lor", "random", "shortest_queue"],
                       help="示教策略列表")
    parser.add_argument("--steps-per-strategy", type=int, default=2000, help="每策略收集步数")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=512, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--hidden-size", type=int, default=256, help="隐藏层大小")
    parser.add_argument("--layer-N", type=int, default=3, help="MLP层数")
    parser.add_argument("--gru-layers", type=int, default=2, help="GRU层数")
    parser.add_argument("--output-dir", type=str, default="./outputs/standalone_pretrain", help="输出目录")
    parser.add_argument("--device", type=str, default="cpu", help="训练设备")
    parser.add_argument("--resume", type=str, help="从检查点恢复训练")

    args = parser.parse_args()

    # 加载配置文件
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
        # 配置文件中的参数覆盖命令行参数
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # 创建预训练器
    pretrainer = StandalonePretrainer(
        state_dim=64,  # 这里应该从实际配置获取
        action_dim=4,  # 这里应该从实际配置获取
        hidden_size=args.hidden_size,
        layer_N=args.layer_N,
        gru_layers=args.gru_layers,
        learning_rate=args.lr,
        device=args.device,
        output_dir=args.output_dir
    )

    # 恢复训练
    if args.resume:
        pretrainer.load_model(args.resume, load_training_state=True)
        print(f"🔄 从检查点恢复训练: {args.resume}")

    # 收集示教数据
    demo_files = args.demo_files or []
    if args.collect_demo:
        demo_file = pretrainer.collect_extended_demo_data(
            strategies=args.strategies,
            steps_per_strategy=args.steps_per_strategy
        )
        demo_files.append(demo_file)

    if not demo_files:
        print("❌ 错误: 没有指定示教数据文件，请使用 --demo-files 或 --collect-demo")
        return

    # 执行训练
    history = pretrainer.train(
        demo_files=demo_files,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("🎉 独立预训练完成!")


if __name__ == "__main__":
    main()