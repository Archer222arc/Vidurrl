#!/usr/bin/env python3
"""
独立预训练模块 - 核心训练逻辑

按照项目规范，将复杂的预训练逻辑模块化到src/中。
脚本只需调用此模块的接口，保持简洁。
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

# 确保可以导入项目模块
repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.core.models.actor_critic import ActorCritic
from src.core.utils.normalizers import RunningNormalizer


class EnhancedDemoDataset(Dataset):
    """增强的示教数据集"""

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

        self.states = []
        self.actions = []
        self.weights = []

        self._preprocess_data()

    def _preprocess_data(self):
        """预处理数据"""
        strategy_counts = {}

        for item in self.demo_data:
            state = np.array(item['state'], dtype=np.float32)
            action = item['action']
            strategy = item.get('strategy', 'unknown')

            if self.normalizer:
                state = self.normalizer.normalize(state)

            self.states.append(state)
            self.actions.append(action)
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # 计算样本权重
        total_samples = len(self.demo_data)
        num_strategies = len(strategy_counts)

        for item in self.demo_data:
            strategy = item.get('strategy', 'unknown')
            strategy_weight = total_samples / (num_strategies * strategy_counts[strategy])
            self.weights.append(strategy_weight)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx].copy()
        action = self.actions[idx]
        weight = self.weights[idx]

        if self.augment_data and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, state.shape)
            state = state + noise

        return torch.tensor(state, dtype=torch.float32), action, weight


class StandalonePretrainer:
    """独立预训练器 - 核心训练逻辑"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 模型配置
        self.model_config = {
            'state_dim': config['state_dim'],
            'action_dim': config['action_dim'],
            'hidden_size': config['hidden_size'],
            'layer_N': config['layer_N'],
            'gru_layers': config['gru_layers'],
        }

        # 创建模型
        self.actor_critic = ActorCritic(**self.model_config).to(self.device)

        # 设置优化器 - 直接访问，缺失属性会自然报错
        actor_params = []
        if self.actor_critic.enable_decoupled:
            # 解耦架构
            actor_params.extend(self.actor_critic.actor_branch.parameters())
            actor_params.extend(self.actor_critic.actor_head.parameters())
        else:
            # 耦合架构
            actor_params.extend(self.actor_critic.actor.parameters())

        self.optimizer = torch.optim.AdamW(
            actor_params,
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=10, verbose=True
        )

        # 训练状态
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0

        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / "tensorboard")

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

            # 初始化隐藏状态 - 正确的GRU格式: (num_layers, batch_size, hidden_size)
            batch_size = states.shape[0]
            hxs = torch.zeros(self.actor_critic.gru_layers, batch_size, self.actor_critic.hidden_size,
                            device=self.device)
            mask = torch.ones(batch_size, 1, device=self.device)

            # 前向传播 - 获取raw logits (保留梯度)
            action_logits, _ = self.actor_critic.forward_actor_logits(states, hxs, mask)

            # 计算损失
            loss = F.cross_entropy(action_logits, actions, reduction='none')
            weighted_loss = (loss * weights).mean()

            # 反向传播
            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计
            total_loss += weighted_loss.item()
            pred = action_logits.argmax(dim=1)
            correct += (pred == actions).sum().item()
            total += actions.size(0)

            if batch_idx % 100 == 0:
                self.writer.add_scalar('Train/Batch_Loss', weighted_loss.item(),
                                     self.epoch * len(dataloader) + batch_idx)

        return total_loss / len(dataloader), correct / total if total > 0 else 0.0

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

                batch_size = states.shape[0]
                hxs = torch.zeros(self.actor_critic.gru_layers, batch_size, self.actor_critic.hidden_size,
                                device=self.device)
                mask = torch.ones(batch_size, 1, device=self.device)

                action_logits, _ = self.actor_critic.forward_actor_logits(states, hxs, mask)

                loss = F.cross_entropy(action_logits, actions, reduction='none')
                weighted_loss = (loss * weights).mean()

                total_loss += weighted_loss.item()
                pred = action_logits.argmax(dim=1)
                correct += (pred == actions).sum().item()
                total += actions.size(0)

        return total_loss / len(dataloader), correct / total if total > 0 else 0.0

    def train(self, demo_files: List[str]) -> Dict[str, List[float]]:
        """执行训练"""
        config = self.config

        # 加载数据
        all_demo_data = []
        for demo_file in demo_files:
            with open(demo_file, 'rb') as f:
                data = pickle.load(f)
                all_demo_data.extend(data['demo_data'])

        # 数据集分割
        validation_split = config.get('validation_split', 0.2)
        split_idx = int(len(all_demo_data) * (1 - validation_split))
        train_data = all_demo_data[:split_idx]
        val_data = all_demo_data[split_idx:]

        # 创建数据集
        train_dataset = EnhancedDemoDataset(train_data, augment_data=True)
        val_dataset = EnhancedDemoDataset(val_data, augment_data=False)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

        # 训练循环
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rate': []}
        epochs = config['epochs']
        early_stopping_patience = config.get('early_stopping_patience', 20)

        for epoch in range(epochs):
            self.epoch = epoch
            start_time = time.time()

            # 训练和验证
            train_loss, train_acc = self.train_epoch(train_loader)
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
                self.save_model("best_model.pt", include_training_state=True)
            else:
                self.patience_counter += 1

            # 定期保存
            if (epoch + 1) % config.get('save_interval', 10) == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pt", include_training_state=True)

            # 输出进度
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

            # 早停
            if self.patience_counter >= early_stopping_patience:
                print(f"早停触发 (patience={early_stopping_patience})")
                break

        # 保存最终模型
        self.save_model("final_model.pt", include_training_state=False)

        # 保存训练历史
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

        self.writer.close()
        print(f"独立预训练完成! 最佳验证损失: {self.best_loss:.4f}")

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
        print(f"模型已保存: {save_path}")

    def load_model(self, model_path: str, load_training_state: bool = False):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])

        if load_training_state and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint.get('epoch', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.patience_counter = checkpoint.get('patience_counter', 0)

        print(f"模型已加载: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="独立预训练模块")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--demo-files", nargs='+', help="示教数据文件路径列表")

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)

    # 创建预训练器
    pretrainer = StandalonePretrainer(config)

    # 获取示教数据文件
    demo_files = args.demo_files or []

    # TODO: 这里应该集成示教数据收集模块
    # 暂时需要外部提供示教数据文件

    if not demo_files:
        print("❌ 错误: 没有指定示教数据文件，请使用 --demo-files")
        return

    # 执行训练
    history = pretrainer.train(demo_files)

    print("🎉 独立预训练完成!")


if __name__ == "__main__":
    main()