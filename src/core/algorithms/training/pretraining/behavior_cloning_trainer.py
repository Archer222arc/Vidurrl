#!/usr/bin/env python3
"""
行为克隆训练器 - 统一的预训练核心模块

整合原始pretrain_actor.py的功能，提供统一的BC训练接口。
"""

import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ....models.actor_critic import ActorCritic
from ....utils.normalizers import RunningNormalizer


class DemoDataset(Dataset):
    """示教数据集 - 兼容原始格式"""

    def __init__(
        self,
        demo_data: List[Dict[str, Any]],
        normalizer: Optional[RunningNormalizer] = None,
        augment_data: bool = False,
        noise_std: float = 0.01
    ):
        self.demo_data = demo_data
        self.normalizer = normalizer
        self.augment_data = augment_data
        self.noise_std = noise_std

        # 预处理数据
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

            # 归一化状态
            if self.normalizer:
                state = self.normalizer.normalize(state)

            self.states.append(state)
            self.actions.append(action)
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # 计算样本权重（平衡不同策略）
        total_samples = len(self.demo_data)
        num_strategies = len(strategy_counts)

        for item in self.demo_data:
            strategy = item.get('strategy', 'unknown')
            if num_strategies > 1:
                strategy_weight = total_samples / (num_strategies * strategy_counts[strategy])
            else:
                strategy_weight = 1.0
            self.weights.append(strategy_weight)

        print(f"📊 数据预处理完成: {total_samples} 样本, 策略分布: {strategy_counts}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx].copy()
        action = self.actions[idx]
        weight = self.weights[idx]

        # 数据增强
        if self.augment_data and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, state.shape)
            state = state + noise

        return torch.tensor(state, dtype=torch.float32), action, weight


class BehaviorCloningTrainer:
    """行为克隆训练器 - 统一接口"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        layer_N: int = 2,
        gru_layers: int = 2,
        enable_decoupled: bool = False,
        feature_projection_dim: Optional[int] = None,
        learning_rate: float = 1e-3,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim

        # 模型配置
        self.model_config = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_size': hidden_size,
            'layer_N': layer_N,
            'gru_layers': gru_layers,
            'enable_decoupled': enable_decoupled,
            'feature_projection_dim': feature_projection_dim,
        }

        # 创建Actor-Critic网络
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            layer_N=layer_N,
            gru_layers=gru_layers,
            enable_decoupled=enable_decoupled,
            feature_projection_dim=feature_projection_dim,
        ).to(self.device)

        # 设置优化器 - 只训练Actor参数
        actor_params = []
        if self.actor_critic.enable_decoupled:
            # 解耦架构 - 直接访问，缺失属性会自然报错
            actor_params.extend(self.actor_critic.actor_branch.parameters())
            actor_params.extend(self.actor_critic.actor_head.parameters())
        else:
            # 耦合架构 - 直接访问，缺失属性会自然报错
            actor_params.extend(self.actor_critic.actor.parameters())

        self.optimizer = torch.optim.Adam(actor_params, lr=learning_rate)

        print(f"🤖 BC训练器已创建: {self.model_config}")
        print(f"   - Actor参数数量: {sum(p.numel() for p in actor_params)}")

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.actor_critic.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for states, actions, weights in dataloader:
            states = states.to(self.device)

            # 安全的tensor转换并清理无效action
            if isinstance(actions, torch.Tensor):
                actions = actions.long().to(self.device)
            else:
                actions = torch.tensor(actions, dtype=torch.long).to(self.device)

            # 清理无效的action indices
            invalid_action_mask = (actions < 0) | (actions >= self.actor_critic.action_dim)
            if invalid_action_mask.any():
                invalid_count = invalid_action_mask.sum().item()
                print(f"WARNING: Found {invalid_count} invalid actions in training data, fixing...")
                actions = torch.clamp(actions, 0, self.actor_critic.action_dim - 1)

            if isinstance(weights, torch.Tensor):
                weights = weights.float().to(self.device)
            else:
                weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

            # 初始化隐藏状态 - 正确的GRU格式: (num_layers, batch_size, hidden_size)
            batch_size = states.shape[0]
            hxs = torch.zeros(self.actor_critic.gru_layers, batch_size, self.actor_critic.hidden_size,
                            device=self.device)
            mask = torch.ones(batch_size, 1, device=self.device)

            # 前向传播 - 获取raw logits (保留梯度)
            action_logits, _ = self.actor_critic.forward_actor_logits(states, hxs, mask)

            # 计算加权交叉熵损失
            log_probs = F.log_softmax(action_logits, dim=-1)
            loss = F.nll_loss(log_probs, actions.long(), reduction='none')
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

                # 安全的tensor转换
                if isinstance(actions, torch.Tensor):
                    actions = actions.long().to(self.device)
                else:
                    actions = torch.tensor(actions, dtype=torch.long).to(self.device)

                if isinstance(weights, torch.Tensor):
                    weights = weights.float().to(self.device)
                else:
                    weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

                batch_size = states.shape[0]
                hxs = torch.zeros(self.actor_critic.gru_layers, batch_size, self.actor_critic.hidden_size,
                                device=self.device)
                mask = torch.ones(batch_size, 1, device=self.device)

                action_logits, _ = self.actor_critic.forward_actor_logits(states, hxs, mask)

                # 确保 action_logits 是浮点类型
                action_logits = action_logits.float()

                log_probs = F.log_softmax(action_logits, dim=1)
                loss = F.nll_loss(log_probs, actions.long(), reduction='none')
                weighted_loss = (loss * weights).mean()

                total_loss += weighted_loss.item()
                pred = action_logits.argmax(dim=1)
                correct += (pred == actions).sum().item()
                total += actions.size(0)

        return total_loss / len(dataloader), correct / total if total > 0 else 0.0

    def train(
        self,
        dataset: DemoDataset,
        epochs: int = 10,
        batch_size: int = 256,
        validation_split: float = 0.1
    ) -> Dict[str, List[float]]:
        """执行训练"""
        # 数据集分割
        dataset_size = len(dataset)
        val_size = int(dataset_size * validation_split) if validation_split > 0 else 0
        train_size = dataset_size - val_size

        if val_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
        else:
            train_dataset = dataset
            val_dataset = None

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

        # 训练历史
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        print(f"🚀 开始BC训练: {epochs} epochs, 批大小: {batch_size}")
        print(f"   - 训练集: {train_size} 样本")
        if val_size > 0:
            print(f"   - 验证集: {val_size} 样本")

        for epoch in range(epochs):
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # 验证
            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            else:
                val_loss, val_acc = train_loss, train_acc
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

            # 输出进度
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
                  f"Time: {epoch_time:.1f}s")

        print("✅ BC训练完成")
        return history

    def save_pretrained_model(self, output_path: str) -> None:
        """保存预训练模型 - 兼容原始格式"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用统一的保存格式
        save_dict = {
            'model_state_dict': self.actor_critic.state_dict(),
            'model_config': self.model_config,
            'training_metadata': {
                'save_time': datetime.now().isoformat(),
                'model_type': 'behavior_cloning',
                'architecture': 'actor_critic',
            }
        }

        torch.save(save_dict, output_path)
        print(f"💾 预训练模型已保存: {output_path}")

    def load_pretrained_model(self, model_path: str) -> None:
        """加载预训练模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # 兼容不同的保存格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 加载权重
        missing_keys, unexpected_keys = self.actor_critic.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"⚠️ 缺少的键: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ 意外的键: {unexpected_keys}")

        print(f"📂 预训练模型已加载: {model_path}")


def create_bc_trainer_from_config(config: Dict[str, Any]) -> BehaviorCloningTrainer:
    """从配置创建BC训练器"""
    return BehaviorCloningTrainer(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_size=config.get('hidden_size', 128),
        layer_N=config.get('layer_N', 2),
        gru_layers=config.get('gru_layers', 2),
        enable_decoupled=config.get('enable_decoupled', False),
        feature_projection_dim=config.get('feature_projection_dim', None),
        learning_rate=config.get('learning_rate', 1e-3),
        device=config.get('device', 'cpu')
    )


def train_bc_from_demo_file(
    demo_file: str,
    output_path: str,
    config: Dict[str, Any]
) -> BehaviorCloningTrainer:
    """从示教文件训练BC - 便捷函数"""
    # 加载示教数据
    with open(demo_file, 'rb') as f:
        data = pickle.load(f)

    demo_data = data['demo_data']
    stats = data.get('stats', {})

    print(f"📂 加载示教数据: {demo_file}")
    print(f"📊 数据统计: {stats}")

    # 更新配置
    if 'state_dim' in stats:
        config['state_dim'] = stats['state_dim']
    if 'action_dim' in stats:
        config['action_dim'] = stats['action_dim']

    # 创建数据集和训练器
    dataset = DemoDataset(demo_data, augment_data=config.get('augment_data', False))
    trainer = create_bc_trainer_from_config(config)

    # 训练
    history = trainer.train(
        dataset=dataset,
        epochs=config.get('epochs', 10),
        batch_size=config.get('batch_size', 256),
        validation_split=config.get('validation_split', 0.1)
    )

    # 保存模型
    trainer.save_pretrained_model(output_path)

    return trainer