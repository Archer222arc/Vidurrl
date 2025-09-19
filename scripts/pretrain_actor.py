#!/usr/bin/env python3
"""
Actor预训练脚本 - 行为克隆

使用收集的示教数据对PPO的Actor进行预训练，
为后续的PPO训练提供良好的初始策略。
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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

from src.rl_components.actor_critic import ActorCritic
from src.rl_components.normalizers import RunningNormalizer


class DemoDataset(Dataset):
    """示教数据集"""

    def __init__(self, demo_data: List[Dict[str, Any]], normalizer: RunningNormalizer = None):
        """
        初始化数据集

        Args:
            demo_data: 示教数据列表
            normalizer: 状态归一化器
        """
        self.states = []
        self.actions = []

        for item in demo_data:
            state = np.array(item['state'], dtype=np.float32)
            action = int(item['action'])

            # 应用归一化
            if normalizer:
                state = normalizer.normalize(state)

            self.states.append(state)
            self.actions.append(action)

        self.states = np.array(self.states)
        self.actions = np.array(self.actions)

        print(f"📊 数据集大小: {len(self.states)}")
        print(f"🎯 状态维度: {self.states.shape[1] if len(self.states) > 0 else 0}")
        print(f"📈 动作分布: {np.bincount(self.actions)}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.from_numpy(self.states[idx]), torch.tensor(self.actions[idx], dtype=torch.long)


class BehaviorCloningTrainer:
    """行为克隆训练器"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        hidden_size: int = 128,
        layer_N: int = 2,
        gru_layers: int = 2,
        enable_decoupled: bool = True,
        feature_projection_dim: int = None,
        device: str = "cpu"
    ):
        """
        初始化训练器

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_size: 隐藏层大小
            layer_N: MLP层数
            gru_layers: GRU层数
            enable_decoupled: 是否使用解耦架构
            feature_projection_dim: 特征投影维度
            device: 设备
        """
        self.device = device
        self.action_dim = action_dim

        # 创建Actor-Critic网络（只训练Actor部分）
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            layer_N=layer_N,
            gru_layers=gru_layers,
            enable_decoupled=enable_decoupled,
            feature_projection_dim=feature_projection_dim,
        ).to(device)

        # 只优化Actor参数
        actor_params = []
        if hasattr(self.actor_critic, 'actor_branch') and self.actor_critic.enable_decoupled:
            # 解耦架构：Actor参数在actor_branch和actor_head中
            actor_params.extend(self.actor_critic.actor_branch.parameters())
            actor_params.extend(self.actor_critic.actor_head.parameters())
        elif hasattr(self.actor_critic, 'actor'):
            # 耦合架构：Actor参数在actor中
            actor_params.extend(self.actor_critic.actor.parameters())
        else:
            raise ValueError("无法找到Actor参数，ActorCritic架构不匹配")

        self.optimizer = torch.optim.Adam(actor_params, lr=1e-3)

        print(f"🤖 Actor-Critic网络已创建: state_dim={state_dim}, action_dim={action_dim}")

    def train(
        self,
        dataset: DemoDataset,
        epochs: int = 10,
        batch_size: int = 256,
        validation_split: float = 0.1
    ) -> Dict[str, List[float]]:
        """
        训练Actor网络

        Args:
            dataset: 示教数据集
            epochs: 训练轮数
            batch_size: 批大小
            validation_split: 验证集比例

        Returns:
            训练历史
        """
        # 划分训练集和验证集
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        print(f"🚀 开始行为克隆训练: epochs={epochs}, batch_size={batch_size}")
        print(f"📊 训练集: {train_size}, 验证集: {val_size}")

        for epoch in range(epochs):
            # 训练阶段
            self.actor_critic.train()
            train_loss, train_acc = self._train_epoch(train_loader)

            # 验证阶段
            self.actor_critic.eval()
            val_loss, val_acc = self._validate_epoch(val_loader)

            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        print("✅ 行为克隆训练完成!")
        return history

    def _train_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        """训练一个epoch"""
        total_loss = 0.0
        correct = 0
        total = 0

        for states, actions in dataloader:
            states = states.to(self.device)
            actions = actions.to(self.device)

            # 前向传播（只使用Actor）
            batch_size = states.shape[0]
            hidden_state = torch.zeros(
                self.actor_critic.gru_layers, batch_size, self.actor_critic.hidden_size,
                device=self.device
            )
            mask = torch.ones(batch_size, 1, device=self.device)

            # 通过网络获取logits
            z = self.actor_critic.forward_mlp(states)
            z, _ = self.actor_critic.forward_gru(z, hidden_state, mask)

            # 根据架构类型获取logits
            if hasattr(self.actor_critic, 'actor_branch') and self.actor_critic.enable_decoupled:
                z_actor = self.actor_critic.actor_branch(z)
                logits = self.actor_critic.actor_head(z_actor)
            else:
                logits = self.actor_critic.actor(z)

            # 计算损失
            loss = F.cross_entropy(logits, actions)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == actions).sum().item()
            total += actions.size(0)

        return total_loss / len(dataloader), correct / total

    def _validate_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        """验证一个epoch"""
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for states, actions in dataloader:
                states = states.to(self.device)
                actions = actions.to(self.device)

                # 前向传播
                batch_size = states.shape[0]
                hidden_state = torch.zeros(
                    self.actor_critic.gru_layers, batch_size, self.actor_critic.hidden_size,
                    device=self.device
                )
                mask = torch.ones(batch_size, 1, device=self.device)

                z = self.actor_critic.forward_mlp(states)
                z, _ = self.actor_critic.forward_gru(z, hidden_state, mask)

                # 根据架构类型获取logits
                if hasattr(self.actor_critic, 'actor_branch') and self.actor_critic.enable_decoupled:
                    z_actor = self.actor_critic.actor_branch(z)
                    logits = self.actor_critic.actor_head(z_actor)
                else:
                    logits = self.actor_critic.actor(z)

                # 计算损失
                loss = F.cross_entropy(logits, actions)

                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == actions).sum().item()
                total += actions.size(0)

        return total_loss / len(dataloader), correct / total

    def save_pretrained_model(self, output_path: str) -> None:
        """保存预训练的模型"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'model_config': {
                'state_dim': self.actor_critic.state_dim,
                'action_dim': self.actor_critic.action_dim,
                'hidden_size': self.actor_critic.hidden_size,
                'layer_N': self.actor_critic.layer_N,
                'gru_layers': self.actor_critic.gru_layers,
                'enable_decoupled': getattr(self.actor_critic, 'enable_decoupled', False),
                'feature_projection_dim': getattr(self.actor_critic, 'feature_projection_dim', None),
            }
        }, output_path)

        print(f"💾 预训练模型已保存: {output_path}")


def main():
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

    args = parser.parse_args()

    # 加载示教数据
    print(f"📂 加载示教数据: {args.demo}")
    with open(args.demo, 'rb') as f:
        data = pickle.load(f)

    demo_data = data['demo_data']
    stats = data['stats']

    print(f"📊 数据统计: {stats}")

    # 创建数据集（不使用归一化，因为我们还没有训练数据）
    dataset = DemoDataset(demo_data)

    # 创建训练器
    trainer = BehaviorCloningTrainer(
        state_dim=stats['state_dim'],
        action_dim=len(stats['action_distribution']),
        hidden_size=args.hidden_size,
        layer_N=args.layer_N,
        gru_layers=args.gru_layers,
        device=args.device
    )

    # 训练
    history = trainer.train(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # 保存模型
    trainer.save_pretrained_model(args.output)

    print("🎉 预训练完成!")


if __name__ == "__main__":
    main()
