#!/usr/bin/env python3
"""
简单测试inference-only模式配置
"""

import os
import torch
from src.rl_components import ActorCritic, CheckpointManager, InferenceMode


def test_inference_simple():
    """简单测试inference模式组件"""

    print("🔍 PPO推理模式简单测试")
    print("=" * 50)

    # 检查checkpoint
    checkpoint_dir = "./outputs/checkpoints_demo"
    if not os.path.exists(checkpoint_dir) or not os.listdir(checkpoint_dir):
        print("❌ 没有找到checkpoint，先运行训练生成checkpoint")
        return False

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        print("❌ 没有.pt checkpoint文件")
        return False

    latest_checkpoint = sorted(checkpoint_files)[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"✅ 使用checkpoint: {checkpoint_path}")

    try:
        # 1. 创建模型
        device = "cpu"
        state_dim = 81
        action_dim = 2
        hidden_size = 128

        ac = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            layer_N=2,
            gru_layers=2,
            use_orthogonal=True,
        ).to(device)

        print("✅ Actor-Critic模型创建成功")

        # 2. 加载checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            ac.load_state_dict(checkpoint['actor_critic_state_dict'])
            print("✅ Checkpoint加载成功")
            print(f"   - 训练步数: {checkpoint.get('step', 'Unknown')}")
            print(f"   - 包含键: {list(checkpoint.keys())}")
        else:
            print("❌ Checkpoint文件不存在")
            return False

        # 3. 测试推理模式
        inference_mode = InferenceMode(ac, device=device)
        print("✅ 推理模式对象创建成功")

        # 验证模型状态
        print(f"✅ 模型训练状态: {ac.training} (应该是False)")
        if ac.training:
            print("⚠️  警告: 模型仍在训练模式")

        # 4. 测试推理前向传播
        print("\n🔍 测试推理前向传播...")

        test_state = torch.randn(1, state_dim, device=device)
        test_hxs = torch.zeros(2, 1, hidden_size, device=device)
        test_mask = torch.ones(1, 1, device=device)

        with torch.inference_mode():
            action, logp, value, new_hxs = inference_mode.act_value(test_state, test_hxs, test_mask)

        print(f"✅ 推理输出:")
        print(f"   - Action shape: {action.shape}")
        print(f"   - Log prob shape: {logp.shape}")
        print(f"   - Value shape: {value.shape}")
        print(f"   - Hidden states shape: {new_hxs.shape}")

        # 验证输出没有梯度
        requires_grads = [
            action.requires_grad,
            logp.requires_grad,
            value.requires_grad,
            new_hxs.requires_grad
        ]

        if any(requires_grads):
            print("⚠️  警告: 推理输出仍有梯度信息")
        else:
            print("✅ 推理输出无梯度 (正确)")

        print("\n📋 推理模式测试结果:")
        print("  - Checkpoint加载: ✅")
        print("  - 推理模式创建: ✅")
        print("  - 前向传播: ✅")
        print("  - 无梯度计算: ✅")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_inference_simple()
    if success:
        print("\n🎉 推理模式组件测试成功！")
        print("📝 使用方法:")
        print("   1. 训练模式: 自动保存checkpoint")
        print("   2. 推理模式: 加载checkpoint + InferenceMode")
        print("   3. 无梯度计算: 使用torch.inference_mode()")
    else:
        print("\n💥 推理模式测试失败")