#!/usr/bin/env python3
"""
精确调试PPO梯度问题的脚本
"""

import torch
import numpy as np
from src.rl_components import ActorCritic, PPOTrainer, RolloutBuffer

def debug_ppo_components():
    """逐步调试PPO组件的梯度计算"""

    print("🔍 PPO梯度调试 - 逐步验证")
    print("=" * 60)

    device = "cpu"
    state_dim = 81
    action_dim = 2
    hidden_size = 128
    rollout_len = 8

    # 1. 创建Actor-Critic
    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        layer_N=2,
        gru_layers=2,
        use_orthogonal=True,
    ).to(device)

    # 确保训练模式
    ac.train()

    print(f"1. Actor-Critic训练模式: {ac.training}")

    # 检查参数是否需要梯度
    param_count = 0
    for name, param in ac.named_parameters():
        if param.requires_grad:
            param_count += 1
    print(f"2. 需要梯度的参数数量: {param_count}")

    # 2. 创建PPO trainer
    ppo = PPOTrainer(
        ac,
        lr=0.001,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=4,
        minibatch_size=8,  # 小batch用于测试
        max_grad_norm=0.5,
        device=device,
    )

    print(f"3. PPO trainer创建成功")

    # 3. 创建rollout buffer并添加经验
    buffer = RolloutBuffer(
        state_dim=state_dim,
        rollout_len=rollout_len,
        gamma=0.95,
        gae_lambda=0.95,
        device=device,
    )

    # 初始化hidden states
    hxs = torch.zeros(2, 1, hidden_size, device=device)

    print(f"4. 模拟rollout过程...")

    # 模拟rollout
    for step in range(rollout_len):
        # 创建随机状态
        state = torch.randn(state_dim, device=device)
        mask = torch.ones(1, device=device)

        # 前向传播获取动作和价值
        with torch.no_grad():
            action, logp, value, hxs = ac.act_value(state.unsqueeze(0), hxs, mask.unsqueeze(-1))

        # 添加到buffer
        buffer.add_step(
            s=state,
            a=action.squeeze(0),
            logp=logp.squeeze(0),
            v=value.squeeze(0),
            r=0.1 * step,  # 简单奖励
            mask=mask.squeeze(0)
        )

    print(f"5. Rollout完成，buffer包含 {buffer.ptr} 步经验")

    # 4. 计算GAE
    with torch.no_grad():
        # 最终状态的价值估计
        final_state = torch.randn(1, state_dim, device=device)
        final_mask = torch.ones(1, 1, device=device)
        z = ac.forward_mlp(final_state)
        z, _ = ac.forward_gru(z, hxs, final_mask)
        last_v = ac.critic(z).squeeze(-1)

    s_t, a_t, logp_t, v_t, ret_t, adv_t = buffer.compute_gae(last_v)
    masks = torch.tensor(buffer.masks, dtype=torch.float32, device=device)

    print(f"6. GAE计算完成")
    print(f"   状态tensor shape: {s_t.shape}, requires_grad: {s_t.requires_grad}")
    print(f"   动作tensor shape: {a_t.shape}, requires_grad: {a_t.requires_grad}")
    print(f"   优势tensor shape: {adv_t.shape}, requires_grad: {adv_t.requires_grad}")

    # 5. 手动执行一次PPO更新步骤
    print(f"7. 手动执行PPO更新...")

    # 设置模型为训练模式
    ac.train()

    # 获取batch数据
    bs = s_t[:8]  # 取前8个样本
    ba = a_t[:8].view(-1).to(torch.long)
    blogp = logp_t[:8]
    bret = ret_t[:8]
    badv = adv_t[:8]
    bm = masks[:8]

    print(f"   batch状态: shape={bs.shape}, requires_grad={bs.requires_grad}")

    # 创建新的hidden states
    with torch.no_grad():
        batch_hxs = torch.zeros(2, 8, hidden_size, device=device)

    print(f"   batch hidden states: requires_grad={batch_hxs.requires_grad}")

    # 前向传播
    print(f"8. 执行前向传播...")
    new_logp, entropy, v_pred, _ = ac.evaluate_actions(bs, batch_hxs, bm.unsqueeze(-1), ba)

    print(f"   new_logp: requires_grad={new_logp.requires_grad}")
    print(f"   entropy: requires_grad={entropy.requires_grad}")
    print(f"   v_pred: requires_grad={v_pred.requires_grad}")

    # 检查是否有任何输出有梯度
    if not any([new_logp.requires_grad, entropy.requires_grad, v_pred.requires_grad]):
        print("❌ 所有输出都没有梯度！")

        # 尝试直接测试模型前向传播
        print("9. 直接测试模型前向传播...")
        test_state = torch.randn(1, state_dim, device=device, requires_grad=True)
        test_hxs = torch.zeros(2, 1, hidden_size, device=device)
        test_mask = torch.ones(1, 1, device=device)
        test_action = torch.randint(0, action_dim, (1,), device=device)

        print(f"   test_state requires_grad: {test_state.requires_grad}")

        test_logp, test_entropy, test_v, _ = ac.evaluate_actions(test_state, test_hxs, test_mask, test_action)

        print(f"   直接测试结果:")
        print(f"     logp requires_grad: {test_logp.requires_grad}")
        print(f"     entropy requires_grad: {test_entropy.requires_grad}")
        print(f"     value requires_grad: {test_v.requires_grad}")

        if test_logp.requires_grad:
            print("✅ 直接测试成功！问题在于输入tensor")
        else:
            print("❌ 直接测试也失败！问题在于模型本身")

            # 更深层调试
            print("10. 深层模型调试...")
            ac.eval()
            print(f"    设置eval模式后，training: {ac.training}")

            ac.train()
            print(f"    设置train模式后，training: {ac.training}")

            # 检查特定层
            for name, module in ac.named_modules():
                if hasattr(module, 'training'):
                    print(f"    {name}: training={module.training}")
                    if name in ['gru', 'actor', 'critic']:
                        break
    else:
        print("✅ 梯度计算正常！")

        # 继续测试loss计算
        print(f"9. 测试loss计算...")

        # PPO loss
        ratio = torch.exp(new_logp - blogp)
        surr1 = ratio * badv
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * badv
        pi_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        v_clipped = v_t[:8] + (v_pred - v_t[:8]).clamp(-0.2, 0.2)
        vf_loss1 = (v_pred - bret).pow(2)
        vf_loss2 = (v_clipped - bret).pow(2)
        vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

        # Total loss
        loss = pi_loss + 0.5 * vf_loss - 0.01 * entropy

        print(f"   loss requires_grad: {loss.requires_grad}")

        if loss.requires_grad:
            print("10. 测试反向传播...")
            loss.backward()
            print("✅ 反向传播成功！")
        else:
            print("❌ loss没有梯度")


if __name__ == "__main__":
    debug_ppo_components()