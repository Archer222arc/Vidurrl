#!/usr/bin/env python3
"""
Debug script to test gradient computation in isolation.
"""

import torch
import numpy as np
from src.rl_components import ActorCritic

def test_actor_critic_gradients():
    """Test if gradients flow properly through Actor-Critic network."""

    print("🔍 Testing Actor-Critic gradient computation...")

    # Create a simple test case
    device = "cpu"
    state_dim = 10
    action_dim = 3
    batch_size = 4

    # Initialize Actor-Critic
    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=32,
        layer_N=1,
        gru_layers=1,
        use_orthogonal=True,
    ).to(device)

    # Create test inputs
    states = torch.randn(batch_size, state_dim, requires_grad=False)  # Input states don't need gradients
    actions = torch.randint(0, action_dim, (batch_size,))
    hxs = torch.zeros(1, batch_size, 32)
    masks = torch.ones(batch_size, 1)

    print(f"Input states require_grad: {states.requires_grad}")
    print(f"Actor parameters require_grad: {next(ac.actor.parameters()).requires_grad}")

    # Forward pass through evaluate_actions
    logp, entropy, v_pred, _ = ac.evaluate_actions(states, hxs, masks, actions)

    print(f"logp requires_grad: {logp.requires_grad}")
    print(f"entropy requires_grad: {entropy.requires_grad}")
    print(f"v_pred requires_grad: {v_pred.requires_grad}")

    # Test loss computation
    dummy_targets = torch.randn_like(v_pred)
    dummy_advantages = torch.randn_like(logp)

    # Simple loss similar to PPO
    pi_loss = -(logp * dummy_advantages).mean()
    vf_loss = ((v_pred - dummy_targets) ** 2).mean()
    loss = pi_loss + 0.5 * vf_loss - 0.01 * entropy

    print(f"pi_loss requires_grad: {pi_loss.requires_grad}")
    print(f"vf_loss requires_grad: {vf_loss.requires_grad}")
    print(f"loss requires_grad: {loss.requires_grad}")

    # Test backward pass
    try:
        loss.backward()
        print("✅ Backward pass succeeded!")

        # Check if gradients were computed
        grad_count = 0
        for name, param in ac.named_parameters():
            if param.grad is not None:
                grad_count += 1

        print(f"✅ Gradients computed for {grad_count} parameters")

    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False

    return True

if __name__ == "__main__":
    success = test_actor_critic_gradients()
    if success:
        print("\n🎉 Actor-Critic gradient computation works correctly!")
    else:
        print("\n💥 Actor-Critic gradient computation has issues!")