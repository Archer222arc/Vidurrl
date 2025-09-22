#!/usr/bin/env python3
"""
Test Actor-Critic with real training parameters.
"""

import torch
import numpy as np
from src.rl_components import ActorCritic

def test_with_real_params():
    """Test Actor-Critic with same parameters as real training."""

    print("🔍 Testing with real training parameters...")

    device = "cpu"
    state_dim = 81
    action_dim = 2
    hidden_size = 128
    layer_N = 2
    gru_layers = 2
    batch_size = 8

    # Initialize Actor-Critic exactly like in training
    ac = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        layer_N=layer_N,
        gru_layers=gru_layers,
        use_orthogonal=True,
    ).to(device)

    # Create tensors exactly like in training
    states = torch.randn(batch_size, state_dim, requires_grad=False, dtype=torch.float32)
    actions = torch.randint(0, action_dim, (batch_size,), dtype=torch.int64)
    hxs = torch.zeros(gru_layers, batch_size, hidden_size, dtype=torch.float32)
    masks = torch.ones(batch_size, 1, dtype=torch.float32)

    print(f"Input tensor info:")
    print(f"  states: shape={states.shape}, dtype={states.dtype}, requires_grad={states.requires_grad}")
    print(f"  actions: shape={actions.shape}, dtype={actions.dtype}")
    print(f"  hxs: shape={hxs.shape}, dtype={hxs.dtype}, requires_grad={hxs.requires_grad}")
    print(f"  masks: shape={masks.shape}, dtype={masks.dtype}, requires_grad={masks.requires_grad}")

    # Check Actor-Critic parameters
    print(f"\nActor-Critic parameters:")
    param_count = 0
    for name, param in ac.named_parameters():
        if param.requires_grad:
            param_count += 1
            print(f"  {name}: requires_grad={param.requires_grad}")
        if param_count > 3:  # Just show first few
            print("  ...")
            break

    # Forward pass
    print(f"\n🚀 Forward pass...")
    try:
        logp, entropy, v_pred, updated_hxs = ac.evaluate_actions(states, hxs, masks, actions)

        print(f"Output tensor info:")
        print(f"  logp: shape={logp.shape}, dtype={logp.dtype}, requires_grad={logp.requires_grad}")
        print(f"  entropy: shape={entropy.shape}, dtype={entropy.dtype}, requires_grad={entropy.requires_grad}")
        print(f"  v_pred: shape={v_pred.shape}, dtype={v_pred.dtype}, requires_grad={v_pred.requires_grad}")
        print(f"  updated_hxs: shape={updated_hxs.shape}, dtype={updated_hxs.dtype}, requires_grad={updated_hxs.requires_grad}")

        # Test backward pass
        print(f"\n⚡ Testing backward pass...")
        dummy_targets = torch.randn_like(v_pred)
        dummy_advantages = torch.randn_like(logp)

        pi_loss = -(logp * dummy_advantages).mean()
        vf_loss = ((v_pred - dummy_targets) ** 2).mean()
        loss = pi_loss + 0.5 * vf_loss - 0.01 * entropy

        print(f"Loss components:")
        print(f"  pi_loss: requires_grad={pi_loss.requires_grad}")
        print(f"  vf_loss: requires_grad={vf_loss.requires_grad}")
        print(f"  entropy: requires_grad={entropy.requires_grad}")
        print(f"  loss: requires_grad={loss.requires_grad}")

        loss.backward()
        print("✅ Backward pass succeeded!")

    except Exception as e:
        print(f"❌ Forward/backward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_real_params()