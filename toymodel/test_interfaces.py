#!/usr/bin/env python3
"""
Test script to verify all interfaces work correctly.
"""

import sys
import os
import torch
import numpy as np

# Add toymodel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from toymodel.src.entities import Request, Replica
from toymodel.src.rl_components import (
    SimpleActorCritic,
    SimplePPOTrainer,
    SimpleRolloutBuffer,
    QueueStateBuilder,
    LatencyRewardCalculator
)
from toymodel.schedulers.ppo_scheduler import PPOScheduler


def test_state_builder():
    """Test state builder with new interface."""
    print("Testing State Builder...")
    
    # Create test replicas
    replicas = [
        Replica(replica_id=0, service_rates={0: 10.0, 1: 5.0}, queue=[]),
        Replica(replica_id=1, service_rates={0: 5.0, 1: 10.0}, queue=[])
    ]
    
    # Add some requests to queues
    for i in range(3):
        replicas[0].queue.append(Request(request_id=i, request_type=0, arrival_time=0.0))
        replicas[1].queue.append(Request(request_id=i+10, request_type=1, arrival_time=0.0))
    
    # Create state builder
    state_builder = QueueStateBuilder(num_replicas=2, n_requests=3, normalize=False)
    
    # Test state building
    request = Request(request_id=100, request_type=0, arrival_time=1.0)
    state = state_builder.build_state(request, replicas)
    
    print(f"State shape: {state.shape}")
    print(f"State: {state}")
    print(f"Expected state dim: {state_builder.get_state_dim()}")
    print(f"Expected action dim: {state_builder.get_action_dim()}")
    
    # Verify state structure: [q0_len, q1_len, q0_types(3), q1_types(3), current_type]
    expected_dim = 2 + 2 * 3 + 1  # 9
    assert state.shape[0] == expected_dim, f"Expected state dim {expected_dim}, got {state.shape[0]}"
    
    print("✓ State Builder test passed\n")


def test_reward_calculator():
    """Test reward calculator with latency-only interface."""
    print("Testing Reward Calculator...")
    
    # Create test replicas
    replicas = [
        Replica(replica_id=0, service_rates={0: 10.0, 1: 5.0}, queue=[]),
        Replica(replica_id=1, service_rates={0: 5.0, 1: 10.0}, queue=[])
    ]
    
    # Create reward calculator
    reward_calc = LatencyRewardCalculator(latency_weight=1.0)
    
    # Test reward calculation
    request = Request(request_id=1, request_type=0, arrival_time=0.0)
    latency = 0.1  # 100ms latency
    
    reward = reward_calc.calculate_reward(request, replicas, 0, latency)
    
    print(f"Reward for latency {latency}: {reward}")
    assert isinstance(reward, float), "Reward should be a float"
    
    print("✓ Reward Calculator test passed\n")


def test_actor_critic():
    """Test actor-critic network."""
    print("Testing Actor-Critic Network...")
    
    # Create network
    state_dim = 9  # [q0_len, q1_len, q0_types(3), q1_types(3), current_type]
    action_dim = 2
    hidden_dim = 64
    
    policy = SimpleActorCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    
    # Test forward pass
    state = torch.randn(1, state_dim)
    action, log_prob, value = policy.get_action_and_value(state)
    
    print(f"Action: {action}")
    print(f"Log prob: {log_prob}")
    print(f"Value: {value}")
    
    assert action.shape == (1,), f"Action shape should be (1,), got {action.shape}"
    assert log_prob.shape == (1,), f"Log prob shape should be (1,), got {log_prob.shape}"
    assert value.shape == (1,), f"Value shape should be (1,), got {value.shape}"
    
    print("✓ Actor-Critic test passed\n")


def test_ppo_scheduler():
    """Test PPO scheduler."""
    print("Testing PPO Scheduler...")
    
    # Create scheduler
    scheduler = PPOScheduler(num_replicas=2, n_requests=3)
    
    # Create test replicas
    replicas = [
        Replica(replica_id=0, service_rates={0: 10.0, 1: 5.0}, queue=[]),
        Replica(replica_id=1, service_rates={0: 5.0, 1: 10.0}, queue=[])
    ]
    
    # Add some requests to queues
    for i in range(2):
        replicas[0].queue.append(Request(request_id=i, request_type=0, arrival_time=0.0))
        replicas[1].queue.append(Request(request_id=i+10, request_type=1, arrival_time=0.0))
    
    # Test scheduling
    request = Request(request_id=100, request_type=0, arrival_time=1.0)
    action = scheduler.schedule(request, replicas)
    
    print(f"Scheduled to replica: {action}")
    assert action in [0, 1], f"Action should be 0 or 1, got {action}"
    
    # Test action probabilities
    probs = scheduler.get_action_probabilities(request, replicas)
    print(f"Action probabilities: {probs}")
    assert len(probs) == 2, f"Should have 2 action probabilities, got {len(probs)}"
    assert abs(sum(probs) - 1.0) < 1e-6, f"Probabilities should sum to 1, got {sum(probs)}"
    
    print("✓ PPO Scheduler test passed\n")


def test_rollout_buffer():
    """Test rollout buffer."""
    print("Testing Rollout Buffer...")
    
    # Create buffer
    buffer = SimpleRolloutBuffer(state_dim=9, buffer_size=10)
    
    # Add some experiences
    for i in range(5):
        state = torch.randn(9)
        action = torch.tensor(i % 2)
        reward = float(i)
        value = torch.tensor(0.5)
        log_prob = torch.tensor(-0.5)
        done = i == 4
        
        buffer.add(state, action, reward, value, log_prob, done)
    
    print(f"Buffer size: {buffer.size}")
    assert buffer.size == 5, f"Expected buffer size 5, got {buffer.size}"
    
    # Test GAE computation
    next_value = torch.tensor(0.5)
    states, actions, log_probs, values, returns, advantages = buffer.compute_gae(next_value)
    
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Returns shape: {returns.shape}")
    print(f"Advantages shape: {advantages.shape}")
    
    assert states.shape[0] == 5, f"Expected 5 states, got {states.shape[0]}"
    assert actions.shape[0] == 5, f"Expected 5 actions, got {actions.shape[0]}"
    
    print("✓ Rollout Buffer test passed\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing PPO Interfaces")
    print("=" * 60)
    
    try:
        test_state_builder()
        test_reward_calculator()
        test_actor_critic()
        test_ppo_scheduler()
        test_rollout_buffer()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

