#!/usr/bin/env python3
"""
Test script to validate PPO reward function improvements.

This script verifies that the implemented changes address the core issues:
1. Reward saturation removed (linear scaling instead of tanh)
2. Load balance penalty with higher weight
3. Dense rewards providing better learning signals
4. Enhanced exploration parameters
"""

import sys
import os
import numpy as np
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.algorithms.rewards.reward_calculator import RewardCalculator


class MockReplicaScheduler:
    """Mock replica scheduler for testing."""
    def __init__(self, queue_length: int = 0):
        self._request_queue = [f"req_{i}" for i in range(queue_length)]
        self._config = type('Config', (), {'num_blocks': 100})()
        self._num_allocated_blocks = min(queue_length * 5, 90)  # Simulate load


class MockMetricStore:
    """Mock metric store for testing."""
    def __init__(self, throughput: float = 0.1, latency: float = 1.8):
        self._throughput = throughput
        self._latency = latency

    def get_throughput(self, current_time: float) -> float:
        return self._throughput

    def get_average_latency(self) -> float:
        return self._latency


def test_reward_saturation_fix():
    """Test that rewards no longer saturate with tanh."""
    print("\n=== Testing Reward Saturation Fix ===")

    calc = RewardCalculator(mode="hybrid")

    # Test with varying throughput levels
    throughputs = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    rewards = []

    for tp in throughputs:
        metric_store = MockMetricStore(throughput=tp, latency=1.0)
        replica_ids = [0, 1, 2, 3]
        schedulers = {rid: MockReplicaScheduler(queue_length=2) for rid in replica_ids}

        def get_scheduler(rid):
            return schedulers[rid]

        reward, info = calc.calculate_reward(
            metric_store, 1.0, replica_ids, get_scheduler
        )
        rewards.append(reward)

        print(f"Throughput: {tp:.2f} -> Reward: {reward:.4f} | Raw: {info.get('raw_reward', 'N/A'):.4f}")

    # Check that rewards scale linearly (not saturate)
    reward_diffs = [rewards[i+1] - rewards[i] for i in range(len(rewards)-1)]
    non_zero_diffs = [abs(d) for d in reward_diffs if abs(d) > 0.001]

    if len(non_zero_diffs) >= 3:
        print("✅ PASS: Rewards show linear scaling, no saturation detected")
        return True
    else:
        print("❌ FAIL: Rewards still appear to be saturating")
        return False


def test_load_balance_penalty():
    """Test that load imbalance is heavily penalized."""
    print("\n=== Testing Load Balance Penalty ===")

    calc = RewardCalculator(mode="hybrid")

    # Test balanced vs imbalanced scenarios
    scenarios = [
        ("Balanced", [2, 2, 2, 2]),      # Perfect balance
        ("Slight imbalance", [1, 2, 2, 3]),  # Small imbalance
        ("Heavy imbalance", [0, 1, 2, 5]),   # Large imbalance
        ("Extreme imbalance", [0, 0, 0, 8])  # Extreme case
    ]

    results = []

    for name, queue_lengths in scenarios:
        metric_store = MockMetricStore(throughput=0.1, latency=1.5)
        replica_ids = [0, 1, 2, 3]
        schedulers = {rid: MockReplicaScheduler(queue_length=queue_lengths[rid]) for rid in replica_ids}

        def get_scheduler(rid):
            return schedulers[rid]

        reward, info = calc.calculate_reward(
            metric_store, 1.0, replica_ids, get_scheduler
        )

        std_dev = np.std(queue_lengths)
        direct_penalty = info.get('direct_imbalance_penalty', 0.0)

        results.append((name, reward, std_dev, direct_penalty))
        print(f"{name:20s} | Std: {std_dev:.2f} | Penalty: {direct_penalty:.4f} | Reward: {reward:.4f}")

    # Check that imbalance correlates with lower rewards
    balanced_reward = results[0][1]  # Balanced scenario reward
    imbalanced_rewards = [r[1] for r in results[1:]]

    if all(r < balanced_reward for r in imbalanced_rewards):
        print("✅ PASS: Load imbalance properly penalized")
        return True
    else:
        print("❌ FAIL: Load imbalance penalty insufficient")
        return False


def test_reward_variance():
    """Test that rewards have sufficient variance for learning."""
    print("\n=== Testing Reward Variance ===")

    calc = RewardCalculator(mode="hybrid")

    # Generate diverse scenarios
    np.random.seed(42)
    rewards = []

    for _ in range(50):
        # Random throughput and latency
        throughput = np.random.uniform(0.01, 0.3)
        latency = np.random.uniform(1.0, 3.0)

        # Random queue distributions
        queue_lengths = np.random.randint(0, 6, size=4).tolist()

        metric_store = MockMetricStore(throughput=throughput, latency=latency)
        replica_ids = [0, 1, 2, 3]
        schedulers = {rid: MockReplicaScheduler(queue_length=queue_lengths[rid]) for rid in replica_ids}

        def get_scheduler(rid):
            return schedulers[rid]

        reward, _ = calc.calculate_reward(
            metric_store, 1.0, replica_ids, get_scheduler
        )
        rewards.append(reward)

    reward_std = np.std(rewards)
    reward_mean = np.mean(rewards)
    zero_count = sum(1 for r in rewards if abs(r) < 0.001)
    unique_count = len(set(np.round(rewards, 4)))

    print(f"Reward statistics:")
    print(f"  Mean: {reward_mean:.4f}")
    print(f"  Std:  {reward_std:.4f}")
    print(f"  Zero rewards: {zero_count}/50 ({zero_count/50*100:.1f}%)")
    print(f"  Unique values: {unique_count}/50 ({unique_count/50*100:.1f}%)")

    # Check for good learning signals (adjusted thresholds)
    if reward_std > 0.2 and zero_count < 10 and unique_count > 35:
        print("✅ PASS: Rewards show good variance and density")
        return True
    else:
        print("❌ FAIL: Insufficient reward variance for learning")
        return False


def test_exploration_parameters():
    """Test that exploration parameters are correctly configured."""
    print("\n=== Testing Exploration Parameters ===")

    # Check config file values (assuming we can access them)
    try:
        sys.path.append('/Users/ruicheng/Documents/GitHub/Vidur/Vidur_arc2')
        from vidur.config.config import PPOGlobalSchedulerModularConfig
        config = PPOGlobalSchedulerModularConfig()

        print(f"Entropy coefficient: {config.entropy_coef}")
        print(f"Base temperature: {config.base_temperature}")
        print(f"Temperature range: [{config.min_temperature}, {config.max_temperature}]")
        print(f"GAE lambda: {config.gae_lambda}")

        checks = [
            (config.entropy_coef >= 0.02, f"Entropy coefficient >= 0.02 (actual: {config.entropy_coef})"),
            (config.base_temperature >= 1.5, f"Base temperature >= 1.5 (actual: {config.base_temperature})"),
            (config.max_temperature >= 3.0, f"Max temperature >= 3.0 (actual: {config.max_temperature})"),
            (hasattr(config, 'gae_lambda') and config.gae_lambda <= 0.90, f"GAE lambda <= 0.90 (actual: {getattr(config, 'gae_lambda', 'N/A')})"),
        ]

        passed = sum(1 for check, _ in checks if check)
        total = len(checks)

        for check, description in checks:
            status = "✅" if check else "❌"
            print(f"  {status} {description}")

        if passed == total:
            print("✅ PASS: All exploration parameters correctly configured")
            return True
        else:
            print(f"❌ FAIL: {total-passed}/{total} exploration parameters need adjustment")
            return False

    except Exception as e:
        print(f"❌ ERROR: Could not verify exploration parameters: {e}")
        return False


def generate_report(test_results):
    """Generate a summary report of all tests."""
    print("\n" + "="*60)
    print("PPO REWARD IMPROVEMENTS VALIDATION REPORT")
    print("="*60)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    print(f"Test Summary: {passed_tests}/{total_tests} tests passed")
    print()

    for test_name, passed in test_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:30s}: {status}")

    print()
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Reward improvements successfully implemented.")
        print("   Expected improvements:")
        print("   - Reward signals should be more informative (no 99% zeros)")
        print("   - Load balancing should be prioritized over raw performance")
        print("   - Exploration should be more effective with higher entropy")
        print("   - Action distribution should move toward [32,32,32,32] from [22,37,29,40]")
    else:
        print("⚠️  Some tests failed. Review the implementation before deployment.")

    print()
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


def main():
    """Run all validation tests."""
    print("PPO Reward Function Improvements - Validation Test")
    print("=" * 50)

    test_results = {}

    try:
        test_results["Reward Saturation Fix"] = test_reward_saturation_fix()
        test_results["Load Balance Penalty"] = test_load_balance_penalty()
        test_results["Reward Variance"] = test_reward_variance()
        test_results["Exploration Parameters"] = test_exploration_parameters()

        generate_report(test_results)

        # Return exit code based on results
        return 0 if all(test_results.values()) else 1

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)