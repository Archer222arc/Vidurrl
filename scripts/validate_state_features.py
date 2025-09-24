#!/usr/bin/env python3
"""
State Feature Validation Script

Validates the effectiveness and discriminative power of enhanced state features
to ensure the agent can distinguish between different scheduling scenarios.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.models.state_builder import StateBuilder
from core.models.actor_critic import ActorCritic


def create_mock_request(arrived_at: float, priority: str = "normal") -> object:
    """Create a mock request object for testing."""
    class MockRequest:
        def __init__(self, arrived_at: float, priority: str):
            self.arrived_at = arrived_at
            self.priority = priority
            self.num_prefill_tokens = 100
            self.num_processed_tokens = 50
            self.num_decode_tokens = 50
            self.completed = False

    return MockRequest(arrived_at, priority)


def create_mock_replica_scheduler(queue_requests: List, allocation_ratio: float = 0.5) -> object:
    """Create a mock replica scheduler for testing."""
    class MockConfig:
        def __init__(self):
            self.num_blocks = 100
            self.block_size = 16
            self.batch_size_cap = 32

    class MockReplicaScheduler:
        def __init__(self, queue_requests: List, allocation_ratio: float):
            self._request_queue = queue_requests
            self._num_allocated_blocks = int(100 * allocation_ratio)
            self._num_running_batches = len(queue_requests) // 4
            self._preempted_requests = []
            self._allocation_map = {}
            self._config = MockConfig()
            self._num_stages = 2

    return MockReplicaScheduler(queue_requests, allocation_ratio)


def generate_test_scenarios() -> Dict[str, Dict]:
    """
    Generate different scheduling scenarios to test feature discrimination.

    Returns scenarios that should produce clearly different state vectors.
    """
    current_time = 100.0

    scenarios = {
        "low_load_balanced": {
            "description": "Low load, all replicas balanced",
            "replicas": [
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 1.0, "normal")
                ], allocation_ratio=0.3),
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 1.2, "normal")
                ], allocation_ratio=0.3),
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 0.8, "normal")
                ], allocation_ratio=0.3),
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 1.1, "normal")
                ], allocation_ratio=0.3),
            ]
        },

        "high_load_imbalanced": {
            "description": "High load, severely imbalanced",
            "replicas": [
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 5.0, "high"),
                    create_mock_request(current_time - 4.5, "urgent"),
                    create_mock_request(current_time - 4.0, "normal"),
                    create_mock_request(current_time - 3.5, "high"),
                ], allocation_ratio=0.9),
                create_mock_replica_scheduler([], allocation_ratio=0.1),
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 0.5, "normal")
                ], allocation_ratio=0.2),
                create_mock_replica_scheduler([], allocation_ratio=0.1),
            ]
        },

        "medium_load_old_requests": {
            "description": "Medium load with very old waiting requests",
            "replicas": [
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 8.0, "normal"),
                    create_mock_request(current_time - 2.0, "normal"),
                ], allocation_ratio=0.6),
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 7.5, "high"),
                    create_mock_request(current_time - 1.5, "normal"),
                ], allocation_ratio=0.6),
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 1.0, "normal"),
                ], allocation_ratio=0.4),
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 0.5, "normal"),
                ], allocation_ratio=0.4),
            ]
        },

        "urgent_priority_mixed": {
            "description": "Mixed load with urgent priority requests",
            "replicas": [
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 3.0, "urgent"),
                    create_mock_request(current_time - 2.5, "critical"),
                ], allocation_ratio=0.7),
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 1.0, "normal"),
                    create_mock_request(current_time - 0.5, "low"),
                ], allocation_ratio=0.4),
                create_mock_replica_scheduler([
                    create_mock_request(current_time - 2.0, "high"),
                ], allocation_ratio=0.5),
                create_mock_replica_scheduler([], allocation_ratio=0.2),
            ]
        }
    }

    return scenarios, current_time


def mock_get_replica_scheduler_fn(replicas: List) -> callable:
    """Create a mock function to get replica schedulers."""
    def get_scheduler(replica_id: int):
        return replicas[replica_id]

    # Add __self__ attribute for compatibility
    get_scheduler.__self__ = type('MockScheduler', (), {
        '_request_queue': []
    })()

    return get_scheduler


class MockMetricStore:
    """Mock metric store for testing."""
    def get_throughput(self, current_time: float) -> float:
        return 2.5

    def get_average_latency(self) -> float:
        return 3.2


def validate_state_features():
    """
    Main validation function to test state feature effectiveness.
    """
    print("🔍 Validating Enhanced State Features...")
    print("=" * 60)

    # Initialize enhanced state builder
    state_builder = StateBuilder(
        max_queue_requests=4,
        history_window=5,
        qps_window=10,
        enable_enhanced_features=True,
        enable_queue_delay_features=True
    )

    # Generate test scenarios
    scenarios, current_time = generate_test_scenarios()
    metric_store = MockMetricStore()

    # Build state vectors for each scenario
    state_vectors = {}
    feature_names = state_builder.get_feature_names(num_replicas=4)

    print(f"📊 Total state dimensions: {len(feature_names)}")
    print(f"🎯 Testing {len(scenarios)} different scenarios...")
    print()

    for scenario_name, scenario_data in scenarios.items():
        print(f"Testing scenario: {scenario_name}")
        print(f"Description: {scenario_data['description']}")

        # Mock replica dictionary
        replicas_dict = {i: f"replica_{i}" for i in range(4)}
        get_scheduler_fn = mock_get_replica_scheduler_fn(scenario_data['replicas'])

        # Build state vector
        state_vector = state_builder.build_global_state(
            replicas=replicas_dict,
            get_replica_scheduler_fn=get_scheduler_fn,
            current_time=current_time,
            metric_store=metric_store
        )

        state_vectors[scenario_name] = state_vector
        print(f"✅ State vector shape: {state_vector.shape}")
        print()

    # Analyze feature discrimination
    print("🔬 Feature Discrimination Analysis:")
    print("=" * 40)

    # Calculate pairwise distances between scenarios
    scenario_names = list(state_vectors.keys())
    n_scenarios = len(scenario_names)

    distance_matrix = np.zeros((n_scenarios, n_scenarios))

    for i, name1 in enumerate(scenario_names):
        for j, name2 in enumerate(scenario_names):
            if i != j:
                # Calculate L2 distance
                dist = np.linalg.norm(state_vectors[name1] - state_vectors[name2])
                distance_matrix[i, j] = dist

    # Print distance matrix
    print("Pairwise L2 distances between scenarios:")
    print(f"{'Scenario':<25} " + " ".join([f"{name[:8]:<8}" for name in scenario_names]))
    for i, name in enumerate(scenario_names):
        distances_str = " ".join([f"{distance_matrix[i, j]:<8.2f}" for j in range(n_scenarios)])
        print(f"{name:<25} {distances_str}")

    print()

    # Identify most discriminative features
    print("🎯 Most Discriminative Features:")
    print("-" * 30)

    feature_variances = []
    for feature_idx in range(len(feature_names)):
        feature_values = [state_vectors[name][feature_idx] for name in scenario_names]
        variance = np.var(feature_values)
        feature_variances.append((feature_idx, feature_names[feature_idx], variance))

    # Sort by variance (higher variance = more discriminative)
    feature_variances.sort(key=lambda x: x[2], reverse=True)

    print("Top 10 most discriminative features:")
    for i, (idx, name, variance) in enumerate(feature_variances[:10]):
        print(f"{i+1:2d}. {name:<35} (variance: {variance:.4f})")

    print()

    # Check for potential issues
    print("⚠️  Feature Quality Check:")
    print("-" * 25)

    # Check for features with zero variance (useless features)
    zero_variance_features = [name for _, name, var in feature_variances if var < 1e-8]
    if zero_variance_features:
        print(f"❌ Found {len(zero_variance_features)} features with zero variance:")
        for name in zero_variance_features[:5]:  # Show first 5
            print(f"   - {name}")
        if len(zero_variance_features) > 5:
            print(f"   ... and {len(zero_variance_features) - 5} more")
    else:
        print("✅ No zero-variance features found")

    # Check for features with very low variance (potentially weak features)
    low_variance_features = [name for _, name, var in feature_variances if 1e-8 <= var < 0.01]
    if low_variance_features:
        print(f"⚠️  Found {len(low_variance_features)} features with low variance (< 0.01):")
        for name in low_variance_features[:3]:
            print(f"   - {name}")
    else:
        print("✅ No excessively low-variance features found")

    print()

    # Test network architecture compatibility
    print("🧠 Network Architecture Validation:")
    print("-" * 35)

    try:
        # Test ActorCritic network with enhanced features
        state_dim = len(feature_names)
        network = ActorCritic(
            state_dim=state_dim,
            action_dim=4,
            hidden_size=320,
            layer_N=3,
            gru_layers=3,
            enable_decoupled=True,
            enable_cross_replica_attention=True,
            num_replicas=4,
            attention_heads=4
        )

        # Test forward pass with a sample state
        sample_state = torch.FloatTensor(state_vectors[scenario_names[0]]).unsqueeze(0)

        # Test policy evaluation
        hidden_states = (
            torch.zeros(3, 1, 320),  # Actor GRU states
            torch.zeros(1, 1, 320)   # Critic GRU states
        )
        masks = torch.ones(1, 1)

        action, log_prob, value, new_hidden = network.act_value(
            sample_state, hidden_states, masks, temperature=1.0
        )

        print("✅ Network forward pass successful")
        print(f"   Sample action: {action.item()}")
        print(f"   Log probability: {log_prob.item():.4f}")
        print(f"   Value estimate: {value.item():.4f}")

    except Exception as e:
        print(f"❌ Network validation failed: {e}")

    print()
    print("🎉 State Feature Validation Complete!")

    # Summary recommendations
    print("\n📋 Recommendations:")
    print("-" * 18)

    min_distance = np.min(distance_matrix[distance_matrix > 0])
    max_distance = np.max(distance_matrix)

    if min_distance < 1.0:
        print("⚠️  Some scenarios are very similar (distance < 1.0)")
        print("   Consider enhancing feature sensitivity")
    else:
        print("✅ Good scenario discrimination (minimum distance > 1.0)")

    if max_distance > 10.0:
        print("✅ Strong feature range (maximum distance > 10.0)")
    else:
        print("⚠️  Consider increasing feature sensitivity for better discrimination")

    high_variance_count = sum(1 for _, _, var in feature_variances if var > 0.1)
    print(f"📊 {high_variance_count}/{len(feature_names)} features show strong discrimination (variance > 0.1)")

    return {
        'state_vectors': state_vectors,
        'distance_matrix': distance_matrix,
        'feature_variances': feature_variances,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    results = validate_state_features()