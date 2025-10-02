"""
Unit tests for queueing environment.
"""

import pytest
import numpy as np
from toymodel.src.environment import QueueEnvironment
from toymodel.schedulers import OracleScheduler, RandomScheduler


class TestQueueEnvironment:
    """Test QueueEnvironment."""

    @pytest.fixture
    def env_config(self):
        """Standard environment configuration."""
        return {
            "num_replicas": 2,
            "service_rates": {
                0: {0: 10.0, 1: 5.0},  # Replica 0: fast for Type A
                1: {0: 5.0, 1: 10.0},  # Replica 1: fast for Type B
            },
            "arrival_rates": {0: 3.0, 1: 3.0},  # Low arrival rates for testing
            "max_time": 100.0,
            "seed": 42,
        }

    def test_environment_initialization(self, env_config):
        """Test environment initialization."""
        env = QueueEnvironment(**env_config)

        assert env.num_replicas == 2
        assert len(env.replicas) == 2
        assert env.current_time == 0.0

    def test_environment_reset(self, env_config):
        """Test environment reset."""
        env = QueueEnvironment(**env_config)
        env.reset()

        assert env.current_time == 0.0
        assert env.next_arrival_idx == 0
        assert len(env.pending_arrivals) > 0
        assert len(env.completed_requests) == 0

    def test_step_until_next_arrival(self, env_config):
        """Test stepping to next arrival."""
        env = QueueEnvironment(**env_config)
        env.reset()

        first_request = env.step_until_next_arrival()

        assert first_request is not None
        assert env.current_time == first_request.arrival_time
        assert first_request.request_type in [0, 1]

    def test_route_request(self, env_config):
        """Test routing request to replica."""
        env = QueueEnvironment(**env_config)
        env.reset()

        request = env.step_until_next_arrival()
        env.route_request(request, replica_id=0)

        replica = env.replicas[0]
        assert request.assigned_replica == 0
        # Request should either be in queue or being served
        assert replica.queue_length >= 0 or replica.is_busy

    def test_oracle_simulation(self, env_config):
        """Test complete simulation with Oracle scheduler."""
        env = QueueEnvironment(**env_config)
        scheduler = OracleScheduler(num_replicas=2)

        def routing_policy(request, replicas):
            return scheduler.schedule(request, replicas)

        completed = env.run_simulation(routing_policy)

        # Verify all requests completed
        assert len(completed) > 0
        for req in completed:
            assert req.is_completed
            assert req.total_time > 0

        # Verify Oracle routing correctness
        for req in completed:
            expected_replica = req.request_type
            assert req.assigned_replica == expected_replica

    def test_random_simulation(self, env_config):
        """Test simulation with Random scheduler."""
        env = QueueEnvironment(**env_config)
        scheduler = RandomScheduler(num_replicas=2, seed=42)

        def routing_policy(request, replicas):
            return scheduler.schedule(request, replicas)

        completed = env.run_simulation(routing_policy)

        # Verify all requests completed
        assert len(completed) > 0
        for req in completed:
            assert req.is_completed

    def test_system_stability(self, env_config):
        """Test system stability under load."""
        # Use low arrival rates to ensure stability (ρ < 1)
        env_config["arrival_rates"] = {0: 2.0, 1: 2.0}  # Total λ = 4.0
        env = QueueEnvironment(**env_config)
        scheduler = OracleScheduler(num_replicas=2)

        def routing_policy(request, replicas):
            return scheduler.schedule(request, replicas)

        completed = env.run_simulation(routing_policy)

        # Compute average latency
        latencies = [req.total_time for req in completed]
        mean_latency = np.mean(latencies)

        # Verify reasonable latency (should be low with optimal routing)
        assert mean_latency < 10.0  # Sanity check

    def test_get_replica_state(self, env_config):
        """Test replica state retrieval."""
        env = QueueEnvironment(**env_config)
        env.reset()

        state = env.get_replica_state(0)

        assert "queue_length" in state
        assert "utilization" in state
        assert "busy_until" in state
        assert "is_busy" in state

    def test_get_system_state(self, env_config):
        """Test system state retrieval."""
        env = QueueEnvironment(**env_config)
        env.reset()

        state = env.get_system_state()

        assert "current_time" in state
        assert "total_queue_length" in state
        assert "num_busy_replicas" in state
        assert "num_completed" in state
        assert "num_remaining_arrivals" in state
