"""
Unit tests for schedulers.
"""

import pytest
from toymodel.src.entities import Request, Replica
from toymodel.schedulers import (
    OracleScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    ShortestQueueScheduler,
)


class TestOracleScheduler:
    """Test Oracle scheduler."""

    def test_oracle_type_a_routing(self):
        """Test Oracle routes Type A to Replica 0."""
        scheduler = OracleScheduler(num_replicas=2)
        request = Request(request_id=1, request_type=0, arrival_time=10.0)
        replicas = [
            Replica(replica_id=0, service_rates={0: 10.0, 1: 5.0}, queue=[]),
            Replica(replica_id=1, service_rates={0: 5.0, 1: 10.0}, queue=[]),
        ]

        replica_id = scheduler.schedule(request, replicas)

        assert replica_id == 0

    def test_oracle_type_b_routing(self):
        """Test Oracle routes Type B to Replica 1."""
        scheduler = OracleScheduler(num_replicas=2)
        request = Request(request_id=1, request_type=1, arrival_time=10.0)
        replicas = [
            Replica(replica_id=0, service_rates={0: 10.0, 1: 5.0}, queue=[]),
            Replica(replica_id=1, service_rates={0: 5.0, 1: 10.0}, queue=[]),
        ]

        replica_id = scheduler.schedule(request, replicas)

        assert replica_id == 1

    def test_oracle_requires_two_replicas(self):
        """Test Oracle requires exactly 2 replicas."""
        with pytest.raises(ValueError, match="exactly 2 replicas"):
            OracleScheduler(num_replicas=3)


class TestRandomScheduler:
    """Test Random scheduler."""

    def test_random_distribution(self):
        """Test Random scheduler produces uniform distribution."""
        scheduler = RandomScheduler(num_replicas=2, seed=42)
        request = Request(request_id=1, request_type=0, arrival_time=10.0)
        replicas = [
            Replica(replica_id=0, service_rates={0: 10.0, 1: 5.0}, queue=[]),
            Replica(replica_id=1, service_rates={0: 5.0, 1: 10.0}, queue=[]),
        ]

        # Generate many routing decisions
        decisions = [scheduler.schedule(request, replicas) for _ in range(1000)]

        # Check approximate uniformity (should be close to 50-50)
        count_0 = sum(1 for d in decisions if d == 0)
        count_1 = sum(1 for d in decisions if d == 1)

        assert 400 < count_0 < 600  # Roughly uniform
        assert 400 < count_1 < 600


class TestRoundRobinScheduler:
    """Test Round-robin scheduler."""

    def test_round_robin_order(self):
        """Test Round-robin cycles through replicas in order."""
        scheduler = RoundRobinScheduler(num_replicas=2)
        request = Request(request_id=1, request_type=0, arrival_time=10.0)
        replicas = [
            Replica(replica_id=0, service_rates={0: 10.0, 1: 5.0}, queue=[]),
            Replica(replica_id=1, service_rates={0: 5.0, 1: 10.0}, queue=[]),
        ]

        decisions = [scheduler.schedule(request, replicas) for _ in range(6)]

        assert decisions == [0, 1, 0, 1, 0, 1]

    def test_round_robin_reset(self):
        """Test Round-robin reset."""
        scheduler = RoundRobinScheduler(num_replicas=2)
        request = Request(request_id=1, request_type=0, arrival_time=10.0)
        replicas = [
            Replica(replica_id=0, service_rates={0: 10.0, 1: 5.0}, queue=[]),
            Replica(replica_id=1, service_rates={0: 5.0, 1: 10.0}, queue=[]),
        ]

        # Make some decisions
        scheduler.schedule(request, replicas)
        scheduler.schedule(request, replicas)

        # Reset and verify starts from 0 again
        scheduler.reset()
        decision = scheduler.schedule(request, replicas)

        assert decision == 0


class TestShortestQueueScheduler:
    """Test Shortest Queue scheduler."""

    def test_shortest_queue_selection(self):
        """Test scheduler selects replica with shortest queue."""
        scheduler = ShortestQueueScheduler(num_replicas=2)
        request = Request(request_id=1, request_type=0, arrival_time=10.0)

        # Replica 1 has shorter queue
        replicas = [
            Replica(
                replica_id=0,
                service_rates={0: 10.0, 1: 5.0},
                queue=[Request(i, 0, 0.0) for i in range(5)],
            ),
            Replica(
                replica_id=1,
                service_rates={0: 5.0, 1: 10.0},
                queue=[Request(i, 0, 0.0) for i in range(2)],
            ),
        ]

        replica_id = scheduler.schedule(request, replicas)

        assert replica_id == 1

    def test_shortest_queue_tie_breaking(self):
        """Test tie breaking by replica ID."""
        scheduler = ShortestQueueScheduler(num_replicas=2)
        request = Request(request_id=1, request_type=0, arrival_time=10.0)

        # Both replicas have same queue length
        replicas = [
            Replica(
                replica_id=0,
                service_rates={0: 10.0, 1: 5.0},
                queue=[Request(i, 0, 0.0) for i in range(3)],
            ),
            Replica(
                replica_id=1,
                service_rates={0: 5.0, 1: 10.0},
                queue=[Request(i, 0, 0.0) for i in range(3)],
            ),
        ]

        replica_id = scheduler.schedule(request, replicas)

        # Should select lower ID
        assert replica_id == 0
