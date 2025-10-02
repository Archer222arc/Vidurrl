"""
Unit tests for toy model entities.
"""

import pytest
from toymodel.src.entities import Request, Replica


class TestRequest:
    """Test Request entity."""

    def test_request_creation(self):
        """Test basic request creation."""
        req = Request(
            request_id=1,
            request_type=0,
            arrival_time=10.0,
        )
        assert req.request_id == 1
        assert req.request_type == 0
        assert req.arrival_time == 10.0
        assert not req.is_completed

    def test_request_queue_time(self):
        """Test queue time calculation."""
        req = Request(
            request_id=1,
            request_type=0,
            arrival_time=10.0,
            service_start_time=15.0,
        )
        assert req.queue_time == 5.0

    def test_request_service_time(self):
        """Test service time calculation."""
        req = Request(
            request_id=1,
            request_type=0,
            arrival_time=10.0,
            service_start_time=15.0,
            completion_time=18.0,
        )
        assert req.service_time == 3.0

    def test_request_total_time(self):
        """Test total time calculation."""
        req = Request(
            request_id=1,
            request_type=0,
            arrival_time=10.0,
            service_start_time=15.0,
            completion_time=18.0,
        )
        assert req.total_time == 8.0
        assert req.is_completed


class TestReplica:
    """Test Replica entity."""

    def test_replica_creation(self):
        """Test basic replica creation."""
        replica = Replica(
            replica_id=0,
            service_rates={0: 10.0, 1: 5.0},
            queue=[],
        )
        assert replica.replica_id == 0
        assert replica.queue_length == 0
        assert not replica.is_busy

    def test_replica_add_to_queue(self):
        """Test adding request to queue."""
        replica = Replica(
            replica_id=0,
            service_rates={0: 10.0, 1: 5.0},
            queue=[],
        )
        req = Request(request_id=1, request_type=0, arrival_time=10.0)

        replica.add_to_queue(req)

        assert replica.queue_length == 1
        assert req.assigned_replica == 0
        assert replica.queue[0] == req

    def test_replica_service_rates(self):
        """Test service rate retrieval."""
        replica = Replica(
            replica_id=0,
            service_rates={0: 10.0, 1: 5.0},
            queue=[],
        )
        assert replica.get_service_rate(0) == 10.0
        assert replica.get_service_rate(1) == 5.0

    def test_replica_start_service(self):
        """Test starting service."""
        replica = Replica(
            replica_id=0,
            service_rates={0: 10.0, 1: 5.0},
            queue=[],
        )
        req = Request(request_id=1, request_type=0, arrival_time=10.0)

        replica.start_service(req, current_time=15.0, service_duration=2.0)

        assert replica.is_busy
        assert replica.current_request == req
        assert replica.busy_until == 17.0
        assert req.service_start_time == 15.0

    def test_replica_complete_service(self):
        """Test completing service."""
        replica = Replica(
            replica_id=0,
            service_rates={0: 10.0, 1: 5.0},
            queue=[],
        )
        req = Request(request_id=1, request_type=0, arrival_time=10.0)

        replica.start_service(req, current_time=15.0, service_duration=2.0)
        completed = replica.complete_service(current_time=17.0)

        assert completed == req
        assert completed.completion_time == 17.0
        assert not replica.is_busy
        assert replica.current_request is None
