"""
Basic entities for toy model simulation.

Defines Request and Replica classes for M/M/1 queueing system.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Request:
    """
    Request entity for toy model.

    Attributes:
        request_id: Unique identifier
        request_type: Request type (0=Type A, 1=Type B)
        arrival_time: Time when request arrived at system
        assigned_replica: Replica ID assigned to this request
        service_start_time: Time when service started
        completion_time: Time when service completed
    """

    request_id: int
    request_type: int  # 0 or 1
    arrival_time: float
    assigned_replica: Optional[int] = None
    service_start_time: Optional[float] = None
    completion_time: Optional[float] = None

    @property
    def queue_time(self) -> float:
        """Time spent waiting in queue."""
        if self.service_start_time is None:
            return 0.0
        return self.service_start_time - self.arrival_time

    @property
    def service_time(self) -> float:
        """Time spent in service."""
        if self.completion_time is None or self.service_start_time is None:
            return 0.0
        return self.completion_time - self.service_start_time

    @property
    def total_time(self) -> float:
        """Total time in system (queue + service)."""
        if self.completion_time is None:
            return 0.0
        return self.completion_time - self.arrival_time

    @property
    def is_completed(self) -> bool:
        """Check if request has completed service."""
        return self.completion_time is not None


@dataclass
class Replica:
    """
    Replica entity for toy model.

    Attributes:
        replica_id: Unique identifier
        service_rates: Service rates for each request type {type: rate}
                      Higher rate = faster service
        queue: FIFO queue of pending requests
        current_request: Request currently being served
        busy_until: Simulation time when current service completes
    """

    replica_id: int
    service_rates: dict[int, float]  # {request_type: rate}
    queue: list[Request]
    current_request: Optional[Request] = None
    busy_until: float = 0.0

    def __post_init__(self):
        """Initialize queue if not provided."""
        if self.queue is None:
            self.queue = []

    @property
    def queue_length(self) -> int:
        """Current queue length (excluding request in service)."""
        return len(self.queue)

    @property
    def is_busy(self) -> bool:
        """Check if replica is currently serving a request."""
        return self.current_request is not None

    @property
    def utilization(self, current_time: float) -> float:
        """
        Compute current utilization (0-1).

        Args:
            current_time: Current simulation time

        Returns:
            1.0 if busy, 0.0 if idle
        """
        return 1.0 if self.busy_until > current_time else 0.0

    def get_service_rate(self, request_type: int) -> float:
        """
        Get service rate for given request type.

        Args:
            request_type: Request type (0 or 1)

        Returns:
            Service rate (requests per unit time)
        """
        return self.service_rates[request_type]

    def add_to_queue(self, request: Request) -> None:
        """
        Add request to FIFO queue.

        Args:
            request: Request to add
        """
        request.assigned_replica = self.replica_id
        self.queue.append(request)

    def start_service(self, request: Request, current_time: float, service_duration: float) -> None:
        """
        Start serving a request.

        Args:
            request: Request to serve
            current_time: Current simulation time
            service_duration: Duration of service
        """
        self.current_request = request
        request.service_start_time = current_time
        self.busy_until = current_time + service_duration

    def complete_service(self, current_time: float) -> Request:
        """
        Complete current service.

        Args:
            current_time: Current simulation time

        Returns:
            Completed request
        """
        completed = self.current_request
        completed.completion_time = current_time
        self.current_request = None
        self.busy_until = current_time
        return completed
