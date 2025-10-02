"""
Base scheduler interface for routing policies.
"""

from abc import ABC, abstractmethod
from toymodel.src.entities import Request, Replica


class BaseScheduler(ABC):
    """
    Abstract base class for routing schedulers.

    All schedulers must implement the schedule() method.
    """

    def __init__(self, num_replicas: int):
        """
        Initialize scheduler.

        Args:
            num_replicas: Number of replicas in system
        """
        self.num_replicas = num_replicas

    @abstractmethod
    def schedule(self, request: Request, replicas: list[Replica]) -> int:
        """
        Make routing decision for incoming request.

        Args:
            request: Incoming request
            replicas: List of available replicas

        Returns:
            Replica ID to route request to (0 to num_replicas-1)
        """
        pass

    def reset(self) -> None:
        """
        Reset scheduler state (optional).

        Override if scheduler maintains internal state.
        """
        pass
