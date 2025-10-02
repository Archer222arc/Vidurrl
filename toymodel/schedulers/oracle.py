"""
Oracle scheduler: optimal routing policy.

Routes Type A to Replica 0, Type B to Replica 1.
"""

from toymodel.src.entities import Request, Replica
from toymodel.schedulers.base import BaseScheduler


class OracleScheduler(BaseScheduler):
    """
    Oracle scheduler with perfect knowledge.

    Routing policy:
    - Type A (request_type=0) → Replica 0 (faster for Type A)
    - Type B (request_type=1) → Replica 1 (faster for Type B)

    This represents the theoretical optimal policy for the toy model.
    """

    def __init__(self, num_replicas: int = 2):
        """
        Initialize Oracle scheduler.

        Args:
            num_replicas: Number of replicas (must be 2)
        """
        super().__init__(num_replicas)

        if num_replicas != 2:
            raise ValueError(f"Oracle scheduler requires exactly 2 replicas, got {num_replicas}")

    def schedule(self, request: Request, replicas: list[Replica]) -> int:
        """
        Make optimal routing decision.

        Args:
            request: Incoming request
            replicas: List of replicas (unused, decision based only on type)

        Returns:
            Replica ID: 0 for Type A, 1 for Type B
        """
        return request.request_type
