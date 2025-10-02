"""
Baseline schedulers for comparison.
"""

import numpy as np
from toymodel.src.entities import Request, Replica
from toymodel.schedulers.base import BaseScheduler


class RandomScheduler(BaseScheduler):
    """
    Random routing: select replica uniformly at random.
    """

    def __init__(self, num_replicas: int = 2, seed: int = None):
        """
        Initialize Random scheduler.

        Args:
            num_replicas: Number of replicas
            seed: Random seed
        """
        super().__init__(num_replicas)
        self.rng = np.random.RandomState(seed)

    def schedule(self, request: Request, replicas: list[Replica]) -> int:
        """
        Select replica uniformly at random.

        Args:
            request: Incoming request (unused)
            replicas: List of replicas (unused)

        Returns:
            Random replica ID
        """
        return self.rng.randint(0, self.num_replicas)


class RoundRobinScheduler(BaseScheduler):
    """
    Round-robin routing: cycle through replicas in order.
    """

    def __init__(self, num_replicas: int = 2):
        """
        Initialize Round-robin scheduler.

        Args:
            num_replicas: Number of replicas
        """
        super().__init__(num_replicas)
        self.counter = 0

    def schedule(self, request: Request, replicas: list[Replica]) -> int:
        """
        Select next replica in round-robin order.

        Args:
            request: Incoming request (unused)
            replicas: List of replicas (unused)

        Returns:
            Next replica ID in sequence
        """
        replica_id = self.counter % self.num_replicas
        self.counter += 1
        return replica_id

    def reset(self) -> None:
        """Reset counter to 0."""
        self.counter = 0


class ShortestQueueScheduler(BaseScheduler):
    """
    Shortest queue routing: select replica with shortest queue.

    Ties broken by replica ID (lower ID preferred).
    """

    def __init__(self, num_replicas: int = 2):
        """
        Initialize Shortest Queue scheduler.

        Args:
            num_replicas: Number of replicas
        """
        super().__init__(num_replicas)

    def schedule(self, request: Request, replicas: list[Replica]) -> int:
        """
        Select replica with shortest queue.

        Args:
            request: Incoming request (unused)
            replicas: List of replicas

        Returns:
            Replica ID with shortest queue (ties broken by lower ID)
        """
        min_queue = min(r.queue_length for r in replicas)
        for replica in replicas:
            if replica.queue_length == min_queue:
                return replica.replica_id
        return 0  # Fallback (should never reach)
