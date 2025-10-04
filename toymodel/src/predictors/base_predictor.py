"""
Base class for latency predictors.

All predictors should inherit from BaseLatencyPredictor and implement
the abstract methods.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from ..entities import Request, Replica


class BaseLatencyPredictor(ABC):
    """
    Abstract base class for latency predictors.
    """

    @abstractmethod
    def predict_latency(
        self,
        request: Request,
        replica: Replica,
        replicas: List[Replica]
    ) -> float:
        """
        Predict the processing latency for a request when assigned to a replica.

        Args:
            request: The request to be assigned
            replica: The target replica
            replicas: All replicas (for context)

        Returns:
            Predicted latency (queue_time + service_time) - total time in system
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset predictor state."""
        pass
