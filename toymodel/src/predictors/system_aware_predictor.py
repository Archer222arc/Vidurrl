"""
System-aware latency predictor that considers impact on other requests.
"""

from typing import List, Tuple
from .base_predictor import BaseLatencyPredictor
from ..entities import Request, Replica


class SystemAwareLatencyPredictor(BaseLatencyPredictor):
    """
    System-aware latency predictor that considers both:
    1. Self latency: predicted completion time for the current request
    2. Impact on others: average impact on other queued requests

    This predictor enables system-level optimization by considering the
    global effect of scheduling decisions.
    """

    def __init__(self, prediction_weight: float = 1.0, impact_weight: float = 1.0):
        """
        Initialize system-aware latency predictor.

        Args:
            prediction_weight: Weight for prediction in reward calculation
            impact_weight: Weight for impact on other requests
        """
        self.prediction_weight = prediction_weight
        self.impact_weight = impact_weight
        self.reset()

    def predict_latency(
        self,
        request: Request,
        replica: Replica,
        replicas: List[Replica]
    ) -> float:
        """
        Predict latency considering both self completion time and impact on others.
        """
        replica_id = replica.replica_id
        request_type = request.request_type

        # 1. Predict self latency
        avg_processing_time = self._get_average_processing_time(replica_id, request_type)

        self_latency = 0.0
        for queued_request in replica.queue:
            queued_type = queued_request.request_type
            queued_avg_time = self._get_average_processing_time(replica_id, queued_type)
            self_latency += queued_avg_time
        self_latency += avg_processing_time

        # 2. Predict impact on other requests
        impact_on_others = 0.0
        queue_length = len(replica.queue)

        if queue_length > 0:
            # Each queued request will be delayed by the new request's processing time
            impact_on_others = avg_processing_time

        # 3. Combine self latency and impact
        combined_latency = self_latency + impact_on_others * self.impact_weight

        return combined_latency

    def predict_detailed(
        self,
        request: Request,
        replica: Replica,
        replicas: List[Replica]
    ) -> Tuple[float, float]:
        """
        Predict latency with detailed breakdown.

        Returns:
            Tuple of (self_latency, avg_impact_on_others)
        """
        replica_id = replica.replica_id
        request_type = request.request_type

        avg_processing_time = self._get_average_processing_time(replica_id, request_type)

        # Self latency
        self_latency = 0.0
        for queued_request in replica.queue:
            queued_type = queued_request.request_type
            queued_avg_time = self._get_average_processing_time(replica_id, queued_type)
            self_latency += queued_avg_time
        self_latency += avg_processing_time

        # Average impact on others
        queue_length = len(replica.queue)
        avg_impact = avg_processing_time if queue_length > 0 else 0.0

        return self_latency, avg_impact

    def update_processing_time(
        self,
        replica_id: int,
        request_type: int,
        actual_processing_time: float
    ):
        """Update average processing time based on actual completion time."""
        key = (replica_id, request_type)

        if key in self.processing_times:
            old_avg, count = self.processing_times[key]
            new_count = count + 1
            new_avg = (old_avg * count + actual_processing_time) / new_count
            self.processing_times[key] = (new_avg, new_count)
        else:
            self.processing_times[key] = (actual_processing_time, 1)

    def _get_average_processing_time(self, replica_id: int, request_type: int) -> float:
        """Get average processing time for a specific replica-request combination."""
        key = (replica_id, request_type)

        if key in self.processing_times:
            avg_time, _ = self.processing_times[key]
            return avg_time
        else:
            return 1.0  # Default 1.0 time unit

    def reset(self):
        """Reset predictor state."""
        self.processing_times = {}
