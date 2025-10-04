"""
Simple latency predictor based on historical average processing times.
"""

from typing import List
from .base_predictor import BaseLatencyPredictor
from ..entities import Request, Replica


class SimpleLatencyPredictor(BaseLatencyPredictor):
    """
    Latency predictor based on historical average processing times.

    Records average processing times for request-replica combinations.
    Predicts latency by estimating completion time for all tasks in queue.
    """

    def __init__(self, prediction_weight: float = 1.0):
        """
        Initialize latency predictor.

        Args:
            prediction_weight: Weight for prediction in reward calculation
        """
        self.prediction_weight = prediction_weight
        self.reset()

    def predict_latency(
        self,
        request: Request,
        replica: Replica,
        replicas: List[Replica]
    ) -> float:
        """
        Predict latency by estimating completion time for all tasks in queue.
        """
        replica_id = replica.replica_id
        request_type = request.request_type

        # Get average processing time for this combination
        avg_processing_time = self._get_average_processing_time(replica_id, request_type)

        # Calculate predicted completion time for all tasks in queue
        predicted_completion_time = 0.0

        # Add processing time for all existing tasks in queue
        for queued_request in replica.queue:
            queued_type = queued_request.request_type
            queued_avg_time = self._get_average_processing_time(replica_id, queued_type)
            predicted_completion_time += queued_avg_time

        # Add processing time for the new request itself
        predicted_completion_time += avg_processing_time

        return predicted_completion_time

    def update_processing_time(
        self,
        replica_id: int,
        request_type: int,
        actual_processing_time: float
    ):
        """
        Update average processing time based on actual completion time.
        """
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
