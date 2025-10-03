"""
Latency predictor for toy model queue scheduling.

Predicts the processing time for a request when assigned to a specific replica.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

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


class SimpleLatencyPredictor(BaseLatencyPredictor):
    """
    Latency predictor based on historical average processing times.
    
    Records average processing times for 4 combinations:
    - Queue 0 processing Request Type 0
    - Queue 0 processing Request Type 1  
    - Queue 1 processing Request Type 0
    - Queue 1 processing Request Type 1
    
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
        
        Args:
            request: The request to be assigned
            replica: The target replica
            replicas: All replicas (for context)
            
        Returns:
            Predicted latency (queue_time + service_time) - total time in system
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
        
        Args:
            replica_id: ID of the replica that processed the request
            request_type: Type of the processed request
            actual_processing_time: Actual time taken to process the request
        """
        key = (replica_id, request_type)
        
        if key in self.processing_times:
            # Update using simple arithmetic average
            old_avg, count = self.processing_times[key]
            new_count = count + 1
            new_avg = (old_avg * count + actual_processing_time) / new_count
            self.processing_times[key] = (new_avg, new_count)
        else:
            # Initialize with first observation
            self.processing_times[key] = (actual_processing_time, 1)
    
    def _get_average_processing_time(self, replica_id: int, request_type: int) -> float:
        """
        Get average processing time for a specific replica-request combination.
        
        Args:
            replica_id: ID of the replica
            request_type: Type of the request
            
        Returns:
            Average processing time, or default estimate if no data available
        """
        key = (replica_id, request_type)
        
        if key in self.processing_times:
            avg_time, count = self.processing_times[key]
            return avg_time
        else:
            # Return default estimate based on service rate
            # This is used when we don't have historical data yet
            return 1.0  # Default 1.0 time unit
    
    def reset(self):
        """Reset predictor state."""
        # Dictionary to store average processing times and counts
        # Key: (replica_id, request_type), Value: (average_processing_time, count)
        self.processing_times = {}


class NoOpLatencyPredictor(BaseLatencyPredictor):
    """
    No-operation predictor that always returns 0.
    
    Used when prediction is disabled.
    """
    
    def predict_latency(
        self,
        request: Request,
        replica: Replica,
        replicas: List[Replica]
    ) -> float:
        """Return 0 (no prediction)."""
        return 0.0
    
    def reset(self):
        """No-op reset."""
        pass


def create_latency_predictor(
    predictor_type: str,
    prediction_weight: float = 1.0,
    **kwargs
) -> BaseLatencyPredictor:
    """
    Factory function to create latency predictors.
    
    Args:
        predictor_type: Type of predictor ('simple', 'none', etc.)
        prediction_weight: Weight for prediction in reward calculation
        **kwargs: Additional arguments for specific predictors
        
    Returns:
        Configured latency predictor instance
    """
    if predictor_type == "simple":
        return SimpleLatencyPredictor(prediction_weight=prediction_weight)
    elif predictor_type == "none":
        return NoOpLatencyPredictor()
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")
