"""
Latency predictor for toy model queue scheduling.

Predicts the processing time for a request when assigned to a specific replica.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

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

        Args:
            request: The request to be assigned
            replica: The target replica
            replicas: All replicas (for context)

        Returns:
            Combined latency metric (self_latency + weighted_impact)
        """
        replica_id = replica.replica_id
        request_type = request.request_type

        # 1. Predict self latency (same as SimpleLatencyPredictor)
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
            impact_on_others = avg_processing_time * queue_length / queue_length  # avg impact

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

        Args:
            request: The request to be assigned
            replica: The target replica
            replicas: All replicas (for context)

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
        """
        Update average processing time based on actual completion time.

        Args:
            replica_id: ID of the replica that processed the request
            request_type: Type of the processed request
            actual_processing_time: Actual time taken to process the request
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
            return 1.0  # Default 1.0 time unit

    def reset(self):
        """Reset predictor state."""
        self.processing_times = {}


class NeuralLatencyPredictor(BaseLatencyPredictor):
    """
    Neural network-based latency predictor trained on historical data.

    This predictor learns to predict both self latency and impact on others
    from state features, handling randomness and non-linear relationships
    better than expectation-based predictors.
    """

    def __init__(
        self,
        num_replicas: int = 2,
        num_request_types: int = 2,
        hidden_dim: int = 128,
        prediction_weight: float = 1.0,
        impact_weight: float = 1.0,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize neural latency predictor.

        Args:
            num_replicas: Number of replicas in the system
            num_request_types: Number of request types
            hidden_dim: Hidden layer dimension
            prediction_weight: Weight for prediction in reward calculation
            impact_weight: Weight for impact on other requests
            checkpoint_path: Path to pretrained model checkpoint
        """
        self.num_replicas = num_replicas
        self.num_request_types = num_request_types
        self.hidden_dim = hidden_dim
        self.prediction_weight = prediction_weight
        self.impact_weight = impact_weight

        # Calculate input dimension
        # Features: current_request_type + per_replica(queue_length + type_counts + service_rate)
        self.input_dim = 1 + num_replicas * (1 + num_request_types + 1)

        # Build neural network
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # [self_latency, avg_impact]
        )

        # Load pretrained weights if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)

        self.model.eval()  # Set to evaluation mode by default

    def _extract_state_features(
        self,
        request: Request,
        replica: Replica,
        replicas: List[Replica]
    ) -> torch.Tensor:
        """Extract state features for neural network input."""
        features = [float(request.request_type)]

        for r in replicas:
            # Queue length
            features.append(float(len(r.queue)))

            # Request type counts in queue
            request_types = [req.request_type for req in r.queue]
            for req_type in range(self.num_request_types):
                features.append(float(request_types.count(req_type)))

            # Service rate for current request type
            features.append(r.get_service_rate(request.request_type))

        return torch.tensor(features, dtype=torch.float32)

    def predict_latency(
        self,
        request: Request,
        replica: Replica,
        replicas: List[Replica]
    ) -> float:
        """
        Predict latency using neural network.

        Args:
            request: The request to be assigned
            replica: The target replica
            replicas: All replicas (for context)

        Returns:
            Combined latency metric (self_latency + weighted_impact)
        """
        state = self._extract_state_features(request, replica, replicas)

        with torch.no_grad():
            prediction = self.model(state)

        self_latency = prediction[0].item()
        avg_impact = prediction[1].item()

        # Ensure non-negative predictions
        self_latency = max(0.0, self_latency)
        avg_impact = max(0.0, avg_impact)

        return self_latency + avg_impact * self.impact_weight

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
        state = self._extract_state_features(request, replica, replicas)

        with torch.no_grad():
            prediction = self.model(state)

        self_latency = max(0.0, prediction[0].item())
        avg_impact = max(0.0, prediction[1].item())

        return self_latency, avg_impact

    def load_checkpoint(self, checkpoint_path: str):
        """Load pretrained model weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained predictor from {checkpoint_path}")

        if 'train_stats' in checkpoint:
            stats = checkpoint['train_stats']
            print(f"  Training loss: {stats.get('final_loss', 'N/A')}")
            print(f"  Val MSE: {stats.get('val_mse', 'N/A')}")

    def save_checkpoint(self, checkpoint_path: str, train_stats: Optional[Dict] = None):
        """Save model weights and training statistics."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_replicas': self.num_replicas,
            'num_request_types': self.num_request_types,
            'hidden_dim': self.hidden_dim,
            'input_dim': self.input_dim,
        }

        if train_stats:
            checkpoint['train_stats'] = train_stats

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved predictor checkpoint to {checkpoint_path}")

    def reset(self):
        """Reset predictor state (no-op for neural predictor)."""
        pass


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
    impact_weight: float = 1.0,
    num_replicas: int = 2,
    num_request_types: int = 2,
    hidden_dim: int = 128,
    checkpoint_path: Optional[str] = None,
    **kwargs
) -> BaseLatencyPredictor:
    """
    Factory function to create latency predictors.

    Args:
        predictor_type: Type of predictor ('simple', 'system_aware', 'learned', 'none')
        prediction_weight: Weight for prediction in reward calculation
        impact_weight: Weight for impact on other requests
        num_replicas: Number of replicas (for learned predictor)
        num_request_types: Number of request types (for learned predictor)
        hidden_dim: Hidden dimension (for learned predictor)
        checkpoint_path: Path to pretrained model (for learned predictor)
        **kwargs: Additional arguments for specific predictors

    Returns:
        Configured latency predictor instance
    """
    if predictor_type == "simple":
        return SimpleLatencyPredictor(prediction_weight=prediction_weight)
    elif predictor_type == "system_aware":
        return SystemAwareLatencyPredictor(
            prediction_weight=prediction_weight,
            impact_weight=impact_weight
        )
    elif predictor_type == "learned":
        return NeuralLatencyPredictor(
            num_replicas=num_replicas,
            num_request_types=num_request_types,
            hidden_dim=hidden_dim,
            prediction_weight=prediction_weight,
            impact_weight=impact_weight,
            checkpoint_path=checkpoint_path
        )
    elif predictor_type == "none":
        return NoOpLatencyPredictor()
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")
