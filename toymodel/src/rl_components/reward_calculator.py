"""
Reward calculator for toy model queue scheduling.

Computes rewards based on latency and optionally predicted latency.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from ..entities import Request, Replica
from ..predictors import BaseLatencyPredictor


class LatencyRewardCalculator:
    """
    Reward calculator for toy model queue scheduling.
    
    Computes rewards based on latency and optionally predicted latency.
    """

    def __init__(
        self,
        latency_weight: float = 1.0,
        latency_scale: float = 1.0,
        use_prediction: bool = False,
        prediction_weight: float = 0.5,
        latency_predictor: Optional[BaseLatencyPredictor] = None
    ):
        """
        Initialize reward calculator.
        
        Args:
            latency_weight: Weight for actual latency-based reward
            latency_scale: Scaling factor for latency normalization
            use_prediction: Whether to use predicted latency in reward calculation
            prediction_weight: Weight for predicted latency in reward calculation
            latency_predictor: Latency predictor instance (optional)
        """
        self.latency_weight = latency_weight
        self.latency_scale = latency_scale
        self.use_prediction = use_prediction
        self.prediction_weight = prediction_weight
        self.latency_predictor = latency_predictor
        
        # Running statistics for reward normalization
        self.latency_mean = 0.0
        self.latency_std = 1.0
        self.latency_count = 0
        
        # Running statistics for prediction normalization
        self.prediction_mean = 0.0
        self.prediction_std = 1.0
        self.prediction_count = 0
    
    def calculate_reward(
        self,
        request: Request,
        replicas: List[Replica],
        assigned_replica: int,
        latency: float
    ) -> Dict[str, float]:
        """
        Calculate reward for routing decision based on latency and optionally predicted latency.

        Args:
            request: The routed request
            replicas: List of replica states
            assigned_replica: Assigned replica ID
            latency: Request latency (queue_time + service_time) - total time in system

        Returns:
            Dictionary containing:
                - 'total': Total reward (negative, lower latency = higher reward)
                - 'latency': Actual latency reward component
                - 'prediction': Predicted latency reward component (if enabled)
        """
        # Actual latency-based reward (negative, lower latency = higher reward)
        latency_reward = -self._normalize_latency(latency) * self.latency_weight

        total_reward = latency_reward
        prediction_reward = 0.0

        # Add predicted latency reward if enabled
        if self.use_prediction and self.latency_predictor is not None:
            predicted_latency = self.latency_predictor.predict_latency(
                request, replicas[assigned_replica], replicas
            )
            prediction_reward = -self._normalize_prediction(predicted_latency) * self.prediction_weight
            total_reward += prediction_reward

        return {
            'total': total_reward,
            'latency': latency_reward,
            'prediction': prediction_reward
        }
    
    def _normalize_latency(self, latency: float) -> float:
        """
        Normalize latency using running statistics.
        
        Args:
            latency: Raw latency value
            
        Returns:
            Normalized latency
        """
        # Update running statistics
        self.latency_count += 1
        delta = latency - self.latency_mean
        self.latency_mean += delta / self.latency_count
        delta2 = latency - self.latency_mean
        self.latency_std += (delta * delta2 - self.latency_std) / self.latency_count
        
        # Normalize
        normalized_latency = (latency - self.latency_mean) / (self.latency_std + 1e-8)
        
        return normalized_latency * self.latency_scale
    
    def _normalize_prediction(self, prediction: float) -> float:
        """
        Normalize prediction using running statistics.
        
        Args:
            prediction: Raw prediction value
            
        Returns:
            Normalized prediction
        """
        # Update running statistics
        self.prediction_count += 1
        delta = prediction - self.prediction_mean
        self.prediction_mean += delta / self.prediction_count
        delta2 = prediction - self.prediction_mean
        self.prediction_std += (delta * delta2 - self.prediction_std) / self.prediction_count
        
        # Normalize
        normalized_prediction = (prediction - self.prediction_mean) / (self.prediction_std + 1e-8)
        
        return normalized_prediction * self.latency_scale  # Use same scale as latency
    
    def reset_statistics(self):
        """Reset running statistics."""
        self.latency_mean = 0.0
        self.latency_std = 1.0
        self.latency_count = 0
        
        self.prediction_mean = 0.0
        self.prediction_std = 1.0
        self.prediction_count = 0
