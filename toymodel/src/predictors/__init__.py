"""
Latency predictors for toy model queue scheduling.

Available predictors:
- SimpleLatencyPredictor: Based on historical averages
- SystemAwareLatencyPredictor: Considers impact on other requests
- NeuralLatencyPredictor: Neural network trained on data
"""

from typing import Optional
from .base_predictor import BaseLatencyPredictor
from .simple_predictor import SimpleLatencyPredictor
from .system_aware_predictor import SystemAwareLatencyPredictor
from .neural_predictor import NeuralLatencyPredictor


class NoOpLatencyPredictor(BaseLatencyPredictor):
    """No-operation predictor that always returns 0."""

    def predict_latency(self, request, replica, replicas) -> float:
        return 0.0

    def reset(self):
        pass


def create_latency_predictor(
    predictor_type: str,
    prediction_weight: float = 1.0,
    impact_weight: float = 1.0,
    num_replicas: int = 2,
    num_request_types: int = 2,
    hidden_dim: int = 128,
    max_queue_obs: int = 128,
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
        max_queue_obs: Maximum queue positions to observe (for learned predictor)
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
            max_queue_obs=max_queue_obs,
            checkpoint_path=checkpoint_path
        )
    elif predictor_type == "none":
        return NoOpLatencyPredictor()
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")


__all__ = [
    'BaseLatencyPredictor',
    'SimpleLatencyPredictor',
    'SystemAwareLatencyPredictor',
    'NeuralLatencyPredictor',
    'NoOpLatencyPredictor',
    'create_latency_predictor'
]
