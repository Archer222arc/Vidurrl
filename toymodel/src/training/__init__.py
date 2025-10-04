"""
Training components for toy model.

This package contains training modules for different components:
- predictor: Neural latency predictor training
"""

from .predictor import (
    LatencyDataCollector,
    PredictorTrainer,
    LatencyDataset,
)

__all__ = [
    'LatencyDataCollector',
    'PredictorTrainer', 
    'LatencyDataset',
]
