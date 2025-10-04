"""
Predictor training components.

This module contains training utilities for neural latency predictor:
- Data collection from simulation
- Supervised learning trainer
"""

from .data_collector import LatencyDataCollector
from .predictor_trainer import PredictorTrainer, LatencyDataset

__all__ = [
    'LatencyDataCollector',
    'PredictorTrainer',
    'LatencyDataset',
]
