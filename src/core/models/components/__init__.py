"""
Modular Components for Actor-Critic Architecture

This package provides reusable components that can be integrated
into any neural network architecture with configuration-based control.
"""

from .temporal_lstm import TemporalLSTM, TemporalLSTMConfig, TemporalLSTMFactory

__all__ = [
    'TemporalLSTM',
    'TemporalLSTMConfig',
    'TemporalLSTMFactory'
]