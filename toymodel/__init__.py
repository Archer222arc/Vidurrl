"""
Toy Model for PPO Routing Policy Validation.

This module implements a simplified M/M/1 queueing simulation
to validate PPO-based routing strategies.
"""

from toymodel.src.entities import Request, Replica
from toymodel.src.request_generator import PoissonRequestGenerator
from toymodel.src.environment import QueueEnvironment
from toymodel.src.config import load_config, ToyModelConfig

__all__ = [
    "Request",
    "Replica",
    "PoissonRequestGenerator",
    "QueueEnvironment",
    "load_config",
    "ToyModelConfig",
]
