"""
Reward calculator for toy model queue scheduling.

Computes rewards based only on latency.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from ..entities import Request, Replica


class LatencyRewardCalculator:
    """
    Reward calculator for toy model queue scheduling.
    
    Computes rewards based only on latency (negative reward for high latency).
    """

    def __init__(
        self,
        latency_weight: float = 1.0,
        latency_scale: float = 1.0
    ):
        """
        Initialize reward calculator.
        
        Args:
            latency_weight: Weight for latency-based reward
            latency_scale: Scaling factor for latency normalization
        """
        self.latency_weight = latency_weight
        self.latency_scale = latency_scale
        
        # Running statistics for reward normalization
        self.latency_mean = 0.0
        self.latency_std = 1.0
        self.latency_count = 0
    
    def calculate_reward(
        self,
        request: Request,
        replicas: List[Replica],
        assigned_replica: int,
        latency: float
    ) -> float:
        """
        Calculate reward for routing decision based only on latency.
        
        Args:
            request: The routed request
            replicas: List of replica states
            assigned_replica: Assigned replica ID
            latency: Request latency (queue_time + service_time)
            
        Returns:
            Calculated reward (negative, lower latency = higher reward)
        """
        # Latency-based reward (negative, lower latency = higher reward)
        latency_reward = -self._normalize_latency(latency) * self.latency_weight
        
        return latency_reward
    
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
    
    def reset_statistics(self):
        """Reset running statistics."""
        self.latency_mean = 0.0
        self.latency_std = 1.0
        self.latency_count = 0
