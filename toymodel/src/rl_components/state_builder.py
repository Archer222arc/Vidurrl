"""
State builder for toy model queue scheduling.

Converts queue states and request information into RL state representation.
State includes: queue lengths + first n request types in each queue.
"""

import torch
import numpy as np
from typing import List, Dict, Any

from ..entities import Request, Replica


class QueueStateBuilder:
    """
    State builder for toy model queue scheduling.
    
    Converts queue states and request information into normalized state representation
    suitable for RL training.
    State: [queue_length_0, queue_length_1, first_n_types_queue_0, first_n_types_queue_1, current_request_type]
    """

    def __init__(self, num_replicas: int = 2, n_requests: int = 3, normalize: bool = True):
        """
        Initialize state builder.
        
        Args:
            num_replicas: Number of replicas in the system
            n_requests: Number of request types to include from each queue
            normalize: Whether to normalize state values
        """
        self.num_replicas = num_replicas
        self.n_requests = n_requests
        self.normalize = normalize
        
        # State normalization statistics (updated during training)
        self.state_mean = None
        self.state_std = None
        self.state_count = 0
    
    def build_state(self, request: Request, replicas: List[Replica]) -> torch.Tensor:
        """
        Build state representation for given request and replica states.
        
        Args:
            request: Incoming request
            replicas: List of replica states
            
        Returns:
            State tensor of shape (state_dim,)
        """
        # Extract queue lengths
        queue_lengths = [replica.queue_length for replica in replicas]
        
        # Extract first n request types from each queue
        queue_0_types = self._get_first_n_types(replicas[0], self.n_requests)
        queue_1_types = self._get_first_n_types(replicas[1], self.n_requests)
        
        # Current request type
        current_request_type = request.request_type
        
        # Build state vector: [queue_length_0, queue_length_1, first_n_types_queue_0, first_n_types_queue_1, current_request_type]
        state = np.concatenate([
            [queue_lengths[0], queue_lengths[1]],  # Queue lengths
            queue_0_types,  # First n types in queue 0
            queue_1_types,  # First n types in queue 1
            [current_request_type]  # Current request type
        ], dtype=np.float32)
        
        # Normalize if enabled
        if self.normalize:
            state = self._normalize_state(state)
        
        return torch.tensor(state, dtype=torch.float32)
    
    def _get_first_n_types(self, replica: Replica, n: int) -> np.ndarray:
        """
        Get first n request types from replica queue.
        
        Args:
            replica: Replica object
            n: Number of request types to extract
            
        Returns:
            Array of request types (padded with -1 if queue is shorter than n)
        """
        types = []
        for i in range(min(n, len(replica.queue))):
            types.append(replica.queue[i].request_type)
        
        # Pad with -1 if queue is shorter than n
        while len(types) < n:
            types.append(-1)
        
        return np.array(types, dtype=np.float32)
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state using running statistics.
        
        Args:
            state: Raw state array
            
        Returns:
            Normalized state array
        """
        if self.state_mean is None:
            # Initialize statistics
            self.state_mean = np.zeros_like(state)
            self.state_std = np.ones_like(state)
            self.state_count = 0
        
        # Update running statistics
        self.state_count += 1
        delta = state - self.state_mean
        self.state_mean += delta / self.state_count
        delta2 = state - self.state_mean
        self.state_std += (delta * delta2 - self.state_std) / self.state_count
        
        # Normalize
        normalized_state = (state - self.state_mean) / (self.state_std + 1e-8)
        
        return normalized_state
    
    def reset_normalization(self):
        """Reset normalization statistics."""
        self.state_mean = None
        self.state_std = None
        self.state_count = 0
    
    def get_state_dim(self) -> int:
        """Get state dimension."""
        return 2 + 2 * self.n_requests + 1  # [queue_length_0, queue_length_1, first_n_types_queue_0, first_n_types_queue_1, current_request_type]
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.num_replicas  # Number of replica choices
