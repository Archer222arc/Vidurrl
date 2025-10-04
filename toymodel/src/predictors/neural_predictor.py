"""
Neural network-based latency predictor trained on historical data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .base_predictor import BaseLatencyPredictor
from ..entities import Request, Replica


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
        max_queue_obs: int = 128,
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
            max_queue_obs: Maximum queue positions to observe per replica
            checkpoint_path: Path to pretrained model checkpoint
        """
        self.num_replicas = num_replicas
        self.num_request_types = num_request_types
        self.hidden_dim = hidden_dim
        self.prediction_weight = prediction_weight
        self.impact_weight = impact_weight
        self.max_queue_obs = max_queue_obs

        # Calculate input dimension
        # Features per replica:
        #   - queue_length (1)
        #   - queue request types (max_queue_obs)
        #   - queue position mask (max_queue_obs) - 1 for valid, 0 for padded
        #   - service rates for each request type (num_request_types)
        #   - current serving request type (1, -1 if idle)
        #   - busy_until time (1)
        # Global features:
        #   - current request type (1)
        #
        # Total = num_replicas * (1 + max_queue_obs + max_queue_obs + num_request_types + 1 + 1) + 1
        features_per_replica = 1 + max_queue_obs + max_queue_obs + num_request_types + 2
        self.input_dim = num_replicas * features_per_replica + 1

        # Build neural network with deeper architecture
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # [self_latency, avg_impact]
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
        """
        Extract rich state features for neural network input with masking.

        Features include:
        - Current request type
        - Per-replica features:
            * Queue length
            * Request types in queue (first max_queue_obs positions, padded with 0)
            * Queue position mask (1 for valid positions, 0 for padded)
            * Service rates for all request types
            * Currently serving request type (-1 if idle)
            * Busy until time (normalized)
        """
        # Pre-allocate feature array for better performance
        features = np.zeros(self.input_dim, dtype=np.float32)
        idx = 0

        # Current request type
        features[idx] = float(request.request_type)
        idx += 1

        for r in replicas:
            queue_len = len(r.queue)

            # Queue length
            features[idx] = float(queue_len)
            idx += 1

            # Queue composition: request types (fully vectorized)
            valid_len = min(queue_len, self.max_queue_obs)
            if valid_len > 0:
                # Vectorized extraction using numpy array comprehension
                features[idx:idx + valid_len] = np.array(
                    [r.queue[i].request_type for i in range(valid_len)],
                    dtype=np.float32
                )
            # Remaining positions already zero from pre-allocation
            idx += self.max_queue_obs

            # Queue mask (vectorized assignment)
            if valid_len > 0:
                features[idx:idx + valid_len] = 1.0
            # Remaining mask positions already zero
            idx += self.max_queue_obs

            # Service rates for all request types
            for req_type in range(self.num_request_types):
                features[idx] = r.get_service_rate(req_type)
                idx += 1

            # Currently serving request type (-1 if idle)
            if r.current_request is not None:
                features[idx] = float(r.current_request.request_type)
            else:
                features[idx] = -1.0
            idx += 1

            # Busy until time (normalized by expected service time)
            if r.current_request is not None:
                expected_service_time = 1.0 / r.get_service_rate(r.current_request.request_type)
                features[idx] = r.busy_until / (expected_service_time + 1e-6)
            # else: already zero from pre-allocation
            idx += 1

        return torch.from_numpy(features)

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
            'max_queue_obs': self.max_queue_obs,
            'input_dim': self.input_dim,
        }

        if train_stats:
            checkpoint['train_stats'] = train_stats

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved predictor checkpoint to {checkpoint_path}")

    def reset(self):
        """Reset predictor state (no-op for neural predictor)."""
        pass
