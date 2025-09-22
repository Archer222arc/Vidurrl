"""
Normalization utilities for reinforcement learning components.

This module provides running normalization for state preprocessing,
using Welford's algorithm for numerical stability.
"""

from typing import Dict, Optional

import numpy as np


class RunningNormalizer:
    """
    Welford running normalizer with optional clipping and warmup handling.

    Maintains per-dimension running statistics to standardize observations
    that may arrive with vastly different scales. Supports incremental
    updates and tolerates batched observations.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        clip: float = 5.0,
        min_count: int = 2,
    ) -> None:
        """
        Initialize the running normalizer.

        Args:
            eps: Small epsilon value to prevent division by zero
            clip: Clipping range for normalized values
        """
        self.eps = float(eps)
        self.clip = float(clip)
        self.count = 0
        self.min_count = max(1, int(min_count))
        self.mean: Optional[np.ndarray] = None
        self.m2: Optional[np.ndarray] = None
        self._shape: Optional[tuple[int, ...]] = None

    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with new observation.

        Args:
            x: New observation array
        """
        arr = np.asarray(x, dtype=np.float32)

        if arr.ndim == 0:
            raise ValueError("RunningNormalizer expected array-like input, got scalar")

        if self._shape is None:
            self._shape = tuple(arr.shape[-1:]) if arr.ndim > 1 else tuple(arr.shape)

        flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(1, -1)

        for sample in flat:
            if self.mean is None:
                self.mean = sample.copy()
                self.m2 = np.zeros_like(sample)
                self.count = 1
                continue

            self.count += 1
            delta = sample - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (sample - self.mean)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize input using running statistics.

        Args:
            x: Input array to normalize

        Returns:
            Normalized and clipped array
        """
        if self.mean is None or self.m2 is None:
            return np.clip(x.astype(np.float32, copy=False), -self.clip, self.clip)

        var = self.m2 / max(self.count - 1, 1)
        std = np.sqrt(np.maximum(var, self.eps))

        if self.count < self.min_count:
            std = np.maximum(std, 1.0)

        z = (x.astype(np.float32, copy=False) - self.mean) / std
        return np.clip(z, -self.clip, self.clip)

    def state_dict(self) -> Dict[str, Optional[np.ndarray]]:
        """Return a serialisable snapshot of the running statistics."""
        return {
            "count": self.count,
            "mean": None if self.mean is None else self.mean.copy(),
            "m2": None if self.m2 is None else self.m2.copy(),
            "eps": self.eps,
            "clip": self.clip,
            "min_count": self.min_count,
        }

    def load_state_dict(self, state: Dict[str, Optional[np.ndarray]]) -> None:
        """Restore statistics from :meth:`state_dict`."""
        self.count = int(state.get("count", 0))
        self.mean = None if state.get("mean") is None else np.asarray(state["mean"], dtype=np.float32)
        self.m2 = None if state.get("m2") is None else np.asarray(state["m2"], dtype=np.float32)
        self.eps = float(state.get("eps", self.eps))
        self.clip = float(state.get("clip", self.clip))
        self.min_count = int(state.get("min_count", self.min_count))
