"""
Normalization utilities for reinforcement learning components.

This module provides running normalization for state preprocessing,
using Welford's algorithm for numerical stability. Includes VecNormalize
implementation based on research best practices for preventing PPO collapse.
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


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


class VecNormalize:
    """
    VecNormalize implementation based on research best practices.

    Provides both observation and reward normalization with discount-based
    reward scaling to prevent PPO training collapse. Based on the techniques
    described in the PPO双极坍塌问题深度研究文档.
    """

    def __init__(
        self,
        obs_eps: float = 1e-8,
        obs_clip: float = 10.0,
        reward_eps: float = 1e-8,
        reward_clip: float = 10.0,
        gamma: float = 0.99,
        norm_reward: bool = True,
        norm_obs: bool = True,
        min_count: int = 10
    ):
        """
        Initialize VecNormalize with research-backed parameters.

        Args:
            obs_eps: Observation normalization epsilon
            obs_clip: Observation clipping range [-clip, clip]
            reward_eps: Reward normalization epsilon
            reward_clip: Reward clipping range
            gamma: Discount factor for reward scaling
            norm_reward: Whether to normalize rewards
            norm_obs: Whether to normalize observations
            min_count: Minimum samples before normalization
        """
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.gamma = gamma

        # Observation normalizer with文档推荐配置
        self.obs_rms = RunningNormalizer(eps=obs_eps, clip=obs_clip, min_count=min_count)

        # Reward normalizer with discount-based scaling
        self.reward_rms = RunningNormalizer(eps=reward_eps, clip=reward_clip, min_count=min_count)
        self.ret = 0.0  # Running discounted return

        # Adaptive normalization parameters for non-stationary environments
        self.adaptation_rate = 0.01
        self.stability_threshold = 0.1

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using running statistics."""
        if not self.norm_obs:
            return obs
        return self.obs_rms.normalize(obs)

    def update_obs(self, obs: np.ndarray) -> None:
        """Update observation statistics."""
        if self.norm_obs:
            self.obs_rms.update(obs)

    def normalize_reward(self, reward: float, done: bool = False) -> float:
        """
        Normalize rewards using discount-based scaling.

        Based on research findings that reward normalization with
        discount factor prevents value estimation bias.
        """
        if not self.norm_reward:
            return reward

        # Update running discounted return
        self.ret = self.ret * self.gamma + reward

        # Update reward statistics with discounted return
        self.reward_rms.update(np.array([self.ret]))

        # Reset on episode end
        if done:
            self.ret = 0.0

        # Return normalized reward
        return self.reward_rms.normalize(np.array([reward]))[0]

    def adaptive_update(self, performance_metric: float) -> None:
        """
        Adaptive normalization for non-stationary environments.

        Adjusts normalization parameters based on performance trends
        to handle distribution shifts in load balancing scenarios.
        """
        if not hasattr(self, '_prev_performance'):
            self._prev_performance = performance_metric
            return

        performance_change = abs(performance_metric - self._prev_performance)

        if performance_change > self.stability_threshold:
            # Increase adaptation rate for non-stationary periods
            adapted_rate = min(self.adaptation_rate * 2, 0.1)
            logger.info(f"[VecNormalize] Detected non-stationarity, adapting rate to {adapted_rate}")
        else:
            # Standard adaptation rate for stable periods
            adapted_rate = self.adaptation_rate

        self._prev_performance = performance_metric

    def state_dict(self) -> Dict:
        """Serialize VecNormalize state."""
        return {
            "obs_rms": self.obs_rms.state_dict(),
            "reward_rms": self.reward_rms.state_dict(),
            "ret": self.ret,
            "norm_obs": self.norm_obs,
            "norm_reward": self.norm_reward,
            "gamma": self.gamma
        }

    def load_state_dict(self, state: Dict) -> None:
        """Restore VecNormalize state."""
        self.obs_rms.load_state_dict(state["obs_rms"])
        self.reward_rms.load_state_dict(state["reward_rms"])
        self.ret = float(state.get("ret", 0.0))
        self.norm_obs = bool(state.get("norm_obs", True))
        self.norm_reward = bool(state.get("norm_reward", True))
        self.gamma = float(state.get("gamma", 0.99))
