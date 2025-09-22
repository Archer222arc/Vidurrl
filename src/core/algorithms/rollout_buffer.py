"""
Rollout buffer for storing and processing PPO experiences.

This module implements experience collection and GAE (Generalized Advantage Estimation)
computation for PPO training.
"""

from typing import List, Tuple

import torch


class RolloutBuffer:
    """
    Buffer for storing rollout experiences and computing GAE advantages.

    Collects state-action-reward sequences and computes advantages using
    Generalized Advantage Estimation for PPO training.
    """

    def __init__(
        self,
        state_dim: int,
        rollout_len: int,
        gamma: float,
        gae_lambda: float,
        device: str = "cpu"
    ):
        """
        Initialize rollout buffer.

        Args:
            state_dim: Dimension of state space
            rollout_len: Maximum length of rollout sequences
            gamma: Discount factor for future rewards
            gae_lambda: GAE lambda parameter for bias-variance tradeoff
            device: Device for tensor computations
        """
        self.rollout_len = rollout_len
        self.gamma = gamma
        self.lmbda = gae_lambda
        self.device = device
        self.reset(state_dim)

    def reset(self, state_dim: int) -> None:
        """
        Reset buffer for new rollout collection.

        Args:
            state_dim: Dimension of state space
        """
        self.s: List[torch.Tensor] = []
        self.a: List[torch.Tensor] = []
        self.logp: List[torch.Tensor] = []
        self.v: List[torch.Tensor] = []
        self.r: List[float] = []
        self.masks: List[float] = []  # 1: not done, 0: reset hidden
        self.ptr = 0

    def add_step(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        logp: torch.Tensor,
        v: torch.Tensor,
        r: float,
        mask: torch.Tensor
    ) -> None:
        """
        Add single step experience to buffer.

        Args:
            s: State tensor
            a: Action tensor
            logp: Log probability of action
            v: Value estimate
            r: Reward
            mask: Episode continuation mask
        """
        self.s.append(s)
        self.a.append(a)  # Tensor scalar (1,) or 0-D
        self.logp.append(logp)  # (1,)

        # Ensure value is stored as (1,)
        if torch.is_tensor(v):
            self.v.append(v.view(-1)[0:1])
        else:
            self.v.append(torch.tensor([float(v)], dtype=torch.float32))

        self.r.append(r)

        # Convert mask to float
        self.masks.append(float(mask.item() if torch.is_tensor(mask) else mask))
        self.ptr += 1

    def is_full(self) -> bool:
        """
        Check if buffer has reached maximum capacity.

        Returns:
            True if buffer is full
        """
        return self.ptr >= self.rollout_len

    def compute_gae(self, last_v: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Compute GAE advantages and returns.

        Args:
            last_v: Bootstrap value for final state

        Returns:
            Tuple of (states, actions, log_probs, values, returns, advantages)
        """
        # Stack all experiences into tensors (exactly as original implementation)
        s = torch.stack(self.s, dim=0).to(self.device)                        # (T, D)
        a = torch.stack(self.a, dim=0).view(-1).to(torch.long).to(self.device)  # (T,)
        logp = torch.stack(self.logp, dim=0).view(-1).to(self.device)         # (T,)
        v = torch.stack(self.v, dim=0).to(self.device)                        # (T,1) -> (T,)
        if v.dim() == 2 and v.size(-1) == 1:
            v = v.squeeze(-1)
        r = torch.tensor(self.r, dtype=torch.float32, device=self.device)     # (T,)
        m = torch.tensor(self.masks, dtype=torch.float32, device=self.device) # (T,)

        # Convert last_v to scalar
        if torch.is_tensor(last_v):
            last_v = last_v.view(-1)[0]

        # Compute GAE advantages
        T = v.shape[0]
        adv = torch.zeros(T, device=self.device)
        last_gae = torch.tensor(0.0, device=self.device)

        for t in reversed(range(T)):
            v_next = last_v if t == T - 1 else v[t + 1]
            delta = r[t] + self.gamma * v_next * m[t] - v[t]
            last_gae = delta + self.gamma * self.lmbda * m[t] * last_gae
            adv[t] = last_gae

        ret = adv + v

        # Normalize advantages for stable training
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        return s, a, logp, v, ret, adv