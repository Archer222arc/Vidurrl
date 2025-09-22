"""
Actor-Critic neural network architecture for PPO.

This module implements a multi-layer perceptron with GRU recurrent layers
for both policy (actor) and value (critic) functions.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def init_layer(layer: nn.Module, gain: float = 1.0, use_orthogonal: bool = True) -> None:
    """
    Initialize network layer with orthogonal or Xavier initialization.

    Args:
        layer: Neural network layer to initialize
        gain: Initialization gain factor
        use_orthogonal: Whether to use orthogonal initialization
    """
    if isinstance(layer, nn.Linear):
        if use_orthogonal:
            nn.init.orthogonal_(layer.weight, gain=gain)
        else:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)


class ActorCritic(nn.Module):
    """
    Actor-Critic network with MLP encoder and GRU recurrent layers.

    The network processes state inputs through a multi-layer perceptron,
    followed by GRU layers for temporal modeling, and outputs both
    policy logits (actor) and state values (critic).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        layer_N: int = 2,
        gru_layers: int = 2,
        use_orthogonal: bool = True,
        enable_decoupled: bool = True,
        feature_projection_dim: int = None,
    ):
        """
        Initialize Enhanced Actor-Critic network.

        Args:
            state_dim: Dimension of state input
            action_dim: Number of possible actions
            hidden_size: Hidden layer size
            layer_N: Number of additional MLP layers (total: 1 + layer_N)
            gru_layers: Number of GRU layers
            use_orthogonal: Whether to use orthogonal initialization
            enable_decoupled: Enable decoupled Actor/Critic architecture
            feature_projection_dim: Dimension for feature projection layer (auto if None)
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.layer_N = layer_N
        self.gru_layers = gru_layers
        self.enable_decoupled = enable_decoupled
        self.feature_projection_dim = feature_projection_dim or hidden_size * 2

        # Feature projection layer for multi-dimensional state normalization
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, self.feature_projection_dim),
            nn.GELU(),
            nn.Linear(self.feature_projection_dim, hidden_size),
        )
        init_layer(self.feature_proj[1], gain=math.sqrt(2), use_orthogonal=use_orthogonal)
        init_layer(self.feature_proj[3], gain=math.sqrt(2), use_orthogonal=use_orthogonal)

        # Compressed shared MLP (only 1 layer when decoupled)
        if enable_decoupled:
            # Minimal shared processing for feature extraction
            self.shared_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            )
            init_layer(self.shared_mlp[0], gain=math.sqrt(2), use_orthogonal=use_orthogonal)
        else:
            # Original MLP encoder: Linear -> (LayerNorm + ReLU) x (1 + layer_N)
            mlp = []
            mlp.append(nn.Linear(hidden_size, hidden_size))
            self.mlp_ln0 = nn.LayerNorm(hidden_size)
            init_layer(mlp[-1], gain=math.sqrt(2), use_orthogonal=use_orthogonal)

            self.mlp_h = nn.ModuleList()
            self.mlp_ln = nn.ModuleList()
            for _ in range(layer_N):
                self.mlp_h.append(nn.Linear(hidden_size, hidden_size))
                self.mlp_ln.append(nn.LayerNorm(hidden_size))
                init_layer(self.mlp_h[-1], gain=math.sqrt(2), use_orthogonal=use_orthogonal)
            self.shared_mlp = nn.Sequential(*mlp)

        # GRU recurrent layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=False,  # Use (T,N,H) format
        )
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        self.gru_ln = nn.LayerNorm(hidden_size)

        # Decoupled Actor and Critic branches
        if enable_decoupled:
            # Actor branch: 2 additional layers + enhanced output
            self.actor_branch = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            )
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_size, action_dim),
                nn.LayerNorm(action_dim),
                nn.Tanh()  # Bound logits to reasonable range
            )
            init_layer(self.actor_branch[0], gain=math.sqrt(2), use_orthogonal=use_orthogonal)
            init_layer(self.actor_branch[3], gain=math.sqrt(2), use_orthogonal=use_orthogonal)
            init_layer(self.actor_head[0], gain=0.01, use_orthogonal=use_orthogonal)

            # Critic branch: 2 additional layers + separate GRU
            self.critic_branch = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            )
            self.critic_gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,  # Single layer for critic
                batch_first=False,
            )
            self.critic_head = nn.Linear(hidden_size, 1)
            init_layer(self.critic_branch[0], gain=math.sqrt(2), use_orthogonal=use_orthogonal)
            init_layer(self.critic_branch[3], gain=math.sqrt(2), use_orthogonal=use_orthogonal)
            init_layer(self.critic_head, gain=1.0, use_orthogonal=use_orthogonal)

            # Initialize critic GRU
            for name, param in self.critic_gru.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
        else:
            # Original output heads
            self.actor = nn.Linear(hidden_size, action_dim)
            self.critic = nn.Linear(hidden_size, 1)
            init_layer(self.actor, gain=0.01, use_orthogonal=use_orthogonal)
            init_layer(self.critic, gain=1.0, use_orthogonal=use_orthogonal)

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature projection and shared MLP.

        Args:
            x: Input tensor of shape (N, state_dim)

        Returns:
            Encoded features of shape (N, hidden_size)
        """
        # Feature projection for multi-scale normalization
        x = self.feature_proj(x)

        if self.enable_decoupled:
            # Minimal shared processing
            x = self.shared_mlp(x)
        else:
            # Original MLP processing
            x = self.shared_mlp(x)
            x = F.relu(self.mlp_ln0(x))
            for i in range(self.layer_N):
                x = F.relu(self.mlp_ln[i](self.mlp_h[i](x)))
        return x

    def forward_gru(
        self, x: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through shared GRU layers.

        Args:
            x: Input tensor of shape (N, hidden_size)
            hxs: Hidden states of shape (gru_layers, N, hidden_size)
            masks: Reset masks of shape (N, 1), 0 indicates reset

        Returns:
            Tuple of (output, updated_hidden_states)
        """
        # x: (N, H) -> (T=1, N, H)
        x = x.unsqueeze(0)
        # Reset hidden states based on masks
        if hxs is not None and masks is not None:
            masks_expanded = masks.view(1, -1, 1)
            hxs = hxs * masks_expanded

        out, hxs = self.gru(x, hxs)
        out = out.squeeze(0)  # (N, H)
        out = self.gru_ln(out)
        return out, hxs

    def act_value(
        self, s: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute value for given state with temperature scaling.

        Args:
            s: State tensor of shape (N, state_dim)
            hxs: Hidden states (tuple if decoupled, single tensor if not)
            masks: Reset masks
            temperature: Temperature for logits scaling (higher = more exploration)

        Returns:
            Tuple of (action, log_prob, value, updated_hidden_states)
        """
        z = self.forward_mlp(s)

        if self.enable_decoupled:
            # Separate processing for actor and critic
            z_shared, hxs_actor = self.forward_gru(z, hxs[0] if isinstance(hxs, tuple) else hxs, masks)

            # Actor branch
            z_actor = self.actor_branch(z_shared)
            logits = self.actor_head(z_actor)

            # Critic branch with separate GRU
            z_critic = self.critic_branch(z_shared)
            z_critic = z_critic.unsqueeze(0)  # (1, N, H)
            hxs_critic = hxs[1] if isinstance(hxs, tuple) else hxs
            if hxs_critic is not None and masks is not None:
                masks_expanded = masks.view(1, -1, 1)
                hxs_critic = hxs_critic * masks_expanded
            z_critic_out, hxs_critic = self.critic_gru(z_critic, hxs_critic)
            z_critic_out = z_critic_out.squeeze(0)  # (N, H)
            v = self.critic_head(z_critic_out).squeeze(-1)

            updated_hxs = (hxs_actor, hxs_critic)
        else:
            # Original shared processing
            z, updated_hxs = self.forward_gru(z, hxs, masks)
            logits = self.actor(z)
            v = self.critic(z).squeeze(-1)

        # Apply temperature scaling to logits
        if temperature != 1.0:
            logits = logits / temperature

        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, v, updated_hxs

    def evaluate_actions(
        self, s: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions for policy gradient computation.

        Args:
            s: State tensor
            hxs: Hidden states (tuple if decoupled, single tensor if not)
            masks: Reset masks
            a: Actions to evaluate

        Returns:
            Tuple of (log_prob, entropy, value, updated_hidden_states)
        """
        z = self.forward_mlp(s)

        if self.enable_decoupled:
            # Separate processing for actor and critic
            z_shared, hxs_actor = self.forward_gru(z, hxs[0] if isinstance(hxs, tuple) else hxs, masks)

            # Actor branch
            z_actor = self.actor_branch(z_shared)
            logits = self.actor_head(z_actor)

            # Critic branch with separate GRU
            z_critic = self.critic_branch(z_shared)
            z_critic = z_critic.unsqueeze(0)  # (1, N, H)
            hxs_critic = hxs[1] if isinstance(hxs, tuple) else hxs
            if hxs_critic is not None and masks is not None:
                masks_expanded = masks.view(1, -1, 1)
                hxs_critic = hxs_critic * masks_expanded
            z_critic_out, hxs_critic = self.critic_gru(z_critic, hxs_critic)
            z_critic_out = z_critic_out.squeeze(0)  # (N, H)
            v = self.critic_head(z_critic_out).squeeze(-1)

            updated_hxs = (hxs_actor, hxs_critic)
        else:
            # Original shared processing
            z, updated_hxs = self.forward_gru(z, hxs, masks)
            logits = self.actor(z)
            v = self.critic(z).squeeze(-1)

        dist = Categorical(logits=logits)
        logp = dist.log_prob(a)
        entropy = dist.entropy().mean()
        return logp, entropy, v, updated_hxs

    def forward_actor_logits(
        self, s: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get raw actor logits for behavior cloning.

        Args:
            s: State tensor of shape (N, state_dim)
            hxs: Hidden states (tuple if decoupled, single tensor if not)
            masks: Reset masks

        Returns:
            Tuple of (logits, updated_hidden_states)
        """
        z = self.forward_mlp(s)

        if self.enable_decoupled:
            # Separate processing for actor and critic
            z_shared, hxs_actor = self.forward_gru(z, hxs[0] if isinstance(hxs, tuple) else hxs, masks)

            # Actor branch
            z_actor = self.actor_branch(z_shared)
            logits = self.actor_head(z_actor)

            # Return actor hidden state (for consistency with act_value)
            updated_hxs = hxs_actor
        else:
            # Original shared processing
            z, updated_hxs = self.forward_gru(z, hxs, masks)
            logits = self.actor(z)

        return logits, updated_hxs