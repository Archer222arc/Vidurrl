"""
Actor-Critic neural network architecture for PPO.

This module implements a multi-layer perceptron with GRU recurrent layers
for both policy (actor) and value (critic) functions.
"""

import math
from typing import Tuple, Optional

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


class CrossReplicaAttention(nn.Module):
    """
    Cross-replica attention mechanism as recommended in PDF.

    This module allows the policy to explicitly compare all replica states
    and pinpoint which one should be scheduled next, implementing a learned
    "least outstanding requests" strategy.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize cross-replica attention module.

        Args:
            feature_dim: Dimension of replica features
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # Multi-head attention components
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        # Normalization and regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim) if use_layer_norm else nn.Identity()

        # Initialize weights
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            init_layer(module, gain=1.0 / math.sqrt(2), use_orthogonal=True)

    def forward(self, replica_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-replica attention.

        Args:
            replica_features: Tensor of shape (batch_size, num_replicas, feature_dim)

        Returns:
            Attended features of shape (batch_size, num_replicas, feature_dim)
        """
        batch_size, num_replicas, feature_dim = replica_features.shape

        # Linear projections
        q = self.q_proj(replica_features)  # (B, N, D)
        k = self.k_proj(replica_features)  # (B, N, D)
        v = self.v_proj(replica_features)  # (B, N, D)

        # Reshape for multi-head attention
        q = q.view(batch_size, num_replicas, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D/H)
        k = k.view(batch_size, num_replicas, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D/H)
        v = v.view(batch_size, num_replicas, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D/H)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, N, N)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, N, D/H)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_replicas, feature_dim
        )  # (B, N, D)

        # Final projection and residual connection
        output = self.out_proj(attn_output)
        output = self.layer_norm(output + replica_features)  # Residual connection

        return output


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
        # NEW: Cross-replica attention parameters
        enable_cross_replica_attention: bool = True,
        num_replicas: int = 4,
        attention_heads: int = 4,
    ):
        """
        Initialize Enhanced Actor-Critic network with cross-replica attention.

        Args:
            state_dim: Dimension of state input
            action_dim: Number of possible actions
            hidden_size: Hidden layer size
            layer_N: Number of additional MLP layers (total: 1 + layer_N)
            gru_layers: Number of GRU layers
            use_orthogonal: Whether to use orthogonal initialization
            enable_decoupled: Enable decoupled Actor/Critic architecture
            feature_projection_dim: Dimension for feature projection layer (auto if None)
            enable_cross_replica_attention: Enable cross-replica attention mechanism (PDF recommendation)
            num_replicas: Number of replicas in the system
            attention_heads: Number of attention heads for cross-replica attention
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.layer_N = layer_N
        self.gru_layers = gru_layers
        self.enable_decoupled = enable_decoupled
        self.feature_projection_dim = feature_projection_dim or hidden_size * 2
        # NEW: Cross-replica attention parameters
        self.enable_cross_replica_attention = enable_cross_replica_attention
        self.num_replicas = num_replicas
        self.attention_heads = attention_heads

        # Feature projection layer for multi-dimensional state normalization
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, self.feature_projection_dim),
            nn.GELU(),
            nn.Linear(self.feature_projection_dim, hidden_size),
        )
        init_layer(self.feature_proj[1], gain=math.sqrt(2), use_orthogonal=use_orthogonal)
        init_layer(self.feature_proj[3], gain=math.sqrt(2), use_orthogonal=use_orthogonal)

        # NEW: Cross-replica attention mechanism (PDF recommendation)
        if self.enable_cross_replica_attention:
            # Calculate the correct feature dimension per replica
            replica_feature_dim = hidden_size // num_replicas
            self.cross_replica_attention = CrossReplicaAttention(
                feature_dim=replica_feature_dim,
                num_heads=attention_heads,
                dropout=0.1,
                use_layer_norm=True
            )

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
        Forward pass through feature projection, cross-replica attention, and shared MLP.

        Args:
            x: Input tensor of shape (N, state_dim)

        Returns:
            Encoded features of shape (N, hidden_size)
        """
        # Feature projection for multi-scale normalization
        x = self.feature_proj(x)

        # NEW: Cross-replica attention mechanism (PDF recommendation)
        if self.enable_cross_replica_attention:
            batch_size = x.shape[0]
            # Calculate the correct feature dimension per replica after projection
            replica_feature_dim = self.hidden_size // self.num_replicas

            # Always project to hidden_size first, then split by replicas
            if x.shape[1] == self.hidden_size:
                # Already projected to hidden_size, split by replicas
                x_replicas = x.view(batch_size, self.num_replicas, replica_feature_dim)
            else:
                # First project entire state to hidden_size, then split by replicas
                # This avoids the dimension mismatch when state_dim is not divisible by num_replicas
                x_projected = x  # x is already projected by feature_proj above
                x_replicas = x_projected.view(batch_size, self.num_replicas, replica_feature_dim)

            # Apply cross-replica attention
            x_attended = self.cross_replica_attention(x_replicas)  # (B, N, D)

            # Flatten back to (batch_size, hidden_size)
            # Option 1: Reshape to original hidden_size by concatenating replicas and projecting
            # Option 2: Mean pooling and then project back to hidden_size
            # We choose Option 1 for better information preservation
            x_flattened = x_attended.view(batch_size, -1)  # (B, N*D) = (B, num_replicas * replica_feature_dim) = (B, hidden_size)
            x = x_flattened  # Already the right size: hidden_size

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
            # Extract single layer for critic from multi-layer hxs
            if isinstance(hxs, tuple):
                hxs_critic = hxs[1]  # Use second element if tuple
            elif hxs.dim() == 3 and hxs.shape[0] > 1:
                hxs_critic = hxs[-1].unsqueeze(0)  # Use last layer for critic: (1, N, H)
            else:
                hxs_critic = hxs  # Use as-is if already single layer
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
            # Extract single layer for critic from multi-layer hxs
            if isinstance(hxs, tuple):
                hxs_critic = hxs[1]  # Use second element if tuple
            elif hxs.dim() == 3 and hxs.shape[0] > 1:
                hxs_critic = hxs[-1].unsqueeze(0)  # Use last layer for critic: (1, N, H)
            else:
                hxs_critic = hxs  # Use as-is if already single layer
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