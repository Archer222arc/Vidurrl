"""
Explicit Architecture Actor-Critic Network

This module provides a clean, explicit architecture where all components
are clearly defined by configuration parameters, eliminating implicit
dependencies and runtime surprises.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ActorCriticConfig:
    """Explicit configuration for Actor-Critic architecture."""

    # Basic parameters
    state_dim: int
    action_dim: int
    hidden_size: int = 128
    layer_N: int = 2
    gru_layers: int = 2
    use_orthogonal: bool = True

    # Architecture components - explicit control
    enable_decoupled: bool = True
    feature_projection_dim: Optional[int] = None

    # Cross-replica attention - explicit control
    enable_cross_replica_attention: bool = False
    num_replicas: int = 4
    attention_heads: int = 4

    # Temporal LSTM - explicit control
    enable_temporal_lstm: bool = False
    temporal_feature_chunks: int = 4
    temporal_hidden_ratio: float = 0.25
    temporal_bidirectional: bool = True

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.enable_cross_replica_attention:
            assert self.num_replicas > 0, "num_replicas must be > 0 when attention enabled"
            assert self.hidden_size % self.attention_heads == 0, "hidden_size must be divisible by attention_heads"

        if self.enable_temporal_lstm:
            assert self.temporal_feature_chunks > 0, "temporal_feature_chunks must be > 0"
            assert 0 < self.temporal_hidden_ratio <= 1, "temporal_hidden_ratio must be in (0, 1]"

        if self.feature_projection_dim is None:
            self.feature_projection_dim = self.hidden_size * 2


def init_layer(layer: nn.Module, gain: float = 1.0, use_orthogonal: bool = True) -> None:
    """Initialize layer weights with specified method."""
    if isinstance(layer, (nn.Linear, nn.Conv1d)):
        if use_orthogonal:
            nn.init.orthogonal_(layer.weight, gain=gain)
        else:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


class CrossReplicaAttention(nn.Module):
    """Cross-replica attention mechanism for load balancing awareness."""

    def __init__(self, config: ActorCriticConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.hidden_size
        self.num_heads = config.attention_heads
        self.head_dim = self.feature_dim // self.num_heads

        assert self.feature_dim % self.num_heads == 0, "feature_dim must be divisible by num_heads"

        # Multi-head attention components
        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.k_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.v_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.out_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(self.feature_dim)

        # Initialize weights
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            init_layer(layer, gain=0.1, use_orthogonal=config.use_orthogonal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-replica attention."""
        batch_size = x.shape[0]

        # Reshape to separate replica features: (N, num_replicas, feature_dim)
        x_reshaped = x.view(batch_size, self.config.num_replicas, -1)

        # Apply multi-head attention
        q = self.q_proj(x_reshaped)
        k = self.k_proj(x_reshaped)
        v = self.v_proj(x_reshaped)

        # Reshape for multi-head attention: (N, num_heads, num_replicas, head_dim)
        q = q.view(batch_size, self.config.num_replicas, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, self.config.num_replicas, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, self.config.num_replicas, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (N, num_replicas, feature_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, self.config.num_replicas, self.feature_dim
        )

        # Output projection and residual connection
        output = self.out_proj(attn_output)
        output = self.layer_norm(x_reshaped + output)

        # Flatten back to original shape
        return output.view(batch_size, -1)


class TemporalLSTM(nn.Module):
    """Temporal LSTM for sequence pattern modeling."""

    def __init__(self, config: ActorCriticConfig):
        super().__init__()
        self.config = config
        self.feature_chunks = config.temporal_feature_chunks

        # Calculate LSTM dimensions
        self.chunk_size = config.hidden_size // self.feature_chunks
        self.lstm_hidden_size = max(int(self.chunk_size * config.temporal_hidden_ratio), 32)

        # Create LSTM
        self.lstm = nn.LSTM(
            input_size=self.chunk_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=config.temporal_bidirectional,
            dropout=0.0
        )

        # Back projection to original hidden size
        lstm_output_size = self.lstm_hidden_size * (2 if config.temporal_bidirectional else 1)
        total_output_size = self.feature_chunks * lstm_output_size

        self.back_projection = nn.Sequential(
            nn.Linear(total_output_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )

        # Initialize weights
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        init_layer(self.back_projection[0], gain=1.0, use_orthogonal=config.use_orthogonal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal LSTM processing."""
        batch_size = x.shape[0]

        # Split into chunks for temporal processing
        x_chunks = x.view(batch_size, self.feature_chunks, self.chunk_size)

        # Apply LSTM
        lstm_out, _ = self.lstm(x_chunks)

        # Flatten and project back
        lstm_flat = lstm_out.contiguous().view(batch_size, -1)
        temporal_features = self.back_projection(lstm_flat)

        # Residual connection
        return x + temporal_features


class ExplicitActorCritic(nn.Module):
    """
    Explicit Architecture Actor-Critic Network.

    All components are clearly defined by configuration, eliminating
    implicit dependencies and runtime surprises.
    """

    def __init__(self, config: ActorCriticConfig):
        super().__init__()

        # Validate and store configuration
        config.validate()
        self.config = config

        # Basic properties
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.hidden_size = config.hidden_size
        self.layer_N = config.layer_N
        self.gru_layers = config.gru_layers

        # Feature projection layer
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.feature_projection_dim),
            nn.GELU(),
            nn.Linear(config.feature_projection_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )

        # Initialize feature projection
        for layer in self.feature_proj:
            if isinstance(layer, nn.Linear):
                init_layer(layer, gain=1.0, use_orthogonal=config.use_orthogonal)

        # Optional: Cross-replica attention
        self.cross_replica_attention = None
        if config.enable_cross_replica_attention:
            self.cross_replica_attention = CrossReplicaAttention(config)

        # Optional: Temporal LSTM
        self.temporal_lstm = None
        if config.enable_temporal_lstm:
            self.temporal_lstm = TemporalLSTM(config)

        # Shared GRU layers
        self.shared_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=0.1 if config.gru_layers > 1 else 0.0
        )
        self.gru_ln = nn.LayerNorm(config.hidden_size)

        # Initialize GRU
        for name, param in self.shared_gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        if config.enable_decoupled:
            # Decoupled architecture: separate actor and critic paths
            self._build_decoupled_heads(config)
        else:
            # Shared architecture: common processing with separate heads
            self._build_shared_heads(config)

    def _build_decoupled_heads(self, config: ActorCriticConfig):
        """Build decoupled actor and critic heads."""
        # Minimal shared processing
        self.shared_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )
        init_layer(self.shared_mlp[0], gain=1.0, use_orthogonal=config.use_orthogonal)

        # Separate Actor path
        actor_layers = []
        current_size = config.hidden_size
        for i in range(config.layer_N):
            actor_layers.extend([
                nn.Linear(current_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU()
            ])
        actor_layers.append(nn.Linear(config.hidden_size, config.action_dim))

        self.actor = nn.Sequential(*actor_layers)

        # Separate Critic path
        critic_layers = []
        current_size = config.hidden_size
        for i in range(config.layer_N):
            critic_layers.extend([
                nn.Linear(current_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU()
            ])
        critic_layers.append(nn.Linear(config.hidden_size, 1))

        self.critic = nn.Sequential(*critic_layers)

        # Initialize actor and critic
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    if layer == self.actor[-1]:  # Actor output layer
                        init_layer(layer, gain=0.1, use_orthogonal=config.use_orthogonal)
                    elif layer == self.critic[-1]:  # Critic output layer
                        init_layer(layer, gain=1.0, use_orthogonal=config.use_orthogonal)
                    else:  # Hidden layers
                        init_layer(layer, gain=1.0, use_orthogonal=config.use_orthogonal)

    def _build_shared_heads(self, config: ActorCriticConfig):
        """Build shared architecture with separate output heads."""
        # Shared MLP layers
        shared_layers = []
        current_size = config.hidden_size
        for i in range(config.layer_N):
            shared_layers.extend([
                nn.Linear(current_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU()
            ])

        self.shared_mlp = nn.Sequential(*shared_layers)

        # Separate output heads
        self.actor = nn.Linear(config.hidden_size, config.action_dim)
        self.critic = nn.Linear(config.hidden_size, 1)

        # Initialize layers
        for layer in self.shared_mlp:
            if isinstance(layer, nn.Linear):
                init_layer(layer, gain=1.0, use_orthogonal=config.use_orthogonal)

        init_layer(self.actor, gain=0.1, use_orthogonal=config.use_orthogonal)
        init_layer(self.critic, gain=1.0, use_orthogonal=config.use_orthogonal)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> tuple:
        """
        Forward pass through the network.

        Args:
            x: Input state tensor
            state: Optional hidden state for GRU

        Returns:
            (action_logits, value, new_state)
        """
        batch_size = x.shape[0]

        # Feature projection
        x = self.feature_proj(x)

        # Optional: Cross-replica attention
        if self.cross_replica_attention is not None:
            x = self.cross_replica_attention(x)

        # GRU processing
        if state is not None:
            x_gru, new_state = self.shared_gru(x.unsqueeze(1), state)
            x = x_gru.squeeze(1)
        else:
            x_gru, new_state = self.shared_gru(x.unsqueeze(1))
            x = x_gru.squeeze(1)

        x = self.gru_ln(x)

        # Optional: Temporal LSTM (only for decoupled architecture)
        if self.temporal_lstm is not None and self.config.enable_decoupled:
            # Minimal shared processing first
            x = self.shared_mlp(x)
            # Then temporal processing
            x = self.temporal_lstm(x)
        elif self.config.enable_decoupled:
            # Decoupled without temporal
            x = self.shared_mlp(x)
        else:
            # Shared architecture
            x = self.shared_mlp(x)

        # Generate outputs
        action_logits = self.actor(x)
        value = self.critic(x)

        return action_logits, value.squeeze(-1), new_state

    def get_architecture_info(self) -> Dict[str, Any]:
        """Get detailed architecture information for debugging."""
        info = {
            "config": self.config.__dict__,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "components": {
                "feature_projection": True,
                "cross_replica_attention": self.cross_replica_attention is not None,
                "temporal_lstm": self.temporal_lstm is not None,
                "architecture_type": "decoupled" if self.config.enable_decoupled else "shared"
            }
        }

        # Get component parameter counts
        if self.cross_replica_attention:
            info["cross_replica_attention_params"] = sum(p.numel() for p in self.cross_replica_attention.parameters())

        if self.temporal_lstm:
            info["temporal_lstm_params"] = sum(p.numel() for p in self.temporal_lstm.parameters())

        return info