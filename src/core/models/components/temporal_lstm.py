"""
Temporal LSTM Component

Modular temporal LSTM component for sequence pattern modeling
between feature groups. Can be integrated into any architecture.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class TemporalLSTMConfig:
    """Configuration for Temporal LSTM component."""

    def __init__(
        self,
        enable: bool = False,
        feature_chunks: int = 4,
        hidden_size_ratio: float = 0.25,
        bidirectional: bool = True,
        residual_connections: bool = True,
        dropout: float = 0.0,
        use_layer_norm: bool = True
    ):
        self.enable = enable
        self.feature_chunks = feature_chunks
        self.hidden_size_ratio = hidden_size_ratio
        self.bidirectional = bidirectional
        self.residual_connections = residual_connections
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # Validation
        if self.enable:
            assert self.feature_chunks > 0, "feature_chunks must be > 0"
            assert 0 < self.hidden_size_ratio <= 1, "hidden_size_ratio must be in (0, 1]"


def init_layer(layer: nn.Module, gain: float = 1.0, use_orthogonal: bool = True) -> None:
    """Initialize layer weights."""
    if isinstance(layer, nn.Linear):
        if use_orthogonal:
            nn.init.orthogonal_(layer.weight, gain=gain)
        else:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


class TemporalLSTM(nn.Module):
    """
    Temporal LSTM component for sequence pattern modeling.

    This component splits input features into chunks and processes them
    as a temporal sequence, allowing the model to capture dependencies
    between different feature groups.
    """

    def __init__(self, hidden_size: int, config: TemporalLSTMConfig):
        super().__init__()

        self.config = config
        self.hidden_size = hidden_size

        if not config.enable:
            # If disabled, this component becomes a no-op
            return

        # Calculate dimensions
        self.feature_chunks = config.feature_chunks
        self.chunk_size = hidden_size // self.feature_chunks
        self.lstm_hidden_size = max(int(self.chunk_size * config.hidden_size_ratio), 32)

        # Ensure chunk_size is valid
        if self.chunk_size * self.feature_chunks != hidden_size:
            # Adjust to make it divisible
            self.chunk_size = hidden_size // self.feature_chunks
            self.adjusted_input_size = self.chunk_size * self.feature_chunks
        else:
            self.adjusted_input_size = hidden_size

        # Create LSTM
        self.lstm = nn.LSTM(
            input_size=self.chunk_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.dropout > 0 else 0.0
        )

        # Calculate output size after LSTM
        lstm_output_size = self.lstm_hidden_size * (2 if config.bidirectional else 1)
        total_lstm_output = self.feature_chunks * lstm_output_size

        # Back projection to original hidden_size
        projection_layers = [nn.Linear(total_lstm_output, hidden_size)]

        if config.use_layer_norm:
            projection_layers.append(nn.LayerNorm(hidden_size))

        projection_layers.append(nn.GELU())

        self.temporal_projection = nn.Sequential(*projection_layers)

        # Back projection layer (for compatibility with existing checkpoints)
        self.temporal_back_projection = nn.Linear(total_lstm_output, hidden_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all component weights."""
        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        # Initialize projection layers
        for layer in self.temporal_projection:
            if isinstance(layer, nn.Linear):
                init_layer(layer, gain=1.0, use_orthogonal=True)

        # Initialize back projection
        init_layer(self.temporal_back_projection, gain=1.0, use_orthogonal=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal LSTM processing.

        Args:
            x: Input tensor of shape (batch_size, hidden_size)

        Returns:
            Processed tensor of shape (batch_size, hidden_size)
        """
        if not self.config.enable:
            return x

        batch_size = x.shape[0]

        # Handle input size adjustment if needed
        if x.shape[1] != self.adjusted_input_size:
            # Truncate or pad to match expected size
            if x.shape[1] > self.adjusted_input_size:
                x_adjusted = x[:, :self.adjusted_input_size]
            else:
                padding = torch.zeros(batch_size, self.adjusted_input_size - x.shape[1], device=x.device)
                x_adjusted = torch.cat([x, padding], dim=1)
        else:
            x_adjusted = x

        # Store original for residual connection
        x_original = x

        # Split into chunks for temporal processing
        x_chunks = x_adjusted.view(batch_size, self.feature_chunks, self.chunk_size)

        # Apply LSTM
        lstm_out, _ = self.lstm(x_chunks)  # (batch_size, feature_chunks, lstm_hidden_size * directions)

        # Flatten LSTM output
        lstm_flat = lstm_out.contiguous().view(batch_size, -1)

        # Project back to original size using the new projection layer
        temporal_features = self.temporal_projection(lstm_flat)

        # Apply residual connection if enabled
        if self.config.residual_connections:
            # Ensure shapes match for residual connection
            if temporal_features.shape[1] == x_original.shape[1]:
                output = x_original + temporal_features
            else:
                # If shapes don't match, use projection only
                output = temporal_features
        else:
            output = temporal_features

        return output

    def get_component_info(self) -> dict:
        """Get information about this component."""
        if not self.config.enable:
            return {
                "enabled": False,
                "parameters": 0
            }

        return {
            "enabled": True,
            "feature_chunks": self.feature_chunks,
            "chunk_size": self.chunk_size,
            "lstm_hidden_size": self.lstm_hidden_size,
            "bidirectional": self.config.bidirectional,
            "residual_connections": self.config.residual_connections,
            "parameters": sum(p.numel() for p in self.parameters()),
            "lstm_parameters": sum(p.numel() for p in self.lstm.parameters()),
            "projection_parameters": sum(p.numel() for p in self.temporal_projection.parameters())
        }


class TemporalLSTMFactory:
    """Factory for creating TemporalLSTM components from configuration."""

    @staticmethod
    def from_dict(config_dict: dict, hidden_size: int) -> TemporalLSTM:
        """
        Create TemporalLSTM from configuration dictionary.

        Args:
            config_dict: Configuration dictionary with temporal_lstm settings
            hidden_size: Hidden size of the parent model

        Returns:
            TemporalLSTM component
        """
        temporal_config = TemporalLSTMConfig(
            enable=config_dict.get("enable", False),
            feature_chunks=config_dict.get("feature_chunks", 4),
            hidden_size_ratio=config_dict.get("hidden_size_ratio", 0.25),
            bidirectional=config_dict.get("bidirectional", True),
            residual_connections=config_dict.get("residual_connections", True),
            dropout=config_dict.get("dropout", 0.0),
            use_layer_norm=config_dict.get("use_layer_norm", True)
        )

        return TemporalLSTM(hidden_size, temporal_config)

    @staticmethod
    def from_ppo_config(ppo_config, hidden_size: int) -> TemporalLSTM:
        """
        Create TemporalLSTM from PPO configuration object.

        Args:
            ppo_config: PPO configuration object with temporal LSTM attributes
            hidden_size: Hidden size of the parent model

        Returns:
            TemporalLSTM component
        """
        temporal_config = TemporalLSTMConfig(
            enable=getattr(ppo_config, 'enable_temporal_lstm', False),
            feature_chunks=getattr(ppo_config, 'temporal_lstm_feature_chunks', 4),
            hidden_size_ratio=getattr(ppo_config, 'temporal_lstm_hidden_ratio', 0.25),
            bidirectional=getattr(ppo_config, 'temporal_lstm_bidirectional', True),
            residual_connections=True,  # Always enable for compatibility
            dropout=0.0,  # No dropout for single layer
            use_layer_norm=True
        )

        return TemporalLSTM(hidden_size, temporal_config)