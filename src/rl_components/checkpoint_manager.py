"""
Checkpoint management for PPO training.

This module provides model saving, loading, and inference-only mode support
for PPO training continuity and deployment.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import numpy as np

from .actor_critic import ActorCritic
from .normalizers import RunningNormalizer


class CheckpointManager:
    """
    Manages PPO model checkpoints and training state.

    Provides functionality for saving/loading model weights, training state,
    and supporting inference-only deployment mode.
    """

    def __init__(
        self,
        checkpoint_dir: str = "./outputs/checkpoints",
        save_interval: int = 100,
        max_checkpoints: int = 5,
        auto_save: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            save_interval: Steps between automatic saves
            max_checkpoints: Maximum number of checkpoints to keep
            auto_save: Whether to automatically save during training
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.auto_save = auto_save

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track save state
        self.last_save_step = 0
        self.saved_checkpoints = []

    def should_save(self, step: int) -> bool:
        """
        Check if checkpoint should be saved at current step.

        Args:
            step: Current training step

        Returns:
            True if checkpoint should be saved
        """
        if not self.auto_save:
            return False

        return (step - self.last_save_step) >= self.save_interval

    def save_checkpoint(
        self,
        step: int,
        actor_critic: ActorCritic,
        normalizer: RunningNormalizer,
        training_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save complete training checkpoint.

        Args:
            step: Current training step
            actor_critic: Actor-critic network to save
            normalizer: State normalizer to save
            training_state: Training state dictionary
            metadata: Additional metadata to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_step_{step:08d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Prepare checkpoint data
        checkpoint_data = {
            "step": step,
            "model_state_dict": actor_critic.state_dict(),
            "normalizer_state": self._serialize_normalizer(normalizer),
            "training_state": training_state,
            "metadata": metadata or {},
            "model_config": {
                "state_dim": actor_critic.state_dim,
                "action_dim": actor_critic.action_dim,
                "hidden_size": actor_critic.hidden_size,
                "layer_N": actor_critic.layer_N,
                "gru_layers": actor_critic.gru_layers,
                "enable_decoupled": getattr(actor_critic, 'enable_decoupled', False),
                "feature_projection_dim": getattr(actor_critic, 'feature_projection_dim', None),
            },
        }

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)

        # Update tracking
        self.last_save_step = step
        self.saved_checkpoints.append(checkpoint_path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        # Save latest checkpoint link
        self._update_latest_link(checkpoint_path)

        print(f"💾 Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        load_latest: bool = True,
    ) -> Dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Specific checkpoint path to load
            load_latest: Whether to load latest checkpoint if path not specified

        Returns:
            Loaded checkpoint data

        Raises:
            FileNotFoundError: If checkpoint not found
            RuntimeError: If checkpoint loading fails
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            if load_latest:
                checkpoint_path = self._get_latest_checkpoint()
            else:
                raise ValueError("Must specify checkpoint_path or set load_latest=True")

        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            print(f"📂 Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")

    def create_actor_critic_from_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        device: str = "cpu",
    ) -> ActorCritic:
        """
        Create and load Actor-Critic from checkpoint.

        Args:
            checkpoint_data: Loaded checkpoint data
            device: Device to load model on

        Returns:
            Loaded Actor-Critic network
        """
        model_config = checkpoint_data["model_config"]

        actor_critic = ActorCritic(
            state_dim=model_config["state_dim"],
            action_dim=model_config["action_dim"],
            hidden_size=model_config["hidden_size"],
            layer_N=model_config["layer_N"],
            gru_layers=model_config["gru_layers"],
            enable_decoupled=model_config.get("enable_decoupled", False),
            feature_projection_dim=model_config.get("feature_projection_dim", None),
        ).to(device)

        actor_critic.load_state_dict(checkpoint_data["model_state_dict"])
        return actor_critic

    def create_normalizer_from_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
    ) -> RunningNormalizer:
        """
        Create and load normalizer from checkpoint.

        Args:
            checkpoint_data: Loaded checkpoint data

        Returns:
            Loaded normalizer
        """
        normalizer = RunningNormalizer()
        self._deserialize_normalizer(normalizer, checkpoint_data["normalizer_state"])
        return normalizer

    def list_checkpoints(self) -> list[str]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint paths sorted by step number
        """
        checkpoints = []
        for path in self.checkpoint_dir.glob("checkpoint_step_*.pt"):
            checkpoints.append(str(path))

        # Sort by step number
        checkpoints.sort(key=lambda x: int(Path(x).stem.split("_")[-1]))
        return checkpoints

    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint without loading full model.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Checkpoint metadata and info
        """
        try:
            # Load only metadata
            checkpoint_data = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

            info = {
                "step": checkpoint_data.get("step", "unknown"),
                "metadata": checkpoint_data.get("metadata", {}),
                "model_config": checkpoint_data.get("model_config", {}),
                "file_size": Path(checkpoint_path).stat().st_size,
                "created_time": Path(checkpoint_path).stat().st_mtime,
            }

            return info

        except Exception as e:
            return {"error": str(e)}

    def _serialize_normalizer(self, normalizer: RunningNormalizer) -> Dict[str, Any]:
        """Serialize normalizer state to dictionary."""
        return {
            "eps": normalizer.eps,
            "clip": normalizer.clip,
            "count": normalizer.count,
            "mean": normalizer.mean.tolist() if normalizer.mean is not None else None,
            "m2": normalizer.m2.tolist() if normalizer.m2 is not None else None,
        }

    def _deserialize_normalizer(
        self, normalizer: RunningNormalizer, state: Dict[str, Any]
    ) -> None:
        """Deserialize normalizer state from dictionary."""
        normalizer.eps = state["eps"]
        normalizer.clip = state["clip"]
        normalizer.count = state["count"]
        normalizer.mean = np.array(state["mean"]) if state["mean"] is not None else None
        normalizer.m2 = np.array(state["m2"]) if state["m2"] is not None else None

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.saved_checkpoints) <= self.max_checkpoints:
            return

        # Remove oldest checkpoints
        checkpoints_to_remove = self.saved_checkpoints[:-self.max_checkpoints]
        for checkpoint_path in checkpoints_to_remove:
            try:
                Path(checkpoint_path).unlink()
                print(f"🗑️  Removed old checkpoint: {checkpoint_path}")
            except FileNotFoundError:
                pass  # Already removed

        # Update tracking
        self.saved_checkpoints = self.saved_checkpoints[-self.max_checkpoints:]

    def _update_latest_link(self, checkpoint_path: Path) -> None:
        """Update 'latest' symlink to point to newest checkpoint."""
        latest_link = self.checkpoint_dir / "latest.pt"

        try:
            # Remove existing link
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()

            # Create new link
            latest_link.symlink_to(checkpoint_path.name)

        except Exception:
            # Fallback: copy file if symlinks not supported
            try:
                shutil.copy2(checkpoint_path, latest_link)
            except Exception:
                pass  # Best effort

    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        latest_link = self.checkpoint_dir / "latest.pt"

        if latest_link.exists():
            return str(latest_link)

        # Fallback: find latest by filename
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None


class InferenceMode:
    """
    Provides inference-only mode for trained PPO models.

    Loads a trained model checkpoint and disables training,
    suitable for deployment and evaluation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        """
        Initialize inference mode.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device for inference
        """
        self.device = device
        self.checkpoint_manager = CheckpointManager()

        # Load checkpoint
        self.checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        # Create model components
        self.actor_critic = self.checkpoint_manager.create_actor_critic_from_checkpoint(
            self.checkpoint_data, device
        )
        self.normalizer = self.checkpoint_manager.create_normalizer_from_checkpoint(
            self.checkpoint_data
        )

        # Set to evaluation mode
        self.actor_critic.eval()

        # Initialize hidden state
        self.hidden_state = torch.zeros(
            self.actor_critic.gru_layers, 1, self.actor_critic.hidden_size, device=device
        )

        print(f"🎯 Inference mode initialized from step {self.checkpoint_data['step']}")

    def predict_action(
        self, state: np.ndarray, deterministic: bool = True
    ) -> tuple[int, torch.Tensor]:
        """
        Predict action for given state.

        Args:
            state: State observation
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action_index, updated_hidden_state)
        """
        # Normalize state
        state_norm = self.normalizer.normalize(state)
        state_tensor = torch.from_numpy(state_norm).float().unsqueeze(0).to(self.device)

        # Predict action
        with torch.no_grad():
            mask = torch.ones(1, 1, device=self.device)  # No reset

            if deterministic:
                # Use policy mean (argmax of logits)
                z = self.actor_critic.forward_mlp(state_tensor)
                z, self.hidden_state = self.actor_critic.forward_gru(z, self.hidden_state, mask)
                logits = self.actor_critic.actor(z)
                action = torch.argmax(logits, dim=-1)
            else:
                # Sample from policy
                action, _, _, self.hidden_state = self.actor_critic.act_value(
                    state_tensor, self.hidden_state, mask
                )

        return int(action.item()), self.hidden_state

    def reset_hidden_state(self) -> None:
        """Reset hidden state for new episode."""
        self.hidden_state = torch.zeros(
            self.actor_critic.gru_layers, 1, self.actor_critic.hidden_size, device=self.device
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        return {
            "step": self.checkpoint_data["step"],
            "model_config": self.checkpoint_data["model_config"],
            "metadata": self.checkpoint_data.get("metadata", {}),
        }