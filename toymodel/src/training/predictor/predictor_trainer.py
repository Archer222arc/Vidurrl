"""
Trainer for neural latency predictor.

Implements supervised learning training loop with validation and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple

from ...predictors import NeuralLatencyPredictor


class LatencyDataset(Dataset):
    """PyTorch dataset for latency prediction."""

    def __init__(self, states: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.

        Args:
            states: State features (N, state_dim)
            labels: Latency labels (N, 2) - [self_latency, avg_impact]
        """
        self.states = torch.from_numpy(states).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.labels[idx]


class PredictorTrainer:
    """Trainer for neural latency predictor."""

    def __init__(
        self,
        predictor: NeuralLatencyPredictor,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu"
    ):
        """
        Initialize trainer.

        Args:
            predictor: Neural latency predictor
            learning_rate: Learning rate
            weight_decay: L2 regularization weight
            device: Device for training
        """
        self.predictor = predictor
        self.device = device
        self.predictor.model.to(device)
        self.predictor.model.train()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.predictor.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        # Loss function (MSE for regression)
        self.criterion = nn.MSELoss()

        # Training statistics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.predictor.model.train()
        total_loss = 0.0
        num_batches = 0

        for states, labels in train_loader:
            states = states.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            predictions = self.predictor.model(states)

            # Compute loss
            loss = self.criterion(predictions, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.predictor.model.parameters(), 
                max_norm=1.0
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.predictor.model.eval()
        total_loss = 0.0
        total_mse_self = 0.0
        total_mse_impact = 0.0
        num_batches = 0

        with torch.no_grad():
            for states, labels in val_loader:
                states = states.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                predictions = self.predictor.model(states)

                # Compute loss
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()

                # Per-component MSE
                mse_self = torch.mean((predictions[:, 0] - labels[:, 0]) ** 2).item()
                mse_impact = torch.mean((predictions[:, 1] - labels[:, 1]) ** 2).item()
                total_mse_self += mse_self
                total_mse_impact += mse_impact

                num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'mse_self_latency': total_mse_self / num_batches,
            'mse_impact': total_mse_impact / num_batches
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        early_stop_patience: int = 20
    ) -> Dict[str, Any]:
        """
        Train predictor.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            early_stop_patience: Early stopping patience

        Returns:
            Training statistics
        """
        print(f"Training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.predictor.model.parameters())}")

        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  Val MSE (self): {val_metrics['mse_self_latency']:.6f}")
                print(f"  Val MSE (impact): {val_metrics['mse_impact']:.6f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best validation loss: {self.best_val_loss:.6f}")
                break

        # Return statistics
        return {
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
