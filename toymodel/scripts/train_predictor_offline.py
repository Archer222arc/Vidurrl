#!/usr/bin/env python3
"""
Train neural latency predictor using collected offline data.
"""

import os
import sys
import argparse
import pickle
from torch.utils.data import random_split, DataLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from toymodel.src.predictors import NeuralLatencyPredictor
from toymodel.src.training import PredictorTrainer, LatencyDataset


def load_dataset(data_path: str):
    """Load dataset from file."""
    print(f"Loading dataset from: {data_path}")

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    states = dataset['states']
    labels = dataset['labels']
    config = dataset['config']
    stats = dataset['stats']

    print(f"Dataset loaded successfully!")
    print(f"  Samples: {len(states)}")
    print(f"  State dim: {states.shape[1]}")
    print(f"  Label dim: {labels.shape[1]}")
    print(f"  Policy: {config['policy']}")
    print(f"\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return states, labels, config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train neural latency predictor'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='toymodel/data/latency_training_data.pkl',
        help='Path to training data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='toymodel/outputs/predictor_checkpoint.pt',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=128,
        help='Hidden layer dimension'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cpu/cuda)'
    )

    args = parser.parse_args()

    # Load dataset
    states, labels, config = load_dataset(args.data)

    # Create predictor
    predictor = NeuralLatencyPredictor(
        num_replicas=config['num_replicas'],
        num_request_types=config['num_request_types'],
        hidden_dim=args.hidden_dim,
        max_queue_obs=128
    )

    print(f"\nPredictor architecture:")
    print(f"  Input dim: {predictor.input_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Output dim: 2 (self_latency, avg_impact)")

    # Create dataset
    full_dataset = LatencyDataset(states, labels)

    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split:")
    print(f"  Train: {train_size} samples")
    print(f"  Val: {val_size} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create trainer
    trainer = PredictorTrainer(
        predictor,
        learning_rate=args.learning_rate,
        device=args.device
    )

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    train_stats = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        early_stop_patience=20
    )

    # Save model
    print(f"\nSaving trained model to: {args.output}")
    predictor.save_checkpoint(args.output, train_stats=train_stats)

    print("\n✅ Training completed successfully!")
    print(f"Best validation loss: {train_stats['best_val_loss']:.6f}")


if __name__ == '__main__':
    import torch
    main()
