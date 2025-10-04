#!/usr/bin/env python3
"""
Collect latency prediction training data using policy-independent simulation.
"""

import os
import sys
import json
import argparse
import pickle

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from toymodel.src.config import load_config
from toymodel.src.training import LatencyDataCollector


def save_dataset(collector: LatencyDataCollector, states, labels, output_path: str):
    """Save collected dataset."""
    dataset = {
        'states': states,
        'labels': labels,
        'config': {
            'num_replicas': collector.config.environment.num_replicas,
            'num_request_types': len(collector.config.environment.arrival_rates),
            'service_rates': collector.config.environment.service_rates,
            'arrival_rates': collector.config.environment.arrival_rates,
            'policy': collector.policy
        },
        'stats': collector.get_statistics()
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to: {output_path}")
    print(f"Dataset statistics:")
    for key, value in dataset['stats'].items():
        print(f"  {key}: {value}")


def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(
        description='Collect latency prediction training data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='toymodel/configs/ppo_config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--policy',
        type=str,
        default='mixed',
        choices=['random', 'round_robin', 'mixed'],
        help='Data collection policy'
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=100,
        help='Number of episodes to collect'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='toymodel/data/latency_training_data.pkl',
        help='Output path for dataset'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create data collector
    collector = LatencyDataCollector(config, policy=args.policy)

    # Collect data
    states, labels = collector.collect_data(args.num_episodes)

    # Save dataset
    save_dataset(collector, states, labels, args.output)

    print("\n✅ Data collection completed successfully!")


if __name__ == '__main__':
    main()
