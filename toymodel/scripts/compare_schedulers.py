#!/usr/bin/env python3
"""
Compare all scheduler types on specified config.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

from toymodel.src.config import load_config
from toymodel.src.environment import QueueEnvironment
from toymodel.schedulers import (
    OracleScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    ShortestQueueScheduler,
    PPOScheduler,
)


def compare_schedulers(config_path: str):
    """Compare all schedulers on given config."""
    # Load config
    config = load_config(config_path)

    print(f"Experiment: {config.experiment.name}")
    print(f"Description: {config.experiment.description}")
    print(f"Arrival rates: Type A={config.environment.arrival_rates[0]}, Type B={config.environment.arrival_rates[1]}")
    print(f"Simulation time: {config.environment.max_time}")
    print()

    # Schedulers to compare
    schedulers = [
        ('Oracle', OracleScheduler(num_replicas=2)),
        ('Random', RandomScheduler(num_replicas=2, seed=config.experiment.seed)),
        ('Round-Robin', RoundRobinScheduler(num_replicas=2)),
        ('Shortest Queue', ShortestQueueScheduler(num_replicas=2)),
    ]
    
    # Add PPO scheduler if model exists
    ppo_model_path = "toymodel/outputs/models/ppo_model_latest.pt"
    if Path(ppo_model_path).exists():
        schedulers.append(('PPO', PPOScheduler(
            num_replicas=2,
            model_path=ppo_model_path,
            n_requests=3
        )))
        print(f"Found PPO model: {ppo_model_path}")
    else:
        print(f"PPO model not found: {ppo_model_path} (skipping PPO comparison)")

    results = []

    for name, scheduler in schedulers:
        # Create environment (disable tensorboard for comparison to avoid conflicts)
        env = QueueEnvironment(
            num_replicas=config.environment.num_replicas,
            service_rates=config.environment.service_rates,
            arrival_rates=config.environment.arrival_rates,
            max_time=config.environment.max_time,
            seed=config.experiment.seed,
            tensorboard_enabled=False,
        )

        # Run simulation
        def routing_policy(request, replicas):
            return scheduler.schedule(request, replicas)

        completed = env.run_simulation(routing_policy)
        env.close()

        # Compute metrics
        latencies = [r.total_time for r in completed]
        queue_times = [r.queue_time for r in completed]
        routing_accuracy = sum(
            1 for r in completed if r.assigned_replica == r.request_type
        ) / len(completed) * 100

        results.append({
            'name': name,
            'num_requests': len(completed),
            'mean_latency': np.mean(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p99_latency': np.percentile(latencies, 99),
            'mean_queue_time': np.mean(queue_times),
            'routing_accuracy': routing_accuracy,
        })

    # Print comparison table
    print('Scheduler Comparison Results:')
    print('-' * 100)
    print(f"{'Scheduler':<15} {'Requests':<10} {'Mean Lat':<10} {'P50 Lat':<10} {'P99 Lat':<10} {'Queue Time':<12} {'Accuracy':<10}")
    print('-' * 100)

    for r in results:
        print(f"{r['name']:<15} {r['num_requests']:<10} {r['mean_latency']:<10.4f} {r['p50_latency']:<10.4f} {r['p99_latency']:<10.4f} {r['mean_queue_time']:<12.4f} {r['routing_accuracy']:<10.1f}%")

    print('-' * 100)

    # Find best scheduler (excluding Oracle)
    non_oracle = [r for r in results if r['name'] != 'Oracle']
    best = min(non_oracle, key=lambda x: x['mean_latency'])
    oracle = [r for r in results if r['name'] == 'Oracle'][0]

    print(f"\nBest non-Oracle scheduler: {best['name']} (latency: {best['mean_latency']:.4f})")
    print(f"Oracle performance: {oracle['mean_latency']:.4f}")
    print(f"Gap to Oracle: {(best['mean_latency'] / oracle['mean_latency'] - 1) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Compare scheduler performance")
    parser.add_argument(
        "config",
        type=str,
        nargs='?',
        default="toymodel/configs/config.json",
        help="Path to config file (default: toymodel/configs/config.json)"
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    print("=" * 60)
    compare_schedulers(args.config)
    print("=" * 60)


if __name__ == "__main__":
    main()
