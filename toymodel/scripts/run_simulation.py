#!/usr/bin/env python3
"""
Run toy model simulation with specified config file.
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
)


def run_simulation(config_path: str):
    """Run simulation with given config."""
    # Load config
    config = load_config(config_path)

    print(f"Experiment: {config.experiment.name}")
    print(f"Description: {config.experiment.description}")
    print()

    # Create environment
    env = QueueEnvironment(
        num_replicas=config.environment.num_replicas,
        service_rates=config.environment.service_rates,
        arrival_rates=config.environment.arrival_rates,
        max_time=config.environment.max_time,
        seed=config.experiment.seed,
        tensorboard_enabled=config.tensorboard.enabled,
        tensorboard_log_dir=config.tensorboard.log_dir,
    )

    # Get scheduler
    scheduler_type = config.scheduler.type
    scheduler_map = {
        'oracle': OracleScheduler(num_replicas=2),
        'random': RandomScheduler(num_replicas=2, seed=config.experiment.seed),
        'round_robin': RoundRobinScheduler(num_replicas=2),
        'shortest_queue': ShortestQueueScheduler(num_replicas=2),
    }

    if scheduler_type not in scheduler_map:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    scheduler = scheduler_map[scheduler_type]

    # Run simulation
    def routing_policy(request, replicas):
        return scheduler.schedule(request, replicas)

    completed = env.run_simulation(routing_policy)

    # Close environment (cleanup tensorboard)
    env.close()

    # Compute metrics
    latencies = [r.total_time for r in completed]
    queue_times = [r.queue_time for r in completed]
    routing_accuracy = sum(
        1 for r in completed if r.assigned_replica == r.request_type
    ) / len(completed) * 100

    print(f"Scheduler: {scheduler_type}")
    print(f"Requests completed: {len(completed)}")
    print(f"Mean latency: {np.mean(latencies):.4f}")
    print(f"P50 latency: {np.percentile(latencies, 50):.4f}")
    print(f"P99 latency: {np.percentile(latencies, 99):.4f}")
    print(f"Mean queue time: {np.mean(queue_times):.4f}")
    print(f"Routing accuracy: {routing_accuracy:.1f}%")

    # Save metrics if configured
    if config.metrics.save_csv:
        import os
        import csv
        from datetime import datetime

        os.makedirs(os.path.dirname(config.metrics.csv_path), exist_ok=True)

        with open(config.metrics.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['experiment_name', config.experiment.name])
            writer.writerow(['scheduler_type', scheduler_type])
            writer.writerow(['num_requests', len(completed)])
            writer.writerow(['mean_latency', np.mean(latencies)])
            writer.writerow(['p50_latency', np.percentile(latencies, 50)])
            writer.writerow(['p99_latency', np.percentile(latencies, 99)])
            writer.writerow(['mean_queue_time', np.mean(queue_times)])
            writer.writerow(['routing_accuracy', routing_accuracy])
            writer.writerow(['timestamp', datetime.now().isoformat()])

        print(f"\nMetrics saved to: {config.metrics.csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run toy model simulation")
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
    run_simulation(args.config)
    print("=" * 60)


if __name__ == "__main__":
    main()
