#!/usr/bin/env python3
"""
Demo script for toy model environment.

Demonstrates basic usage of QueueEnvironment with different schedulers.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from toymodel.src.config import load_config
from toymodel.src.environment import QueueEnvironment
from toymodel.schedulers import (
    OracleScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    ShortestQueueScheduler,
)


def run_scheduler(scheduler_name, scheduler, config):
    """Run simulation with given scheduler."""
    env = QueueEnvironment(
        num_replicas=config.environment.num_replicas,
        service_rates=config.environment.service_rates,
        arrival_rates=config.environment.arrival_rates,
        max_time=config.environment.max_time,
        seed=config.experiment.seed,
    )

    def routing_policy(request, replicas):
        return scheduler.schedule(request, replicas)

    completed = env.run_simulation(routing_policy)

    # Compute metrics
    latencies = [r.total_time for r in completed]
    queue_times = [r.queue_time for r in completed]

    routing_accuracy = sum(
        1 for r in completed if r.assigned_replica == r.request_type
    ) / len(completed) * 100

    print(f"\n{scheduler_name}:")
    print(f"  Requests completed: {len(completed)}")
    print(f"  Mean latency: {np.mean(latencies):.4f}")
    print(f"  P50 latency: {np.percentile(latencies, 50):.4f}")
    print(f"  P99 latency: {np.percentile(latencies, 99):.4f}")
    print(f"  Mean queue time: {np.mean(queue_times):.4f}")
    print(f"  Routing accuracy: {routing_accuracy:.1f}%")


def main():
    """Run demo with multiple schedulers."""
    print("=" * 60)
    print("Toy Model Environment Demo")
    print("=" * 60)

    # Load configuration from JSON
    config = load_config('configs/toymodel/config.json')

    print(f"\nExperiment: {config.experiment.name}")
    print(f"Description: {config.experiment.description}")
    print("\nConfiguration:")
    print(f"  Replicas: {config.environment.num_replicas}")
    print(f"  Service rates (Replica 0): Type A={config.environment.service_rates[0][0]}, Type B={config.environment.service_rates[0][1]}")
    print(f"  Service rates (Replica 1): Type A={config.environment.service_rates[1][0]}, Type B={config.environment.service_rates[1][1]}")
    print(f"  Arrival rates: Type A={config.environment.arrival_rates[0]}, Type B={config.environment.arrival_rates[1]}")
    print(f"  Simulation time: {config.environment.max_time}")
    print(f"  Scheduler: {config.scheduler.type}")

    # Run with different schedulers
    schedulers = [
        ("Oracle (Optimal)", OracleScheduler(num_replicas=2)),
        ("Random", RandomScheduler(num_replicas=2, seed=42)),
        ("Round-Robin", RoundRobinScheduler(num_replicas=2)),
        ("Shortest Queue", ShortestQueueScheduler(num_replicas=2)),
    ]

    for name, scheduler in schedulers:
        run_scheduler(name, scheduler, config)

    print("\n" + "=" * 60)
    print("Note: Oracle achieves 100% routing accuracy and lowest latency")
    print("Goal: PPO should learn to match Oracle performance")
    print("=" * 60)


if __name__ == "__main__":
    main()
