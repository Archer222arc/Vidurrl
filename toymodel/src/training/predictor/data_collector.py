"""
Data collector for neural latency predictor training.

Collects (state, latency) pairs using policy-independent simulation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from ...environment import QueueEnvironment
from ...entities import Request, Replica
from ...predictors import NeuralLatencyPredictor


class LatencyDataCollector:
    """
    Collect training data for latency predictor.
    
    Per-replica design: For each request, collects state features for ALL replicas,
    making the predictor policy-independent.
    """

    def __init__(self, config: Dict[str, Any], policy: str = "random"):
        """
        Initialize data collector.

        Args:
            config: Environment configuration
            policy: Data collection policy ("random", "round_robin", "mixed")
        """
        self.config = config
        self.policy = policy

        # Create environment
        self.env = QueueEnvironment(
            num_replicas=config.environment.num_replicas,
            service_rates=config.environment.service_rates,
            arrival_rates=config.environment.arrival_rates,
            max_time=config.environment.max_time,
            seed=config.experiment.seed
        )

        # Create predictor for feature extraction (not for prediction)
        self.predictor = NeuralLatencyPredictor(
            num_replicas=config.environment.num_replicas,
            num_request_types=len(config.environment.arrival_rates),
            max_queue_obs=128
        )

        # Data storage
        self.data_points = []
        self.round_robin_counter = 0

    def select_replica(self, request: Request, replicas: List[Replica]) -> int:
        """
        Select replica using specified policy.

        Args:
            request: Incoming request
            replicas: List of replicas

        Returns:
            Selected replica ID
        """
        if self.policy == "random":
            return np.random.randint(0, len(replicas))
        elif self.policy == "round_robin":
            replica_id = self.round_robin_counter % len(replicas)
            self.round_robin_counter += 1
            return replica_id
        elif self.policy == "mixed":
            # 50% random, 50% round-robin
            if np.random.rand() < 0.5:
                return np.random.randint(0, len(replicas))
            else:
                replica_id = self.round_robin_counter % len(replicas)
                self.round_robin_counter += 1
                return replica_id
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

    def collect_episode(self) -> int:
        """
        Collect data from one episode.

        For each incoming request, we collect data for ALL replicas (not just selected one).
        This makes predictor per-replica and policy-independent.

        Returns:
            Number of data points collected
        """
        self.env.reset()
        episode_data = 0
        pending_labels = []

        while True:
            # Get next request
            request = self.env.step_until_next_arrival()
            if request is None:
                break

            # For EACH replica, extract state
            for replica_id in range(len(self.env.replicas)):
                # Extract state features for this replica
                state_features = self.predictor._extract_state_features(
                    request, self.env.replicas[replica_id], self.env.replicas
                )

                # Store data point (will be labeled later)
                data_idx = len(self.data_points)
                self.data_points.append({
                    'state': state_features.numpy(),
                    'request_id': request.request_id,
                    'replica_id': replica_id,
                    'request_type': request.request_type,
                    'arrival_time': request.arrival_time,
                    'queue_length_at_arrival': len(self.env.replicas[replica_id].queue),
                    'labeled': False
                })

                # Track for labeling
                pending_labels.append({
                    'request_id': request.request_id,
                    'replica_id': replica_id,
                    'data_idx': data_idx,
                    'queue_snapshot': list(self.env.replicas[replica_id].queue)
                })

                episode_data += 1

            # Now actually route the request using the policy
            selected_replica = self.select_replica(request, self.env.replicas)
            self.env.route_request(request, selected_replica)

        # Process remaining requests
        self.env._drain_queues()

        # Label data points with actual latencies
        labeled = 0
        for pending in pending_labels:
            request_id = pending['request_id']
            replica_id = pending['replica_id']
            data_idx = pending['data_idx']

            # Find completed request
            completed_req = None
            for req in self.env.completed_requests:
                if req.request_id == request_id:
                    completed_req = req
                    break

            if completed_req is None:
                continue

            # Only label if request was actually routed to this replica
            if completed_req.assigned_replica == replica_id:
                # Actual measurement from simulation
                self_latency = completed_req.total_time
                avg_impact = completed_req.service_time

                self.data_points[data_idx]['self_latency'] = self_latency
                self.data_points[data_idx]['avg_impact'] = avg_impact
                self.data_points[data_idx]['total_latency'] = completed_req.total_time
                self.data_points[data_idx]['queue_time'] = completed_req.queue_time
                self.data_points[data_idx]['service_time'] = completed_req.service_time
                self.data_points[data_idx]['labeled'] = True
                labeled += 1

        return labeled

    def collect_data(self, num_episodes: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect data from multiple episodes.

        Args:
            num_episodes: Number of episodes to collect

        Returns:
            Tuple of (states, labels) arrays
        """
        print(f"Collecting data using '{self.policy}' policy...")
        print(f"Number of episodes: {num_episodes}")

        for episode in tqdm(range(num_episodes), desc="Collecting episodes"):
            labeled = self.collect_episode()

        # Extract labeled data
        states = []
        labels = []

        for data_point in self.data_points:
            if data_point['labeled']:
                states.append(data_point['state'])
                # Label is [self_latency, avg_impact]
                labels.append([
                    data_point['self_latency'],
                    data_point['avg_impact']
                ])

        states = np.array(states, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        print(f"\nData collection completed!")
        print(f"Total labeled samples: {len(states)}")
        print(f"State shape: {states.shape}")
        print(f"Label shape: {labels.shape}")

        return states, labels

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        labeled_data = [d for d in self.data_points if d['labeled']]
        
        if not labeled_data:
            return {}
        
        self_latencies = [d['self_latency'] for d in labeled_data]
        impacts = [d['avg_impact'] for d in labeled_data]
        
        return {
            'num_samples': len(labeled_data),
            'mean_self_latency': float(np.mean(self_latencies)),
            'std_self_latency': float(np.std(self_latencies)),
            'mean_impact': float(np.mean(impacts)),
            'std_impact': float(np.std(impacts))
        }
