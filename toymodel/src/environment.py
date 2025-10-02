"""
M/M/1 queueing environment for toy model simulation.

Event-driven simulation with routing control interface.
"""

import numpy as np
from typing import Optional, Callable

from toymodel.src.entities import Request, Replica
from toymodel.src.request_generator import PoissonRequestGenerator
from toymodel.src.tensorboard_monitor import ToyModelTensorBoardMonitor


class QueueEnvironment:
    """
    M/M/1 queueing environment with multiple replicas.

    Simulates exponential service times and supports external routing control.
    """

    def __init__(
        self,
        num_replicas: int,
        service_rates: dict[int, dict[int, float]],
        arrival_rates: dict[int, float],
        max_time: float = 1000.0,
        seed: int = None,
        tensorboard_enabled: bool = False,
        tensorboard_log_dir: str = "outputs/toymodel/tensorboard",
    ):
        """
        Initialize queueing environment.

        Args:
            num_replicas: Number of replicas (fixed at 2 for toy model)
            service_rates: Service rates {replica_id: {request_type: rate}}
                          Example: {0: {0: 10.0, 1: 5.0}, 1: {0: 5.0, 1: 10.0}}
            arrival_rates: Arrival rates {request_type: rate}
                          Example: {0: 6.0, 1: 6.0}
            max_time: Maximum simulation time
            seed: Random seed
            tensorboard_enabled: Enable TensorBoard monitoring
            tensorboard_log_dir: Directory for TensorBoard logs
        """
        self.num_replicas = num_replicas
        self.service_rates = service_rates
        self.arrival_rates = arrival_rates
        self.max_time = max_time
        self.rng = np.random.RandomState(seed)

        # TensorBoard monitor
        self.tb_monitor = None
        if tensorboard_enabled:
            from toymodel.src.config import load_config
            try:
                config = load_config('toymodel/configs/config.json')
                self.tb_monitor = ToyModelTensorBoardMonitor(
                    log_dir=tensorboard_log_dir,
                    enabled=tensorboard_enabled,
                    auto_start=tensorboard_enabled,
                    clean_previous_runs=config.tensorboard.clean_previous_runs,
                )
            except:
                # Fallback if config not available
                self.tb_monitor = ToyModelTensorBoardMonitor(
                    log_dir=tensorboard_log_dir,
                    enabled=tensorboard_enabled,
                    auto_start=tensorboard_enabled,
                    clean_previous_runs=True,
                )

        # Initialize replicas
        self.replicas = [
            Replica(
                replica_id=i,
                service_rates=service_rates[i],
                queue=[],
            )
            for i in range(num_replicas)
        ]

        # Request generator
        self.request_generator = PoissonRequestGenerator(
            arrival_rates=arrival_rates,
            seed=seed,
        )

        # Simulation state
        self.current_time = 0.0
        self.pending_arrivals = []
        self.completed_requests = []
        self.next_arrival_idx = 0

    def reset(self) -> None:
        """Reset environment to initial state."""
        self.current_time = 0.0
        self.next_arrival_idx = 0
        self.completed_requests = []

        # Generate arrival sequence
        self.pending_arrivals = self.request_generator.generate_arrivals(self.max_time)

        # Reset replicas
        for replica in self.replicas:
            replica.queue = []
            replica.current_request = None
            replica.busy_until = 0.0

    def step_until_next_arrival(self) -> Optional[Request]:
        """
        Advance simulation to next request arrival.

        Processes any service completions that occur before next arrival.

        Returns:
            Next arriving request, or None if simulation complete
        """
        if self.next_arrival_idx >= len(self.pending_arrivals):
            return None

        next_request = self.pending_arrivals[self.next_arrival_idx]
        next_arrival_time = next_request.arrival_time

        # Process all service completions before next arrival
        while True:
            earliest_completion, replica_idx = self._find_earliest_completion()

            if earliest_completion is None or earliest_completion > next_arrival_time:
                break

            # Advance time to completion
            self.current_time = earliest_completion
            self._complete_service(replica_idx)

        # Advance time to arrival
        self.current_time = next_arrival_time
        self.next_arrival_idx += 1

        return next_request

    def route_request(self, request: Request, replica_id: int) -> None:
        """
        Route request to specified replica.

        Args:
            request: Request to route
            replica_id: Target replica ID
        """
        replica = self.replicas[replica_id]
        replica.add_to_queue(request)

        # Try to start service immediately if replica is idle
        self._try_start_service(replica_id)

    def _try_start_service(self, replica_id: int) -> None:
        """
        Try to start service on replica if idle and queue non-empty.

        Args:
            replica_id: Replica to check
        """
        replica = self.replicas[replica_id]

        if replica.is_busy or replica.queue_length == 0:
            return

        # Pop from queue (FIFO)
        request = replica.queue.pop(0)

        # Generate service time (exponential distribution)
        service_rate = replica.get_service_rate(request.request_type)
        service_duration = self.rng.exponential(scale=1.0 / service_rate)

        # Start service
        replica.start_service(request, self.current_time, service_duration)

    def _complete_service(self, replica_id: int) -> None:
        """
        Complete service on replica.

        Args:
            replica_id: Replica completing service
        """
        replica = self.replicas[replica_id]

        completed = replica.complete_service(self.current_time)
        self.completed_requests.append(completed)

        # Log request completion metrics (every 5 requests for better visibility)
        if self.tb_monitor and len(self.completed_requests) % 5 == 0:
            self.tb_monitor.log_request_metrics(
                request_type=completed.request_type,
                assigned_replica=completed.assigned_replica,
                queue_time=completed.queue_time,
                service_time=completed.service_time,
                total_time=completed.total_time,
                step=len(self.completed_requests)
            )

            # Log queue state for all replicas
            for rep in self.replicas:
                self.tb_monitor.log_queue_metrics(
                    replica_id=rep.replica_id,
                    queue_length=rep.queue_length,
                    utilization=1.0 if rep.is_busy else 0.0,
                    step=len(self.completed_requests)
                )

        # Try to start next service
        self._try_start_service(replica_id)

    def _find_earliest_completion(self) -> tuple[Optional[float], Optional[int]]:
        """
        Find earliest service completion across all replicas.

        Returns:
            Tuple of (completion_time, replica_id), or (None, None) if all idle
        """
        earliest_time = None
        earliest_replica = None

        for replica in self.replicas:
            if replica.is_busy and replica.busy_until > self.current_time:
                if earliest_time is None or replica.busy_until < earliest_time:
                    earliest_time = replica.busy_until
                    earliest_replica = replica.replica_id

        return earliest_time, earliest_replica

    def run_simulation(
        self,
        routing_policy: Callable[[Request, list[Replica]], int],
    ) -> list[Request]:
        """
        Run complete simulation with given routing policy.

        Args:
            routing_policy: Function that takes (request, replicas) and returns replica_id

        Returns:
            List of completed requests
        """
        self.reset()

        # Log system state periodically
        log_interval = 20  # Log every 20 requests (more frequent for better visibility)

        while True:
            # Get next arrival
            next_request = self.step_until_next_arrival()

            if next_request is None:
                break

            # Apply routing policy
            replica_id = routing_policy(next_request, self.replicas)

            # Route request
            self.route_request(next_request, replica_id)

            # Periodic system state and aggregate metrics logging
            if self.tb_monitor and self.next_arrival_idx % log_interval == 0:
                total_in_system = sum(r.queue_length + (1 if r.is_busy else 0) for r in self.replicas)
                self.tb_monitor.log_system_state(
                    current_time=self.current_time,
                    total_requests_completed=len(self.completed_requests),
                    total_requests_in_system=total_in_system,
                    step=self.next_arrival_idx
                )

                # Log rolling aggregate metrics including routing accuracy
                if len(self.completed_requests) > 0:
                    self._log_rolling_metrics(step=self.next_arrival_idx)
                    self._log_routing_accuracy(step=self.next_arrival_idx)

        # Process remaining requests in queues
        self._drain_queues()

        # Log final aggregate metrics
        if self.tb_monitor:
            self._log_final_metrics()

        return self.completed_requests

    def _drain_queues(self) -> None:
        """Process all remaining requests in queues until empty."""
        max_drain_time = self.current_time + 10 * self.max_time  # Safety limit

        while self.current_time < max_drain_time:
            earliest_completion, replica_idx = self._find_earliest_completion()

            if earliest_completion is None:
                break

            self.current_time = earliest_completion
            self._complete_service(replica_idx)

    def _log_routing_accuracy(self, step: int) -> None:
        """Log routing accuracy and distribution metrics."""
        if not self.completed_requests or not self.tb_monitor:
            return

        # Calculate routing ratio by request type to each replica
        for req_type in [0, 1]:
            type_requests = [r for r in self.completed_requests if r.request_type == req_type]
            if type_requests:
                for replica_id in range(self.num_replicas):
                    type_to_replica = [r for r in type_requests if r.assigned_replica == replica_id]
                    routing_ratio = (len(type_to_replica) / len(type_requests)) * 100
                    if self.tb_monitor.writer:
                        self.tb_monitor.writer.add_scalar(
                            f"Routing/Type_{req_type}_to_Replica_{replica_id}_Ratio",
                            routing_ratio,
                            step
                        )

                # Calculate accuracy for this request type
                optimal_count = sum(1 for r in type_requests if r.assigned_replica == r.request_type)
                accuracy = (optimal_count / len(type_requests)) * 100
                if self.tb_monitor.writer:
                    self.tb_monitor.writer.add_scalar(
                        f"Routing/Type_{req_type}_Accuracy",
                        accuracy,
                        step
                    )

    def _log_rolling_metrics(self, step: int) -> None:
        """Log rolling aggregate metrics during simulation."""
        import numpy as np

        if not self.completed_requests:
            return

        # Use recent window for rolling metrics
        window_size = min(200, len(self.completed_requests))
        recent_requests = self.completed_requests[-window_size:]

        latencies = [r.total_time for r in recent_requests]
        queue_times = [r.queue_time for r in recent_requests]
        routing_accuracy = sum(
            1 for r in recent_requests if r.assigned_replica == r.request_type
        ) / len(recent_requests) * 100

        aggregate_metrics = {
            "MeanLatency": np.mean(latencies),
            "P50Latency": np.percentile(latencies, 50),
            "P99Latency": np.percentile(latencies, 99),
            "MeanQueueTime": np.mean(queue_times),
            "RoutingAccuracy": routing_accuracy,
            "TotalCompleted": len(self.completed_requests),
        }

        self.tb_monitor.log_aggregate_metrics(
            metrics=aggregate_metrics,
            step=step
        )

        # Log latency metrics by replica
        for replica_id in range(self.num_replicas):
            replica_requests = [r for r in recent_requests if r.assigned_replica == replica_id]
            if replica_requests:
                replica_latencies = [r.total_time for r in replica_requests]
                replica_queue_times = [r.queue_time for r in replica_requests]

                if self.tb_monitor.writer:
                    self.tb_monitor.writer.add_scalar(
                        f"Latency/Replica_{replica_id}_Mean",
                        np.mean(replica_latencies),
                        step
                    )
                    self.tb_monitor.writer.add_scalar(
                        f"Latency/Replica_{replica_id}_P99",
                        np.percentile(replica_latencies, 99),
                        step
                    )
                    self.tb_monitor.writer.add_scalar(
                        f"QueueTime/Replica_{replica_id}_Mean",
                        np.mean(replica_queue_times),
                        step
                    )

    def _log_final_metrics(self) -> None:
        """Log final aggregate metrics to TensorBoard."""
        import numpy as np

        if not self.completed_requests:
            return

        latencies = [r.total_time for r in self.completed_requests]
        queue_times = [r.queue_time for r in self.completed_requests]
        routing_accuracy = sum(
            1 for r in self.completed_requests if r.assigned_replica == r.request_type
        ) / len(self.completed_requests) * 100

        aggregate_metrics = {
            "MeanLatency": np.mean(latencies),
            "P50Latency": np.percentile(latencies, 50),
            "P99Latency": np.percentile(latencies, 99),
            "MeanQueueTime": np.mean(queue_times),
            "RoutingAccuracy": routing_accuracy,
            "TotalCompleted": len(self.completed_requests),
        }

        self.tb_monitor.log_aggregate_metrics(
            metrics=aggregate_metrics,
            step=len(self.completed_requests)
        )

    def close(self) -> None:
        """Clean up resources."""
        if self.tb_monitor:
            self.tb_monitor.close()

    def get_replica_state(self, replica_id: int) -> dict:
        """
        Get current state of replica.

        Args:
            replica_id: Replica ID

        Returns:
            Dictionary with queue_length, utilization, busy_until
        """
        replica = self.replicas[replica_id]
        return {
            "queue_length": replica.queue_length,
            "utilization": 1.0 if replica.busy_until > self.current_time else 0.0,
            "busy_until": replica.busy_until,
            "is_busy": replica.is_busy,
        }

    def get_system_state(self) -> dict:
        """
        Get current system-wide state.

        Returns:
            Dictionary with global statistics
        """
        total_queue = sum(r.queue_length for r in self.replicas)
        total_busy = sum(1 for r in self.replicas if r.is_busy)

        return {
            "current_time": self.current_time,
            "total_queue_length": total_queue,
            "num_busy_replicas": total_busy,
            "num_completed": len(self.completed_requests),
            "num_remaining_arrivals": len(self.pending_arrivals) - self.next_arrival_idx,
        }
