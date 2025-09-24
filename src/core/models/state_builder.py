"""
State building components for reinforcement learning schedulers.

This module provides functionality to convert complex scheduler states
into feature vectors suitable for neural network processing with enhanced
state features for capturing high-frequency variations.
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from vidur.entities import Replica, Request


class MetricStore(Protocol):
    """Protocol for metric store interface."""

    def get_throughput(self, current_time: float) -> float:
        """Get current throughput."""
        ...

    def get_average_latency(self) -> float:
        """Get average latency."""
        ...


def safe_get(obj: Any, name: str, default: Any) -> Any:
    """Direct attribute access - no fallback allowed per CLAUDE.md regulations."""
    return getattr(obj, name)


def get_replica_queue(replica_scheduler: Any) -> List[Request]:
    """
    Get request queue from replica scheduler.

    Direct access to _request_queue - no fallback allowed per CLAUDE.md regulations.
    """
    return list(replica_scheduler._request_queue)


def normalize_priority(priority: Any) -> float:
    """
    Normalize priority value to float.

    Direct access to priority value - no fallback allowed per CLAUDE.md regulations.
    """
    if priority is None:
        return 0.0

    # Handle Enum values - direct access to .value attribute
    if hasattr(priority, "value"):
        return float(priority.value)

    # Handle numeric types directly
    if isinstance(priority, (int, float)):
        return float(priority)

    # Handle string priorities - direct mapping
    priority_str = str(priority).strip().lower()
    priority_map = {
        "low": 1.0, "normal": 2.0, "med": 2.0, "medium": 2.0,
        "high": 3.0, "urgent": 4.0, "critical": 5.0,
    }
    # Direct access - if key doesn't exist, let it fail
    return priority_map[priority_str]


def get_request_priority(request: Any) -> Any:
    """
    Get priority from request - explicit check for attribute existence.

    No fallback allowed per CLAUDE.md - if priority doesn't exist, returns None.
    """
    # Explicit check for attribute existence
    if hasattr(request, 'priority'):
        return request.priority
    return None


class StateBuilder:
    """
    Builds enhanced state vectors for reinforcement learning from scheduler state.

    Converts complex scheduler state into normalized feature vectors with
    enhanced features for capturing high-frequency variations and trends.
    """

    def __init__(
        self,
        max_queue_requests: int = 4,
        history_window: int = 5,
        qps_window: int = 10,
        enable_enhanced_features: bool = True,
        enable_queue_delay_features: bool = False,
        queue_delay_normalization: Optional[Dict[str, float]] = None
    ):
        """
        Initialize enhanced state builder.

        Args:
            max_queue_requests: Maximum number of requests to include in state
            history_window: Number of historical steps to track for each replica
            qps_window: Window size for QPS computation (in steps)
            enable_enhanced_features: Whether to enable enhanced state features
            enable_queue_delay_features: Whether to enable queue delay features
            queue_delay_normalization: Normalization parameters for queue delay features
        """
        self.max_queue_requests = max_queue_requests
        self.history_window = history_window
        self.qps_window = qps_window
        self.enable_enhanced_features = enable_enhanced_features

        # NEW: Queue delay features configuration
        self.enable_queue_delay_features = enable_queue_delay_features
        self.queue_delay_normalization = queue_delay_normalization or {
            "max_wait_time_seconds": 10.0,
            "urgency_scale": 10.0,
            "priority_weight": 1.0
        }

        # Historical tracking
        self._replica_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self._queue_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self._global_history: deque = deque(maxlen=qps_window)
        self._request_arrival_times: deque = deque(maxlen=qps_window * 2)

        # Previous state for delta computation
        self._prev_replica_states: Dict[int, Dict[str, float]] = defaultdict(dict)
        self._prev_global_state: Dict[str, float] = {}

        # Step counter
        self._step_count: int = 0

    def build_replica_state(
        self,
        replica_id: int,
        replica_scheduler: Any,
        current_time: float
    ) -> List[float]:
        """
        Build enhanced state vector for single replica.

        Returns:
            State vector with:
            - 11 base resource/runtime features
            - Enhanced features (if enabled): historical load, queue variations, trends
            - K request features (7 dimensions each)
        """
        vec: List[float] = []
        request_queue = get_replica_queue(replica_scheduler)

        # Resource utilization features
        num_alloc = int(safe_get(replica_scheduler, "_num_allocated_blocks", 0))
        num_running_batches = int(safe_get(replica_scheduler, "_num_running_batches", 0))
        preempted_requests = safe_get(replica_scheduler, "_preempted_requests", [])
        allocation_map = safe_get(replica_scheduler, "_allocation_map", {})

        # Configuration features
        config = safe_get(replica_scheduler, "_config", None)
        num_blocks = int(getattr(config, "num_blocks", 1)) if config else 1
        num_blocks = max(1, num_blocks)
        block_size = int(getattr(config, "block_size", 1)) if config else 1
        batch_cap = int(getattr(config, "batch_size_cap", 1)) if config else 1
        num_stages = int(safe_get(replica_scheduler, "_num_stages", 0))

        # Computed features
        alloc_frac = float(num_alloc) / float(num_blocks)
        avail_frac = float(max(num_blocks - num_alloc, 0)) / float(num_blocks)
        queue_len = float(len(request_queue))

        # ENHANCED: Add queue delay features as recommended in PDF
        oldest_request_wait_time = self._get_oldest_request_wait_time(request_queue, current_time)
        avg_queue_wait_time = self._get_average_queue_wait_time(request_queue, current_time)
        queue_urgency_score = self._compute_queue_urgency(request_queue, current_time)

        # Base features (14 dimensions - EXPANDED from 11)
        vec.extend([
            queue_len,                         # Queue length
            float(num_alloc),                  # Allocated blocks
            float(num_blocks),                 # Total blocks
            alloc_frac,                        # Utilization fraction
            avail_frac,                        # Available fraction
            float(num_running_batches),        # Running batches
            float(len(preempted_requests)),    # Preempted requests
            float(len(allocation_map)),        # Allocation map size
            float(batch_cap),                  # Batch capacity
            float(block_size),                 # Block size
            float(num_stages),                 # Number of stages
            oldest_request_wait_time,          # NEW: Oldest request waiting time (PDF recommendation)
            avg_queue_wait_time,               # NEW: Average queue waiting time
            queue_urgency_score,               # NEW: Queue urgency indicator
        ])

        # Enhanced features for high-frequency dynamics
        if self.enable_enhanced_features:
            enhanced_features = self._build_enhanced_replica_features(
                replica_id, queue_len, alloc_frac, num_running_batches, current_time
            )
            vec.extend(enhanced_features)

        # Request features (7 dimensions per request)
        for i in range(self.max_queue_requests):
            if i < len(request_queue):
                req = request_queue[i]
                vec.extend(self._extract_request_features(req, current_time))
            else:
                # Pad with zeros
                vec.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return vec

    def _get_oldest_request_wait_time(self, request_queue: List[Request], current_time: float) -> float:
        """
        Get waiting time of the oldest request in queue (PDF recommendation).

        This feature helps the agent infer urgency and avoid sending new traffic
        to an overloaded replica.
        """
        if not request_queue:
            return 0.0

        oldest_request = request_queue[0]  # Assuming queue is FIFO
        # Direct attribute access - no fallback allowed per CLAUDE.md regulations
        arrived_at = float(oldest_request.arrived_at)
        wait_time = max(0.0, float(current_time) - arrived_at)

        # IMPROVED: Use adaptive normalization with softer scaling
        # Log scaling to better distinguish short vs medium wait times
        if wait_time <= 0.1:
            normalized_wait = wait_time * 10.0  # Linear scaling for very short waits
        else:
            # Log scaling for longer waits: log(1 + wait_time) / log(1 + max_time)
            max_expected_wait = self.queue_delay_normalization["max_wait_time_seconds"]
            normalized_wait = min(
                np.log(1.0 + wait_time) / np.log(1.0 + max_expected_wait),
                1.0
            )
        return float(normalized_wait)

    def _get_average_queue_wait_time(self, request_queue: List[Request], current_time: float) -> float:
        """
        Get average waiting time across all requests in queue.

        Provides a sense of overall queue pressure beyond just queue length.
        """
        if not request_queue:
            return 0.0

        total_wait = 0.0
        valid_requests = 0

        for request in request_queue:
            # Direct attribute access - no fallback allowed per CLAUDE.md regulations
            arrived_at = float(request.arrived_at)
            wait_time = max(0.0, float(current_time) - arrived_at)
            total_wait += wait_time
            valid_requests += 1

        if valid_requests == 0:
            return 0.0

        avg_wait = total_wait / valid_requests

        # IMPROVED: More sensitive normalization for average wait times
        # Use square root scaling to better distinguish moderate differences
        max_expected_wait = self.queue_delay_normalization["max_wait_time_seconds"]
        if avg_wait <= 0.1:
            normalized_avg_wait = avg_wait * 10.0  # Linear for very short
        else:
            # Square root scaling for better sensitivity in mid-range
            normalized_avg_wait = min(
                np.sqrt(avg_wait) / np.sqrt(max_expected_wait),
                1.0
            )
        return float(normalized_avg_wait)

    def _compute_queue_urgency(self, request_queue: List[Request], current_time: float) -> float:
        """
        Compute urgency score based on request priorities and wait times.

        High urgency indicates the replica needs immediate attention.
        """
        if not request_queue:
            return 0.0

        urgency_score = 0.0
        for request in request_queue:
            # Direct attribute access - no fallback allowed per CLAUDE.md regulations
            priority = normalize_priority(get_request_priority(request))

            # Get wait time
            arrived_at = float(request.arrived_at)
            wait_time = max(0.0, float(current_time) - arrived_at)

            # Urgency = priority * wait_time_factor
            wait_factor = min(wait_time / 5.0, 2.0)  # Cap at 2x urgency
            request_urgency = priority * (1.0 + wait_factor)
            urgency_score += request_urgency

        # Normalize by queue length and cap at 1.0
        avg_urgency = urgency_score / len(request_queue) if request_queue else 0.0
        normalized_urgency = min(avg_urgency / 10.0, 1.0)  # Assuming max urgency ~10
        return normalized_urgency

    def _extract_request_features(self, request: Request, current_time: float) -> List[float]:
        """
        Extract features from a single request.

        Returns 7-dimensional feature vector:
        [age, num_prefill, num_processed, remaining_prefill, completed, num_decode, priority]
        """
        # Calculate request age
        # Direct attribute access - no fallback allowed per CLAUDE.md regulations
        arrived_at = float(request.arrived_at)
        age = max(0.0, float(current_time) - arrived_at)

        # Token processing features
        num_prefill = float(request.num_prefill_tokens)
        num_processed = float(request.num_processed_tokens)
        remaining_prefill = max(0.0, num_prefill - num_processed)

        # Completion status
        completed = 1.0 if bool(request.completed) else 0.0
        num_decode = float(request.num_decode_tokens)

        # Priority normalization
        priority = normalize_priority(get_request_priority(request))

        return [
            age,
            num_prefill,
            num_processed,
            remaining_prefill,
            completed,
            num_decode,
            priority,
        ]

    def _build_enhanced_replica_features(
        self,
        replica_id: int,
        queue_len: float,
        alloc_frac: float,
        num_running_batches: float,
        current_time: float
    ) -> List[float]:
        """
        Build enhanced features for capturing high-frequency dynamics.

        Returns 8-dimensional feature vector:
        [queue_delta, alloc_delta, load_ema, queue_trend, queue_variance,
         historical_peak, historical_low, time_since_peak]
        """
        # Current state snapshot
        current_state = {
            "queue_len": queue_len,
            "alloc_frac": alloc_frac,
            "num_batches": num_running_batches,
            "timestamp": current_time
        }

        # Get previous state for delta computation - direct access
        prev_state = self._prev_replica_states[replica_id] if replica_id in self._prev_replica_states else {}

        # Delta features (change from previous step)
        queue_delta = queue_len - (prev_state["queue_len"] if "queue_len" in prev_state else queue_len)
        alloc_delta = alloc_frac - (prev_state["alloc_frac"] if "alloc_frac" in prev_state else alloc_frac)

        # Update history tracking
        self._replica_history[replica_id].append(current_state)
        self._queue_history[replica_id].append(queue_len)

        # Historical analysis features
        load_ema = self._compute_load_ema(replica_id, alloc_frac)
        queue_trend = self._compute_queue_trend(replica_id)
        queue_variance = self._compute_queue_variance(replica_id)

        # Peak/valley detection
        historical_peak, historical_low, time_since_peak = self._analyze_historical_peaks(
            replica_id, queue_len, current_time
        )

        # Update previous state
        self._prev_replica_states[replica_id] = current_state

        return [
            queue_delta,          # Queue length change
            alloc_delta,          # Allocation change
            load_ema,             # Exponential moving average of load
            queue_trend,          # Queue length trend (slope)
            queue_variance,       # Queue length variance
            historical_peak,      # Historical peak queue length
            historical_low,       # Historical low queue length
            time_since_peak,      # Time since last peak (normalized)
        ]

    def _compute_load_ema(self, replica_id: int, current_load: float) -> float:
        """Compute exponential moving average of replica load."""
        history = self._replica_history[replica_id]
        if not history:
            return current_load

        # EMA with alpha = 0.3 for responsiveness to recent changes
        alpha = 0.3
        ema = current_load
        for i, state in enumerate(reversed(history)):
            if i == 0:
                continue  # Skip current state
            weight = alpha * ((1 - alpha) ** i)
            ema += weight * state["alloc_frac"]

        return ema

    def _compute_queue_trend(self, replica_id: int) -> float:
        """Compute queue length trend using linear regression slope."""
        queue_hist = list(self._queue_history[replica_id])
        if len(queue_hist) < 2:
            return 0.0

        # Simple linear trend (slope)
        n = len(queue_hist)
        x_sum = sum(range(n))
        y_sum = sum(queue_hist)
        xy_sum = sum(i * queue_hist[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))

        if n * x2_sum - x_sum * x_sum == 0:
            return 0.0

        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope

    def _compute_queue_variance(self, replica_id: int) -> float:
        """Compute queue length variance for volatility measurement."""
        queue_hist = list(self._queue_history[replica_id])
        if len(queue_hist) < 2:
            return 0.0

        mean_queue = sum(queue_hist) / len(queue_hist)
        variance = sum((q - mean_queue) ** 2 for q in queue_hist) / len(queue_hist)
        return variance

    def _analyze_historical_peaks(
        self,
        replica_id: int,
        current_queue: float,
        current_time: float
    ) -> Tuple[float, float, float]:
        """Analyze historical peaks and valleys in queue length."""
        history = self._replica_history[replica_id]
        if not history:
            return (current_queue, current_queue, 0.0)

        queue_values = [state["queue_len"] for state in history]
        timestamps = [state["timestamp"] for state in history]

        historical_peak = max(queue_values) if queue_values else current_queue
        historical_low = min(queue_values) if queue_values else current_queue

        # Find time since last peak
        time_since_peak = 0.0
        for i, (queue_val, timestamp) in enumerate(zip(reversed(queue_values), reversed(timestamps))):
            if queue_val >= historical_peak * 0.9:  # Within 90% of peak
                time_since_peak = current_time - timestamp
                break

        # Normalize time_since_peak (assume 60 seconds is "long ago")
        normalized_time = min(time_since_peak / 60.0, 1.0)

        return (historical_peak, historical_low, normalized_time)

    def build_global_state(
        self,
        replicas: Dict[int, Replica],
        get_replica_scheduler_fn,
        current_time: float,
        metric_store: Optional[MetricStore] = None,
    ) -> np.ndarray:
        """
        Build enhanced global state vector from all replicas.

        Returns:
            Global state = replica_states + global_features + enhanced_global_features
        """
        state_parts: List[float] = []

        # Build state for each replica (sorted by ID for consistency)
        for replica_id in sorted(replicas.keys()):
            replica_scheduler = get_replica_scheduler_fn(replica_id)
            replica_state = self.build_replica_state(
                replica_id, replica_scheduler, current_time
            )
            state_parts.extend(replica_state)

        # Base global features
        global_queue_len = self._get_global_queue_length(get_replica_scheduler_fn)
        throughput, latency = self._get_metrics(metric_store, current_time)

        state_parts.extend([global_queue_len, throughput, latency])

        # Enhanced global features for QPS and system dynamics
        if self.enable_enhanced_features:
            enhanced_global = self._build_enhanced_global_features(
                global_queue_len, throughput, latency, current_time, len(replicas)
            )
            state_parts.extend(enhanced_global)

        # Increment step counter for tracking
        self._step_count += 1

        return np.asarray(state_parts, dtype=np.float32)

    def _get_global_queue_length(self, get_replica_scheduler_fn) -> float:
        """Get global queue length from scheduler instance."""
        # Direct access to scheduler - no fallback allowed per CLAUDE.md regulations
        scheduler = get_replica_scheduler_fn.__self__
        request_queue = scheduler._request_queue
        return float(len(request_queue))

    def _get_metrics(
        self,
        metric_store: Optional[MetricStore],
        current_time: float
    ) -> tuple[float, float]:
        """Get throughput and latency metrics."""
        if metric_store is None:
            return (0.0, 0.0)

        # Direct access to metrics - no fallback allowed per CLAUDE.md regulations
        throughput = float(metric_store.get_throughput(current_time))
        latency = float(metric_store.get_average_latency())

        return (throughput, latency)

    def _build_enhanced_global_features(
        self,
        global_queue_len: float,
        throughput: float,
        latency: float,
        current_time: float,
        num_replicas: int
    ) -> List[float]:
        """
        Build enhanced global features for system-level dynamics.

        Returns 7-dimensional feature vector:
        [qps_current, qps_ema, qps_trend, qps_variance, system_load_balance,
         global_queue_delta, completion_rate]
        """
        # Track request arrivals for QPS computation
        self._request_arrival_times.append(current_time)

        # Current global state
        current_global = {
            "global_queue": global_queue_len,
            "throughput": throughput,
            "latency": latency,
            "timestamp": current_time
        }

        # Update global history
        self._global_history.append(current_global)

        # QPS Features
        qps_current = self._compute_current_qps(current_time)
        qps_ema = self._compute_qps_ema(qps_current)
        qps_trend = self._compute_qps_trend()
        qps_variance = self._compute_qps_variance()

        # System-level features
        system_load_balance = self._compute_load_balance_score()
        global_queue_delta = self._compute_global_queue_delta(global_queue_len)
        completion_rate = self._compute_completion_rate(throughput)

        return [
            qps_current,          # Current QPS (requests per second)
            qps_ema,              # QPS exponential moving average
            qps_trend,            # QPS trend (increasing/decreasing)
            qps_variance,         # QPS variance (volatility)
            system_load_balance,  # Load balance across replicas
            global_queue_delta,   # Global queue length change
            completion_rate,      # Request completion rate
        ]

    def _compute_current_qps(self, current_time: float) -> float:
        """Compute current QPS based on recent request arrivals."""
        if len(self._request_arrival_times) < 2:
            return 0.0

        # Count requests in the last second
        recent_arrivals = [
            t for t in self._request_arrival_times
            if current_time - t <= 1.0
        ]

        return float(len(recent_arrivals))

    def _compute_qps_ema(self, current_qps: float) -> float:
        """Compute exponential moving average of QPS."""
        if not self._global_history:
            return current_qps

        # Extract historical QPS values
        alpha = 0.2  # Smoothing factor for QPS
        ema = current_qps

        for i, state in enumerate(reversed(self._global_history)):
            if i == 0:
                continue  # Skip current
            # Approximate historical QPS from throughput
            hist_qps = state["throughput"]
            weight = alpha * ((1 - alpha) ** i)
            ema += weight * hist_qps

        return ema

    def _compute_qps_trend(self) -> float:
        """Compute QPS trend using recent global history."""
        if len(self._global_history) < 3:
            return 0.0

        # Use throughput as proxy for QPS trend
        recent_throughput = [
            state["throughput"]
            for state in list(self._global_history)[-3:]
        ]

        if len(recent_throughput) < 2:
            return 0.0

        # Simple linear trend
        n = len(recent_throughput)
        x_sum = sum(range(n))
        y_sum = sum(recent_throughput)
        xy_sum = sum(i * recent_throughput[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))

        if n * x2_sum - x_sum * x_sum == 0:
            return 0.0

        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope

    def _compute_qps_variance(self) -> float:
        """Compute QPS variance for demand volatility measurement."""
        if len(self._global_history) < 2:
            return 0.0

        throughput_values = [
            state["throughput"]
            for state in self._global_history
        ]

        mean_throughput = sum(throughput_values) / len(throughput_values)
        variance = sum(
            (tp - mean_throughput) ** 2 for tp in throughput_values
        ) / len(throughput_values)

        return variance

    def _compute_load_balance_score(self) -> float:
        """
        Compute load balance score across replicas.

        Uses coefficient of variation of replica queue lengths.
        Lower values indicate better balance.
        """
        replica_queues = []
        for replica_id in self._queue_history:
            if self._queue_history[replica_id]:
                recent_queue = list(self._queue_history[replica_id])[-1]
                replica_queues.append(recent_queue)

        if len(replica_queues) < 2:
            return 1.0  # Perfect balance with single replica

        mean_queue = sum(replica_queues) / len(replica_queues)
        if mean_queue == 0:
            return 1.0  # Perfect balance when all queues empty

        # Coefficient of variation (normalized standard deviation)
        variance = sum((q - mean_queue) ** 2 for q in replica_queues) / len(replica_queues)
        std_dev = variance ** 0.5
        cv = std_dev / mean_queue

        # Convert to balance score (higher = better balance)
        balance_score = 1.0 / (1.0 + cv)
        return balance_score

    def _compute_global_queue_delta(self, current_global_queue: float) -> float:
        """Compute change in global queue length."""
        prev_global = self._prev_global_state["global_queue"] if "global_queue" in self._prev_global_state else current_global_queue
        delta = current_global_queue - prev_global

        # Update previous state
        self._prev_global_state["global_queue"] = current_global_queue

        return delta

    def _compute_completion_rate(self, current_throughput: float) -> float:
        """
        Compute request completion rate as ratio of throughput to arrival rate.

        Values > 1.0 indicate system is catching up,
        Values < 1.0 indicate system is falling behind.
        """
        current_qps = self._compute_current_qps(
            self._global_history[-1]["timestamp"] if self._global_history else 0.0
        )

        if current_qps == 0:
            return 1.0  # No arrivals, perfect completion rate

        completion_rate = current_throughput / current_qps
        return min(completion_rate, 2.0)  # Cap at 2.0 for numerical stability

    def get_state_dimension(self, num_replicas: int) -> int:
        """
        Calculate total state dimension with enhanced features.

        Args:
            num_replicas: Number of replicas in the system

        Returns:
            Total dimension of state vector
        """
        # Base replica features (UPDATED: increased from 11 to 14)
        replica_base_features = 14

        # Enhanced replica features (if enabled)
        enhanced_replica_features = 8 if self.enable_enhanced_features else 0

        # Request features per replica
        request_features_per_replica = self.max_queue_requests * 7

        # Total per replica
        replica_state_dim = (
            replica_base_features +
            enhanced_replica_features +
            request_features_per_replica
        )

        # Base global features
        base_global_features = 3  # global_queue_len, throughput, latency

        # Enhanced global features (if enabled)
        enhanced_global_features = 7 if self.enable_enhanced_features else 0

        total_global_features = base_global_features + enhanced_global_features

        # NOTE: Queue delay features are already included in replica_base_features (14 total)
        # No need to add them separately here
        return num_replicas * replica_state_dim + total_global_features

    def get_feature_names(self, num_replicas: int) -> List[str]:
        """
        Get descriptive names for all state features.

        Args:
            num_replicas: Number of replicas in the system

        Returns:
            List of feature names corresponding to state dimensions
        """
        feature_names = []

        # Replica features for each replica
        for replica_id in range(num_replicas):
            prefix = f"replica_{replica_id}_"

            # Base replica features (UPDATED: added 3 new queue delay features)
            feature_names.extend([
                f"{prefix}queue_length",
                f"{prefix}allocated_blocks",
                f"{prefix}total_blocks",
                f"{prefix}utilization_fraction",
                f"{prefix}available_fraction",
                f"{prefix}running_batches",
                f"{prefix}preempted_requests",
                f"{prefix}allocation_map_size",
                f"{prefix}batch_capacity",
                f"{prefix}block_size",
                f"{prefix}num_stages",
                f"{prefix}oldest_request_wait_time",  # NEW: PDF recommendation
                f"{prefix}avg_queue_wait_time",       # NEW: Queue pressure indicator
                f"{prefix}queue_urgency_score",       # NEW: Priority-weighted urgency
            ])

            # Enhanced replica features
            if self.enable_enhanced_features:
                feature_names.extend([
                    f"{prefix}queue_delta",
                    f"{prefix}allocation_delta",
                    f"{prefix}load_ema",
                    f"{prefix}queue_trend",
                    f"{prefix}queue_variance",
                    f"{prefix}historical_peak",
                    f"{prefix}historical_low",
                    f"{prefix}time_since_peak",
                ])

            # Request features
            for req_idx in range(self.max_queue_requests):
                req_prefix = f"{prefix}request_{req_idx}_"
                feature_names.extend([
                    f"{req_prefix}age",
                    f"{req_prefix}num_prefill",
                    f"{req_prefix}num_processed",
                    f"{req_prefix}remaining_prefill",
                    f"{req_prefix}completed",
                    f"{req_prefix}num_decode",
                    f"{req_prefix}priority",
                ])

        # Global features
        feature_names.extend([
            "global_queue_length",
            "global_throughput",
            "global_latency",
        ])

        # Enhanced global features
        if self.enable_enhanced_features:
            feature_names.extend([
                "qps_current",
                "qps_ema",
                "qps_trend",
                "qps_variance",
                "system_load_balance",
                "global_queue_delta",
                "completion_rate",
            ])

        return feature_names

    def reset_history(self) -> None:
        """Reset all historical tracking for new training episodes."""
        self._replica_history.clear()
        self._queue_history.clear()
        self._global_history.clear()
        self._request_arrival_times.clear()
        self._prev_replica_states.clear()
        self._prev_global_state.clear()
        self._step_count = 0