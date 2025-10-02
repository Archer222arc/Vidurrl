"""
Request generator for toy model simulation.

Implements Poisson arrival process with exponential inter-arrival times.
"""

import numpy as np
from typing import Iterator

from toymodel.src.entities import Request


class PoissonRequestGenerator:
    """
    Generate requests following Poisson arrival process.

    Supports multiple request types with independent arrival rates.
    """

    def __init__(
        self,
        arrival_rates: dict[int, float],
        seed: int = None,
    ):
        """
        Initialize Poisson request generator.

        Args:
            arrival_rates: Arrival rates for each request type {type: rate}
                          Example: {0: 6.0, 1: 6.0} means Type A and B both arrive at 6 req/sec
            seed: Random seed for reproducibility
        """
        self.arrival_rates = arrival_rates
        self.request_types = sorted(arrival_rates.keys())
        self.rng = np.random.RandomState(seed)
        self.request_counter = 0

    def generate_arrivals(self, max_time: float) -> list[Request]:
        """
        Generate arrival sequence up to max_time.

        Uses superposition property: independent Poisson processes can be merged.

        Args:
            max_time: Maximum simulation time

        Returns:
            List of requests sorted by arrival time
        """
        all_requests = []

        # Generate arrivals for each request type independently
        for request_type in self.request_types:
            rate = self.arrival_rates[request_type]
            requests = self._generate_poisson_arrivals(request_type, rate, max_time)
            all_requests.extend(requests)

        # Sort by arrival time (superposition)
        all_requests.sort(key=lambda r: r.arrival_time)

        # Assign sequential IDs
        for i, request in enumerate(all_requests):
            request.request_id = i

        return all_requests

    def _generate_poisson_arrivals(
        self,
        request_type: int,
        rate: float,
        max_time: float,
    ) -> list[Request]:
        """
        Generate Poisson arrivals for single request type.

        Args:
            request_type: Request type identifier
            rate: Arrival rate (requests per unit time)
            max_time: Maximum simulation time

        Returns:
            List of requests with exponential inter-arrival times
        """
        requests = []
        current_time = 0.0

        while current_time < max_time:
            # Exponential inter-arrival time with rate parameter
            inter_arrival = self.rng.exponential(scale=1.0 / rate)
            current_time += inter_arrival

            if current_time >= max_time:
                break

            request = Request(
                request_id=-1,  # Will be assigned later
                request_type=request_type,
                arrival_time=current_time,
            )
            requests.append(request)

        return requests

    def generate_stream(self, max_time: float) -> Iterator[Request]:
        """
        Generate arrival stream as iterator.

        More memory-efficient for large simulations.

        Args:
            max_time: Maximum simulation time

        Yields:
            Requests in arrival time order
        """
        requests = self.generate_arrivals(max_time)
        for request in requests:
            yield request

    def get_total_arrival_rate(self) -> float:
        """
        Get total system arrival rate (sum of all types).

        Returns:
            Total arrival rate
        """
        return sum(self.arrival_rates.values())

    def get_type_proportion(self, request_type: int) -> float:
        """
        Get proportion of arrivals for given type.

        Args:
            request_type: Request type

        Returns:
            Proportion in [0, 1]
        """
        total_rate = self.get_total_arrival_rate()
        return self.arrival_rates[request_type] / total_rate if total_rate > 0 else 0.0
